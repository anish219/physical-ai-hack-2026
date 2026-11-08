"""Client of the already-running arm services. Never opens CAN or cameras."""
import json
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
import math
import time
import urllib.request


class ExistingBridge:
    def __init__(self, base='http://127.0.0.1', right_port=8692, left_port=8691, dashboard_port=8081):
        self.right = f'{base}:{right_port}'
        self.left = f'{base}:{left_port}'
        self.dashboard = f'{base}:{dashboard_port}'
        self.connected = True
        self.claims = {'left': False, 'right': False}
        self.control_locks = {side: threading.RLock() for side in ('left', 'right')}
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.frame_locks = {role: threading.Lock() for role in ('wrist_l', 'wrist_r', 'overhead')}
        self.preview_cache = {}

    def request(self, base, path, body=None):
        req = urllib.request.Request(base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read())

    def connect(self):
        self.request(self.right, '/state')
        self.connected = True
        return self.status()

    def disconnect(self):
        # Disconnect the dashboard, not the underlying powered arm/service.
        self.connected = False
        return self.status()

    def states(self):
        def read(base):
            try:
                return self.request(base, '/state')
            except Exception as error:
                return {'arm_open': False, 'error': str(error), 'stamp': 0}
        left = self.pool.submit(read, self.left)
        right = self.pool.submit(read, self.right)
        return {'left': left.result(), 'right': right.result()}

    def assignment(self):
        station = self.request(self.dashboard, '/api/state')
        if 'assigned_side' not in station or station['assigned_side'] not in (None, 'left', 'right'):
            raise RuntimeError('Unable to determine leader assignment')
        return station['assigned_side']

    def status(self):
        arms = self.states()
        try:
            assigned, assignment_error = self.assignment(), None
        except Exception as error:
            assigned, assignment_error = None, str(error)
        access = {}
        for side, st in arms.items():
            with self.control_locks[side]:
                if assignment_error or assigned == side:
                    self.claims[side] = False
                controlled = self.claims[side]
                owner = 'unknown' if assignment_error else 'leader' if assigned == side else 'dashboard' if controlled else 'available'
                access[side] = {'powered': st['arm_open'], 'fresh': self.fresh(st), 'error': st.get('error'),
                                'controlled': controlled, 'owner': owner}
        return dict(mode='existing-service', side='right', connected=self.connected,
                    armPowered=arms['right']['arm_open'], motionEnabled=False,
                    controlsAvailable=True, arms=access, assignmentError=assignment_error)

    def telemetry(self):
        if not self.connected:
            raise RuntimeError('Dashboard not connected')
        states = self.states()
        def positions(st):
            return [math.degrees(v) for v in st['q']] if self.fresh(st) else None
        return dict(arm01=positions(states['left']), arm02=positions(states['right']),
                    timestamps={side: st.get('stamp') for side, st in states.items()},
                    timestamp=states['right'].get('stamp'), units='degrees', simulated=False,
                    armPowered=states['right']['arm_open'], stale=not self.fresh(states['right']))

    @staticmethod
    def fresh(st):
        q = st.get('q')
        age = time.time() - st.get('stamp', 0)
        return (st.get('arm_open') is True and 0 <= age <= 1 and isinstance(q, list)
                and len(q) == 6 and all(type(v) in (int, float) and math.isfinite(v) for v in q))

    def base_for(self, side):
        if side not in ('left', 'right'):
            raise ValueError('Unknown arm')
        return self.left if side == 'left' else self.right

    def ensure_control(self, side, require_claim=True):
        self.base_for(side)
        if not self.connected:
            raise RuntimeError('Connect the dashboard first')
        if self.assignment() == side:
            self.claims[side] = False
            raise RuntimeError(f'{side.title()} arm is assigned to the leader; release it in 8081 first')
        if require_claim and not self.claims[side]:
            raise RuntimeError(f'Take control of the {side} arm first')

    def acquire(self, side, takeover=False):
        self.base_for(side)
        if type(takeover) is not bool:
            raise ValueError('takeover must be a boolean')
        with self.control_locks[side]:
            if takeover and self.assignment() == side:
                # Explicit user handover only. Stop the leader child, not the arm service.
                self.request(self.dashboard, '/api/assignment', {'side': None})
                if self.assignment() == side:
                    raise RuntimeError('Leader did not release this arm')
                st = self.request(self.base_for(side), '/state')
                if st.get('arm_open'):
                    if not self.fresh(st):
                        raise RuntimeError('Leader released, but no fresh pose available to establish hold')
                    self.request(self.base_for(side), '/move', {'q': st['q'], 'grip': st['grip'], 'rate': 0.1})
            self.ensure_control(side, require_claim=False)
            self.request(self.base_for(side), '/state')
            self.claims[side] = True
        return self.status()

    def release(self, side):
        base = self.base_for(side)
        with self.control_locks[side]:
            if self.claims[side]:
                if self.assignment() != side:
                    st = self.request(base, '/state')
                    if st.get('arm_open'):
                        if not self.fresh(st):
                            raise RuntimeError('Cannot confirm a hold with stale telemetry; use the station controls')
                        self.request(base, '/move', {'q': st['q'], 'grip': st['grip'], 'rate': 0.1})
                self.claims[side] = False
        return self.status()

    def power(self, side, enabled):
        base = self.base_for(side)
        if type(enabled) is not bool:
            raise ValueError('enabled must be a boolean')
        with self.control_locks[side]:
            self.ensure_control(side)
            self.request(base, '/power', {'enabled': enabled})
        return self.status()

    def stop_motion(self, side):
        base = self.base_for(side)
        with self.control_locks[side]:
            self.ensure_control(side)
            st = self.request(base, '/state')
            if not self.fresh(st):
                raise RuntimeError('Arm is off or telemetry stale')
            # Freeze joint AND gripper goals; upstream /stop only freezes joints.
            self.request(base, '/move', {'q': st['q'], 'grip': st['grip'], 'rate': 0.1})
        return {'requested': True}

    def jog(self, side, joint, delta):
        base = self.base_for(side)
        if type(joint) is not int or not 0 <= joint < 6 or type(delta) not in (int, float) or delta not in (-5, -1, 1, 5):
            raise ValueError('Jog requires joint 0..5 and deltaDegrees ±1 or ±5')
        with self.control_locks[side]:
            self.ensure_control(side)
            st = self.request(base, '/state')
            if not self.fresh(st) or st.get('error') or st.get('moving'):
                raise RuntimeError('Arm must be powered, stationary and have fresh error-free telemetry')
            q = list(st['q'])
            q[joint] += math.radians(delta)
            limits = [(-160,160),(-180,0),(0,180),(-123,81),(-85,85),(-145,145)]
            for index, (value, (low, high)) in enumerate(zip(q, limits)):
                degrees = math.degrees(value)
                # Feedback may sit a fraction beyond a hard zero due to encoder
                # offsets. Tolerate that only on untouched axes; never relax the
                # bound on the axis the operator is actively jogging.
                tolerance = 0.0 if index == joint else 0.5
                if degrees < low - tolerance or degrees > high + tolerance:
                    message = 'Target exceeds vendor joint limits' if index == joint else 'Measured pose exceeds vendor joint limits'
                    raise ValueError(message)
            self.request(base, '/move', {'q': q, 'grip': st['grip'], 'rate': 0.1})
        return {'requested': True}

    def command_target(self, side, q, grip):
        base = self.base_for(side)
        if not isinstance(q, (list, tuple)) or len(q) != 6 or any(type(v) not in (int, float) or not math.isfinite(v) for v in q):
            raise ValueError('A robot target needs six finite joint angles in radians')
        if type(grip) not in (int, float) or not math.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError('Gripper target must be between 0 and 1')
        limits = [(-160,160),(-180,0),(0,180),(-123,81),(-85,85),(-145,145)]
        if not all(low <= math.degrees(value) <= high for value,(low,high) in zip(q,limits)):
            raise ValueError('Dream target exceeds a Metal arm joint limit')
        with self.control_locks[side]:
            self.ensure_control(side)
            st = self.request(base, '/state')
            if not self.fresh(st) or st.get('error'):
                raise RuntimeError('Arm is off or feedback is stale')
            # The shared controller clamps this rate to 0.6 rad/s and slews the
            # gripper independently, so Dream rows cannot become instantaneous jumps.
            self.request(base, '/move', {'q': list(q), 'grip': float(grip), 'rate': 0.3})
        return {'requested': True}

    def frame(self, role, preview=False):
        routes = {'wrist_r': [(self.right, '/wrist.jpg')],
                  'wrist_l': [(self.left, '/wrist.jpg')],
                  'overhead': [(self.right, '/overhead.jpg'), (self.left, '/overhead.jpg')]}
        if role not in routes:
            raise ValueError('Unknown camera role')
        for base, path in routes[role]:
            try:
                with urllib.request.urlopen(base + path, timeout=3) as response:
                    stamp = float(response.headers['X-Stamp'])
                    if not 0 <= time.time() - stamp <= 1:
                        raise RuntimeError('Camera frame stale')
                    data = response.read()
                    frame = dict(data=data, timestamp=stamp, observationId=f'{base}-{path}-{stamp}')
                    return self.preview(role, frame) if preview else frame
            except (OSError, ValueError, TypeError, RuntimeError):
                continue
        raise RuntimeError('No fresh camera frame; its owning arm service may be off')

    def preview(self, role, frame):
        # Snapshot route retains original bytes; only browser previews are resized.
        with self.frame_locks[role]:
            cached = self.preview_cache.get(role)
            if cached and cached['observationId'] == frame['observationId']:
                return cached
            cv2 = importlib.import_module('cv2')
            np = importlib.import_module('numpy')
            image = cv2.imdecode(np.frombuffer(frame['data'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Invalid upstream JPEG')
            h, w = image.shape[:2]
            if w > 640:
                image = cv2.resize(image, (640, max(1, round(h*640/w))), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if not ok:
                raise ValueError('Cannot encode preview')
            result = dict(frame, data=jpeg.tobytes())
            self.preview_cache[role] = result
            return result

    def close(self):
        self.connected = False
        self.pool.shutdown(wait=True)
