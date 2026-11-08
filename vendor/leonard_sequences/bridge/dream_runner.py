"""Dream API action loop for one explicitly claimed arm.

All hardware reads and writes still pass through the already-running arm HTTP
services. Dream receives observations from both sides, but only right-arm action
rows are ever applied by this runner.
"""
import base64
import importlib
import json
import math
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class DreamRunner:
    def __init__(self, bridge):
        self.bridge = bridge
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.run_id = 0
        self.state = 'idle'
        self.prompt = ''
        self.checkpoint = 'dreamlabs/dream-1-beta'
        self.started = None
        self.duration = 0
        self.error = None
        self.last_action_at = None
        self.arm_model = 'metal'
        self.embodiment = None

    def status(self):
        with self.lock:
            elapsed = max(0, time.monotonic() - self.started) if self.started and self.state == 'running' else 0
            return {'state': self.state, 'prompt': self.prompt, 'checkpoint': self.checkpoint,
                    'durationSeconds': self.duration, 'elapsedSeconds': round(elapsed, 1),
                    'error': self.error, 'lastActionAt': self.last_action_at, 'side': 'right'}

    def start(self, prompt, checkpoint='dreamlabs/dream-1-beta', duration_seconds=20):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
            raise ValueError('Enter a Dream task prompt (up to 1000 characters)')
        if not isinstance(checkpoint, str) or not checkpoint.strip() or len(checkpoint) > 160:
            raise ValueError('Enter a valid Dream checkpoint ID')
        arm_model = os.environ.get('LEONARD_DREAM_ARM_FILE') or 'metal'
        if arm_model != 'metal' and not os.path.isfile(arm_model):
            raise RuntimeError('LEONARD_DREAM_ARM_FILE points to a file that does not exist')
        embodiment_path = Path(__file__).resolve().parent / 'embodiments' / 'makermods.json'
        try:
            embodiment = json.loads(embodiment_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f'Cannot load the MakerMods Dream embodiment: {error}') from error
        if not isinstance(embodiment.get('video'), dict):
            raise RuntimeError('MakerMods Dream embodiment must map camera roles in its video section')
        if type(duration_seconds) not in (int, float) or not math.isfinite(duration_seconds) or not 1 <= duration_seconds <= 300:
            raise ValueError('Action window must be 1–300 seconds')
        with self.lock:
            if self.state == 'running':
                raise RuntimeError('A Dream action is already running')
            self.bridge.ensure_control('right')
            states = self.bridge.states()
            for side, state in states.items():
                if not self.bridge.fresh(state) or state.get('error'):
                    raise RuntimeError(f'{side.title()} arm needs fresh, error-free feedback for Dream observations')
            if states['right'].get('moving'):
                raise RuntimeError('Wait for the right arm to stop before starting a Dream sequence')
            self.run_id += 1
            self.prompt, self.checkpoint = prompt.strip(), checkpoint.strip()
            self.arm_model = arm_model
            self.embodiment = embodiment
            self.duration, self.started, self.error = float(duration_seconds), time.monotonic(), None
            self.last_action_at = None
            self.stop_event.clear()
            self.state = 'running'
            self.thread = threading.Thread(target=self._run, args=(self.run_id,), name='leonard-dream-right', daemon=True)
            self.thread.start()
        return self.status()

    def stop(self):
        with self.lock:
            if self.state != 'running':
                return self.status()
            self.stop_event.set()
        return self.status()

    def _observations(self, prompt):
        states = self.bridge.states()
        for side, state in states.items():
            if not self.bridge.fresh(state) or state.get('error'):
                raise RuntimeError(f'{side.title()} arm feedback became stale or reported an error')
        # The physical bridge calls its fixed camera `overhead`; the Dream
        # embodiment calls the same scene-facing view `scene`.
        camera_sources = {'scene': 'overhead', 'wrist_l': 'wrist_l', 'wrist_r': 'wrist_r'}
        with ThreadPoolExecutor(max_workers=3) as fetchers:
            frames = {dream_role: fetchers.submit(self.bridge.frame, source_role)
                      for dream_role, source_role in camera_sources.items()}
            jpeg_frames = {role: self._to_dream_jpeg(future.result()['data'])
                           for role, future in frames.items()}
        l, r = states['left'], states['right']
        observation = {
            'state.left_arm': list(l['q']), 'state.left_gripper': [float(l['grip'])],
            'state.left_grip': [float(l['grip'])],
            'state.right_arm': list(r['q']), 'state.right_gripper': [float(r['grip'])],
            'state.right_grip': [float(r['grip'])],
            'prompt': prompt,
        }
        return ({f'video.{role}': data for role, data in jpeg_frames.items()}, observation)

    @staticmethod
    def _to_dream_jpeg(jpeg):
        cv2 = importlib.import_module('cv2')
        np = importlib.import_module('numpy')
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('A camera returned an invalid JPEG')
        height, width = image.shape[:2]
        target_ratio = 320 / 176
        if width / height > target_ratio:
            cropped_width = round(height * target_ratio)
            left = max(0, (width - cropped_width) // 2)
            image = image[:, left:left + cropped_width]
        else:
            cropped_height = round(width / target_ratio)
            top = max(0, (height - cropped_height) // 2)
            image = image[top:top + cropped_height, :]
        image = cv2.resize(image, (320, 176), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('Could not encode a Dream camera frame')
        return base64.b64encode(encoded.tobytes()).decode('ascii')

    @staticmethod
    def _right_chunk(action):
        rows = action.get('q_r_chunk') if isinstance(action, dict) else None
        if not isinstance(rows, list) or not rows:
            raise RuntimeError('Dream returned no right-arm joint action chunk')
        limits = [(-160, 160), (-180, 0), (0, 180), (-123, 81), (-85, 85), (-145, 145)]
        result = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 7:
                raise RuntimeError('Dream right-arm actions must have six joint targets and one gripper target')
            values = [float(v) for v in row]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError('Dream returned a non-finite action')
            q = values[:6]
            if not all(low <= math.degrees(v) <= high for v, (low, high) in zip(q, limits)):
                raise RuntimeError('Dream action exceeds a Metal arm joint limit; the action was stopped')
            grip = values[6]
            if not 0 <= grip <= 1:
                raise RuntimeError('Dream gripper target is outside the 0–1 open/closed range')
            result.append((q, grip))
        return result

    def _hold_right(self):
        try:
            with self.bridge.control_locks['right']:
                self.bridge.ensure_control('right')
                state = self.bridge.request(self.bridge.right, '/state')
                if self.bridge.fresh(state):
                    self.bridge.request(self.bridge.right, '/move', {'q': state['q'], 'grip': state['grip'], 'rate': 0.3})
        except Exception:
            # Do not conceal the original Dream/control error with a secondary hold failure.
            pass

    def _run(self, run_id):
        client = None
        action_pool = None
        try:
            dream = importlib.import_module('dream')
            client = dream.DreamClient()
            client.policy({'checkpoint': self.checkpoint, 'embodiment': self.embodiment,
                           'ik': {'arm': self.arm_model, 'head': 'eef'},
                           'action_hz': 20, 'strict': True})
            deadline = self.started + self.duration
            next_observation = 0.0
            action_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dream-inference')
            inference = None
            chunk, index = [], 0
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_observation:
                    frames, observation = self._observations(self.prompt)
                    client.set_observation(frames, observation, timestamp=now)
                    next_observation = now + 0.2
                if inference is not None and inference.done():
                    chunk = self._right_chunk(inference.result())
                    inference = None
                    index = 0
                if inference is None and (not chunk or index >= len(chunk)):
                    inference = action_pool.submit(client.get_action)
                if chunk and index < len(chunk):
                    q, grip = chunk[index]
                    self.bridge.command_target('right', q, grip)
                    index += 1
                    with self.lock:
                        self.last_action_at = time.time()
                remaining = max(0, self.duration - (time.monotonic() - self.started))
                self.stop_event.wait(min(0.05, remaining) if remaining else 0.01)
            self.stop_event.set()
            self._hold_right()
            with self.lock:
                if run_id == self.run_id:
                    self.state = 'complete' if time.monotonic() >= deadline and not self.error else 'stopped'
        except Exception as error:
            self._hold_right()
            with self.lock:
                if run_id == self.run_id:
                    self.error = str(error)
                    self.state = 'error'
        finally:
            self.stop_event.set()
            if action_pool is not None:
                action_pool.shutdown(wait=False, cancel_futures=True)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            with self.lock:
                if run_id == self.run_id and self.state == 'running':
                    self.state = 'stopped'
