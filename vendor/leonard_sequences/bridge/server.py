"""Local, single-owner observation bridge. No hackathon application imports."""
import argparse
import importlib
import json
import math
import socket
from pathlib import Path
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

MOTORS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll')
ROOT = Path(__file__).resolve().parents[1] / 'dist'


class MetalObserver:
    """Use the installed driver bus; never enable/disable torque or home an arm.

    API based on the station's inspected MetalFollower and DamiaoMotorsBus.
    Reads actively request motor feedback. Do not share a bus with another owner.
    """
    def __init__(self, cfg):
        from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig
        self.device = MetalFollower(MetalFollowerConfig(
            port=cfg['port'], id=cfg['driver_id'], can_interface=cfg['can_interface'],
            cameras={}, disable_torque_on_disconnect=False))
        try:
            self.device.bus.connect(handshake=False)
        except BaseException:
            try:
                self.device.bus.disconnect(False)
            except Exception:
                pass
            raise

    def read(self):
        result = [float(self.device.bus.read('Present_Position', motor)) for motor in MOTORS]
        if not all(math.isfinite(value) for value in result):
            raise ValueError('Driver returned non-finite motor positions')
        return result

    def close(self):
        self.device.bus.disconnect(False)


class Bridge:
    def __init__(self, config, mock=False):
        if config['arm']['side'] != 'right':
            raise ValueError('This bridge is restricted to the right arm')
        if not mock and (not config['arm'].get('port') or 'REPLACE_' in config['arm']['port']):
            raise ValueError('Set the verified right-arm CAN port before running')
        if set(config.get('cameras', {})) - {'wrist_r', 'overhead'}:
            raise ValueError('Only right wrist and explicitly available overhead cameras may be configured')
        self.config, self.mock = config, mock
        self.arm = None
        self.mock_connected = False
        self.lock = threading.Lock()
        self.cameras = {}
        self.camera_lock = threading.Lock()
        self.stop = threading.Event()
        self.threads = []

    def connect(self):
        with self.lock:
            if not self.mock and self.arm is None:
                self.arm = MetalObserver(self.config['arm'])
        if self.mock:
            self.mock_connected = True
        return self.status()

    def disconnect(self):
        with self.lock:
            if self.arm:
                self.arm.close()
                self.arm = None
            self.mock_connected = False
        return self.status()

    def status(self):
        return dict(mode='mock' if self.mock else 'hardware', connected=self.mock_connected if self.mock else self.arm is not None,
                    side=self.config['arm']['side'], motionEnabled=False)

    def telemetry(self):
        with self.lock:
            if not (self.mock_connected if self.mock else self.arm is not None):
                raise RuntimeError('Arm not connected')
            positions = [0, -35, 65, 0, 45, 0] if self.mock else self.arm.read()
        side = self.config['arm']['side']
        return dict(arm01=positions if side == 'left' else None,
                    arm02=positions if side == 'right' else None,
                    timestamp=time.time(), units='degrees', simulated=self.mock)

    def start_cameras(self):
        # Explicitly configured cameras only. No device scan or default captures.
        for role, cfg in self.config.get('cameras', {}).items():
            thread = threading.Thread(target=self.capture, args=(role, cfg), daemon=True)
            self.threads.append(thread)
            thread.start()

    def capture(self, role, cfg):
        cap = None
        sequence = 0
        try:
            cv2 = importlib.import_module('cv2')
            device = cfg['device']
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError('Cannot open configured camera')
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.get('width', 640))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get('height', 480))
            cap.set(cv2.CAP_PROP_FPS, cfg.get('fps', 15))
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera stopped delivering frames')
                ok, data = cv2.imencode('.jpg', frame)
                if ok:
                    sequence += 1
                    with self.camera_lock:
                        self.cameras[role] = dict(data=data.tobytes(), timestamp=time.time(),
                                                  observationId=f'{role}-{sequence}')
        except Exception as error:
            with self.camera_lock:
                self.cameras[role] = dict(error=str(error))
        finally:
            if cap is not None:
                cap.release()

    def frame(self, role, preview=False):
        with self.camera_lock:
            frame = self.cameras.get(role)
        if not frame or 'data' not in frame or time.time() - frame['timestamp'] > 1:
            raise RuntimeError('Camera unavailable or frame stale')
        return frame

    def close(self):
        self.stop.set()
        self.disconnect()
        for thread in self.threads:
            thread.join(timeout=2)


def handler_for(bridge):
    from video import VideoHub
    video = VideoHub(bridge)
    dream_runner = None
    if hasattr(bridge, 'command_target'):
        from dream_runner import DreamRunner
        dream_runner = DreamRunner(bridge)
    class Handler(SimpleHTTPRequestHandler):
        video_hub = video
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if not path.startswith('/api/'):
                return super().do_GET()
            try:
                if path == '/api/status':
                    return self.reply(200, dict(bridge.status(), video=video.stats(), dream=dream_runner.status() if dream_runner else {'state': 'unavailable'}))
                if path == '/api/telemetry':
                    return self.reply(200, bridge.telemetry())
                if path == '/api/dream/status' and dream_runner:
                    return self.reply(200, dream_runner.status())
                if path.startswith('/api/cameras/') and path.endswith('/stream.mjpg'):
                    role = path.split('/')[3]
                    if role not in video.streams:
                        return self.reply(404, {'error': 'Unknown camera'})
                    return self.stream_video(video.streams[role])
                if path.startswith('/api/cameras/') and path.endswith('/frame.jpg'):
                    role = path.split('/')[3]
                    frame = bridge.frame(role, preview=parse_qs(urlsplit(self.path).query).get("preview") == ["1"])
                    self.send_response(200)
                    for key, value in [('Content-Type', 'image/jpeg'), ('Cache-Control', 'no-store'),
                                       ('Content-Length', str(len(frame['data']))),
                                       ('X-Observation-Id', frame['observationId']),
                                       ('X-Capture-Timestamp', str(frame['timestamp']))]:
                        self.send_header(key, value)
                    self.end_headers()
                    return self.wfile.write(frame['data'])
                self.reply(404, {'error': 'Unknown endpoint'})
            except Exception as error:
                self.reply(503, {'error': str(error)})

        def stream_video(self, stream):
            stream.subscribe()
            try:
                sequence, frame = stream.next(0, timeout=4)
                self.connection.settimeout(1)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                while True:
                    started = time.monotonic()
                    header = (f'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame["data"])}\r\n'
                              f'X-Capture-Timestamp: {frame["timestamp"]}\r\n\r\n').encode()
                    self.wfile.write(header + frame['data'] + b'\r\n')
                    self.wfile.flush()
                    time.sleep(max(0, 1/30-(time.monotonic()-started)))
                    after = sequence
                    sequence, frame = stream.next(after)
                    if sequence == after:
                        raise RuntimeError('Camera stopped delivering new frames')
            except (OSError, RuntimeError):
                # Close a stalled/slow stream rather than buffering old video.
                self.close_connection = True
            finally:
                stream.unsubscribe()

        def do_POST(self):
            # Local same-origin UI only; no CORS or cross-site form actions.
            origin = self.headers.get('Origin')
            if self.headers.get('X-Leonard-Client') != '1' or (origin and urlsplit(origin).netloc != self.headers.get('Host')):
                return self.reply(403, {'error': 'Same-origin bridge client required'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 4096:
                    return self.reply(400, {'error': 'Request body too large'})
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/api/connect':
                    return self.reply(200, bridge.connect())
                if self.path == '/api/disconnect':
                    return self.reply(200, bridge.disconnect())
                if hasattr(bridge, 'acquire'):
                    parts = self.path.strip('/').split('/')
                    if len(parts) == 4 and parts[:2] == ['api', 'arms'] and parts[2] in ('left', 'right'):
                        side, action = parts[2:]
                        if action == 'acquire':
                            return self.reply(200, bridge.acquire(side, takeover=body.get('takeover', False)))
                        if action == 'release':
                            return self.reply(200, bridge.release(side))
                        if action == 'power':
                            return self.reply(200, bridge.power(side, body.get('enabled')))
                        if action == 'stop':
                            return self.reply(200, bridge.stop_motion(side))
                        if action == 'jog':
                            return self.reply(200, bridge.jog(side, body.get('joint'), body.get('deltaDegrees')))
                path = urlsplit(self.path).path
                if dream_runner and path == '/api/dream/start':
                    return self.reply(200, dream_runner.start(body.get('prompt'), body.get('checkpoint', 'dreamlabs/dream-1-beta'), body.get('durationSeconds', 20)))
                if dream_runner and path == '/api/dream/stop':
                    return self.reply(200, dream_runner.stop())
                self.reply(501, {'error': 'Motion, Dream inference and recovery are not enabled'})
            except Exception as error:
                self.reply(503, {'error': str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--existing', action='store_true', help='Reuse running services on 8691/8692/8081 without opening hardware')
    parser.add_argument('--port', type=int, default=8094)
    args = parser.parse_args()
    if not args.mock and not args.existing and args.config is None:
        parser.error('Hardware mode requires --config; no default arm or camera is opened')
    config = json.loads(args.config.read_text()) if args.config else {'arm': {'side': 'right'}}
    if config['arm']['side'] not in ('left', 'right'):
        parser.error('arm.side must be left or right')
    if args.existing:
        from existing import ExistingBridge
        bridge = ExistingBridge()
    else:
        bridge = Bridge(config, args.mock)
    if not args.mock and not args.existing:
        bridge.start_cameras()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(bridge))
    print(f'Leonard: http://127.0.0.1:{args.port}/?bridge=1', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.RequestHandlerClass.video_hub.close()
        server.server_close()
        bridge.close()


if __name__ == '__main__':
    main()
