"""Shared latest-frame preview producers; slow viewers cannot queue old frames."""
import threading
import time

ROLES = ('wrist_l', 'overhead', 'wrist_r')


class PreviewStream:
    def __init__(self, source, role):
        self.source, self.role = source, role
        self.condition = threading.Condition()
        self.latest = None
        self.error = None
        self.sequence = 0
        self.listeners = 0
        self.closed = False
        self.received = []
        self.thread = threading.Thread(target=self.run, name=f'preview-{role}', daemon=True)
        self.thread.start()

    def subscribe(self):
        with self.condition:
            if self.listeners == 0:
                self.error = None
            self.listeners += 1
            self.condition.notify_all()

    def unsubscribe(self):
        with self.condition:
            self.listeners = max(0, self.listeners - 1)

    def next(self, after, timeout=1):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.error or (self.sequence > after and self.latest is not None and 0 <= time.time() - self.latest['timestamp'] <= 1), timeout)
            if self.closed or self.error or not self.latest or time.time() - self.latest['timestamp'] > 1:
                raise RuntimeError(self.error or 'No fresh camera frame')
            return self.sequence, self.latest

    def stats(self):
        with self.condition:
            times = [t for t in self.received if t > time.monotonic() - 3]
            fps = (len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0
            age = time.time()-self.latest['timestamp'] if self.latest else None
            return {'fps': round(fps,1), 'ageMs': round(age*1000) if age is not None else None,
                    'fresh': age is not None and 0 <= age <= 1 and not self.error, 'error': self.error}

    def run(self):
        previous = None
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.listeners > 0)
                if self.closed:
                    return
            started = time.monotonic()
            try:
                frame = self.source.frame(self.role, preview=True)
                with self.condition:
                    self.error = None
                    if frame['observationId'] != previous:
                        previous = frame['observationId']
                        self.latest = frame
                        self.sequence += 1
                        self.received.append(time.monotonic())
                        self.received = self.received[-180:]
                        self.condition.notify_all()
            except Exception as error:
                with self.condition:
                    self.error = str(error)
                    self.condition.notify_all()
            # Sample faster than the source to avoid aliasing its ~30 Hz clock.
            with self.condition:
                self.condition.wait(timeout=max(0, 1/60 - (time.monotonic()-started)))

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class VideoHub:
    def __init__(self, source):
        self.streams = {role: PreviewStream(source, role) for role in ROLES}

    def stats(self):
        return {role: stream.stats() for role, stream in self.streams.items()}

    def close(self):
        for stream in self.streams.values():
            stream.close()
