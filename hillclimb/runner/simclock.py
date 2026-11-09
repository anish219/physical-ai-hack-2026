"""The station sim's clock, read from its /clock endpoint (the sim can run faster than the wall clock)."""
import time as _time


class Clock:
    """k times faster than the wall clock, the same in every process that reads /clock. time() is virtual seconds;
    sleep(s) sleeps s / k real seconds; anything else is the time module's."""

    def __init__(self, k=1.0, real0=None):
        self.k, self.real0 = float(k), float(real0 if real0 is not None else _time.time())

    def time(self):
        return self.real0 + (_time.time() - self.real0) * self.k

    def sleep(self, s):
        if s > 0:
            _time.sleep(s / self.k)

    def __getattr__(self, name):
        return getattr(_time, name)
