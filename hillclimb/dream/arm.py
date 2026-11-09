"""Arm access for the Dream backend.

The backend runs on the controller's hillclimb.common.arm.ArmClient (its motion calls pass the controller's gate
guard). That client returns None or False on failure; `Arm` wraps it so the loop's threads get an exception they
catch and count, never a silent None. `Arm.from_url` builds one for standalone sim tests.
"""

from __future__ import annotations

FORBIDDEN_PORTS = (8692, 8081, 8084, 8697, 18081, 18084)


class ArmError(RuntimeError):
    pass


class Arm:
    def __init__(self, client) -> None:
        self.c = client
        base = getattr(client, "base", "")
        port = base.rsplit(":", 1)[-1].split("/")[0] if base.count(":") >= 2 else ""
        if port.isdigit() and int(port) in FORBIDDEN_PORTS:
            raise ValueError(f"refusing to drive {base}: that port is not the left arm")

    @classmethod
    def from_url(cls, url: str) -> Arm:
        from hillclimb.common.arm import ArmClient  # noqa: PLC0415
        return cls(ArmClient(url))

    @property
    def base(self) -> str:
        return getattr(self.c, "base", "")

    def state(self) -> dict:
        s = self.c.state()
        if s is None:
            raise ArmError(getattr(self.c, "last_error", None) or "no /state")
        return s

    def frame(self, name: str) -> bytes:
        f = self.c.frame(name)
        if f is None:
            raise ArmError(getattr(self.c, "last_error", None) or f"no /{name}.jpg")
        return f[0] if isinstance(f, tuple) else f

    def target(self, q, grip: float) -> bool:
        """False when refused (gate guard) or failed; the loop counts it, the arm keeps its last target."""
        return bool(self.c.target([float(v) for v in q], float(grip)))

    def move(self, q, grip: float, rate: float = 0.6) -> bool:
        return bool(self.c.move([float(v) for v in q], float(grip), rate=float(rate)))

    def stop(self) -> None:
        try:
            self.c.stop()
        except Exception:  # noqa: BLE001
            pass
