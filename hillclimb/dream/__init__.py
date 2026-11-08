"""Dream stacking backend for the left arm (see README.md)."""

from __future__ import annotations

from .backend import DEFAULT_CHECKPOINT, DreamBackend  # noqa: F401


def make_backend(settings: dict, arm) -> DreamBackend:
    """What hillclimb.left builds it with. settings: the controller's settings dict (checkpoint, wording, and the
    optional dream_hz, keep_out, tower_xy); arm: the controller's hillclimb.common.arm.ArmClient. The policy starts
    opening and warming at once; place_next answers "policy warming" until it is ready."""
    s = dict(settings or {})
    be = DreamBackend(arm, s.get("checkpoint") or DEFAULT_CHECKPOINT, wording=s.get("wording") or "",
                      hz=float(s.get("dream_hz") or 20.0), keep_out=bool(s.get("keep_out", True)),
                      tower_xy=s.get("tower_xy"))
    be.prepare({})
    return be
