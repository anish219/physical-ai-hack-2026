"""Latch and jaw logic with fake reads: python -m hillclimb.gate.tests.test_core (no network, no cameras)."""
from __future__ import annotations

from hillclimb.gate import settings
from hillclimb.gate.core import GateCore
from hillclimb.gate.jaw import JawTracker


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    return bool(cond)


def run() -> bool:
    good = True
    g = GateCore(min_gap_s=1.5)
    good &= check("unarmed never opens", not g.try_open(0, "yes", True, True))
    g.arm(0)
    good &= check("fast yes asks for agentic", g.fast_read(1, True, True))
    g.agentic_read(2, now_t=10.0, yes=True, ok=True)
    good &= check("one agentic yes is not enough", not g.try_open(2, "unknown", True, True))
    g.agentic_read(3, now_t=10.5, yes=True, ok=True)
    good &= check("a second yes on a frame only 0.5 s newer does not count", g.streak == 1)
    g.agentic_read(4, now_t=12.0, yes=True, ok=True)
    good &= check("two yeses 2 s apart make a streak of 2", g.streak == 2)
    good &= check("no jaw release keeps it shut", not g.try_open(4, "unknown", False, True))
    good &= check("unhealthy keeps it shut", not g.try_open(4, "unknown", True, False))
    good &= check("all good opens (socket unknown, not required)", g.try_open(5, "unknown", True, True))
    g.fast_read(6, False, True)
    g.agentic_read(7, now_t=20.0, yes=False, ok=True)
    good &= check("latched: later no reads do not close it", g.open and g.try_open(8, "unknown", False, False))

    g = GateCore(min_gap_s=1.5, socket_required=True)
    g.arm(0)
    g.fast_read(1, True, True)
    g.agentic_read(2, now_t=10.0, yes=True, ok=True)
    g.agentic_read(3, now_t=12.0, yes=True, ok=True)
    good &= check("socket required: unknown blocks", not g.try_open(3, "unknown", True, True))
    good &= check("socket required: yes opens", g.try_open(3, "yes", True, True))

    g = GateCore(min_gap_s=1.5)
    g.arm(0)
    g.fast_read(1, True, True)
    g.agentic_read(2, now_t=10.0, yes=True, ok=True)
    g.agentic_read(3, now_t=12.0, yes=False, ok=False)  # a timeout
    good &= check("a timeout between yeses resets the streak", g.streak == 0)
    g.fast_read(4, True, True)
    g.agentic_read(5, now_t=14.0, yes=True, ok=True)
    g.fast_read(6, False, True)
    good &= check("a fast no between agentic yeses resets the streak", g.streak == 0)
    g.fast_read(7, True, False)
    good &= check("a fast yes from a failed read does not trigger agentic", g.fast == "no")
    g.agentic_read(8, now_t=16.0, yes=True, ok=True)
    g.agentic_read(9, now_t=18.0, yes=True, ok=True)
    g.arm(10)
    good &= check("re-arming resets everything", not g.open and g.streak == 0)

    # jaw: normalized grip, hold band 0.03 to 0.085, open over 0.10
    j = JawTracker()
    j.reset(0.0)
    for k in range(10):
        j.add(0.1 * k, 0.5)  # open, approaching
    good &= check("open without a hold is not a release", not j.released_since(0))
    for k in range(10, 20):
        j.add(0.1 * k, 0.05)  # 1 s holding a plug
    for k in range(20, 26):
        j.add(0.1 * k, 0.3)  # opens for 0.6 s
    good &= check("hold then open is a release", j.released_since(0))
    j.reset(5.0)
    for k in range(50, 55):
        j.add(0.1 * k, 0.05)  # only 0.4 s holding
    for k in range(55, 60):
        j.add(0.1 * k, 0.3)
    good &= check("a hold shorter than 0.5 s does not count", not j.released_since(5.0))
    j.reset(10.0)
    for k in range(100, 110):
        j.add(0.1 * k, 0.05)
    for k in range(110, 113):
        j.add(0.1 * k, 0.012)  # closed on nothing
    for k in range(113, 120):
        j.add(0.1 * k, 0.3)
    good &= check("hold, closed on nothing, then open is not a release", not j.released_since(10.0))
    good &= check("defaults are as documented", settings.get("JAW_OPEN_MIN") == 0.10)
    print("ALL PASS" if good else "SOME FAILED")
    return good


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
