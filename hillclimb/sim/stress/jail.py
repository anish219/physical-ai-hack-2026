"""Import this FIRST, before anything from hillclimb: it starts the fake left and right arm servers, points every
hillclimb URL, log and state path at them and at a temp folder, and installs a socket jail that refuses every
connection except to the fakes' own loopback ports. The station's tunnels on this Mac (8691, 8697, 18081, 18084) and
api.openai.com are unreachable from these tests by construction.
"""
from __future__ import annotations

import os
import socket
import tempfile

from hillclimb.sim.stress.fake_arm import FakeArm  # noqa: E402  (fake_arm imports nothing from hillclimb)

TMP = tempfile.mkdtemp(prefix="hc_stress_")
LEFT = FakeArm("left")
RIGHT = FakeArm("right")
DEAD_PORT = 1  # nothing listens on port 1; also refused by the jail
ALLOWED = {LEFT.port, RIGHT.port}
BLOCKED_SEEN: list[str] = []

for k in ("OPENAI_API_KEY", "GATE_URL", "RUNNER_SIM_PORT"):
    os.environ.pop(k, None)
os.environ.update({
    "LEFT_URL": LEFT.url, "RIGHT_URL": RIGHT.url, "OVERHEAD_FROM": "left",
    "LOG_DIR": os.path.join(TMP, "logs"), "STATE_FILE": os.path.join(TMP, "logs", "left_state.json"),
    "GATE_FILE": os.path.join(TMP, "usb_gate_open"), "RELAY_DIR": os.path.join(TMP, "relay"),
    "KEYS_FILE": os.path.join(TMP, "no_keys.env"), "SOCKET_VERIFIER_DIR": os.path.join(TMP, "no_verifier"),
    "HILLCLIMB_PORT": "0",
})
os.makedirs(os.path.join(TMP, "relay"), exist_ok=True)

_orig_connect, _orig_connect_ex, _orig_gai = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo


def _ok(addr) -> bool:
    try:
        host, port = addr[0], int(addr[1])
    except Exception:  # noqa: BLE001  (unix sockets and the like)
        return True
    return host in ("127.0.0.1", "::1", "localhost") and port in ALLOWED


def _connect(self, addr):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _ok(addr):
        BLOCKED_SEEN.append(f"{addr[0]}:{addr[1]}")
        raise ConnectionRefusedError(f"stress jail: {addr[0]}:{addr[1]} is not a fake")
    return _orig_connect(self, addr)


def _connect_ex(self, addr):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _ok(addr):
        BLOCKED_SEEN.append(f"{addr[0]}:{addr[1]}")
        return 111
    return _orig_connect_ex(self, addr)


def _gai(host, *a, **k):
    if host not in ("127.0.0.1", "::1", "localhost", None, "0.0.0.0", ""):
        BLOCKED_SEEN.append(f"dns:{host}")
        raise socket.gaierror(f"stress jail: no DNS for {host}")
    return _orig_gai(host, *a, **k)


socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = _connect, _connect_ex, _gai


def allow(port: int) -> None:
    ALLOWED.add(int(port))


RESULTS: list[tuple[str, str, str]] = []  # (verdict, name, detail): verdict is OK, BUG or NOTE


def record(verdict: str, name: str, detail: str = "") -> None:
    RESULTS.append((verdict, name, detail))
    print(f"{verdict:4} {name}" + (f"  ({detail})" if detail else ""), flush=True)


def summary() -> None:
    bugs = [r for r in RESULTS if r[0] == "BUG"]
    print(f"\n{len(RESULTS)} checks, {len(bugs)} BUG, blocked connections: {sorted(set(BLOCKED_SEEN))[:10]}", flush=True)
