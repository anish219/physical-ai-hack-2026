"""Client for the left arm's hillclimb server.

The left arm's HTTP start link is this server. START arms a match round and
posts `/left/go`. Pressing START again posts `/left/stop`. The same `start`
and `stop` functions are what `hillclimb/left/match.py:start` runs if that
path is used from the Script input. If nothing is already listening on the
hillclimb port, `start` brings the server up in the background.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from hillclimb.common.config import CFG

SCRIPT_REF = "hillclimb/left/match.py:start"
ARM = {
    "backend": "scripted",
    "mode": "match",
    "drive": "auto",
    "round_s": 300,
    "max_goal": 6,
    "crunch_at_s": 90,
    "crunch_goal": 5,
    "freeze_s": 20,
}
_lock = threading.Lock()


def base_url():
    return f"http://127.0.0.1:{CFG['HILLCLIMB_PORT']}"


def serves(url):
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlparse(url.strip())
    expected = urlparse(base_url())
    return (
        parsed.scheme == expected.scheme
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.port == expected.port
    )


def _request(path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url() + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"hillclimb {path} failed: {error.code} {detail}") from error
    return json.loads(body) if body else {}


def server_up():
    try:
        _request("/api/state", timeout=0.4)
    except (OSError, RuntimeError, TimeoutError, json.JSONDecodeError):
        return False
    return True


def ensure_server():
    if server_up():
        return
    with _lock:
        if server_up():
            return
        from hillclimb.left.server import build, gate_module

        server, controller = build()
        controller.start()
        gate = gate_module()
        if gate is not None and hasattr(gate, "start"):
            try:
                gate.start()
            except Exception:
                pass
        threading.Thread(target=server.serve_forever, name="hillclimb", daemon=True).start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if server_up():
            return
        time.sleep(0.05)
    raise RuntimeError("Hillclimb server did not start")


def start(context=None):
    ensure_server()
    armed = _request("/left/arm", ARM)
    if not armed.get("ok"):
        raise RuntimeError(armed.get("why") or "could not arm the left hillclimb round")
    started = _request("/left/go", {})
    if not started.get("ok"):
        raise RuntimeError(started.get("why") or "could not start the left hillclimb round")
    return started


def stop(context=None):
    if not server_up():
        return None
    return _request("/left/stop", {}, timeout=10)
