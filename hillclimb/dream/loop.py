"""Our own Dream-1 control loop for the left arm (the team's dream_caller is not used or edited).

Threads:
- observer: reads the arm's /state, /overhead.jpg and /wrist.jpg at obs_hz and hands them to the Dream client;
- worker: the only thread that calls get_action() (one call at a time, as the client requires). While running it
  fetches chunks back to back as rows run out; while idle it makes a keep-warm call every 30 s and discards the rows;
- player: plays rows to /target at hz. Each chunk plays from its row 0, like the team's caller that the sim
  experiments ran with; the next chunk is fetched while the current one plays (when about 1.5 s of rows are left), so
  there is no pause between chunks. A row filter (the keep-out) may refuse a row; the arm then keeps its last target.

Dream refusals never escape: pool_warming and capacity refusals set status "warming" and retry after retry_after_s;
an expired session restarts the episode; timeouts and network errors retry. Only a refused key or checkpoint stops
the loop, with status "error".
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from .arm import Arm

WARM_EVERY_S = 30.0
PREFETCH_S = 1.5


def hold_prompt(checkpoint: str) -> str:
    """The do-nothing prompt for keep-warm calls; their rows are never played."""
    return "hold the arm still"


def _jpeg(img: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("could not encode a frame")
    return enc.tobytes()


class DreamLoop:
    def __init__(self, arm: Arm, checkpoint: str, *, hz: float = 20.0, obs_hz: float = 10.0,
                 width: int = 320, height: int = 180, playback: str = "row0", log: Callable[[dict], None] | None = None,
                 client_factory: Callable[[str, float], object] | None = None) -> None:
        self.arm, self.checkpoint, self.hz, self.obs_hz = arm, checkpoint, hz, obs_hz
        self.playback = playback  # row0: each chunk from its first row (as in the experiments); due: from the row due now
        self.width, self.height = width, height
        self.log = log or (lambda rec: None)
        self.client_factory = client_factory or open_client
        self.blank = _jpeg(np.zeros((height, width, 3), np.uint8))
        self.lock = threading.RLock()
        self.client = None
        self.embodiment = None
        self.prompt = hold_prompt(checkpoint)
        self.running = False
        self.row_filter: Callable[[list[float], dict], bool] | None = None  # True = send the row
        self.chunk: tuple[float, list[list[float]]] | None = None  # (time it started playing, rows)
        self.next_rows: list[list[float]] | None = None
        self.executed = 0
        self.last_obs = None
        self.last_state: dict = {}
        self.status = {"mode": "closed", "error": "", "warming_since": None, "calls": 0, "rows_sent": 0,
                       "rows_refused": 0, "last_call_ms": None, "last_warm_ok": None, "obs_age_s": None}
        self.stop_all = threading.Event()
        self.threads: list[threading.Thread] = []

    # ---------------------------------------------------------------- lifecycle
    def open(self) -> None:
        """Starts the threads; the worker opens the policy and warms it up in the background."""
        if self.threads:
            return
        for fn, name in ((self._observe, "observe"), (self._work, "dream"), (self._play, "play")):
            t = threading.Thread(target=fn, daemon=True, name=f"hillclimb-{name}")
            t.start()
            self.threads.append(t)

    def close(self) -> None:
        self.running = False
        self.stop_all.set()
        for t in self.threads:
            t.join(timeout=5)
        with self.lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:  # noqa: BLE001
                    pass
                self.client = None

    def ready(self) -> bool:
        return self.status["mode"] in ("ready", "running")

    def wait_ready(self, timeout: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.ready():
                return True
            if self.status["mode"] == "error":
                return False
            time.sleep(0.5)
        return self.ready()

    # ---------------------------------------------------------------- one prompt
    def start(self, prompt: str) -> None:
        with self.lock:
            self.prompt = prompt
            self.chunk = None
            self.next_rows = None
            self.executed = 0
            if self.client is not None:
                try:
                    self.client.reset_episode()
                except Exception as e:  # noqa: BLE001
                    self.log({"ev": "reset_episode_failed", "err": str(e)[:200]})
            self.running = True
        self.log({"ev": "start", "prompt": prompt})

    def stop(self) -> None:
        """Stop playing rows and hold; the policy stays open and warm."""
        with self.lock:
            self.running = False
            self.chunk = None
            self.next_rows = None
            self.prompt = hold_prompt(self.checkpoint)
        self.arm.stop()
        self.log({"ev": "stop"})

    # ---------------------------------------------------------------- threads
    def _observe(self) -> None:
        period = 1.0 / self.obs_hz
        while not self.stop_all.is_set():
            t0 = time.monotonic()
            try:
                st = self.arm.state()
                scene = self._resize(self.arm.frame("overhead"))
                wrist = self._resize(self.arm.frame("wrist"))
                with self.lock:
                    self.last_state = st
                    self.last_obs = (st, scene, wrist, time.time())
                    if self.client is not None:
                        frames, obs = self._observation(st, scene, wrist)
                        self.client.set_observation(frames, obs, timestamp=time.monotonic())
                    self.status["obs_age_s"] = 0.0
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.status["obs_error"] = str(e)[:200]
                    if self.last_obs is not None:
                        self.status["obs_age_s"] = round(time.time() - self.last_obs[3], 2)
            self.stop_all.wait(max(0.0, period - (time.monotonic() - t0)))

    def _resize(self, enc: bytes) -> bytes:
        img = cv2.imdecode(np.frombuffer(enc, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("bad JPEG from the arm server")
        return _jpeg(cv2.resize(img, (self.width, self.height)))

    def _observation(self, st: dict, scene: bytes, wrist: bytes):
        q = [float(v) for v in st["q"]]
        g = float(st["grip"])
        state = {"prompt": self.prompt,
                 "state.left_arm": q, "state.left_gripper": [g * 100.0], "state.left_grip": [g],
                 "state.right_arm": [0.0] * 6, "state.right_gripper": [0.0], "state.right_grip": [0.0]}
        frames = {"video.scene": scene, "video.wrist_l": wrist, "video.wrist_r": self.blank}
        frames = {k: base64.b64encode(v).decode("ascii") for k, v in frames.items()}
        emb = self.embodiment
        if emb:
            frames = {f"video.{k}": frames[f"video.{k}"] for k in emb["video"]}
            state = {"prompt": state["prompt"], **{f"state.{k}": state[f"state.{k}"] for k in emb["state"]}}
        return frames, state

    def _open_policy(self) -> bool:
        try:
            client, emb = self.client_factory(self.checkpoint, self.hz)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            with self.lock:
                self.status.update(mode="error" if "auth" in msg.lower() or "unknown_checkpoint" in msg else "opening",
                                   error=msg[:300])
            self.log({"ev": "open_failed", "err": msg[:300]})
            return False
        with self.lock:
            self.client, self.embodiment = client, emb
            if self.last_obs is not None:
                st, scene, wrist, _ = self.last_obs
                frames, obs = self._observation(st, scene, wrist)
                self.client.set_observation(frames, obs, timestamp=time.monotonic())
        self.log({"ev": "policy_open", "checkpoint": self.checkpoint})
        return True

    def _call(self) -> list[list[float]] | None:
        """One get_action. Rows, or None after a refusal (the status says why)."""
        import dream  # noqa: PLC0415

        t0 = time.monotonic()
        try:
            rows = self.client.get_rows()
        except dream.DreamSessionExpired:
            with self.lock:
                try:
                    self.client.reset_episode()
                except Exception:  # noqa: BLE001
                    pass
            self.log({"ev": "session_expired"})
            return None
        except dream.DreamAuthError as e:
            with self.lock:
                self.status.update(mode="error", error=f"auth: {e}"[:300])
            return None
        except dream.DreamServerError as e:
            wait = getattr(e, "retry_after_s", None) or 2.0
            reason = getattr(e, "reason", "") or str(e)
            if "pool_warming" in str(e) or "NO_CAPACITY" in str(e) or getattr(e, "retryable", False) or "rate_limited" in str(e):
                with self.lock:
                    if self.status["mode"] != "warming":
                        self.status["warming_since"] = time.time()
                    self.status.update(mode="warming", error=str(reason)[:200])
                self.stop_all.wait(min(wait, 30.0))
            else:
                with self.lock:
                    self.status.update(error=str(e)[:300])
                self.stop_all.wait(2.0)
            return None
        except Exception as e:  # noqa: BLE001  timeouts, network, a malformed reply: retry
            with self.lock:
                self.status.update(error=str(e)[:300])
            self.stop_all.wait(1.0)
            return None
        ms = (time.monotonic() - t0) * 1000
        with self.lock:
            self.status.update(calls=self.status["calls"] + 1, last_call_ms=round(ms), error="",
                               mode="running" if self.running else "ready", warming_since=None)
        return rows

    def _work(self) -> None:
        last_warm = 0.0
        while not self.stop_all.is_set():
            if self.client is None:
                with self.lock:
                    self.status["mode"] = "opening"
                if not self._open_policy():
                    self.stop_all.wait(5.0)
                continue
            if self.status["mode"] == "error":
                self.stop_all.wait(5.0)
                continue
            if self.last_obs is None:
                self.stop_all.wait(0.2)
                continue
            if self.running:
                need = self.next_rows is None and (self.chunk is None or self._remaining() <= int(self.hz * (1.2 if self.playback == "due" else PREFETCH_S)))
                if not need:
                    self.stop_all.wait(0.03)
                    continue
                t_req = time.monotonic()
                rows = self._call()
                if rows is not None and self.running:
                    with self.lock:
                        if self.playback == "due":
                            self.chunk, self.next_rows = (t_req, rows), None
                        elif self.chunk is None:
                            self.chunk = (time.monotonic(), rows)
                        else:
                            self.next_rows = rows
                continue
            # idle: warm up (first call) or keep warm
            if self.status["mode"] in ("opening", "warming") or time.time() - last_warm > WARM_EVERY_S:
                with self.lock:
                    self.prompt = hold_prompt(self.checkpoint)
                rows = self._call()
                last_warm = time.time()
                with self.lock:
                    self.status["last_warm_ok"] = rows is not None
                if rows is not None:
                    self.log({"ev": "warm", "ms": self.status["last_call_ms"]})
                continue
            self.stop_all.wait(0.2)

    def _remaining(self) -> int:
        if self.chunk is None:
            return 0
        t_play, rows = self.chunk
        done = int((time.monotonic() - t_play) * self.hz)
        return max(0, len(rows) - done)

    def _play(self) -> None:
        period = 1.0 / self.hz
        while not self.stop_all.is_set():
            t0 = time.monotonic()
            row = None
            with self.lock:
                if self.running and self.chunk is not None:
                    t_play, rows = self.chunk
                    i = int((t0 - t_play) * self.hz)
                    if i >= len(rows) and self.next_rows is not None:
                        self.chunk, self.next_rows = (t0, self.next_rows), None
                        t_play, rows, i = t0, self.chunk[1], 0
                    if i < len(rows):
                        row = rows[i]
                st = self.last_state
            if row is not None:
                send = True
                if self.row_filter is not None:
                    try:
                        send = self.row_filter(row, st)
                    except Exception:  # noqa: BLE001
                        send = True
                if send:
                    try:
                        if self.arm.target(row[:6], row[6]):
                            self.status["rows_sent"] += 1
                        else:
                            self.status["rows_failed"] = self.status.get("rows_failed", 0) + 1
                    except Exception as e:  # noqa: BLE001
                        self.status["error"] = f"target: {e}"[:200]
                else:
                    self.status["rows_refused"] += 1
            self.stop_all.wait(max(0.0, period - (time.monotonic() - t0)))


class _JointPolicy:
    """A fine-tuned joint checkpoint: picks its embodiment's camera and state slots and converts its per-key reply
    into [rows, 7] (six joints in radians, then the jaw normalized 0 closed to 1 open)."""

    def __init__(self, client, embodiment: dict) -> None:
        self.client, self.embodiment = client, embodiment

    def set_observation(self, frames, state, *, timestamp):
        self.client.set_observation(frames, state, timestamp=timestamp)

    def reset_episode(self):
        self.client.reset_episode()

    def get_rows(self) -> list[list[float]]:
        reply = self.client.get_action()["per_key"]
        joints = np.asarray(reply["action.left_arm"], float)
        grip = np.asarray(reply["action.left_gripper"], float).reshape(-1)
        spec = self.embodiment["parts"]["action"]["left_gripper"]
        norm = np.clip((grip - spec["closed"]) / (spec["open"] - spec["closed"]), 0.0, 1.0)
        if joints.shape != (len(grip), 6) or not np.isfinite(joints).all():
            raise ValueError("checkpoint returned invalid joint rows")
        return np.column_stack([joints, norm]).tolist()

    def close(self):
        self.client.close()


def open_client(checkpoint: str, hz: float):
    """Opens a left-arm joint fine-tune. Returns (policy, embodiment)."""
    import dream  # noqa: PLC0415

    client = dream.DreamClient()
    try:
        model = next((m for m in client.get_models() if m.id == checkpoint), None)
        if model is None:
            raise ValueError(f"unknown_checkpoint: {checkpoint} is not listed for this account")
        if not model.servable:
            raise ValueError(f"{checkpoint} is not servable yet")
        body = model.embodiment_config
        if not body or any("eef" in k for k in body["action"]) or "left_arm" not in body["parts"]["action"]:
            raise ValueError(f"{checkpoint} is not a left-arm joint checkpoint; this loop drives only those")
        client.policy({"checkpoint": checkpoint, "embodiment": body, "action_hz": hz})
    except Exception:
        client.close()
        raise
    return _JointPolicy(client, body), body
