"""Vision checks for the left arm with GPT-6.1 Terra (reasoning off, JSON answers). Every call has a timeout and
returns None on any failure, which callers treat as "unknown", never as success.

- grew_and_stands: before/after frames from the look pose; did the tower grow by one and stand. Used as a cross-check
  beside the harness's own placement count (Terra never decides height on its own).
- choose_next: Terra picks from a candidate list the harness built, by index, so it can never name a block that is
  not there; the harness writes the prompt from the wording table.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request

from hillclimb.common import keys
from hillclimb.common.config import CFG

URL = "https://api.openai.com/v1/chat/completions"


def call(prompt: str, images: list[bytes], timeout: float | None = None) -> tuple[dict | None, float]:
    """(parsed JSON answer or None, milliseconds)."""
    key = keys.get("OPENAI_API_KEY")
    t0 = time.time()
    if not key:
        return None, 0.0
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}} for b in images if b]
    body = {"model": CFG["VLM_MODEL"], "reasoning_effort": "none", "response_format": {"type": "json_object"},
            "max_completion_tokens": 800, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout or CFG["VLM_TIMEOUT_S"]) as r:
            j = json.loads(r.read())
        txt = j["choices"][0]["message"]["content"]
        return json.loads(txt[txt.index("{"): txt.rindex("}") + 1]), (time.time() - t0) * 1000
    except Exception:  # noqa: BLE001  timeout, HTTP error, bad JSON: all "unknown"
        return None, (time.time() - t0) * 1000


GREW = """Images of a robot station, taken with the arm backed off to a look pose.
Image 1: overhead camera BEFORE. Image 2: overhead camera AFTER. Image 3: wrist camera BEFORE. Image 4: wrist camera AFTER.
Foam cubes with a round hole are blocks; sticks, balls, cables and the basket are not.
The tower was {before} blocks high BEFORE. Judge the AFTER state: has the tower grown by exactly one block, with the new block resting on top, released (not held by the gripper), and is the tower standing (nothing fell)?
Answer only JSON: {{"grew_by_one": bool, "standing": bool, "tower_blocks_after": int, "something_fell": bool, "confidence": number}}"""


def grew_and_stands(before: dict, after: dict, tower_before: int) -> dict | None:
    """before/after: {"overhead": jpeg, "wrist": jpeg}. Returns Terra's answer plus "ms", or None."""
    ans, ms = call(GREW.format(before=tower_before),
                   [before.get("overhead"), after.get("overhead"), before.get("wrist"), after.get("wrist")])
    if ans is None or not isinstance(ans.get("grew_by_one"), bool):
        return None
    ans["ms"] = round(ms)
    return ans


STANDING = """Images of a robot station. Image 1: overhead camera. Image 2: wrist camera from a look pose.
Foam cubes with a round hole are blocks; sticks, balls, cables and the basket are not.
The robot expects a tower of {expected} blocks standing. Is that tower standing, and how many blocks high is it now?
Answer only JSON: {{"standing": bool, "tower_blocks": int, "fell": bool, "confidence": number}}"""


def tower_standing(now: dict, expected: int) -> dict | None:
    ans, ms = call(STANDING.format(expected=expected), [now.get("overhead"), now.get("wrist")])
    if ans is None or not isinstance(ans.get("standing"), bool):
        return None
    ans["ms"] = round(ms)
    return ans


CHOOSE = """Images of a robot station. Image 1: overhead camera. Image 2: wrist camera.
A robot arm is building one tower of foam cubes. These are the loose blocks it may pick next (index: colour, position in the arm's frame, cm):
{cands}
Choose the ONE block that is easiest to pick cleanly: free on all sides, not touching other blocks, not under anything, within easy reach, not near the tower. Sticks, balls and cables are not blocks.
Answer only JSON: {{"index": int, "reason": string, "confidence": number}}"""


def choose_next(now: dict, candidates: list[dict]) -> int | None:
    """Index into candidates, or None (the caller then takes its own first choice)."""
    if not candidates:
        return None
    lines = "\n".join(f"{i}: {c['colour']} at ({c['x'] * 100:.0f}, {c['y'] * 100:.0f})" + (" lying on its side" if c.get("lying") else "")
                      for i, c in enumerate(candidates))
    ans, _ = call(CHOOSE.format(cands=lines), [now.get("overhead"), now.get("wrist")])
    try:
        i = int(ans["index"])
        return i if 0 <= i < len(candidates) else None
    except Exception:  # noqa: BLE001
        return None
