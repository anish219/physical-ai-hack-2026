"""The two vision checks, both on the fixed overhead camera, both GPT-6.1 Terra with reasoning off.

fast_check: one call on the reference-versus-now composite (the prompt that gave 0 false "go"s in 562 certain-"no"
reads). agentic_check: the same question with tools to look again, zoom, or see frames from up to 3 s before or 2 s
after now, then a verdict (0 false "go"s in 326 checks). Any error, timeout, refusal or missing verdict is a "no".
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request

from . import imaging, settings

URL = "https://api.openai.com/v1/chat/completions"

FAST_PROMPT = """Four images from ONE fixed overhead camera at a robot station with a black USB power strip and orange USB cables.
Top row: zoomed view around the power strip. Bottom row: the full overhead view.
LEFT column is a reference taken earlier; RIGHT column is now. The camera does not move between them.
Cables are either plugged into the strip's USB ports, held in the robot gripper, or lying loose on the table. Some loose cables may already be on the table in the reference.
The question that matters: since the reference, has at least one more cable been pulled out of the strip AND is at least one of those cables now lying on the table, released, with no gripper holding it?
If the strip or cables are hidden by the arm, answer false.
Answer only JSON: {"plugged_ref": int, "plugged_now": int, "loose_on_table_ref": int, "loose_on_table_now": int, "gripper_holding_cable_now": bool, "new_cable_unplugged_and_resting_on_table": bool, "confidence": number 0-1}"""


def agentic_system(max_tools: int) -> str:
    return f"""You verify one fact for a robot competition, where a false yes disqualifies the team: since the reference moment, has a USB cable been pulled OUT of the black power strip's ports AND is that cable now lying on the table, released (no gripper holding it)?
You see a fixed overhead camera. The first image: top row is a zoom on the power strip (reference left, now right); bottom row is the full overhead view (reference left, now right). The camera does not move.
Loose cables may already lie on the table in the reference; they do not count. Only a cable that was plugged in at the reference and is now out and resting on the table counts.
You may call tools (at most {max_tools} calls) to look at frames from a few seconds before now, or up to 2 s after now, or to zoom in.
If anything is hidden, unclear or ambiguous, answer false. Finish by calling submit_verdict."""


TOOLS = [
    {"type": "function", "function": {"name": "get_frame", "description": "A full frame from a camera at now + t_offset_s (between -3 and +2 seconds).",
                                      "parameters": {"type": "object", "properties": {"camera": {"type": "string", "enum": ["overhead"]}, "t_offset_s": {"type": "number"}},
                                                     "required": ["camera", "t_offset_s"]}}},
    {"type": "function", "function": {"name": "zoom", "description": "A zoomed crop of a camera frame at now + t_offset_s. Box in normalized image coordinates (0 to 1).",
                                      "parameters": {"type": "object", "properties": {"camera": {"type": "string", "enum": ["overhead"]}, "t_offset_s": {"type": "number"},
                                                                                      "x0": {"type": "number"}, "y0": {"type": "number"}, "x1": {"type": "number"}, "y1": {"type": "number"}},
                                                     "required": ["camera", "t_offset_s", "x0", "y0", "x1", "y1"]}}},
    {"type": "function", "function": {"name": "submit_verdict", "description": "The final answer.",
                                      "parameters": {"type": "object", "properties": {"unplugged_and_on_table": {"type": "boolean"}, "confidence": {"type": "number"}, "evidence": {"type": "string"}},
                                                     "required": ["unplugged_and_on_table", "confidence", "evidence"]}}},
]


class VlmError(Exception):
    pass


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _post(body: dict, timeout_s: float) -> dict:
    key = settings.api_key()
    if not key:
        raise VlmError("no API key")
    if timeout_s <= 0.05:
        raise VlmError("no time left")
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", "authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001
            msg = ""
        raise VlmError(f"HTTP {e.code}: {msg[:160]}") from None
    except Exception as e:  # noqa: BLE001  (timeouts, resets, bad JSON)
        raise VlmError(f"{type(e).__name__}: {str(e)[:160]}") from None


def _parse_json(text: str):
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def fast_check(composite_jpeg: bytes, timeout_s: float | None = None) -> dict:
    """Returns {"yes": bool, "answer": dict|None, "ms": int, "error": str|None, "usage": dict}. Never raises."""
    t0 = time.time()
    timeout_s = float(timeout_s if timeout_s is not None else settings.get("FAST_TIMEOUT_S"))
    out = {"yes": False, "answer": None, "error": None, "usage": {}}
    try:
        if not composite_jpeg:
            raise VlmError("no composite")
        j = _post({"model": settings.get("VLM_MODEL"), "reasoning_effort": "none", "response_format": {"type": "json_object"},
                   "max_completion_tokens": 400,
                   "messages": [{"role": "user", "content": [{"type": "text", "text": FAST_PROMPT},
                                                             {"type": "image_url", "image_url": {"url": _data_url(composite_jpeg)}}]}]},
                  timeout_s)
        out["usage"] = j.get("usage", {})
        ans = _parse_json(j["choices"][0]["message"].get("content") or "")
        if not isinstance(ans, dict):
            raise VlmError("unparseable answer")
        out["answer"] = ans
        out["yes"] = ans.get("new_cable_unplugged_and_resting_on_table") is True
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:200]
    out["ms"] = int((time.time() - t0) * 1000)
    return out


def agentic_check(source, ref_jpeg: bytes, now_t: float, strip_box, budget_s: float | None = None,
                  max_tools: int | None = None) -> dict:
    """source: a frames.FrameSource; now_t: the time of the "now" frame. Returns
    {"yes", "confidence", "evidence", "tools", "asked", "ms", "error", "now_t"}. Never raises; anything but a clean
    submit_verdict of true is a "no"."""
    t0 = time.time()
    budget = float(budget_s if budget_s is not None else settings.get("AGENTIC_BUDGET_S"))
    max_tools = int(max_tools if max_tools is not None else settings.get("AGENTIC_MAX_TOOLS"))
    deadline = t0 + budget
    out = {"yes": False, "confidence": 0.0, "evidence": "", "tools": 0, "asked": [], "error": None, "now_t": now_t, "usage_in": 0, "usage_out": 0}
    try:
        now = source.at(now_t)
        if now is None:
            raise VlmError("no now frame")
        comp = imaging.composite(ref_jpeg, now.jpeg, strip_box)
        if not comp:
            raise VlmError("no composite")
        messages = [{"role": "system", "content": agentic_system(max_tools)},
                    {"role": "user", "content": [{"type": "text", "text": "Reference and now. Decide."},
                                                 {"type": "image_url", "image_url": {"url": _data_url(comp)}}]}]
        for turn in range(max_tools + 1):
            force = turn == max_tools
            j = _post({"model": settings.get("VLM_MODEL"), "reasoning_effort": "none", "max_completion_tokens": 600, "messages": messages,
                       "tools": TOOLS, "parallel_tool_calls": False,
                       "tool_choice": {"type": "function", "function": {"name": "submit_verdict"}} if force else "required"},
                      deadline - time.time())
            out["usage_in"] += j.get("usage", {}).get("prompt_tokens", 0)
            out["usage_out"] += j.get("usage", {}).get("completion_tokens", 0)
            msg = j["choices"][0]["message"]
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            calls = msg.get("tool_calls") or []
            final = next((c for c in calls if c["function"]["name"] == "submit_verdict"), None)
            if final:
                v = json.loads(final["function"]["arguments"])
                out["yes"] = v.get("unplugged_and_on_table") is True
                out["confidence"] = float(v.get("confidence") or 0.0)
                out["evidence"] = str(v.get("evidence", ""))[:300]
                break
            if not calls:
                raise VlmError("no tool call and no verdict")
            images = []
            for c in calls:
                out["tools"] += 1
                try:
                    a = json.loads(c["function"]["arguments"] or "{}")
                except ValueError:
                    a = {}
                off = max(-3.0, min(2.0, float(a.get("t_offset_s") or 0.0)))
                wait = max(0.0, min(off + 0.5, deadline - time.time() - 2.0))
                f = source.at(now_t + off, wait_s=wait)
                img = None
                if f is not None:
                    if c["function"]["name"] == "zoom":
                        img = imaging.zoom(f.jpeg, [a.get("x0", 0), a.get("y0", 0), a.get("x1", 1), a.get("y1", 1)])
                    else:
                        img = imaging.scaled(f.jpeg)
                out["asked"].append(f"{c['function']['name']}:{off:+.1f}")
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": f"Image for {c['function']['name']} overhead at now{off:+.1f} s follows." if img else "Frame unavailable."})
                if img:
                    images.append({"type": "image_url", "image_url": {"url": _data_url(img)}})
            if images:
                messages.append({"role": "user", "content": [{"type": "text", "text": "Requested image(s):"}, *images]})
        else:
            raise VlmError("no verdict")
    except Exception as e:  # noqa: BLE001
        out["yes"] = False
        out["error"] = str(e)[:200]
    out["ms"] = int((time.time() - t0) * 1000)
    return out
