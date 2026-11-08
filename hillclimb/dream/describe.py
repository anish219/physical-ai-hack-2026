"""What the Dream backend needs from the scene when the controller gives no block positions: the tower top's colour
and the loose blocks it could name, read by Terra from the current frames, then turned into a prompt with the
checkpoint's wording table (hillclimb.common.wording). Terra only chooses among labels it is told are allowed, and a
name the table cannot make unambiguous is never sent.
"""

from __future__ import annotations

from hillclimb.common import vision
from hillclimb.common.wording import TABLES, stack_prompt

COLOURS = ("orange", "purple", "green", "yellow", "blue")
AVOID = ("purple", "yellow")  # B misses these most often in the sim; they go last

PROMPT = """Two camera views of a robot station: image 1 is an overhead camera, image 2 the robot arm's wrist camera.
Foam cubes with a round hole are blocks (colours: orange, purple, green, yellow, blue; some repeat). Long foam sticks,
golf balls, cables, the power strip and the basket are NOT blocks: never count or choose them.
One stack of blocks is the tower; it is {height} block(s) high. In the overhead image the robot's forward is up the
image and the robot's right is the image's right.
List every LOOSE block (on the table, nothing on top of it, not part of the tower). For each, say whether another loose
block has the same colour, and if so which of these words singles it out: {quals}.
Answer only JSON: {{"tower_top_colour": str, "tower_top_lying": bool, "loose": [{{"colour": str, "lying": bool,
"same_colour_other": bool, "qualifier": str}}], "confidence": number}}"""


def read_scene(frames: dict, height: int, table: str) -> dict | None:
    """frames: {"overhead": jpeg, "wrist": jpeg}. Returns {"top": {...}, "loose": [...]} or None."""
    quals = [q for q in TABLES[table]["qualifiers"]]
    ans, ms = vision.call(PROMPT.format(height=height, quals=", ".join(f'"{q}"' for q in quals)),
                          [frames.get("overhead"), frames.get("wrist")])
    if not ans or ans.get("tower_top_colour") not in COLOURS or not isinstance(ans.get("loose"), list):
        return None
    loose = []
    for b in ans["loose"]:
        if b.get("colour") not in COLOURS:
            continue
        q = b.get("qualifier") or ""
        if b.get("same_colour_other") and q not in quals:
            continue  # cannot be named unambiguously
        loose.append({"colour": b["colour"], "lying": bool(b.get("lying")), "qualifier": q if b.get("same_colour_other") else ""})
    return {"top": {"colour": ans["tower_top_colour"], "lying": bool(ans.get("tower_top_lying"))}, "loose": loose, "ms": round(ms)}


def prompt_from_read(read: dict, table: str, tower_height: int) -> tuple[str, dict] | None:
    """The first nameable loose block (colours B misses most go last), formatted with the table's strings."""
    t = TABLES[table]
    top_c = read["top"]["colour"]
    order = sorted(read["loose"], key=lambda b: (b["colour"] == top_c, b["colour"] in AVOID))
    for b in order:
        x = f"{b['colour']} block"
        if b["lying"] and t["lying"]:
            x += t["lying"]
        if b["qualifier"]:
            x += " " + b["qualifier"]
        if table == "synth" and tower_height >= 2:
            return t["tower"].format(x=x), b
        top = read["top"]
        y = f"{top['colour']} block"
        if top.get("lying") and t["lying"]:
            y += t["lying"]
        if any(o["colour"] == top["colour"] for o in read["loose"]):
            if tower_height >= 2:
                y += " " + t.get("tower_qualifier", "at the top of the tower")
            else:
                continue  # a lone support with a same-colour twin needs positions to name; try another block
        return t["stack"].format(x=x, y=y), b
    return None


def prompt_from_positions(loose: list[dict], top: dict, table: str, hand_xy=(0.277, 0.0)) -> tuple[str, dict] | None:
    """When the scene carries positions (sim tests, or a perception that provides them). Order: never the top's
    colour (same-colour stacks confuse naming and checks), then colours B misses most last, then nearest the start
    pose's hand."""
    import math  # noqa: PLC0415
    cands = [b for b in loose if b["colour"] != top.get("colour")] or list(loose)
    order = sorted(cands, key=lambda b: (b["colour"] in AVOID, math.hypot(b["x"] - hand_xy[0], b["y"] - hand_xy[1])))
    for b in order:
        p = stack_prompt(b, top, loose, table)
        if p:
            return p, b
    return None
