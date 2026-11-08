"""Prompt wording, one table per checkpoint, copied from the training labels (never paraphrase these strings).

- "real": checkpoints B (blocks-left-fresh500) and C (stack-left-scripted500), from the station's ~/ft1/wording_B_C.txt.
- "synth": fine-tune #2 (...-realsynth500-20261108), from pah-synth sim/synth.py COMMANDS and QUALIFIERS at 5fb8a79.

Block names: "<colour> block", plus " lying on its side" for the real table when the block lies on its side, plus one
qualifier only when another candidate of the same colour exists. Left/right uses the left arm's base frame (y to the
robot's left) and needs a 3 cm margin over every other same-colour candidate; when no qualifier separates them the
name is ambiguous and the caller must choose another block.
"""
from __future__ import annotations

import math

TABLES = {
    "real": {
        "stack": "stack the {x} on top of the {y}",
        "unstack": "take the {x} off the {y} and put it on the table",
        "unstack_top": "take the top {colour} block off the tower and put it on the table",
        "move": "move the {x} to an open spot on the table",
        "tower_qualifier": "at the top of the tower",
        "qualifiers": ("on the robot's left", "on the robot's right"),
        "lying": " lying on its side",
    },
    "synth": {
        "stack": "stack the {x} on top of the {y}",
        "tower": "put the {x} on top of the tower",
        "table": "put the {x} down on the table",
        "straighten": "straighten the {x} on top of the tower",
        "qualifiers": ("nearest the robot's base", "farthest from the robot's base", "on the robot's left", "on the robot's right"),
        "lying": "",
    },
}
CHECKPOINT_TABLE = {
    "dream-1-ft:f9ed26ed-blocks-left-fresh500-20261108": "real",
    "dream-1-ft:f9ed26ed-stack-left-scripted500-20261108": "real",
    "dream-1-ft:f9ed26ed-blocks-left-b-realsynth500-20261108": "synth",
    "dream-1-ft:f9ed26ed-blocks-left-realsynth500-20261108": "synth",
}
MARGIN = 0.03


def table_for(checkpoint: str) -> str:
    return CHECKPOINT_TABLE.get(checkpoint, "real")


def name_block(block: dict, others: list[dict], table: str = "real") -> str | None:
    """'<colour> block[ lying on its side][ qualifier]', or None when same-colour candidates can't be told apart.
    block and others: {"id", "colour", "x", "y", "lying"} in the left arm's base frame (metres)."""
    t = TABLES[table]
    name = f"{block['colour']} block"
    if block.get("lying") and t["lying"]:
        name += t["lying"]
    same = [o for o in others if o is not block and o.get("id") != block.get("id") and o.get("colour") == block.get("colour")]
    if not same:
        return name
    y, r = block["y"], math.hypot(block["x"], block["y"])
    if all(y - o["y"] >= MARGIN for o in same):
        return f"{name} on the robot's left"
    if all(o["y"] - y >= MARGIN for o in same):
        return f"{name} on the robot's right"
    if table == "synth":
        if all(math.hypot(o["x"], o["y"]) - r >= MARGIN for o in same):
            return f"{name} nearest the robot's base"
        if all(r - math.hypot(o["x"], o["y"]) >= MARGIN for o in same):
            return f"{name} farthest from the robot's base"
    return None


def stack_prompt(block: dict, top: dict | None, loose: list[dict], table: str = "real", tower_ids: set | None = None) -> str | None:
    """The prompt that puts `block` on the tower. None if a name would be ambiguous."""
    t = TABLES[table]
    x = name_block(block, loose, table)
    if x is None:
        return None
    if top is None:  # no tower yet: the synth table has a word for starting one, the real table does not
        return t["table"].format(x=x) if "table" in t else None
    if table == "synth" and "tower" in t:
        return t["tower"].format(x=x)
    y = f"{top['colour']} block"
    if top.get("lying") and t["lying"]:
        y += t["lying"]
    if any(o.get("colour") == top.get("colour") for o in loose):
        y += " " + t.get("tower_qualifier", "at the top of the tower")
    return t["stack"].format(x=x, y=y)
