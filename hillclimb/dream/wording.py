"""Prompt wording, one table per checkpoint family. Never retype these strings: the policy only knows the exact
wording it was trained on.

- "b_c": checkpoints blocks-left-fresh500 (B) and stack-left-scripted500 (C), trained on the real night labels
  (station ~/ft1/wording_B_C.txt). Same colour: "on the robot's left" / "on the robot's right" (base-frame y, 3 cm
  margin); a support inside a tower whose colour is not unique: "... at the top of the tower"; "lying on its side"
  with no comma; no size words.
- "synth": fine-tune #2 (the ...-realsynth500-20261108 checkpoints), trained on pah-synth sim/synth.py COMMANDS and
  QUALIFIERS at commit 5fb8a79, copied below verbatim and pinned by a hash.

A name that cannot single out its block returns None: the caller must pick another block, never send it.
"""

from __future__ import annotations

import hashlib
import json
import math

# ---- copied verbatim from pah-synth sim/synth.py at 5fb8a79 (do not edit; the hash below guards it)
SYNTH_COMMANDS = {
    "stack": "stack the {x} on top of the {y}",
    "tower": "put the {x} on top of the tower",
    "table": "put the {x} down on the table",
    "straighten": "straighten the {x} on top of the tower",
}
SYNTH_QUALIFIERS = ("nearest the robot's base", "farthest from the robot's base", "on the robot's left", "on the robot's right")
SYNTH_HASH = "7cda4d72"  # first 8 hex of sha256 over json.dumps([COMMANDS, QUALIFIERS], sort_keys=True)

# ---- the real night labels' forms (B and C), from ~/ft1/wording_B_C.txt
BC_COMMANDS = {
    "stack": "stack the {x} on top of the {y}",
    "unstack": "take the {x} off the {y} and put it on the table",
    "unstack_top": "take the top {colour} block off the tower and put it on the table",
    "table": "move the {x} to an open spot on the table",
}
BC_QUALIFIERS = ("on the robot's left", "on the robot's right")
TOWER_TOP = "at the top of the tower"
LYING = "lying on its side"
MARGIN = 0.03

CHECKPOINT_FAMILY = {
    "dream-1-ft:f9ed26ed-blocks-left-fresh500-20261108": "b_c",
    "dream-1-ft:f9ed26ed-stack-left-scripted500-20261108": "b_c",
    "dream-1-ft:f9ed26ed-stacking-left-500-20261107": "b_c",
    "dream-1-ft:f9ed26ed-blocks-left-b-realsynth500-20261108": "synth",
    "dream-1-ft:f9ed26ed-blocks-left-realsynth500-20261108": "synth",
}


def synth_hash() -> str:
    return hashlib.sha256(json.dumps([SYNTH_COMMANDS, list(SYNTH_QUALIFIERS)], sort_keys=True).encode()).hexdigest()[:8]


def family(checkpoint: str) -> str:
    fam = CHECKPOINT_FAMILY.get(checkpoint)
    if fam is None:
        fam = "synth" if "realsynth" in checkpoint else "b_c"
    if fam == "synth" and synth_hash() != SYNTH_HASH:
        raise RuntimeError("synth wording table changed: re-copy it from sim/synth.py and update SYNTH_HASH")
    return fam


def _single_out(block: dict, among: list[dict], qualifiers: tuple[str, ...]) -> str | None:
    """'' when the colour is unique among candidates, a qualifier that is true with 3 cm to spare, or None."""
    others = [b for b in among if b["id"] != block["id"]]
    if not others:
        return ""
    r = lambda b: math.hypot(b["x"], b["y"])  # noqa: E731  distance from the left arm's base
    checks = {
        "nearest the robot's base": all(r(block) + MARGIN < r(o) for o in others),
        "farthest from the robot's base": all(r(block) - MARGIN > r(o) for o in others),
        "on the robot's left": all(block["y"] - MARGIN > o["y"] for o in others),
        "on the robot's right": all(block["y"] + MARGIN < o["y"] for o in others),
    }
    for q in qualifiers:
        if checks[q]:
            return q
    return None


def _name(block: dict, qual: str, fam: str) -> str:
    parts = [f"{block['colour']} block"]
    if qual:
        parts.append(qual)
    if fam == "b_c" and block.get("lying"):
        parts.append(LYING)
    return " ".join(parts)


def stack_prompt(checkpoint: str, block: dict, top: dict, loose: list[dict], tower_count: int = 1) -> str | None:
    """The prompt that puts `block` (loose, nothing on it) on `top` (the tower's top block), or None if either name
    is ambiguous. Blocks are dicts with id, colour, x, y (m, left arm base frame) and optional lying (bool, from
    perception only). `loose` holds every loose uncovered block, `block` included. `tower_count` is the number of
    stacks of 2 or more standing (the synth "tower" form needs exactly one)."""
    fam = family(checkpoint)
    same = [b for b in loose if b["colour"] == block["colour"]]
    if fam == "synth":
        qx = _single_out(block, same, SYNTH_QUALIFIERS)
        if qx is None:
            return None
        x = _name(block, qx, fam)
        if top.get("level", 0) >= 1 and tower_count == 1:
            return SYNTH_COMMANDS["tower"].format(x=x)
        tops = [b for b in loose if b["colour"] == top["colour"] and b["id"] != block["id"]] + [top]
        qy = _single_out(top, tops, SYNTH_QUALIFIERS)
        if qy is None:
            return None
        return SYNTH_COMMANDS["stack"].format(x=x, y=_name(top, qy, fam))
    qx = _single_out(block, same, BC_QUALIFIERS)
    if qx is None:
        return None
    x = _name(block, qx, fam)
    others_same = [b for b in loose if b["colour"] == top["colour"] and b["id"] != block["id"]]
    if not others_same:
        y = _name(top, "", fam)
    elif top.get("level", 0) >= 1:
        y = _name(top, "", fam) + " " + TOWER_TOP
    else:
        qy = _single_out(top, others_same + [top], BC_QUALIFIERS)
        if qy is None:
            return None
        y = _name(top, qy, fam)
    return BC_COMMANDS["stack"].format(x=x, y=y)


def hold_prompt(checkpoint: str) -> str:
    """The do-nothing prompt for keep-warm calls; its rows are never played."""
    return "hold the arm still"
