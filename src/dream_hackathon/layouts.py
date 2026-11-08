"""How each arm family's recordings are laid out: the vector order, the gripper unit and the camera keys.

The layouts match what `dreamcli robot run` sends a fine-tuned model on each arm, so a model trained on these
recordings runs there unchanged:

- YAM: `[right joints x6, right gripper, left joints x6, left gripper]`, the gripper from 0 (closed) to 1 (open).
- MakerMods (Metal and Maker arms): `[left joints x6, left gripper, right joints x6, right gripper]`, the gripper as
  the jaw stroke in millimetres (Metal: 100 mm open, Maker: 80 mm open). `tools/add_eef.py` then adds the tool poses
  Dream's MakerMods models train on.

Joints are radians in every layout.
"""

import json
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt

## The arm types a station config may name, each with the family that drives and records it.
FAMILIES: dict[str, str] = {"yam": "yam", "metal": "makermods", "maker": "makermods"}
## The arm sides in the order the recorded vector holds them.
SIDE_ORDER: dict[str, tuple[str, str]] = {"yam": ("right", "left"), "makermods": ("left", "right")}
## What the recorded gripper reads fully open, per arm type: the loop's gripper 1.0 times this.
GRIP_SCALE: dict[str, float] = {"yam": 1.0, "metal": 100.0, "maker": 80.0}
## The station's camera roles: one fixed camera over the workspace and one on each wrist.
CAMERA_ROLES: tuple[str, ...] = ("overhead", "wrist_l", "wrist_r")
## Each family's Dream camera role for the station's overhead camera: `dreamcli setup` names a YAM's `overhead` and a
## MakerMods arm's `scene`, and a fine-tune runs only on the camera roles it was registered with.
_SCENE_ROLE: dict[str, str] = {"yam": "overhead", "makermods": "scene"}
## Joints per arm.
JOINTS = 6
## Width of the recorded state and action vectors: per arm its joints and its gripper.
WIDTH = 2 * (JOINTS + 1)

_EMBODIMENTS_DIR = pathlib.Path(__file__).parent / "embodiments"


def embodiment(family: str) -> dict[str, Any]:
    """The Dream embodiment config a family's fine-tunes are registered under.

    @param family `"yam"` or `"makermods"`.
    @return The config dict (`state`, `action`, `video`), as `dream.DreamClient.add_embodiment` takes it.
    """
    return json.loads((_EMBODIMENTS_DIR / f"{family}.json").read_text())


def video_keys(family: str) -> dict[str, str]:
    """Each station camera role's dataset video key, e.g. `{"overhead": "observation.images.overhead", ...}`."""
    video = embodiment(family)["video"]
    return {"overhead": video[_SCENE_ROLE[family]], "wrist_l": video["wrist_l"], "wrist_r": video["wrist_r"]}


def to_vector(
    arm: str,
    joints_rad: dict[str, npt.NDArray[np.float64]],
    grip: dict[str, float],
) -> npt.NDArray[np.float32]:
    """Packs the supplied arms into one recorded vector, in the family's order.

    @param arm The arm type, `"yam"`, `"metal"` or `"maker"`.
    @param joints_rad Per side, six joint angles in radians.
    @param grip Per side, the gripper from 0 (closed) to 1 (open).
    @return Float32, seven values per supplied arm, in the family's order and gripper unit.
    """
    parts = []
    for side in SIDE_ORDER[FAMILIES[arm]]:
        if side not in joints_rad:
            continue
        parts.append(np.asarray(joints_rad[side], dtype=np.float64))
        parts.append([grip[side] * GRIP_SCALE[arm]])
    return np.concatenate(parts).astype(np.float32)
