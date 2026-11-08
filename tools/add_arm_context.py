"""Export seven-value single-arm recordings with an explicitly supplied static opposite-arm pose.

Writes a new 14-value dataset; the source is unchanged. The supplied pose is static
context, not a claim that the opposite arm was measured during historical episodes.
Run tools/add_eef.py on the result before using the MakerMods Dream embodiment.
"""

import argparse
import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from add_eef import _link_or_copy, _stats


def convert(source, target, side, pose):
    info = json.loads((source / "meta/info.json").read_text())
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        expected = [f"{side}_joint_{i}" for i in range(6)] + [f"{side}_gripper"]
        if feature["shape"] != [7] or feature["names"] != expected:
            raise ValueError(f"{key} is not the seven-value {side} layout")
    other = "left" if side == "right" else "right"
    joints = np.asarray(pose["joints_rad"], dtype=np.float32)
    if pose["side"] != other or joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError(f"Expected a measured {other} context pose")
    grip = float(pose["grip"])
    if not np.isfinite(grip):
        raise ValueError("Context grip must be finite")
    from dream_hackathon.layouts import GRIP_SCALE

    context = np.append(joints, grip * GRIP_SCALE[info["robot_type"]]).astype(np.float32)
    # copytree refuses an existing output; never replace the source or old exports.
    shutil.copytree(source, target, copy_function=_link_or_copy, ignore=shutil.ignore_patterns("*.parquet"))
    stats = [json.loads(line) for line in (source / "meta/episodes_stats.jsonl").read_text().splitlines()]
    by_episode = {row["episode_index"]: row for row in stats}
    for path in sorted(source.glob("data/chunk-*/episode_*.parquet")):
        table = pq.read_table(path)
        for key in ("observation.state", "action"):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            static = np.broadcast_to(context, (len(values), 7))
            values = np.concatenate((static, values) if side == "right" else (values, static), axis=1)
            column = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), 14)
            table = table.set_column(table.schema.get_field_index(key), key, column)
            by_episode[table["episode_index"][0].as_py()]["stats"][key] = _stats(values)
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
    names = [name for s in ("left", "right") for name in
             ([f"{s}_joint_{i}" for i in range(6)] + [f"{s}_gripper"])]
    for key in ("observation.state", "action"):
        info["features"][key] = {"dtype": "float32", "shape": [14], "names": names}
    info["static_arm_context"] = {"controlled_side": side, "pose": pose,
                                  "source": "explicit static context supplied at export"}
    (target / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (target / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--context-pose", required=True, type=pathlib.Path)
    args = parser.parse_args()
    convert(args.dataset.expanduser(), args.out.expanduser(), args.side,
            json.loads(args.context_pose.expanduser().read_text()))
    print(f"Wrote {args.out}; next run tools/add_eef.py on this export")


if __name__ == "__main__":
    main()
