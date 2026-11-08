"""Lists a dataset's episodes, and drops the ones you do not want to train on.

```
uv run dhk-dataset list ~/hackathon_data/myteam-cubes
uv run dhk-dataset drop ~/hackathon_data/myteam-cubes 3 7
```

B throws away the episode being recorded; this is for one you notice later (watch them on the dashboard). Dropping
deletes the episodes' files and renumbers the ones after them, so the numbers `list` shows change. It refuses while a
collector records into the same folder.

@see main()
"""

import argparse
import json
import pathlib
import sys

from dream_hackathon import dataset, station_lock


def list_command(root: pathlib.Path) -> None:
    """Prints one line per episode, then the totals."""
    episodes = dataset.list_episodes(root)
    if not episodes:
        print(f"{root}: no episodes yet")
        return
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    for episode in episodes:
        tasks = "; ".join(episode.tasks)
        print(f"{episode.index:4d}  {episode.frames / fps:6.1f} s  {episode.frames:6d} frames  {tasks}")
    frames = sum(episode.frames for episode in episodes)
    print(f"{len(episodes)} episodes, {frames / fps / 60:.1f} minutes")


def drop_command(root: pathlib.Path, indices: list[int], *, yes: bool) -> None:
    """Drops episodes after asking, unless `yes`."""
    for holder in station_lock.holders():
        if holder.get("dataset") == str(root.resolve()):
            sys.exit(
                f"a collector is recording into {root} right now ({station_lock.describe(holder)}); stop it first"
            )
    episodes = {episode.index: episode for episode in dataset.list_episodes(root)}
    unknown = sorted(set(indices) - set(episodes))
    if unknown:
        sys.exit(f"{root} has no episode {unknown[0]}: `dhk-dataset list {root}` shows them")
    for index in sorted(set(indices)):
        print(f"  drop episode {index}: {episodes[index].frames} frames, {'; '.join(episodes[index].tasks)}")
    if not yes and input("delete these episodes' files and renumber the rest? [y/N] ").strip().lower() != "y":
        print("nothing changed")
        return
    left = dataset.drop_episodes(root, set(indices))
    print(f"dropped; {left} episodes left, numbered 0 to {left - 1}")
    robot = json.loads((root / "meta" / "info.json").read_text()).get("robot_type")
    if robot in ("metal", "maker"):
        print(f"then rebuild the dataset you fine-tune on: tools/add_eef.py {root} --arm {robot} --replace")


def main(argv: list[str] | None = None) -> None:
    """The `dhk-dataset` command."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list the episodes")
    listing.add_argument("dataset", type=pathlib.Path)
    dropping = commands.add_parser("drop", help="delete episodes and renumber the rest")
    dropping.add_argument("dataset", type=pathlib.Path)
    dropping.add_argument("episodes", type=int, nargs="+", help="the episode numbers `list` shows")
    dropping.add_argument("--yes", action="store_true", help="do not ask first")
    args = parser.parse_args(argv)
    root = args.dataset.expanduser()
    if not (root / "meta" / "info.json").exists():
        sys.exit(f"{root} is not a dataset (no meta/info.json)")
    if args.command == "list":
        list_command(root)
    else:
        drop_command(root, args.episodes, yes=args.yes)


if __name__ == "__main__":
    main()
