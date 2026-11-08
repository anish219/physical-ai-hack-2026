"""Per-arm ownership, compatible with dreamcli's exclusive whole-station lock.

Scoped controllers share `~/.dreamlabs/arms.lock` and exclusively lock `arms.left.lock` and/or `arms.right.lock`.
Legacy acquire() callers and dreamcli own the global file exclusively. Opposite-arm controllers can coexist.
The kernel releases locks on exit.

The file's first line is `<pid> <holder>`, the format `dreamcli` writes; ours adds a second line of JSON saying who
started it, from where, and which dataset it records into.

@see acquire(), holder()
"""

import datetime
import fcntl
import getpass
import json
import os
import pathlib
from collections.abc import Iterable
from contextlib import ExitStack
from typing import IO, Any

## dreamcli's lock on the arms (`robots.arms_lock`, in its home `~/.dreamlabs`).
LOCK_PATH = pathlib.Path("~/.dreamlabs/arms.lock").expanduser()


class ArmLocks(ExitStack):
    """Scoped locks, with recording metadata that follows a dashboard selection change."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[IO[str]] = []

    def set_dataset(self, dataset: pathlib.Path) -> None:
        for handle in self.records:
            handle.seek(0)
            first, _, rest = handle.read().partition("\n")
            details = json.loads(rest)
            details["dataset"] = str(dataset)
            handle.seek(0)
            handle.truncate()
            handle.write(first + "\n" + json.dumps(details) + "\n")
            handle.flush()


def acquire_sides(
    holder: str, sides: Iterable[str], dataset: pathlib.Path | None = None, path: pathlib.Path | None = None
) -> ArmLocks:
    """Own specific sides. Opposite-side controllers coexist; legacy whole-station owners stay exclusive.

    Hold a shared lock on dreamcli's file, and an exclusive lock on each requested side file.
    Closing the returned stack releases all locks. Custom controllers must use this same API.
    """
    path = path or LOCK_PATH
    chosen = sorted(set(sides))
    if not chosen or any(side not in ("left", "right") for side in chosen):
        raise ValueError("sides must contain left and/or right")
    stack = ArmLocks()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        shared = stack.enter_context(open(path, "a+", encoding="utf-8"))  # noqa: SIM115 - lifetime is the returned stack
        try:
            fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StationBusyError(describe(_parse(_read(path)))) from None
        for side in chosen:
            record = stack.enter_context(acquire(holder, dataset, path=path.with_name(f"arms.{side}.lock")))
            stack.records.append(record)
    except BaseException:
        stack.close()
        raise
    return stack


class StationBusyError(RuntimeError):
    """Another program holds the station's arms; the message says which."""


def acquire(holder: str, dataset: pathlib.Path | None = None, path: pathlib.Path | None = None) -> IO[str]:
    """Takes the station's lock; it is held until the returned file is closed or the process exits.

    @param holder What holds it, e.g. `"dhk-collect"`.
    @param dataset The dataset folder being recorded into, if any.
    @param path The lock file; None for LOCK_PATH.
    @return The open lock file.
    @throws StationBusyError Another program holds the lock.
    """
    path = path or LOCK_PATH
    handle = _try_lock(path)
    if handle is None:
        found = holders(path)
        raise StationBusyError(describe(found[0] if found else _parse(_read(path))))
    details = {
        "user": getpass.getuser(),
        "from": os.environ.get("SSH_CONNECTION", "").split(" ")[0] or "this computer",
        "since": datetime.datetime.now().strftime("%H:%M"),
        "dataset": str(dataset) if dataset is not None else None,
    }
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()} {holder}\n{json.dumps(details)}\n")
    handle.flush()
    return handle


def holder(path: pathlib.Path | None = None) -> dict[str, Any] | None:
    """The first current owner, or None. Use holders() when checking all recording datasets."""
    found = holders(path)
    return found[0] if found else None


def holders(path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    """Who holds the lock now, without keeping it.

    @param path The lock file; None for LOCK_PATH.
    @return One `{"pid", "holder", ...}` record per owner, or an empty list when free.
    """
    path = path or LOCK_PATH
    handle = _try_lock(path)
    if handle is not None:
        handle.close()
        return []
    # A scoped owner holds a shared global lock and writes its details in a side file.
    found = []
    for side in ("left", "right"):
        side_path = path.with_name(f"arms.{side}.lock")
        side_handle = _try_lock(side_path)
        if side_handle is None:
            found.append(_parse(_read(side_path)))
        else:
            side_handle.close()
    return found or [_parse(_read(path))]


def describe(found: dict[str, Any]) -> str:
    """The refusal line for a held lock, from its holder() record."""
    who = found["holder"] or "another program"
    pid = f" (pid {found['pid']})" if found["pid"] else ""
    extra = ""
    if found.get("user"):
        extra = f", started by {found['user']} from {found.get('from')} at {found.get('since')}"
    if found.get("dataset"):
        extra += f", recording into {found['dataset']}"
    return f"the station's arms are in use by {who}{pid}{extra}: stop it (Ctrl+C in its terminal), then try again"


def _try_lock(path: pathlib.Path) -> IO[str] | None:
    """The locked file, or None when someone else holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a+" never empties the file on opening, so a refused opener cannot wipe out the holder's details.
    handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - held open: the lock lasts as long as the file
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse(text: str) -> dict[str, Any]:
    """`{"pid", "holder"}` from the first line, plus our JSON details from the second when they are there."""
    first, _, rest = text.strip().partition("\n")
    pid, _, name = first.partition(" ")
    found: dict[str, Any] = {"pid": pid if pid.isdigit() else None, "holder": name.strip() or None}
    try:
        details = json.loads(rest) if rest else {}
    except ValueError:
        details = {}
    if isinstance(details, dict):
        found.update(details)
    return found
