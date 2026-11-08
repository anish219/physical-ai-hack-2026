"""The station lock: one holder at a time, shared with dreamcli's, and a refusal that says who holds it."""

import pathlib
import subprocess
import sys

import pytest

from dream_hackathon import station_lock


def test_one_holder_at_a_time_and_released_on_close(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "arms.lock"
    lock = station_lock.acquire("dhk-collect", tmp_path / "cubes", path=path)
    held = station_lock.holder(path)
    assert held["holder"] == "dhk-collect" and held["dataset"] == str(tmp_path / "cubes")
    with pytest.raises(station_lock.StationBusyError, match=r"in use by dhk-collect \(pid \d+\).*recording into"):
        station_lock.acquire("scripts/check_arms.py", path=path)
    lock.close()
    assert station_lock.holder(path) is None
    station_lock.acquire("scripts/check_arms.py", path=path).close()


def test_reads_the_line_dreamcli_writes(tmp_path: pathlib.Path) -> None:
    found = station_lock._parse("4242 dreamcli robot run\n")
    assert found == {"pid": "4242", "holder": "dreamcli robot run"}
    assert station_lock.describe(found).startswith("the station's arms are in use by dreamcli robot run (pid 4242)")


def test_our_first_line_is_dreamclis_format(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "arms.lock"
    lock = station_lock.acquire("dhk-collect", path=path)
    pid, _, holder = path.read_text().splitlines()[0].partition(" ")
    assert pid.isdigit() and holder == "dhk-collect"
    lock.close()


def test_opposite_arms_can_be_owned_by_separate_processes(tmp_path):
    path = tmp_path / "arms.lock"
    with station_lock.acquire_sides("Oculus", ["left"], path=path):
        code = (
            "from dream_hackathon import station_lock; import pathlib,sys; "
            "station_lock.acquire_sides('custom', [sys.argv[2]], path=pathlib.Path(sys.argv[1])).close()"
        )
        right = subprocess.run([sys.executable, "-c", code, str(path), "right"], capture_output=True, text=True)
        assert right.returncode == 0, right.stderr
        left = subprocess.run([sys.executable, "-c", code, str(path), "left"], capture_output=True, text=True)
        assert left.returncode != 0 and "Oculus" in left.stderr
        with pytest.raises(station_lock.StationBusyError):
            station_lock.acquire("legacy dreamcli", path=path)
    station_lock.acquire("legacy dreamcli", path=path).close()


def test_whole_station_owner_conflicts_with_either_scoped_owner(tmp_path):
    path = tmp_path / "arms.lock"
    with station_lock.acquire("dreamcli", path=path):
        for side in ("left", "right"):
            with pytest.raises(station_lock.StationBusyError, match="dreamcli"):
                station_lock.acquire_sides("custom", [side], path=path)
    with station_lock.acquire_sides("custom", ["left", "right"], path=path):
        assert station_lock.holder(path)["holder"] == "custom"


def test_each_scoped_dataset_is_visible_and_metadata_can_follow_selection(tmp_path):
    path = tmp_path / "arms.lock"
    with (
        station_lock.acquire_sides("Oculus", ["left"], tmp_path / "left", path=path) as left,
        station_lock.acquire_sides("custom", ["right"], tmp_path / "right", path=path),
    ):
        assert {h["dataset"] for h in station_lock.holders(path)} == {
            str(tmp_path / "left"),
            str(tmp_path / "right"),
        }
        left.set_dataset(tmp_path / "both")
        assert station_lock.holders(path)[0]["dataset"] == str(tmp_path / "both")
