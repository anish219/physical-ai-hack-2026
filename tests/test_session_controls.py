"""The dashboard's session controls: typing the task, and deleting chosen episodes. No hardware needed."""

import json
import pathlib
import urllib.error
import urllib.request

import dream
import numpy as np
from conftest import write_episodes
from test_collect import FakeArms, FakeQuest, _collector

from dream_hackathon import collect, dashboard, dataset, layouts


def _controls(root: pathlib.Path, task: str = "pick up the cube") -> collect.SessionControls:
    keys = list(layouts.video_keys("yam").values())
    return collect.SessionControls(
        lambda: dataset.DatasetWriter(root, 30, "yam", [f"v{i}" for i in range(14)], keys, 320, 180), task
    )


def test_the_task_follows_dreams_rules(tmp_path: pathlib.Path) -> None:
    controls = _controls(tmp_path / "data")
    assert controls.set_task("  put the red cube in the bowl ") == (
        True,
        "task set: new episodes record 'put the red cube in the bowl'",
    )
    assert controls.task == "put the red cube in the bowl"
    assert not controls.set_task("two\nlines")[0] and controls.task == "put the red cube in the bowl"
    assert not controls.set_task("x" * 513)[0]
    assert controls.set_task("   ")[0] and controls.task == collect.DEFAULT_TASK


def test_deleting_renumbers_and_the_next_episode_follows_on(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "data"
    write_episodes(root, "yam", episodes=4, frames=60, tasks=["a", "b", "c", "d"])
    controls = _controls(root)
    ok, message = controls.delete([1, 2])
    assert ok and message == "deleted episodes 1, 2; 2 left, the later ones renumbered"
    assert [e.tasks for e in dataset.list_episodes(root)] == [("a",), ("d",)]
    assert controls.writer.total_episodes == 2  # the writer was reloaded from the folder
    episode = controls.writer.start_episode("e")
    frame = np.zeros((180, 320, 3), np.uint8)
    for _ in range(60):
        episode.add_frame(np.zeros(14), np.zeros(14), {key: frame for key in layouts.video_keys("yam").values()})
    assert episode.save() == 2
    assert dream.datasets.validate_dataset(root, layouts.embodiment("yam")).ok
    assert not controls.delete([7])[0]  # no such episode: nothing deleted
    assert len(dataset.list_episodes(root)) == 3


def test_no_delete_while_an_episode_records_and_no_recording_while_deleting(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs)
    controls = collector._controls
    controls.set_task("stack the cups")

    headset.press = {"A"}
    collector.tick()  # starts an episode, with the typed task
    assert collector.episode is not None and collector.episode.task == "stack the cups"
    assert controls.delete([0]) == (False, "an episode is recording: save it (A) or discard it (B) first")
    headset.press = {"B"}
    collector.tick()  # discarded: the dataset is free again
    assert collector.episode is None and not controls.lock.locked()

    assert controls.lock.acquire(blocking=False)  # a delete in progress
    headset.press = {"A"}
    collector.tick()
    assert collector.episode is None and any("being deleted" in line for line in logs)
    controls.lock.release()


def _post(board: dashboard.Dashboard, path: str, body: dict, headers: dict | None = None) -> tuple[int, dict | str]:
    headers = {"X-Dashboard": "1", "Content-Type": "application/json"} if headers is None else headers
    request = urllib.request.Request(
        f"http://127.0.0.1:{board.port}{path}", data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def test_the_page_sets_the_task_and_deletes_chosen_episodes(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "data"
    write_episodes(root, "yam", episodes=3, frames=60, tasks=["a", "b", "c"])
    controls = _controls(root)
    board = dashboard.Dashboard(
        0,
        lambda role: np.zeros((180, 320, 3), np.uint8),
        layouts.video_keys("yam"),
        root,
        title="test station",
        on_task=controls.set_task,
        on_delete=controls.delete,
        task_of=lambda: controls.task,
        on_control=controls.set_controlled_arms,
        control_of=lambda: controls.controlled_arms,
    )
    try:
        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{board.port}/api/status", timeout=5).read())
        assert (status["task"], status["can_set_task"], status["can_delete"]) == ("pick up the cube", True, True)
        assert status["can_select_arms"] and status["controlled_arms"] == "right"
        for selection in ("left", "both", "right"):
            assert _post(board, "/api/control", {"arms": selection})[1]["ok"]
            assert controls.controlled_arms == selection
        assert not _post(board, "/api/control", {"arms": "invalid"})[1]["ok"]
        assert controls.controlled_arms == "right"
        assert _post(board, "/api/control", {"arms": "left"}, headers={})[0] == 403
        assert _post(board, "/api/task", {"task": "open the drawer"})[1]["ok"]
        assert controls.task == "open the drawer"
        assert _post(board, "/api/task", {"task": "x"}, headers={"Content-Type": "application/json"})[0] == 403
        assert _post(board, "/api/episodes/delete", {"episodes": ["one"]})[0] == 400
        status, reply = _post(board, "/api/episodes/delete", {"episodes": [0]})
        assert status == 200 and reply["ok"] and len(dataset.list_episodes(root)) == 2
    finally:
        board.close()


def test_a_page_without_a_session_offers_neither(tmp_path: pathlib.Path) -> None:
    board = dashboard.Dashboard(
        0, lambda role: np.zeros((180, 320, 3), np.uint8), layouts.video_keys("yam"), None, title="camera check"
    )
    try:
        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{board.port}/api/status", timeout=5).read())
        assert (status["task"], status["can_set_task"], status["can_delete"]) == (None, False, False)
        assert _post(board, "/api/task", {"task": "x"})[0] == 404
        assert _post(board, "/api/episodes/delete", {"episodes": [0]})[0] == 404
    finally:
        board.close()
