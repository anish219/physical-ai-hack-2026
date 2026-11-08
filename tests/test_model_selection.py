"""Account model listing stays independent of policies and motor controls."""

import json
import subprocess
import sys
from types import SimpleNamespace

from test_armserver import request

from dream_hackathon import dream_caller, unified_control
from dream_hackathon.so101_dashboard import Controls


def test_catalog_subprocess_can_import_controller_in_separate_sdk_environment(monkeypatch):
    def run(command, **kwargs):
        assert command[-1] == "--list-models"
        assert str(unified_control.pathlib.Path(unified_control.__file__).resolve().parent.parent) in (
            kwargs["env"]["PYTHONPATH"].split(unified_control.os.pathsep)
        )
        return SimpleNamespace(stdout='{"models": []}')

    monkeypatch.setattr(unified_control.subprocess, "run", run)
    assert unified_control.list_models("sdk-python") == {"models": []}


def test_sdk_catalog_uses_exact_checkpoint_ids_without_opening_policy(monkeypatch):
    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_models(self):
            return [SimpleNamespace(id="dream-1-ft:account-task", owner="mine", servable=True)]

    monkeypatch.setitem(sys.modules, "dream", SimpleNamespace(DreamClient=Client))
    assert dream_caller.available_models() == [{"id": "dream-1-ft:account-task", "owner": "mine", "servable": True}]


def test_catalog_endpoint_and_manual_checkpoint_drafts_remain_independent():
    models = {"models": [{"id": "dream-1-ft:account-task", "owner": "mine", "servable": True}]}
    leader = SimpleNamespace(side=None, state=lambda: Controls(1, "none").snapshot())
    callers = {"right": SimpleNamespace(snapshot=lambda: {})}
    controller = unified_control.UnifiedController({}, leader, callers, model_loader=lambda: models)
    page = unified_control.UnifiedPage(controller, 0)
    base = f"http://127.0.0.1:{page.port}"
    try:
        assert json.loads(request(base, "/api/models")[1]) == models
        assert controller.api_drafts["right"]["checkpoint"] == "dreamlabs/dream-1-beta"
        code, body, _ = request(base, "/api/dream/right/draft", {"checkpoint": models["models"][0]["id"]})
        assert code == 200 and json.loads(body)["fields"]["checkpoint"] == "dream-1-ft:account-task"
        assert controller.api_drafts["left"]["checkpoint"] == "dreamlabs/dream-1-beta"
        controller.update_draft("right", {"checkpoint": "manual-id"})
        assert json.loads(request(base, "/api/models")[1]) == models
        assert controller.api_drafts["right"]["checkpoint"] == "manual-id"
        controller.model_loader = lambda: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["sdk"], stderr="private SDK diagnostics")
        )
        code, body, _ = request(base, "/api/models")
        assert code == 502 and b"private SDK diagnostics" not in body
    finally:
        page.close()
