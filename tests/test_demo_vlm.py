"""Protocol/label evaluation tests. These do not claim a real VLM has passed."""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dream_hackathon.demo_vlm import LocalVLM, OpenAIVLM, Verdict, run_cases


@pytest.mark.parametrize(
    "content",
    [
        '{"verdict": true, "reason": "x"}',
        '{"verdict": "yes"}',
        '{"verdict": "yes", "reason": 3}',
        '{"verdict": "maybe", "reason": "x"}',
        "[]",
    ],
)
def test_malformed_verdict_is_not_positive(content):
    with pytest.raises(ValueError):
        Verdict.parse(content)


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_real_http_protocol_and_no_expected_label_in_prompt(tmp_path, provider):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            assert self.path == ("/api/chat" if provider == "ollama" else "/responses")
            if provider == "openai":
                assert self.headers["Authorization"] == "Bearer test-key"
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            response = (
                {"message": {"content": '{"verdict": "yes", "reason": "visible"}'}}
                if provider == "ollama"
                else {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": '{"verdict": "yes", "reason": "visible"}',
                                }
                            ],
                        }
                    ],
                }
            )
            self.wfile.write(json.dumps(response).encode())

    (tmp_path / "frame.jpg").write_bytes(b"\xff\xd8\xff image fixture, not real inference")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            client = LocalVLM("test-model", url) if provider == "ollama" else OpenAIVLM("test-model", "test-key", url)
            report = run_cases(
                client,
                [{"image": "frame.jpg", "question": "Is the USB disconnected?", "expected": "no"}],
                tmp_path,
            )
        finally:
            server.shutdown()
            worker.join()
    assert not report["passed"] and report["correct"] == 0
    request = requests[0]
    if provider == "ollama":
        assert request["messages"][1]["content"] == "Is the USB disconnected?"
        encoded = request["messages"][1]["images"][0]
        assert request["stream"] is False and request["format"]["type"] == "object"
    else:
        content = request["input"][0]["content"]
        assert content[0]["text"] == "Is the USB disconnected?"
        assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
        encoded = content[1]["image_url"].split(",")[1]
        assert request["text"]["format"]["strict"] is True
        assert request["store"] is False
        assert "test-key" not in json.dumps(report)
    assert base64.b64decode(encoded) == (tmp_path / "frame.jpg").read_bytes()


@pytest.mark.parametrize(
    "response",
    [
        {"status": "incomplete"},
        {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
    ],
)
def test_hosted_nonverdict_is_reported_as_error(monkeypatch, response):
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: io.BytesIO(json.dumps(response).encode()))
    with pytest.raises(ValueError):
        OpenAIVLM("test-model", "test-key").verify(b"\xff\xd8\xff fixture", "Is it done?")


def test_unknown_is_recorded_without_turning_into_a_yes(tmp_path):
    class Client:
        model, url = "test", "local test double"

        def verify(self, image, question):
            return Verdict("unknown", "occluded"), {}

    (tmp_path / "frame.jpg").write_bytes(b"fixture")
    report = run_cases(Client(), [{"image": "frame.jpg", "question": "Visible?", "expected": "yes"}], tmp_path)
    assert not report["passed"]


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_custom_scene_schema_round_trips_structured_facts(monkeypatch, provider):
    scene = {"known": True, "block_count": 6, "collapsed": False, "standing_free": True}
    schema = {"type": "object", "properties": {"block_count": {"type": "integer"}}}
    requests = []

    def response(request, **_kwargs):
        requests.append(json.loads(request.data))
        payload = (
            {"message": {"content": json.dumps(scene)}}
            if provider == "ollama"
            else {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(scene)},
                        ],
                    }
                ],
            }
        )
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", response)
    client = LocalVLM("test") if provider == "ollama" else OpenAIVLM("test", "test-key")
    result, _ = client.inspect(b"\xff\xd8\xff fixture", "Inspect the tower", schema)
    assert result == scene
    sent_schema = requests[0]["format"] if provider == "ollama" else requests[0]["text"]["format"]["schema"]
    assert sent_schema == schema
