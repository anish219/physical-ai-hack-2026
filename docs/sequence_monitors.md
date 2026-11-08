# Sequence monitors

Each sequence subphase has an optional **monitor**. The editor on 8081 exposes it
on every Dream task and wait step. Anish's VLM (or any other checker) plugs in
here; the timed sequencer still advances after the window even if the monitor
says `no`. Do not add success gating unless the team agrees to change that.

## Editor field

`step.monitor` is stored with the shared plan in `~/.cache/dream_hackathon/sequences.json`:

```json
{
  "type": "none | prompt | script",
  "prompt": "VLM instruction, used when type is prompt",
  "script": "registered pointer, used when type is script"
}
```

| Type | What the operator fills in | What runs after the subphase window |
| --- | --- | --- |
| `none` | nothing | skip (`result: skipped`) |
| `prompt` | a text prompt on that step | `registry` prompt evaluator, with that prompt |
| `script` | a pointer from the dropdown | the callable registered under that pointer |

Prompt text is required to **run** a `prompt` monitor, not to save a draft.
A `script` pointer must be registered before **run**; unknown pointers can still
be saved so a plan survives a restart before the script module is deployed.

## Integration point

Edit [`src/dream_hackathon/monitor_scripts.py`](../src/dream_hackathon/monitor_scripts.py).
`SequenceManager` imports `register_all(registry)` once when the unified
controller starts. Restart 8081 after changing that file.

```python
from dream_hackathon.monitor_scripts import register_all

def register_all(registry):
    registry.set_prompt_evaluator(evaluate_vlm_prompt)
    registry.register_script("usb_unplugged", evaluate_usb_unplugged, label="USB unplugged")
```

`label` is the dropdown text. `pointer` is the stable id stored on the step
(1–160 characters). The live dropdown is the `monitor_scripts` list on
`GET /api/state`.

## Evaluator contract

```python
def evaluate_vlm_prompt(prompt: str, context: dict) -> dict: ...
def evaluate_usb_unplugged(context: dict) -> dict: ...
```

`context` is:

```python
{
    "selection": "left" | "right" | "both",
    "sides": ("left",) | ("right",) | ("left", "right"),
    "step": { ...subphase including monitor... },
    "arms": {"left": <arm snapshot>, "right": <arm snapshot>},
    "camera_urls": {"left": "http://127.0.0.1:8691", "right": "http://127.0.0.1:8692"},
}
```

Arm JPEG routes on those servers match the existing station cameras (wrist /
workspace as already exposed by each armserver). Pull frames with a normal HTTP
GET; do not open a second CAN adapter or camera.

Return one of:

```python
{"result": "yes", "detail": "optional"}
{"result": "no", "detail": "optional"}
{"result": "uncertain", "detail": "optional"}
```

A bare `"yes"` / `"no"` / `"uncertain"` string is also accepted. Anything else
fails the sequence. If no evaluator is registered, the run records
`{"result": "unwired", "detail": "..."}` and continues.

The latest result is `sequences.<selection>.monitor_result` in `/api/state` and
the `/api/events` `sequences` payload. The editor status line shows the
`result` string.

## What not to change

- Do not gate `SequenceRun.run` on monitor results yet. The hook is
  `SequenceRun.evaluate_monitor` after each window.
- Do not put model weights, API keys, or one-off notebooks in `sequences.py`.
  Keep the VLM client in `monitor_scripts.py` or a module it imports.
- HTML for 8081 is cached at process start; `sequences.js` reloads from disk.
  Restart unified_control after HTML/CSS changes.

## Quick check

```python
from dream_hackathon.sequences import MonitorRegistry, validate_definition

registry = MonitorRegistry()
registry.register_script("usb_unplugged", lambda context: "yes", label="USB unplugged")
plan = validate_definition({
    "phases": [{
        "id": "p", "title": "Check",
        "steps": [{
            "id": "s", "label": "Wait", "type": "wait", "durationSeconds": 0,
            "monitor": {"type": "script", "prompt": "", "script": "usb_unplugged"},
        }],
    }]
}, for_run=True, scripts=registry.pointers())
assert registry.evaluate(plan["phases"][0]["steps"][0]["monitor"], {"step": {}})["result"] == "yes"
```
