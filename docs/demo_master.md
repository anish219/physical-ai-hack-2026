# Local final-demo coordinator (work in progress)

These new modules are an isolated backup implementation. They do not import the
Dream caller, operate motors, change station processes, download models, or write
to a remote repository. Existing control pages remain separate. Code and reports
stay local; hosted VLM inference is allowed by the team's confirmed preference.

## First milestone: prove the VLM on actual camera images

`demo_vlm.py` calls an explicitly selected vision model through either OpenAI's
hosted Responses API (`--provider openai`) or Ollama's local native API
(`--provider ollama`, the original default). It sends image bytes and a question, requests a
structured `yes` / `no` / `unknown` verdict, and saves the original response,
image SHA-256, elapsed time, expected label, and comparison in a local JSON report.
Expected labels are never sent to the model.

API references: [vision](https://docs.ollama.com/capabilities/vision),
[chat](https://docs.ollama.com/api/chat), and
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

Hosted inference uses `OPENAI_API_KEY` from the process environment. The key is
sent only in the request header and is not written to the report. Set a different
variable name with `--api-key-env`; no key is accepted on the command line. The
OpenAI request uses image input and strict structured output, with `store: false`.
See official [image input](https://developers.openai.com/api/docs/guides/images-vision)
and [structured output](https://developers.openai.com/api/docs/guides/structured-outputs) documentation.

From this checkout, after setting the API key in your environment:

```powershell
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe -m dream_hackathon.demo_vlm --provider openai --model gpt-6.1-terra --manifest inspection/demo-frames/cases.json --report inspection/demo-vlm-terra.json
```

For an already running Ollama server and installed vision model:

```powershell
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe -m dream_hackathon.demo_vlm --provider ollama --model YOUR_INSTALLED_VISION_MODEL --image inspection/demo-frames/usb-removed.jpg --question "Is at least one of the four USB plugs fully disconnected from the strip?" --expected yes --report inspection/demo-vlm-report.json
```

Use `--manifest inspection/demo-frames/cases.json` instead of `--image`,
`--question`, and `--expected` to evaluate several labeled photographs. Image paths
are relative to the manifest. Example format (supply real photographs):

```json
[
  {"image": "usb-present.jpg", "question": "Is at least one of the four USB plugs fully disconnected from the strip?", "expected": "no"},
  {"image": "usb-removed.jpg", "question": "Is at least one of the four USB plugs fully disconnected from the strip?", "expected": "yes"},
  {"image": "usb-obscured.jpg", "question": "Is at least one of the four USB plugs fully disconnected from the strip?", "expected": "unknown"}
]
```

The USB question needs all four sockets and plugs visible; otherwise use a
question about a specifically marked socket. Include multiple positive, negative,
and occluded examples for USB extraction, each stack placement, a released tower,
assigned-color ball selection, and a ball inside the white bin. Test a wrong-color
ball too. Include a visible scale for height questions. Record latency to decide
which actions can overlap inference. A passing report proves only those labeled
cases, not general demo reliability. Exit status is 0 if every label matches, 1
for any mismatch; transport/parsing errors surface directly without retries.

**Current status:** protocol and scheduling are tested using local fixtures. Real
VLM accuracy and latency still require provider credentials or a running local
model, and real camera images. No hosted provider key was set when this was built.
A single photograph cannot establish a three-second standing interval.

### Model shortlist (November 7, 2026)

These are candidates to evaluate, not rankings established on our own footage:

* **Gemini Robotics ER 2** (`gemini-robotics-er-2-preview`): first robotics-specific
  candidate. Google documents spatial reasoning, video understanding and task
  success tracking; its nonstreaming API supports structured output. Public preview.
  [Model details](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview?hl=en),
  [success tracking](https://deepmind.google/models/gemini-robotics/embodied-reasoning/).
  A Gemini transport adapter is not implemented in this probe yet.
* **GPT-6.1 Terra** (`gpt-6.1-terra`): practical first run with the implemented
  hosted adapter. Official documentation positions it for efficient visual
  reasoning and confirms image input and structured outputs.
  [Model details](https://developers.openai.com/api/docs/models/gpt-6.1-terra).
* **GPT-6 Luna** (`gpt-6-luna`): lower-cost candidate for focused repeated checks
  such as ball color or bin occupancy, subject to matching the stronger model's
  accuracy on these scenes.
  [Model details](https://developers.openai.com/api/docs/models/gpt-6-luna).

Compare on the same labeled photographs, especially partially extracted USBs,
grippers still supporting the tower, and balls held over rather than inside the
bin. Report false positive completions separately from ordinary accuracy. Select
one verifier after measuring accuracy and latency; no automatic model switching
or model-voting policy is planned. Calibrated height and continuous duration stay
deterministic regardless of VLM choice.

## Scheduling and pass gates

`demo_master.py` is a shared event-driven state machine for two arm workers.
Each task has ordered steps and a completion token identifying arm, task, cycle,
and step. Every step requires a deterministic success and VLM `yes`; an `unknown`
or disagreement leaves that step pending for the caller to handle explicitly.
There is no automatic retry, motor stop, recovery motion, or operator override.

| Phase | Left arm | Right arm | Gate to next phase |
| --- | --- | --- | --- |
| 1 | Wait | Locate, grasp, extract, release, verify USB | At least one removal confirmed by both measurements and vision |
| 2 | Place horizontal base, horizontal second block, vertical top; release and verify | Finish requested USB target, then wait | Calibrated height >=4 inches, continuously free standing >=3 seconds, plus vision confirmation |
| 3 | Repeatedly locate assigned color, grasp, deposit in white bin, verify | Same once USB target reached | None |

Default right-arm stopping point is **one USB**; `usb_target=4` continues removing
the remaining plugs concurrently with stacking. No reinsertion or cylinder
insertion is planned. One horizontal block plus one vertical block is 3.5 inches;
the confirmed three-block plan is 1.5 + 1.5 + 2 = **5 inches**, subject to actual
measurement. Height bonuses are not optimized beyond this initial plan.

Interpretation pending rule clarification: global phases are sequential. Neither
arm starts balls until BOTH gates pass, even if the other arm finishes earlier.
Phase 2 starts immediately after the first verified removal, without waiting for
all four. Once phase 3 opens, a ready arm collects while the other finishes its
USB target. Tower qualification latches permanently after passing; later collapse
does not revoke it. `verified_deposits` counts completed deposit cycles, not an
independent live inventory of balls still in the bin.

## Local execution and worker interface

Show the initial plan without hardware:

```powershell
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe -m dream_hackathon.demo_master --color orange --usb-target 1
```

Two workers can share one `DemoMaster` instance. `snapshot()` supplies each next
step and token. Execute that step, collect the two verification sources, then call
`complete(side, token, deterministic, vlm, evidence)`. Calls are serialized under
one lock; duplicates and completions for old steps cannot advance the machine.
The worker keeps running its current step until evidence accepts it; no retry
or motion policy is implemented here.

Example local completion event for the initial right-arm locate step:

```json
{"side":"right","token":"right:usb:0:0","deterministic":true,"vlm":"yes","evidence":{"frame":"usb-locate.jpg","measurement":"detector-output.json"}}
```

`--events path/to/events.jsonl` replays these events and prints every resulting
state. Gate events are final-step completions with additional evidence fields:

* USB: `measured_removed` and `visual_removed`, integer cumulative counts.
* Tower: `height_inches`, `continuous_freestanding_s`,
  `visual_freestanding: true`, and `visual_height_at_least_4: true`.

Evidence providers are still to be implemented. Use calibrated image geometry or
depth for height, a continuous deterministic support/standing tracker for duration,
and connector occupancy changes for USB counts. Count neither elapsed VLM call
time nor the gap between two isolated photos as continuous standing time. Other
step checks need measured target/gripper state plus the corresponding visual
question. Ball verification should establish correct color and a new deposit,
not simply detect a preexisting ball in the bin. Keep image/measurement references
in `evidence` so local `history` is reviewable.

No motion primitives, remote arm transport, live detector, persistent service,
or trained VLM weights are bundled. Hosted inference transport is provided separately
from arm execution. Wire workers to the existing independent arm
servers only after the real-image VLM experiment and deterministic checks are
working. This keeps the first milestone runnable without touching either arm.

Validation:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_demo_master.py tests/test_demo_vlm.py -q
```
