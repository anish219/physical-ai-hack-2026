# Full Auto orchestration

The **Orchestration** tab edits Left, Right, and Both-arm flow graphs. The main
**Controller** tab controls execution: select **Full Auto Control**, then use its
**Go**, **Stop**, and **Edit flow** panel. Selecting Full Auto does not start the
graph or power an arm. Left and Right flows can run independently; a Both flow
can fork into parallel operations.

Launch the existing `dream_hackathon.unified_control` entry point with your usual
station configuration and leader port. The new `--flow-file` option defaults to
`~/.cache/dream_hackathon/flows.json`. All three graphs persist there. Connected
browsers receive shared definitions and execution updates. A running flow uses
a snapshot: edits apply on the next Go. Unsaved edits are saved before Go.
Concurrent edit conflicts retain the local draft and offer **Load saved**.

For a laptop UI preview without robot devices or cameras:

```powershell
$env:PYTHONPATH = "src"
D:/PERSONAL_PROJECTS/MakerModsHackathon/.venv/Scripts/python.exe scripts/preview_orchestration.py --port 8096
```

The preview has simulated arm status, the real scheduler, and script/HTTP
operations. It has no Dream arm workers. Its loop example only emits a local
signal and needs no external services.

## Editing

Add blocks from the toolbar and drag their headers to position them. Click an
output port and then an input port to connect them. Click a connection to delete
it; Escape cancels a pending connection. Click a block to edit its settings.
**Configuration JSON** supports nested trigger expressions and operation options.
**Export/Import** transfers a graph as JSON. **Load saved** discards unsaved edits.
**Loop example** replaces the draft with a three-iteration graph. Neither saving
nor loading a graph starts it.

| Block | Behavior / outputs |
| --- | --- |
| Go | Exactly one per runnable graph; emits `next` on Go |
| Trigger | Waits for its condition; emits `next` or optional `timeout` |
| Operation | Script/API/VLM/signal/variable; emits `next`, `error`, or `timeout` |
| Condition | Immediately chooses `true` or `false` |
| Loop | Chooses `body` while count/condition permits, otherwise `done` |
| Parallel | Activates all connected `next` branches concurrently |
| Join | Waits for every incoming branch in the same Parallel activation |
| End | Ends a branch; run completes after all branches end |

Connect ordinary outputs once; use Parallel for fan-out. Error and timeout edges
are optional. An unhandled error marks the run as an error; already-started
operations can settle, and explicit Stop cancels remaining progress. There is no
automatic recovery motion or fallback branch. Parallel/Join can be nested; match
each Parallel with its Join. A branch ending elsewhere does not satisfy a Join.
Use separate Left/Right Dream workers for concurrent arm actions rather than
overlapping workers for the same arm.

## Triggers and timers

* `{"kind":"delay","seconds":2}`: two seconds after this block is activated.
* `{"kind":"elapsed","seconds":30}`: 30 seconds after this run's Go. Passes
  immediately if reached later than that; page-load time is irrelevant.
* `{"kind":"script_end","name":"script_id","count":1}`: that script node has
  exited at least once this run. Exit does not assert task success. Compare the
  node's `returncode` when success matters.
* `signal` or `vlm`: compare an input signal. `result`: compare an operation's
  output. `variable`: compare a variable set by an operation.

Data conditions take `name`, optional dot `path`, `op`, and JSON `value`.
Comparisons are `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contains`, and `exists`.
Missing data does not satisfy ordinary comparisons. Dot paths can include list
indices, such as `body.items.0.color`.

AND/OR use `all`/`any` with a `conditions` array; NOT uses `not` with `condition`.
Example trigger configuration combining deterministic and visual evidence:

```json
{"condition":{"kind":"all","conditions":[
  {"kind":"signal","name":"usb_measurement","path":"removed","op":"gte","value":1},
  {"kind":"vlm","name":"usb_vision","path":"verdict","op":"eq","value":"yes"}
]}}
```

Optional trigger `timeoutSeconds` follows its `timeout` edge. Blank means no
deadline. Timeout does not assert that a pass condition succeeded. Conditions
evaluate once; Triggers wait. Signals/results reset on Go and otherwise retain
their latest values. For repeated verification, use execution counts or update
signals explicitly so an earlier `yes` is not reused unintentionally.

## Operations

Changing **Execute** swaps an operation's settings while preserving its ID and
connections. Operations run on the controller's machine.

**Script:** `{"operation":"script","argv":["python","scripts/task.py","--side","right"],"cwd":"/path/to/checkout"}`.
Arguments run directly; use `["bash","scripts/task.sh"]` for a shell script.
Combined stdout/stderr and `returncode` become the node result. Nonzero exit
selects `error`. Optional `timeoutSeconds` measures script runtime and terminates
the owned process on expiry. Explicit Stop also terminates that process. Child
process lifecycles remain the script's responsibility.

**HTTP API:** `{"operation":"http","url":"http://127.0.0.1:9000/execute","method":"POST","headers":{},"body":{"task":"unplug"}}`.
Results have `status` and `body`; JSON responses are parsed, other responses are
text. Optional `timeoutSeconds` is a network I/O timeout; blank leaves waits
unlimited. Stop discards late replies and prevents downstream operations; it
cannot retract a request already accepted by an external service. Add that
service's cancellation request explicitly when your workflow needs it.

**Dream API:** `{"operation":"dream","selection":"right","task":"Remove the USB","checkpoint":"dreamlabs/dream-1-beta","run":true,"durationSeconds":10}`.
Uses the existing worker. Playback ends at the configured window, then waits for
the pending call to unwind before advancing. `run:false` makes a single inspection
call without playing motor actions. A timed window is not verification of task
success; follow it with a pass gate.

**VLM:** `{"operation":"vlm","provider":"openai","model":"gpt-6.1-terra","imageUrl":"http://127.0.0.1:8081/api/camera/overhead.jpg","question":"Is a USB fully disconnected?","apiKeyEnv":"OPENAI_API_KEY"}`.
Supports OpenAI or Ollama. `imagePath` selects a local image instead of a URL;
`url` overrides the provider base URL. Credentials come from the controller's
environment. Results have `verdict` (`yes`/`no`/`unknown`) and `reason`. Compare
them with `kind:result`, `name:<VLM block ID>`, `path:verdict`. Optional network
timeouts apply separately to image fetch and inference. Other providers can use
HTTP/script operations or feed input signals; no native Gemini adapter is bundled.
Actual VLM accuracy/latency remains unvalidated on demo images.

**Emit signal / set variable:** `{"operation":"emit","name":"event","value":true}`
or `{"operation":"set","name":"finished","value":true}`. Values may be any JSON.

Operation fields support `{{results.inspect.verdict}}`, `{{signals.target}}`,
`{{variables.color}}`, and `{{env.KEY}}`. A whole-value template retains the value's
type; embedded templates become text, e.g. `Bearer {{env.KEY}}`. Expansion happens
only at execution; saved graphs retain expressions. Python expressions are not
evaluated.

## Loops and demo gates

Count loops send `body` N times then `done`. Wire the final body operation back
to the Loop input. While loops reevaluate their condition on each visit. Set
variables or signals to change it. Negating a `finished` signal gives an
until-finished loop. No automatic iteration limit or run-wide cutoff is imposed.

For the competition, place the USB gate before stacking and the tower gate before
ball collection. Combine calibrated height >=4 inches, deterministic continuous
free-standing duration >=3 seconds, and visual confirmation in the tower gate.
The confirmed three-block target is nominally five inches. Traversing a passed
gate does not rewind if a later input changes; collapse does not revoke a passed
tower gate. Timer completion alone establishes neither gate. A single VLM image
does not prove a three-second interval.

## HTTP integration

`{selection}` is `left`, `right`, or `both`:

| Endpoint | Request |
| --- | --- |
| `GET /api/state` | Response includes `flows`: definitions, revisions, run/node status, elapsed time, context |
| `POST /api/flows/{selection}/save` | `{"definition":{"nodes":[],"edges":[]},"revision":0}`; revision optional |
| `POST /api/flows/{selection}/go` | `{}`; requires matching Full Auto input |
| `POST /api/flows/{selection}/stop` | `{}` |
| `POST /api/flows/{selection}/signal` | `{"name":"usb_vision","value":{"verdict":"yes"}}` |
| `GET /api/events` | Existing stream includes `type:"flows"` updates |

The signal form below the canvas uses that same endpoint. Graph nodes contain
`id`, `type`, `label`, `x`, `y`, `config`; edges contain `id`, `source`, `port`,
`target`. Structure/configuration is validated before execution. Existing source
handoff applies: selecting another input, Return to 0, or explicit arm power-off
stops overlapping flows. Stop does not power down arms. Closing a browser leaves
the station-owned run active.

Tests use fake arm workers, actual local subprocesses, and localhost HTTP servers:

```bash
PYTHONPATH=src python -m pytest tests/test_orchestration.py tests/test_demo_vlm.py \
  tests/test_unified_control.py tests/test_sequences.py tests/test_dream_caller.py -q
```
