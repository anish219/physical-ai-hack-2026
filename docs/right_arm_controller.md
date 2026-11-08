# Headless right-arm controller integration

Implementation: `src/dream_hackathon/right_arm_controller.py`, with the script entry
point `scripts/run_right_arm.py`. This implements the independent-worker strategy
from `codex/right-arm-handoff`. Publishing the source does not deploy or start it on
the robot; the verifier binding and physical validation remain necessary.

## Launcher and left-worker contract

Launch from the prepared checkout root with the station's existing Python environment:

```sh
PYTHONPATH=src .venv/bin/python -m dream_hackathon.right_arm_controller \
  --config configs/right_arm_controller.json \
  --run-id UNIQUE_MATCH_ID --go-time SHARED_UNIX_SECONDS \
  --run-dir /tmp/relay/UNIQUE_MATCH_ID
```

The equivalent script entry point is `PYTHONPATH=src .venv/bin/python scripts/run_right_arm.py`
with the same arguments.

The overarching launcher supplies exactly the same unique run ID and Go time to
both workers. Go time is Unix seconds, including fractional seconds. Workers must
run on the same host or use synchronized clocks. A future Go waits without acting;
an expired Go never starts an action. All retries consume the original 300 seconds.
The UI and left worker need no browser or right-worker acknowledgment.

`first-removal.json` is written atomically once, only after fresh known first-removal
evidence. Its fields are `version: 1`, `run_id`, `go_time`, `first_removal: true`,
`unplugged_count` (integer 1–4), `observed_at` (Unix seconds), `inspection_id`, and
`evidence` with `deterministic` and `vlm` references. Left must validate the identity
and Go time against its active run and require the positive boolean and count.
A file from another run, missing file, or `right-status.json` never grants clearance.
Right immediately proceeds to a new hub inspection without waiting for left.
The gate remains latched through subsequent scene changes. Left owns stacking,
collapse handling and all tower decisions.

Use a fresh run directory for every run. `right.lock` prevents duplicate workers
within that directory. A second host-local lock prevents this worker from using
the same configured right URL across run directories. The launcher must still stop
any other target producers before Go and avoid URL aliases for the same listener. This worker never discovers or controls the left listener. Configured external
scripts/providers must honor that same isolation contract.

`right-status.json` atomically reports run ID, right side, state and elapsed seconds.
SIGINT/SIGTERM cancel progress and stop the right listener, without homing or power
changes. The explicit five-minute deadline does the same. Pending HTTP requests
cannot be retracted: providers must honor cancellation/ownership on their side.
Execution exceptions terminate with error; they do not become manipulation retries.
Policy setup/return moves are explicit configuration; no extra exception recovery is defined. A cancelled run must not be resumed
using its old gate.

## Bindings

Copy `configs/right_arm_controller.example.json` to a separate local config and
resolve the placeholders. `--check` with the launch arguments validates configuration
without connecting to hardware or the cloud. The example deliberately cannot run.

- `pull_all` invokes the full `scripts/replay_usb_bundle.py --execute` path, including
  CV alignment and calibrated retargeting. It must explicitly use the right server.
- `unplug_policy` uses `dream-1-ft:f9ed26ed-usb-unplug-right-500-20261107`
  with its exact submitted unplug prompt and the native joint checkpoint adapter.
- `plug_usb` uses `dream-1-ft:f9ed26ed-usb-manip-gen-cos1000-20261108`
  with its generalized training prompt: “Pick up the USB-C cable and insert its
  connector into the target port using the right arm.” It is a policy window,
  followed by the same seated-and-released verification, rather than a recording.
- `return_usb_start` is a named pose operation using the existing right listener's
  rate-limited `/move`. It waits for fresh measured arrival before allowing the
  next policy call, and stops on listener errors/stalls. It does not need a recording.
  It preserves the measured gripper opening unless the pose explicitly specifies one.

Both policy windows are editable `durationSeconds` values (initially three seconds,
retaining the handoff's initial window size). These values are experiment settings,
not established physical completion times. Success is always verified afterward.
- `verify_usb` must be an HTTP or script provider combining automatic deterministic
  measurements with VLM scene assessment. The existing dummy and a lone VLM answer
  cannot satisfy this interface. A script can return JSON stdout via
  `resultPath: "stdout"`; HTTP uses `resultPath: "body"`.

The repeated insertion/unplug strategy conflicts with the project rule forbidding
reinsertion of removed cables. The branch cites reported judge approval. The user explicitly confirmed judge approval in this chat on November 8. The
example records `judge_approved_repeated_cycle: true`; no further confirmation is
needed for this implementation. A different allowed-cable strategy needs explicit transition
changes; it is not silently substituted here.

## Change policy starting positions and retry behavior

All tuning happens before Go. The worker snapshots config and pose definitions at
startup; modifying the files during a run does not change policy behavior.

`configs/right_arm_start_poses.json` holds named profiles. `unplug_start` currently
uses the measured first-frame joints from candidate training episode 16 (original
17, because original 6 was excluded). This is the medoid of the 50 episode starts:
an actually observed pose closest to the other starts, rather than averaged joints.
Its source path/hash and selection method are stored with the profile. Starts vary
substantially across demonstrations, so this is a candidate for testing, not a
proven universal start or a collision-checked return path. No return was executed.

Each pose has six `q` values in radians, optional ordered `waypoints` (six joints
per waypoint), `rate_rad_s`, `tolerance_rad` and `grip`. `grip: "preserve"` keeps
the current measured opening. An explicit grip must be normalized 0–1, not the
training data's 0–100 mm. Default rate 0.3 rad/s comes from the existing listener's
movement default; the listener retains its own limits and lag handling.

Each Dream binding may have:

```json
"prepare": {"pose": "unplug_start", "first": true, "retry_every": 1}
```

This returns before its first window and before every retry. `retry_every: 2`
returns after two unsuccessful windows; `0` disables retry returns. Attempts are
counted separately for hub clearing, cycle unplugging and insertion, and reset after
verified physical success. These counters never reset the match clock. A failure is an unsuccessful physical verification;
provider errors still terminate with error. The explicit post-insertion
`return_usb_start` completes before cycle unplugging regardless of preparation.

The example enables preparation for unplugging. Insertion defaults to
`{"pose": null, "first": false, "retry_every": 0}`, so it starts where the arm is.
To experiment with an insertion setup, capture an `insert_start` pose and set its
binding's `prepare` to that profile with the desired first/retry settings. Set
`prepare.pose` to null to disable policy preparation entirely. To disable the explicit
post-insertion return for an experiment, set that binding's `enabled` to false.
Preparation on the unplug binding is independent, so disable that too if the
experiment should make no setup move.

For a pre-Go measured pose capture on the robot, with the arm already stationary:

```sh
PYTHONPATH=src .venv/bin/python scripts/capture_right_arm_pose.py \
  --name unplug_start --output configs/right_arm_start_poses.json
```

This only GETs `/state` and saves measured joints; it sends no motion, power, or
release command. Use `--name insert_start` to save an insertion candidate. Position
the arm through the team's normal coordinated setup before running capture. Capturing
a pose does not certify a safe return path; add measured intermediate waypoints
if needed during physical validation. Retries preserve the grip and never assume a
held cable has been released or an insertion completed.

## Verifier request and result

Each request supplies `run_id`, new `inspection_id`, and `requested_at`. These are
available to binding templates as `{{variables.run_id}}`,
`{{variables.inspection_id}}` and `{{variables.requested_at}}`. Providers must acquire
new images after the request; copying request fields onto cached evidence is invalid.
Store both measurement and VLM evidence for the same camera observation, and return:

```json
{
  "run_id": "UNIQUE_MATCH_ID",
  "inspection_id": "ECHO_CURRENT_INSPECTION_ID",
  "observed_at": 1794150000.25,
  "known": true,
  "plugged_count": 3,
  "unplugged_count": 1,
  "fully_inserted": false,
  "gripper_released": true,
  "cable_on_table": true,
  "evidence": {"deterministic": "evidence/measurement.json", "vlm": "evidence/vision.json"}
}
```

All booleans must be JSON booleans. Counts must be consistent integers in 0–4;
original ports are the four calibrated hub positions, not unrelated loose cables.
`observed_at` must describe the fresh scene after the request, before response.
Unknown, stale, mismatched, inconsistent or unsupported observations never pass.
Evidence references are audited provenance, not proof of detector accuracy: the
provider's actual camera freshness and real-scene accuracy still need validation.
Fields are not merged across attempts. A provider failure is an execution error;
uncertainty about a physical scene must be reported as `known: false`.

Unknown/zero first removal retries bundle pull. Unknown hub clearing waits
`poll_seconds` before inspection; known partial clearing invokes one policy window.
All four removed begins insertion. Seated + released requires the configured return operation to
complete before the unplug policy. Cycle unplug requires zero occupied ports,
cable resting on table, and released gripper before the next insertion. There are
no ball operations or dependencies on tower observations.

## Hardware-free verification

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_right_arm_controller.py \
  tests/test_orchestration.py tests/test_demo_strategy.py tests/test_bundle_pull.py \
  tests/test_model_selection.py tests/test_dream_joint_finetune.py -q
```

These checks validate transitions/adapters, not actual policy success or physical
replay convergence. Before deployment, validate the candidate return path and selected policy windows,
validate the combined verifier on real failure/occlusion scenes, integrate the
launcher contract, and coordinate a fresh physical test with the operator through
the shared terminal.
