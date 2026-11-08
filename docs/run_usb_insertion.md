# One-command right-arm USB insertion run

Entry point: `scripts/run_usb_insertion.py`, or `python -m dream_hackathon.usb_run`.
This coordinates one complete attempt using the existing right-arm server and
Dream API backend. It does not require an open browser.

## Controller Script input

In the Right arm **Script** panel, use:

```text
scripts/usb_insertion_controller.py:start
```

The adapter is installed inside the live NUC project so the loader accepts it.
It starts the isolated coordinator copy with `--script-input`, which keeps Script
selected and calls the existing Dream worker at `127.0.0.1:8084` directly. This
avoids switching its own panel back to Dream and canceling the run. Panel Stop
interrupts the child; child completion ends this one Script attempt. The log is
`/tmp/usb-insertion-controller.log`. Loading the file alone does not start motion.
The direct worker URL can be overridden with `--policy-url` when calling the
coordinator directly.

## Command line

From a checkout of **`codex/scripted-usb-insertion`** with the station configuration:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_usb_insertion.py
```

On the NUC, a self-contained copy is also available under
`/home/pah/laptop-handoff-20261108/continuous-insertion-review`. From that folder:

```bash
PYTHONPATH=src /home/pah/physical-ai-hacks/.venv/bin/python scripts/run_usb_insertion.py \
  --station-config /home/pah/physical-ai-hacks/configs/station.yaml
```

Calling this entry point executes the run. The existing controller backend at
`127.0.0.1:8081` and powered right-arm server at `127.0.0.1:8692` supply API access,
measured joints and cameras. Use the requested independent Left/Right setup. The
runner selects Dream on right through the existing backend; that backend exits
Both mode if Both was selected. No left-arm API request is issued by this runner.

## Sequence and completion

1. Read the saved right start pose and current API task/checkpoint. CLI/profile
   overrides take precedence over the existing API draft and current worker.
2. Select right Dream input, stop any current right API run, and wait for it to
   yield before moving to the start pose.
3. Interpolate to the start pose and wait for the measured joints to arrive.
   The default is the controller's saved **right zero**, not the near-port pose
   from the insertion recording. The default joint rate is 0.5 rad/s and measured
   arrival tolerance is 0.01 rad. No zero is overwritten.
4. Connect the right API observation worker and start its grasp/approach policy.
5. Start the visual checker. It runs while that policy is moving. When the USB
   is near the opening, stop the right API and wait for its busy state to clear.
6. Run visual alignment and continuous image-corrected world-Z insertion, as
   described in [scripted_usb_insertion.md](scripted_usb_insertion.md).
7. After the configured stroke, write the result and exit. The API remains
   stopped; motors retain the final commanded pose/grip. There is no automatic
   repeat, return-to-start, policy restart or power-off.

Normal completion and operator Ctrl+C stop the owned right API policy. Other
exceptions propagate with their error rather than introducing recovery moves.
Before visual takeover, that policy remains the motion owner; an observation
failure at that point does not install an additional automatic shutdown path.

The result reports `run_complete`, `api_stopped`, and the actual start joints,
along with the insertion controller's result. `insertion_verified` remains false:
finishing a stroke does not certify successful physical insertion. Keep the
team's external seating/release verification before counting a successful cycle.

## Editable settings

Default profile: `configs/usb_insertion_run.json`. Supply another with `--profile`.
Its `plan` path is relative to the profile file; the station-config argument is
relative to the working directory unless absolute.

| Setting | Default behavior |
| --- | --- |
| `controller_url` | Existing controller backend at `http://127.0.0.1:8081`. |
| `right_url` | Existing right-arm listener at `http://127.0.0.1:8692`. |
| `start_q` | Null uses the saved right zero; set six joint radians to use another start. |
| `start_grip` | Null preserves the current measured grip during the start move. Set a normalized opening explicitly to change it. |
| `start_speed_rad_s` | 0.5 rad/s for interpolation to start. |
| `start_tolerance_rad` | 0.01 rad measured arrival tolerance. |
| `start_grip_tolerance` | 0.02 normalized opening when a start grip is requested. |
| `task`, `checkpoint` | Null reuses the current right API draft/worker. Set values to choose a different policy or prompt. |
| `poll_seconds` | 0.1 seconds between completion/observation checks. |

CLI overrides are `--start-q` followed by six radians, `--start-grip`, `--task`,
and `--checkpoint`. Motion/vision settings and optional final release remain in
the referenced insertion plan. Gripper values sent to the arm are normalized;
raw recorded gripper values are not reused.

Add **`--align-only`** to perform the same start-pose and API-grasp sequence, then
finish after visual alignment. It skips Cartesian probe calibration, descent and
release. This leaves the API stopped so the operator can explicitly test whether
the API can finish insertion from that aligned pose.

Use `--output DIRECTORY` to select logs. Otherwise the runner creates a timestamped
folder under `inspection/usb-insertion-runs`. `events.jsonl` records the stages
and visual/controller events; `alignment.jpg` shows the latest observation;
`result.json` contains final completion data. `--quiet` emits one final JSON object
on stdout, suitable for a script operation with `resultPath: stdout`.

## Handoff and validation

The coordinator calls the existing insertion `execute()` function instead of
duplicating visual or Cartesian logic. `ControllerAPI` isolates the backend
contract, and `run()` accepts injectable arm, API, vision, kinematics and reporting
dependencies. Teammates can replace an adapter without rewriting the sequence.

`tests/test_usb_run.py` verifies measured start-pose completion before API start,
profile overrides, API cancellation before scripted motion, align-only routing,
operator interruption, observation readiness, right-only endpoints and JSON
completion output. These are simulated tests, not a hardware execution of the
overarching run.
