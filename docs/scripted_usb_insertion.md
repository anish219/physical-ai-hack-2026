# USB insertion: continuous CV and world-frame Cartesian control

Work is published on `codex/scripted-usb-insertion` in
`anish219/physical-ai-hack-2026`. This implementation is standalone and does not
require a browser. The grabbing/approach policy remains separate.

## Current behavior

1. Watch fresh right-wrist images while the grabbing policy approaches.
2. Localize the labeled port face using a multiview, affine-augmented SIFT model.
   Reacquire it independently in each image instead of relying on optical flow
   initialized close to one recorded viewpoint.
3. Search image-wide orange components for the held USB body/end and axis. Use
   association with the wrist gripper, component width and shape to distinguish
   the held connector from thin cable segments and loose connectors. There is no
   small fixed live USB crop. The camera-to-gripper mounting remains fixed.
4. Once the port/USB geometry is near, yield the right-only approach policy using
   its stop endpoint/callback. Wait for its busy status to clear when configured.
5. Use the existing visual servo to align to the recorded pre-insertion geometry,
   preserving the current live grip. This initial alignment can correct pose;
   the subsequent descent keeps the aligned world orientation fixed.
6. Measure the local image response to small world X/Y/Z translations using FK/IK
   and measured joint feedback. The default calibration probes are ±0.6 mm along
   each world axis and return to the aligned pose between probe pairs.
7. Descend along **negative world Z**. For **every new control-frame image**,
   localize the port and held USB again, estimate the lateral correction, and
   command a fresh world X/Y/Z target through IK. World X and Y follow CV while
   Z follows the requested downward stroke. This is not open-loop playback of
   recorded joint angles.
8. Finish the configured stroke and return a result for the existing right-arm
   inspection step. Successful motion is not itself verified insertion.

The feedback uses the four port-face anchors relative to the live USB end and
axis. Its measured image/Cartesian Jacobian separates lateral error from image
motion due to changing depth. Only the X/Y correction is applied; Z is commanded
explicitly. Holding an unchanging 2-D pixel location during downward motion would
otherwise feed depth/perspective changes into sideways steering. A configurable
Broyden update adapts the Jacobian using successive measured movements and images.

Every descent target follows a fresh image and actual joint measurement. The
configured Hz is a maximum control rate; actual throughput also depends on image
acquisition, feature localization and IK. A localization error remains an
execution error; no guessed correction, recovery movement, motor shutdown or
policy restart is added.

## Run alongside the right-only API policy

From a checkout of this branch **on the NUC**, with the existing arm server and
policy backend running:

```bash
PYTHONPATH=src .venv/bin/python scripts/insert_usb.py run \
  recordings/usb-insertion/plan.json \
  --server http://127.0.0.1:8692 \
  --config configs/station.yaml --execute \
  --policy-stop-url http://127.0.0.1:8081/api/dream/right/stop \
  --policy-status-url http://127.0.0.1:8081/api/state \
  --policy-busy-path dream.right.busy
```

The optional URLs address an existing backend; a GUI does not need to be open.
For an independent policy owner, substitute its stop/status URLs and JSON dot
path. Or call `execute(..., policy_stop=...)` with its handoff callback. If the
policy already yielded before invoking the script, omit the policy URL options.
A callback/endpoint without status polling must complete the handoff before
returning. The policy remains yielded throughout scripted control.

The script issues targets to the selected right listener only. It does not
power motors, home, change arm input selection, start a grasp policy, retract,
or control left. Use the existing calibrated station configuration for joint
signs, FK/IK and measured TCP.

## Align only, then test policy insertion separately

Add `--align-only` to the command above. It finishes after initial visual alignment
without world-axis probe calibration, downward motion or release. This mode
leaves the right policy yielded and the grip unchanged. The operator/owner can
then explicitly run the API insertion test; this script does not resume it.

## Motion and vision settings

Settings are in `recordings/usb-insertion/plan.json`. The current explicit
Cartesian defaults are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `distance_m` | 0.028 | Total downward world-Z stroke, based on the previous model review of the insertion demo. |
| `speed_m_s` | 0.0015 | Downward speed: 1.5 mm/s. |
| `probe_m` | 0.0006 | Per-axis displacement for image/Cartesian calibration. |
| `lateral_gain` | 0.3 | Fraction of estimated world-X/Y error applied per frame. |
| `lateral_step_m` | 0.001 | Maximum magnitude of one X/Y correction. |
| `adapt_gain` | 0.1 | Per-frame Jacobian adaptation gain; zero keeps calibration fixed. |
| `motion_rad_s` | 0.15 | Joint interpolation rate used during calibration. |
| `settle_s` | 0.15 | Settling interval for each calibration measurement. |
| `hz` | 20 | Maximum descent control frequency. |
| `depth_tolerance_m` | 0.0004 | Measured endpoint tolerance for stroke completion. |

The `servo` section controls the initial alignment's existing joint probes and
correction gains. The `vision` section controls the port model, descriptor ratio,
independent inliers, spatial coverage, orange-color range, held-component shape,
gripper association, near-opening thresholds and alignment tolerance. These are
explicit editable algorithm parameters, not hidden runtime cutoffs.

The labeled port face includes the distinctive USB row as well as the lower
panel to reduce repeated-hole correspondence ambiguity. Affine training views
cover rotation, scale and compressed perspectives. `view_images` adds registered
actual views to the descriptor bank. A completely hidden port or held connector
still cannot be localized; this is an instance-specific model for the visible
orange USB and this strip, not an arbitrary USB detector.

`release_grip` defaults to null: hold the current normalized grip throughout and
after the stroke. Set it explicitly to a normalized opening between 0 and 1 to
command release at the final pose. `--align-only` does not release, even if that
field is set. No withdrawal motion is added.

## Source recordings and offline inspection

Assets come from episode index 25 (human run 26) of the insertion dataset. Frame
480 is the alignment reference; frames 420, 450, 460 and 600 exercise distant,
near and changed-view localization. The earlier optical-flow detector rejected
all inspected live frames before reaching its proximity check; that matcher has
been replaced. Frame 450 is now a valid near-port observation, rather than being
rejected because reference tracking failed.

The saved 480–660 joint window remains available for provenance and offline
comparison. It is **not** the runtime downward trajectory anymore. Its prior
calibrated-model review indicated about 28 mm downward travel and about 3 mm on
each horizontal axis; that result informed the initial stroke distance. The
recording exporter and relative-trajectory averaging helper remain available
for offline analysis. Recorded gripper values use a different representation;
the controller uses the current live normalized grip rather than replaying them.

Inspect without contacting an arm:

```bash
PYTHONPATH=src .venv/bin/python scripts/insert_usb.py run recordings/usb-insertion/plan.json
PYTHONPATH=src .venv/bin/python scripts/insert_usb.py run recordings/usb-insertion/plan.json \
  --check-image recordings/usb-insertion/near-frame-460.png
```

To export a different explicitly reviewed recording interval, use the existing
`prepare DATASET/right --window episode:start:end --output DIRECTORY` subcommand.
It writes a reference and editable plan. Port labels and color/gripper parameters
must correspond to that reference; current default geometry is specific to the
included 640×360 wrist view. Additional `--window` selections preserve other
reviewed demonstrations for offline relative-TCP comparisons. The original
insertion data is backed up at
`/home/pah/laptop-handoff-20261108/downloads/usb-c-insertion/right` on the NUC.

## Integration, results and validation

Keep the right-arm strategy's existing insertion verification and retry edges.
After the stroke, inspect fully seated and gripper released before advancing.
For an orchestration Script binding, target right and include `--quiet` so stdout
is one final JSON object; use `resultPath: stdout`. Without that option progress
also goes to stderr, which the existing orchestration executor merges into script
output. `events.jsonl` records detections, takeover, alignment, axis calibration,
each descent frame's XY correction, measured/target world positions, and joint
targets. `alignment.jpg` is updated even for unknown waiting observations, so
missed approaches can be inspected.

Relevant tests cover rotated/scaled port reacquisition, shifted USB localization
outside the old crop, loose-cable rejection, actual recorded views, policy yielding
before motion, align-only behavior, measured Cartesian calibration, lateral
correction independent of depth flow, moving-port corrections on every control
frame, fixed world orientation, preserved grip, and localization loss. They use
fake hardware. Actual continuous descent and seating still require live testing;
the previous recorded-path model check does not validate this new controller.
