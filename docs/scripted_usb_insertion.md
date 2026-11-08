# Scripted USB insertion after policy grasp

The new entry point is `scripts/insert_usb.py` (or the module
`dream_hackathon.usb_insertion`). It replaces the near-port alignment and insertion
part of an insertion policy. The policy still grabs and carries the USB toward
the opening. This is a standalone script; no controller GUI is required.

## Behavior

1. Read the right wrist camera while the policy approaches. A local OpenCV
   reference matcher tracks the taught port face and segments the held orange
   USB. No hosted model, detector service or training step is required.
2. When the observed port/plug geometry is inside the configured near-opening
   region, hand control over from the policy. If configured, POST to its stop
   endpoint and wait for its busy field to become false. If called by a controller
   that already paused the policy, omit those endpoint options.
3. Capture the current measured joint angles and normalized grip, then hold that
   state. Do not replay the recording's approach or grab, or move to its absolute
   starting joints.
4. Align the port face relative to the live USB tip and axis. The visual servo
   measures the image/joint Jacobian with explicit positive and negative joint
   probes, then corrects the visual error until within the configured tolerance.
   The live grip remains unchanged throughout alignment and pushing.
5. Retarget the taught insertion trajectory from the live aligned TCP pose and
   execute it with the demonstrated timing. With multiple explicitly selected
   successful windows, average their relative TCP paths at normalized phases.
   Do not average absolute joint angles from different starting configurations.
6. Return a completion result for the outer right-arm strategy to inspect. This
   motion does not itself certify successful seating or release. The existing
   `verify_usb` step must still check fully inserted and gripper released.

The script does not start a grasp policy, power motors, home, change arm input
selection, retract, or issue commands to left. The selected right-arm HTTP server
continues owning the hardware. Its existing station configuration supplies FK/IK,
joint signs and measured TCP.

## Included reference and taught push

`recordings/usb-insertion/plan.json` and its PNGs are self-contained runtime and
offline-test assets, exported from the laptop's `downloads/usb-c-insertion/right`
dataset. The same original dataset is backed up on the NUC under
`/home/pah/laptop-handoff-20261108/downloads/usb-c-insertion/right`.

The initial candidate is episode index **25** (human run 26), frames **480–660**,
approximately 16–22 seconds. Frame 480 provides the aligned near-port reference;
the following measured joint samples provide a six-second insertion segment.
Frames 450 and 460 are included as earlier/near-opening detector examples. Later
recorded release motion is not included in the push.

This selection was visually inspected and evaluated offline; it has not been
executed by this script on the robot. The dataset task label is not a certified
success label. Other episodes and some other intervals contain frozen camera
frames, so selecting arbitrary clips or averaging all 37 episodes would not
establish a valid insertion path. The earlier camera QA report found no frozen
interval within this selected 480–660 window.

The saved dataset gripper values use a different representation from the arm
server's normalized grip. The exporter uses only the six measured joint columns;
execution retains the current live normalized grip. It never sends those raw
recorded gripper values to the server.

## Inspect without moving

From a checkout of this branch, with the project's normal Python environment:

```bash
PYTHONPATH=src .venv/bin/python scripts/insert_usb.py run recordings/usb-insertion/plan.json

PYTHONPATH=src .venv/bin/python scripts/insert_usb.py run recordings/usb-insertion/plan.json \
  --check-image recordings/usb-insertion/near-frame-460.png
```

Neither command contacts an arm. The second writes a detector result and annotated
image under `inspection/usb-insertion-run/`. The aligned reference should pass
alignment; frame 460 should trigger near-opening detection but require correction;
frame 450 should not trigger takeover.

## Execute from the right-arm owner

After the grabbing policy has yielded control near the port:

```bash
PYTHONPATH=src .venv/bin/python scripts/insert_usb.py run recordings/usb-insertion/plan.json \
  --server http://127.0.0.1:8692 --config configs/station.yaml --execute --quiet
```

To monitor alongside the existing unified controller's right-only Dream policy
and take over automatically when near, add:

```text
--policy-stop-url http://127.0.0.1:8081/api/dream/right/stop
--policy-status-url http://127.0.0.1:8081/api/state
--policy-busy-path dream.right.busy
```

Those optional URLs use an existing backend; no browser needs to be open. For an
independent policy owner, replace the URLs and JSON dot path with its equivalents,
or call `execute(..., policy_stop=...)` with a callback that yields that policy.
The policy must remain yielded while the script sends targets. Without the status
option, the stop callback/endpoint must complete the handoff before returning.
Do not use a bimanual policy session as the right-only owner.

By default `release_grip` in the plan is null, so the script holds the current
grip after pushing. To include release in this operation, explicitly set that
field to the desired normalized opening, between 0 and 1. Release is then
commanded at the final insertion pose. No withdrawal movement is added. The final
inspection still needs to verify release physically.

## Replace the insertion API operation

The outer right-arm state machine keeps its current transitions. Run the grasp/
approach policy until the script takes over, then await script completion before
the existing insertion inspection. Failed seating returns to that strategy's
next insertion attempt. This script adds no new retry choreography.

For an orchestration Script binding, select right, supply the execution command
above as the argument list, and include `--quiet`. The command emits one final
JSON object on stdout; use the binding's `resultPath: stdout`. Without `--quiet`,
progress is printed to stderr as well, and the current orchestration executor
merges that stream into script output, so it is unsuitable for JSON parsing.

The final result distinguishes `push_complete`, `release_commanded` and
`insertion_verified` (false until the external inspection). Events, near-port
observations, takeover state, aligned state and the retargeted push targets are
also saved in `events.jsonl`; `alignment.jpg` shows the latest visual observation.

## Change recordings, target port or servo settings

Export another explicitly selected approach-contact/insertion window:

```bash
PYTHONPATH=src .venv/bin/python scripts/insert_usb.py prepare DATASET/right \
  --window 25:480:660 --output recordings/usb-insertion
```

Repeat `--window episode:start:end` to average several reviewed successful
insertion segments. Each must start at the same insertion phase and end seated,
before release/withdrawal. The first window supplies the visual reference. Runtime
averaging operates on relative TCP transforms using the calibrated station model.
`pyarrow` is needed for export; runtime uses the exported rows and PNG only.

The initial vision geometry is specific to this reference's 640×360 right wrist
view and orange USB. Edit the plan's `port_polygon`, `port_point`, `plug_roi`, HSV
range and `plug_axis_hint` for a different reference, connector color or target
port. `port_polygon` encloses textured points on the port face; the plug ROI is
excluded from that tracking mask. Four port anchors are expressed in the USB's
live tip/axis frame, rather than treating the whole image or the gripper as a
stationary target. Background people do not supply tracking features.

Near-opening thresholds, pixel alignment tolerance, probe displacement, servo
gain, joint correction step, settling time, motion rate, push duration and sample
count are explicit plan settings. No additional deadline or automatic recovery
path is installed. This is a local matcher around a taught approach, not a
general detector for arbitrary USB types, port layouts or distant viewpoints.

## Validation

The initial implementation passed 22 offline tests on the NUC. Retargeting the
101-point taught path with the NUC's calibrated model preserved its starting
joints exactly and produced approximately 28 mm downward TCP travel, with about
3 mm displacement along each horizontal base axis. These are model calculations;
zero hardware commands were issued. Additional handoff/CLI regression coverage
is included with this implementation.

`tests/test_usb_insertion.py` covers real recorded approach images, missing USB,
plug-relative image geometry, relative-pose averaging, preserved live grip,
policy yielding before motion, and offline operation without arm requests. The
existing `tests/test_bundle_pull.py` suite covers the reused servo, interpolation
and kinematics infrastructure. Live visual-servo convergence and actual seating
remain hardware verification work; offline tests do not establish those results.
