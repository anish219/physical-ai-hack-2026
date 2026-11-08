# Local CV-aligned USB bundle pull

The entry point is `scripts/replay_usb_bundle.py`. The self-contained replay
assets are in `recordings/usb-bundle-pull/`: 506 original edited targets, the
7-second wrist reference, and `plan.json`. Nothing is deployed or started by
the existing dashboard.

The sequence is:

1. Move to the recording's starting joints using its existing 0.5 rad/s approach
   rate, then play the recorded approach through 7.0 seconds.
2. Match features on the strip's top face and the orange-wire centroid near the
   seated connectors against the reference. Image coordinates use 426 x 240.
3. If correction is needed, measure an image/joint Jacobian with +0.012 and
   -0.012 rad probes on each of the six joints, returning to the checkpoint
   after each pair. Gripper aperture stays at its recorded checkpoint value.
   Use measured joint motion and image response to align. Corrections have a
   0.5 gain, a 0.02 rad step, and a 0.2 rad/s motion rate. These are editable
   servo settings in the plan.
4. Once all four strip-anchor errors are within 2 pixels and the orange-wire
   centroid is within 3 pixels, compute the rigid tool-pose correction
   `T_aligned @ inverse(T_recorded_checkpoint)`. Apply it to the remaining
   recorded tool poses and solve their joints using the station's own URDF,
   joint signs, and TCP. Continue the original approach, grasp, and pull with
   the original timing and gripper targets through approximately 14.98 seconds.
5. Move directly from the corrected pull endpoint to the original measured
   release joints, with the gripper closed, then play the 0.6-second opening
   at that fixed pose. The transfer retains its original minimum duration and
   0.5 rad/s joint rate. Alignment corrections do not move the drop location.

## Inspect locally

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/replay_usb_bundle.py recordings/usb-bundle-pull/plan.json
```

This reads local files only and writes `inspection/bundle-pull-run/events.jsonl`
and `alignment.jpg`. An offline image can be checked with
`--check-image path/to/wrist.png`; green circles are the recorded targets and
red circles are the observed positions. `--output` changes the output folder.

## Execute on the station when ready

Use the already running right-arm HTTP server, which defaults to
`http://127.0.0.1:8692`. It supplies `/state`, `/wrist.jpg`, and `/target`.
The existing station configuration must contain that arm's calibrated model,
joint signs, and measured TCP. It is the kit's `configs/station.yaml` format.

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_usb_bundle.py \
  recordings/usb-bundle-pull/plan.json \
  --config configs/station.yaml --execute
```

`--server http://HOST:PORT` overrides the server URL. The script uses the
existing arm session; it sends no power, torque, home, tracking, or shutdown
commands. Run it with that arm assigned to this script rather than another
target producer. Ctrl+C exits the script without a release or return motion.
Logs identify the current phase and alignment error; `alignment.jpg` shows
the most recent CV result.

This implementation makes small alignment corrections around the taught scene.
The wrist mount, strip appearance, and orange bundle must match the reference.
It checks visible geometry, not grasp force or whether every plug detached.
If it cannot measure the strip or orange wires it raises the actual vision
error; there is no blind-playback fallback or automatic power action. Alignment
continues until its image tolerances are met or the operator exits.

## Reference provenance and validation

The reference is the right pane at edited time 7.0 seconds of the existing
`inspection/usb-review/usb-bundle-pull-clean/review-preview.mp4`. That pane
contains the original filmed approach, not the unfilmed shortcut. The source
capture is `/home/pah/hackathon_data/usb-pull-demos/20261107-191647`.
The JSON plan preserves the source timeline and provenance. Replacing the
reference or checkpoint requires keeping the image and motion times paired.

The local tests cover real-image perspective perturbations, moving-background
exclusion, missing wire evidence, image-servo convergence, calibrated-model FK
and IK, rigid retargeting, phase sequencing, and the exact shortcut release.

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m pytest tests/test_bundle_pull.py -q
```

No physical run has been performed during this local development. The recorded
arm's actual station config and live wrist image are not present locally, so
hardware convergence and the calibrated real-arm pull remain to be evaluated
on the station.
