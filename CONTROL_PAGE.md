# NUCK-1 arm control page

The SO-101 leader dashboard runs on **NUCK-1, 10.103.0.132**, port **8081**.
The default follower is the **right Metal arm**. The page controls real motors.

## Open the existing page from a teammate's laptop

Run this on your laptop, keeping the terminal open:

```sh
ssh -N -L 8081:127.0.0.1:8081 pah@10.103.0.132
```

Then open [the control page](http://localhost:8081). If port 8081 is already in
use on your laptop, use `-L 8082:127.0.0.1:8081` and open http://localhost:8082.
This is the same shared controller for every teammate; each laptop needs its
own tunnel. Do not start another controller to open another copy of the page.

- **Track and Untrack:** Separate action buttons, with a prominent TRACKED /
  UNTRACKED indicator. Track captures both tools' current poses, then maps SO-101
  handle displacement/rotation to Metal end-effector targets through IK.
  Untrack holds the last arm and gripper targets. Retracking captures new anchors.
- **Translation scale:** 0–3 metres of follower motion per metre of leader motion.
  Rotation follows relative leader rotation. Changing scale re-anchors the mapping.
- **Return to 0:** Uses the existing Home motion to return only the controlled
  arm to the joint pose captured when it was opened, at 0.5 rad/s. The gripper
  target stays unchanged. The arm remains powered and untracked afterward;
  Track captures fresh leader/tool anchors. As with the existing control, save
  an active recording before returning. Kill arm interrupts a return directly.
- **Record example / Stop & save example:** After the arm has been opened with
  Track, records its measured joints/gripper and commanded joints/gripper,
  the live overhead and controlled-arm wrist cameras, and blank frames for the
  uncontrolled arm's wrist. The uncontrolled wrist camera is not opened.
  Stop & save commits a LeRobot v2.1 episode on NUCK-1.
  The page reports the folder and saved episode number. The writer requires at
  least 1.6 seconds (49 frames at 30 fps). Recording works while untracked too.
  `--record-cameras overhead` explicitly selects a one-camera recording;
  the default is `overhead wrist_l wrist_r`, including the inactive wrist's
  blank stream. The overhead and controlled wrist streams require real cameras;
  neither is substituted when missing. Recording errors
  stay visible even after a Home or tracking action changes the general status.
- **Reject example:** While recording (or starting a recording), discards that
  current example. Otherwise the button says **Reject last saved example** and
  removes only the latest saved example, updating the dataset metadata. Reject
  leaves tracking unchanged. Rejecting a pending recording never deletes an
  older saved example.
- **Grip holding torque:** Sets the existing SDK's motor-output holding target
  from 0.1 to 2.0 N·m; the unchanged default is 0.5 N·m. The leader still sets jaw
  opening. After contact, motor torque feedback regulates the hold target by
  correcting the gripper position command. The page shows the applied target,
  measured motor torque, and HOLDING state. This is not a fingertip force sensor
  or an instantaneous peak-torque limit. The SDK's existing behavior remains in
  use; no automatic force escalation is added. Change the slider explicitly for
  stronger squeezing. Torque changes do not re-anchor arm tracking.
- **Kill arm:** Stops this controller's tracking and recording request and sends
  motor torque-disable commands to **only the selected arm and its gripper**.
  The other arm is untouched. It does **not home** the controlled arm.
  This is a software torque-off command; it does not
  disconnect the physical electrical supply. The page reports disable errors.
  Track explicitly reopens and enables only the selected follower afterward.

If Track startup fails, the dashboard remains online and shows the SDK's error.
The failed startup releases its per-arm ownership; another attempt requires an
explicit Track press. Kill remains available and uses the gripper-aware driver
for the selected arm's current motor mode, including before the arm is opened.

New examples default to `<station dataset_dir>/so101/right-context` (or
`left-context`) on NUCK-1. Each frame has the standard 14-value raw MakerMods
layout: left six joint radians + gripper mm, then right six joints + gripper mm.
The selected arm records measured state and commanded action. The opposite arm
records its latest published measured pose in both state and action (hold pose).
It is never commanded by this page. Its wrist video remains black; overhead and
the selected wrist remain live. Old seven-value datasets stay in their original
folders and cannot be appended to with this layout.

Independent controllers publish measurements from `MakerModsArms.read()` to
`~/.cache/dream_hackathon/arm_context/{left,right}.json`. A stopped publisher's
last measurement remains static context. Publishers must use this updated
adapter; an already-running controller must reload it on its next restart.
There is no second CAN reader, expiry cutoff, or invented zero pose. A missing
context file reports a recording error. To capture a static pose when that arm's
adapter is free, run (no enable/disable or target commands):

```sh
.venv/bin/python -m dream_hackathon.arm_context --side left
```

Train left and right separately using the normal bimanual model contract. For
each API call, supply the controlled arm's live state, the other arm's chosen
static measured context, overhead, the controlled wrist, and a blank other wrist.
Send only that call's controlled-side output to hardware; discard its other-side
output. The two calls can run in parallel. Both pose contexts must be expressed
in the normal per-arm bases with the same gripper units and EEF conversion as
training. Do not send raw 14-value vectors to a model expecting EEF features.

Prepare new recordings with the existing converter and validation:

```sh
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/my-task/so101/right-context --arm metal
```

The converter produces 34-value state and 32-value action features and runs
Dream's validator. This validates the format, not learned task performance.
For historical seven-value recordings, explicitly supply the static opposite
pose to a new export, then run `add_eef.py` on that export:

```sh
~/dream-env/bin/python tools/add_arm_context.py ~/hackathon_data/my-task/so101/right \
  --side right --context-pose ~/.cache/dream_hackathon/arm_context/left.json \
  --out ~/hackathon_data/my-task/so101/right-static-context
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/my-task/so101/right-static-context --arm metal
```

The export records the supplied pose's provenance. It does not claim that this
pose was measured during those historical episodes, and does not change their
original files.
Stopping a recording or killing motors saves frames already collected; dataset
validation errors, including an episode that is too short, appear on the page.

## Source for teammate agents

Repository: https://github.com/anish219/physical-ai-hack-2026
Branch: `codex/so101-dashboard`

```sh
git fetch origin codex/so101-dashboard
git switch --track origin/codex/so101-dashboard
```

On a checkout with unfinished work, inspect the branch or cherry-pick its commit
instead of switching that checkout. The relevant files are:

- `src/dream_hackathon/so101.py`: leader FK, tool-space mapping, IK, control loop.
- `so101_dashboard.py`: HTTP page/state/control/kill endpoints.
- `so101_power.py`: selected-arm torque-off; no enable handshake or homing.
- `so101_recording.py`: camera and LeRobot episode recording.
- `so101_gripper.py`: selected-gripper effort updates and cached torque feedback.
- `arms/makermods.py`: selected follower startup and explicit `power_off()`.
- `station_lock.py`: per-side ownership compatible with the legacy station lock.
- `teleop.home_step`: shared return motion used by the original Home control
  and the leader dashboard's Return to 0 button.
- `docs/so101_leader.md`: calibration, model/frame alignment, and command flags.
- `tests/test_so101*.py`: pose mapping, UI API, kill behavior, and real encoded
  dataset tests. The power tests mock hardware; they never press live Kill.

## Station process and restart

On NUCK-1, the repo is `/home/pah/physical-ai-hacks`, the environment is `.venv`,
the running tmux session is `so101-control`, and its log is
`/tmp/so101-control.log`. Inspect with:

```sh
tmux capture-pane -pt so101-control
tail -n 50 /tmp/so101-control.log
curl http://127.0.0.1:8081/api/state
```

If the controller is not already running, start it on the station:

```sh
cd ~/physical-ai-hacks
.venv/bin/python -m dream_hackathon.so101 \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B015818-if00 \
  --side right --dashboard-port 8081
```

It starts untracked. `--side left` selects the other follower;
`--dataset-dir /path/to/examples` and `--task "..."` select the recording folder
and task label. `--dry-run` computes targets without sending them and does not
record live examples. Do not commit `configs/station.yaml` or calibration data.

HTTP endpoints: `GET /api/state`, `POST /api/control` with JSON `tracking`,
`scale`, `grip_torque_nm`, or `recording_requested`, plus `POST /api/home`,
`POST /api/reject`, and `POST /api/kill`
with no body. Browser
controls connect only to the station through the SSH tunnel.
