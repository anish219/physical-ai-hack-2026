# Unified station controller

The unified page is served on NUCK-1 at port **8081**. The existing SSH tunnel
continues to work: [open the page](http://localhost:8081/).

```bash
ssh -N -L 8081:127.0.0.1:8081 pah@10.103.0.132
```

Each arm has an independent **Turn on** / **Turn off / kill** control. Turning on
opens only that arm and holds its measured joint/gripper position. Turning off
disables only that arm's motor torque without homing. It does not switch the
physical electrical supply. Errors remain visible on the corresponding arm card.
An arm's startup or shutdown does not hold the other arm's request or motor mutex.

Each arm card has input buttons for **Dream API**, **SO-101**, **Sequences**, and the disabled
**Full Auto Control** placeholder. Clicking the active input deselects it; neither
arm needs an input. SO-101 can be assigned to only one arm: selecting it on Left
releases Right, and vice versa. Selecting Dream API separately on Left and Right
opens two independent API panels, each with its own task and policy session.
The **Both arms** controls between the arm cards select a combined Dream API
session instead. Both clears the individual inputs; choosing an individual input
clears Both. The combined panel uses all three cameras and both live arm poses.
Both has a matching disabled Full Auto placeholder, and no SO-101 option.
Sequences follows the same selection rules: separate Left and Right buttons
open independent editors, while the center Both button opens one combined editor.
Changing input cancels affected API playback (including pending replies) and
releases the previous leader before the new source can command the arm.

Each arm card also has its own **Gripper holding torque** slider (0.1–2.0 N·m)
and applied/measured motor-torque feedback. The arm server owns this setting for
all inputs, including Dream API and no input. Input changes do not reset it.
The unified SO-101 child uses `--server-gripper-torque`, so its remembered leader
settings never overwrite the arm's torque target. Standalone leader control keeps
its existing torque slider behavior. `POST /api/arms/{side}/grip-torque` accepts
`{"torque_nm": number}` without requiring a leader assignment.

Only panels for selected inputs appear below the arm cards. Without a SO-101
assignment its panel is hidden; with no selected inputs no lower panels appear.
SO-101 selection
connects the leader but starts untracked. **Track** enables the selected arm if
needed and captures fresh hand/tool anchors. **Untrack** holds its last target.
Changing assignment disconnects the old leader worker, saves any active example
under the existing save rules, and starts the new worker untracked. It does not
home, disable, or reopen follower hardware. None leaves powered arms holding their
current targets.

Both arm cards always show **Return to 0** and **Set 0**, independently of input
selection. Return to 0 pauses that arm's input, powers it on if off, and returns
to its saved joint pose at the existing Home speed (0.5 rad/s), keeping the gripper
target. Set 0 stores the powered arm's current measured joint pose. The initial
zero is captured on first power-on; it is replaced only by Set 0, never by Kill
or subsequent power-on. Per-arm zeros persist across service restarts in
`~/.cache/dream_hackathon/zero/{left,right}.json`. Existing zeros are retained
when migrating the service. Kill cancels a Return to 0 without changing its saved pose.

The SO-101 panel retains the leader controls:

- Track / Untrack and the tracked-state indicator.
- Translation scale, 0–3×, with the existing re-anchoring behavior.
- Record example / Stop & save example, frame count, dataset folder and saved
  episode reporting.
- Reject current/pending example or reject last saved example, as before.
- Per-arm gripper holding torque, applied/measured torque and HOLDING feedback
  are in the arm cards and work independently of the selected input.
- Per-arm Kill and Return to 0 are in the arm cards above both input panels.
- Existing recording, startup and torque-error reporting.

Each **Dream API** panel is fixed to its selected Left, Right, or Both input. Turn
on the chosen arms with their existing power controls and press **Connect**. Enter
a task and checkpoint, then **Call once · inspect** to display a prediction without
playing it, or **Run API** to repeatedly infer and play action chunks. **Stop API**
cancels playback, discards pending replies for playback, and holds the selected
arms through their existing stop endpoint. It leaves power on.

Both uses one policy session with live poses for both arms and all three cameras:
top, left wrist, right wrist. Left/Right use top and the chosen wrist, a black image
in the other wrist slot, and six zero joints plus zero grip for the other arm.
Dream's SDK computes tool poses, including the dummy pose, from those joints.
Single-arm playback sends only that side's returned chunk; Both plays matching
left/right rows at 20 Hz through the same direct-target path used by the leader.
Observation streaming continues at 10 Hz during inference and playback.

API task and checkpoint text are shared live with everyone connected. Edits are
sent on input, distributed through `/api/events`, and retained in server memory
for new/reloaded browsers. Left, Right and Both have separate shared drafts.
Editing changes the shared draft, not an already-running policy's task; the next
Call or Run uses the latest shared text. Simultaneous edits use the server's
arrival order. `/api/dream/{left|right|both}/draft` accepts partial `task` and
`checkpoint` updates without invoking inference or moving an arm.

Run requires the matching input selection: two individual API assignments stay
independent, and only Both enables combined playback. Changing between combined
and individual modes stops the previous playback. Turning an arm off
also cancels API playback that includes it, without stopping the other arm's
independent caller. API errors appear in the API panel and are not retried.

The existing Dream API key is loaded server-side from the existing credentials or
environment. No key is entered or displayed in the page. The observation panel
shows pre-SDK state; Dream adds EEF poses and maps camera roles for the checkpoint.

## Sequences

The sequence editor was pulled from NUCK-1's unpushed `/home/pah/leonard-dashboard`
code. Its original bridge and editor are preserved in `vendor/leonard_sequences`,
and the original service at port 8094 is left intact. That legacy interface is
not shown on the main page. It can still be launched independently as documented
in the preserved README and SOURCE.txt.

Select **Sequences** under Left, Right, or Both to show a matching control panel.
Add phases and nested subphases. Each subphase is either a **Dream task** with a
prompt, checkpoint and action window (1–300 seconds), or a **Wait / checkpoint**
(0–300 seconds). Every subphase also has a **Monitor** (`none`, `prompt` for a
VLM instruction, or `script` for a registered pointer). See
[sequence_monitors.md](sequence_monitors.md) for the plug-in Anish's model
should use. Run walks the phases and subphases in their defined order; Stop
cancels the current action and prevents later steps. Source changes, Return to 0
and arm-off also cancel affected sequences. Left and Right can run independently;
Both is one coordinated bimanual sequence and excludes individual inputs.

The original timed semantics are retained: finishing a window does not verify
task success. The older browser-side completion/recovery prototype remains in the
preserved source; no new success verifier or recovery behavior is enabled here.
The unified scheduler runs server-side and uses the existing API workers, direct
target transport, camera/dummy-pose layout and per-arm gripper settings. This
extends the legacy right-only Dream steps to Left and Both without opening another
CAN adapter or camera. Existing Dream credentials are used.

Definitions are shared with connected browsers and stored in
`~/.cache/dream_hackathon/sequences.json` (override with `--sequence-file`). Edits
apply to the next explicit Run; an active sequence uses the definition captured
at its start. The original defaults are retained: Right has an Unplug USB phase,
Left starts empty, and the new Both definition starts empty. Existing plans saved
in the legacy editor's browser storage remain there unchanged.

The shared backend accepts `POST /api/sequences/{left|right|both}/save` with
`{"definition":{"phases":[...]}}`, plus `/run` and `/stop` with an empty object.
Sequence state, current phase/subphase and errors appear in `/api/state` and the
existing `/api/events` stream. Sequence data and the original backend source are
not used as instructions to start hardware automatically.

## Ownership and processes

`managed_arms.py` owns the follower CAN adapters. Each arm has its own worker,
request mutex, power lifecycle and HTTP listener. It holds a shared global lock
and the exclusive selected-side lock, compatible with the per-arm ownership API.
The old whole-station lock remains exclusive against both arms.

| Component | Endpoint | Responsibility |
| --- | --- | --- |
| Unified page | `127.0.0.1:8081` | Per-arm input/power/zero controls, separate SO-101 and Dream API panels |
| Left arm server | `127.0.0.1:8691` | Own and command only the left follower; its camera/state API |
| Right arm server | `127.0.0.1:8692` | Own and command only the right follower; its camera/state API |
| Assigned SO-101 worker | `127.0.0.1:8083` | Existing leader FK/IK, Track, Home, gripper and recording logic |
| Dream API worker | `127.0.0.1:8084` | Independent Left/Right and combined Both policy sessions in `~/dream-env` |

The SO-101 worker uses `remote_arm.py`; it never opens follower CAN adapters or
takes their locks. Its `close()` leaves follower power and held targets unchanged.
This is what lets the selector change without competing with the arm server.
Each side's translation scale and grip setting are retained across assignments.
An untracked remote SO-101 worker reads the server's held targets for recording
and sends no joint/gripper targets. Untrack is serialized with its motor loop,
so an old leader command cannot follow a completed handoff or Return to 0 request.

The independent servers retain the existing `/state`, `/move`, `/stop`, `/quit`,
`/wrist.jpg` and `/overhead.jpg` interfaces. The managed listeners additionally
accept `POST /power {"enabled": true|false}` and
`POST /grip-torque {"torque_nm": number}`. SO-101 sends its computed joint/gripper
targets through `POST /target`, which passes them to the SDK without another joint
rate cap, proportional interpolation, gripper slew or legacy lag/stall override.
The original leader IK/SDK behavior and its Home motion still apply.
`/quit` returns and closes its arm;
the managed listener stays online for an explicit subsequent power-on.

The existing `/move` motor stepping, rate, gripper-slew and lag/stall behavior
remain for legacy arm-server clients. They are not added on top of SO-101's
already-computed targets. No watchdog cutoffs or automatic power
retries are added. The new page reports failures without changing the other arm.

## Cameras and recordings

The arm servers share the overhead capture in-process and each owns its wrist
stream. The recorder borrows JPEG frames through HTTP, rather than reopening
physical cameras. The opposite wrist remains the existing blank recording stream.
The selected arm records measured state and commanded action in the same
seven-value format as the currently running leader controller. It never opens
the other arm or requires a static opposite-arm pose for recording. Missing
camera frames are reported through the existing recording error field.

Examples use `left` and `right` subfolders under the configured recording root.
The NUC launch retains `/home/pah/hackathon_data/training-run-1/so101`, so selecting
Right shows the existing 51 saved examples and new recordings can continue there.
Existing files are neither rewritten nor deleted during migration. As before,
these single-arm recordings need the separately documented context export and
EEF conversion before use with the bimanual training schema.

## Launch

Use the collector/leader environment (it contains the MakerMods stack and IK):

```bash
cd ~/physical-ai-hacks
.venv/bin/python -u -m dream_hackathon.unified_control \
  --config configs/station.yaml \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B015818-if00 \
  --port 8081
```

Alternatively, the replacement `~/pah/armserver.py --unified` entry point accepts
the same arguments. With no `--enable` argument both followers start unopened;
`--enable right` or `--enable left right` explicitly enables those arms on launch.
SO-101 assignment starts at None. Use `--dataset-dir` for the unified recording
root, `--translation-scale` and `--gripper-hold-torque-nm` for initial settings.
`--dream-python` selects the Dream SDK interpreter (default `~/dream-env/bin/python`);
`--dream-port` changes its internal port (default 8084). Its log is
`/tmp/unified-dream-api.log`. The Dream worker starts disconnected with no inference
or motor commands until requested from the page.

Do not run the old direct-CAN leader controller on the same arms simultaneously;
the unified service replaces that process. Its SO-101 child is launched with
`--remote-server` and is managed by the selector. The old standalone arm-server
launch modes and direct-CAN SO-101 CLI remain available separately.
