# Leonard orchestration console

Local-only frontend scaffold. No backend, live feeds, provider calls or hardware commands are implemented. Serve dist with `python3 -m http.server 5173 --directory dist`. Test with `node --test tests/machine.test.mjs`.

## Automatic progression

Start once. The independent monitor advances all phases and challenges after two consecutive positive judgments from distinct source images. Model routing changes automatically per phase. No manual completion button remains. Proposed predicate names are in dist/machine.js; exact success prompts, colors, counts, tolerances and failure criteria still need definition.

The action loop runs every 250 ms and the monitor every 500 ms, with one in-flight request per loop. These are scheduling defaults, not measured throughput. Camera/motor snapshots, active challenge time in milliseconds, phase, model and attempt are passed to adapters. Active time excludes pause and recovery.

## Image verifier

Replace sources.verifyCondition(observation) in dist/adapters.js through your existing backend. It receives the camera slots and a predicate such as usb_unplugged, block_stacked or ball_in_correct_basket. Select relevant images and ask one precise question. Return `{condition: "usb_unplugged", result: "yes", observationId: "camera1-frame123", evidence: "Connector visibly separated from port"}`.

Result is yes, no or uncertain. No means success has not happened; it is not inherently failure. Uncertain/invalid results reset confirmation. Repeated source frames do not add confirmations. Adapter errors pause the run. The placeholder returns uncertain indefinitely, so absent integrations never fabricate success. The real adapter must synchronize inputs, validate freshness, and reject stale/out-of-order observations. Keep VLM API keys out of browser code. No VLM provider has been wired in.

## Recovery

sources.evaluateFailure is an independent hook, currently returning null. Define fast telemetry checks and phase failure conditions later. Failure takes priority over success. The UI also allows manual failure injection.

Running → Stopping → Recovering → Running at the same phase and next attempt. stopCommands must resolve only after queued/in-flight actions are cancelled and both arms are held. Stop failure blocks the run. recover must return completed:true and verifyRecovery must return true before retry; the placeholder leaves the run in Recovery required. Three failed attempts block the run. Reset is the explicit operator escape.

Epoch tokens discard late inference and monitor results after transitions or interruption. Your backend must enforce ordering/cancellation for commands already submitted; frontend guards are not hardware interlocks. A checkpoint is task context, not a physical scene restoration.

Refresh resets session state. Export includes state, model routes, checkpoints and activity. All further development is local only; do not publish automatically.

## Right-arm observation bridge

`bridge/server.py` serves this same web app and imports the installed LeRobot MetalFollower directly. No hackathon runner is imported. The bridge rejects left-arm configuration. Arm connection happens only after the Connect right arm button is clicked. The driver bus is opened with handshake=False and closed with disconnect(False); follower.connect(), homing and torque enable/disable are not called. Motor feedback reads actively send requests; do not share the right bus with another controller. Actual hardware behavior remains unverified until tested on the NUC.

Test without hardware: `python3 bridge/server.py --mock`, then open `http://localhost:8094/?bridge=1`. Mock data is explicitly labeled in bridge mode. Start cycle is disabled while using the observation-only bridge.

For hardware, copy bridge/config.example.json to a private config file and replace port, can_interface and driver_id using the right-arm station mapping. Run using the NUC environment where lerobot is installed: `~/physical-ai-hacks/.venv/bin/python bridge/server.py --config /path/to/right-arm.json`. Server binds only to 127.0.0.1:8094; use `ssh -N -L 8094:127.0.0.1:8094 pah@10.103.0.132` from your Mac, then open `http://localhost:8094/?bridge=1`. Copy this console directory to the NUC separately before running; it has not been deployed there.

Cameras default to an empty configuration and are never scanned. Optional cameras entries use role wrist_r or overhead, e.g. `"wrist_r": {"device":"/dev/v4l/by-id/VERIFIED_RIGHT_CAMERA", "width":640, "height":480, "fps":15}`. Capture starts when the bridge starts and owns that device until shutdown. Do not configure cameras in use by a teammate. Left wrist is intentionally unavailable in right-only mode. The bridge shares its latest JPEG across browser and snapshot consumers. A stale/error camera returns HTTP 503 rather than a previous image.

Endpoints: GET /api/status, GET /api/telemetry, GET /api/cameras/{role}/frame.jpg, POST /api/connect and POST /api/disconnect. POST requires the same-origin browser's X-Leonard-Client header. Movement, Dream inference, VLM verification and physical recovery remain disabled/unimplemented. Test bridge behavior with `python3 -m unittest discover -s tests -p 'test_bridge.py'`.

## Reuse the station's running services (preferred)

Inspection of the live station found `dream_hackathon.unified_control` serving 8081 with persistent per-arm services on 8691 (left) and 8692 (right). Use `python3 bridge/server.py --existing --port 8094` on the NUC. This mode never imports a driver, opens CAN, captures a USB camera, or alters 8081. It reads /state and /wrist.jpg or /overhead.jpg from the existing listeners. Left camera endpoints are read-only; all power and jog mutations target right:8692 only.

Connect/Disconnect only subscribe/unsubscribe this dashboard; they do not power down or close the arm service. Explicit right-arm buttons request power on/off, position hold, or ±1 degree single-joint jogs through that service. Power off releases torque; Hold requests the measured joint and gripper pose instead. Mutations are refused if 8081 reports the leader assigned to the right arm. This check is not an atomic ownership lease; coordinate ownership with teammates. The existing service remains the sole hardware owner and maintains its own control loop. Dream orchestration stays disabled in bridge mode pending its separate integration.

q and /move targets are radians upstream; display and jog increments use degrees. Off or stale feedback is never displayed as current. Camera X-Stamp timestamps identify distinct source frames and must be no more than one second old. The existing service can return old JPEGs after power-off; this bridge rejects those. Cameras depend on their existing owner's lifecycle, so frames may be unavailable when both services are off. No hardware power or jog endpoints were exercised during validation.

A copy is running separately in ~/leonard-dashboard on the NUC. Open via an SSH tunnel: `ssh -N -L 8094:127.0.0.1:8094 pah@10.103.0.132`, then `http://localhost:8094/?bridge=1`. This is LAN/SSH-local only, not a Sites deployment. All state and camera reads are proxied same-origin by the bridge. Tests: `python3 -m unittest discover -s tests -p 'test_*.py'`.

Dual-arm monitoring update: existing-service mode now reads both /state endpoints concurrently and displays both arms in degrees. Left remains read-only while a teammate controls it; all hardware mutation routes still target only right:8692. Failure of one telemetry service does not hide the other's valid readings. Preview camera requests run independently with one in flight per camera, waiting 125 ms after each response/decode instead of refreshing at 1 Hz. The preview=1 route resizes to at most 640 px wide at JPEG quality 65; original snapshot routes remain unchanged for verification. OpenCV/numpy are required for resized previews; run the bridge using the existing physical-ai-hacks/.venv Python. Only our 8094 bridge was restarted, leaving the arm services and teammate controller untouched.

## Per-arm dashboard control

Both arms now start in observation mode. Each has Take control / Disconnect. POST /api/arms/{left|right}/acquire records dashboard command permission without opening hardware or changing power. Power, hold and jog routes require that permission and recheck the live 8081 leader assignment. An arm assigned to the leader is read-only; it must be released in the teammate UI before Leonard can acquire it. If the leader later takes an arm, Leonard drops its permission. This is cooperative dashboard gating, not an atomic exclusive lease enforced on all upstream clients. Coordinate any other custom controllers separately.

Disconnect per arm requests a hold of the measured joints and gripper before relinquishing command permission; it does not release torque or stop the other arm. If a hold cannot be confirmed from fresh state the action fails rather than pretending disconnect succeeded. Both panels expose the same power/hold/joint-jog features after acquisition. Step toggles select 1 or 5 degrees independently per arm; only signed 1/5 degree increments are accepted server-side. No movement/power commands were used for UI verification.

## Video transport

The page now uses one persistent multipart JPEG connection per camera (/api/cameras/{role}/stream.mjpg), paced at a maximum of 30 fps. Each role has one shared producer retaining only the latest unique source frame. Slow network writes time out; stalled frames are hidden and streams reconnect. Preview resizing is shared, and original snapshots remain available for the verifier. The page displays measured producer fps. A direct test of the original station sources measured roughly 15–19 unique fps, so the bridge cannot currently provide 30 distinct frames/second. The shared source uses a one-frame OpenCV capture buffer and does not apply the configured fps explicitly. No changes were made to it: the user requested keeping the shared controller running. A source capture change needs a coordinated restart and fresh measurements before promising 30 fps.

Video reconnect repair: subscribing after an idle period now waits for a fresh source frame instead of immediately returning a retained stale frame. Prior source errors are cleared when the first subscriber returns. The browser no longer uses document.hidden to suppress initial camera requests (embedded side panels may report hidden while visible), and a status watchdog reconnects stalled multipart streams. When the owning arm service is off, the UI labels the camera paused instead of displaying old video. Live validation is limited when the source arm services are powered off; automatic resume is covered by a stale-cache regression test.

Explicit leader handover: a leader-owned arm now offers Take over from leader. Only clicking that action sends /api/assignment with side:null to the existing 8081 controller, waits for unassignment, requests a hold of the selected arm's measured joints/gripper if powered, then grants Leonard command permission. It does not restart the shared controller, home an arm, or change torque/power. Ordinary acquisition still refuses to replace a leader, and acquiring the other arm leaves its assignment intact. This deliberate action interrupts that leader's teleoperation and may finalize its recording. It was unit-tested with fake services, not exercised on a teammate's live arm.

## Independent arm sequences and Dream API

The editor stores separate left and right sequences in browser local storage. Each is an ordered list of phases with nested subphases. A subphase can be a timed checkpoint or a Dream prompt with a model ID and action window. The left sequence defaults empty and only supports no-motion checkpoints. The right sequence defaults to an unplug-USB prompt. Steps advance when their configured time window ends; no image-based success verifier is connected, so a finished window does not mean the task succeeded.

On the NUC, `/api/dream/start` creates a DreamClient using the existing `~/.dreamlabs/credentials`; no key is sent to the browser. The loop sends resized overhead, left-wrist and right-wrist JPEGs plus fresh joint/gripper state for both arms, calls the selected checkpoint, and consumes only `q_r_chunk`. Predicted left-arm actions are ignored. The right arm must be powered, fresh and explicitly claimed by Leonard; leader assignment blocks commands. Targets are checked against the Metal joint limits and sent at 0.3 rad/s, below the existing service's 0.6 rad/s cap. Stop and duration expiry request a hold of the measured right-arm pose.

The base checkpoint `dreamlabs/dream-1-beta` needs a calibrated arm JSON file for Dream IK. Set `LEONARD_DREAM_ARM_FILE` in the Leonard bridge process environment. A Metal fine-tune can use the client's `ik: {arm: metal, head: eef}` path. The NUC has the Dream client and credentials, but this workspace has not confirmed a calibrated arm JSON file. Dream inference and physical motion have not been exercised.
