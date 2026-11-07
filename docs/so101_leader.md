# SO-101 leader to one Metal arm

The mapping is in end-effector space: calibrated SO-101 angles go through the
leader's forward kinematics, its handle's relative translation and rotation
become the Metal tool's pose target, and the existing Metal IK calculates all
six follower joint targets. There is no joint-to-joint mapping or fixed wrist yaw.
Gripper opening is independent and maps from calibrated 0–100 to the Metal
arm's configured closed/open angles.

The leader model is TheRobotStudio's SO-101 leader URDF with middle-of-range
joint zeros, bundled under src/dream_hackathon/models/. FK uses its fixed
handle_frame_link; the trigger angle does not affect that frame. Metal uses
the station config's URDF, joint signs and measured TCP. Both start poses are
captured on each Track press, so their absolute poses need not match.

On NUCK-1, in ~/physical-ai-hacks, the calibrated leader port is:

~~~bash
LEADER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B015818-if00
~~~

Print tool-pose targets and the IK results for the right arm without commanding it:

~~~bash
.venv/bin/python -m dream_hackathon.so101 --leader-port "$LEADER" --dry-run
~~~

The command serves a dashboard on port 8081 and starts untracked. Forward it
from your laptop with `ssh -N -L 8081:localhost:8081 pah@10.103.0.132`, then open
http://localhost:8081. The page has a Track/Untrack button and a translation
scale slider. Track opens the selected follower on first use and captures fresh
leader and follower tool poses on every engagement. Untrack holds the last arm
and gripper targets while you reposition the leader. Changing scale re-anchors
at the current target so the scale change does not jump the target.

Use `--dashboard-port` to change the port, or `--no-dashboard` to start tracking
immediately with command-line settings.

Start the live dashboard for the right arm (the default), or select the left arm:

~~~bash
.venv/bin/python -m dream_hackathon.so101 --leader-port "$LEADER"
.venv/bin/python -m dream_hackathon.so101 --leader-port "$LEADER" --side left
~~~

Only the selected follower is opened. --side right explicitly selects right.
Translation is 1 metre per leader metre by default; --translation-scale 2
doubles it. Orientation follows the leader's relative rotation without scaling.
--leader-to-base-rpy-deg ROLL PITCH YAW specifies the leader base's orientation
in the follower base, in degrees; default 0 0 0 means aligned bases.
For example, --leader-to-base-rpy-deg 0 0 90 maps leader +X motion to follower +Y.
--hz defaults to 30. Metal IK retains the speed and joint-limit settings
already in the station config. Ctrl+C uses the existing selected follower's
return-to-start and disconnect sequence.

Leader-only reads and explicit recalibration use the saved ID hackathon_so101:

~~~bash
.venv/bin/python -m dream_hackathon.so101 --leader-port "$LEADER" --read-leader
.venv/bin/python -m dream_hackathon.so101 --leader-port "$LEADER" --calibrate
~~~

The driver expects servo IDs 1–6, base to gripper. --leader-urdf and
--leader-tool-link allow a different leader model or fixed tool frame.

The page also offers **Record example / Stop & save example** and **Kill arms**.
Recording writes the selected arm's measured state, commands and all three
cameras as LeRobot v2.1 episodes. Use `--dataset-dir` and `--task` to override
the folder and task text. Kill disables both arms' motor torque directly,
without homing; it does not disconnect the physical supply.

Per-side locks let this controller coexist with a compatible controller for the
opposite arm; legacy whole-station controllers remain exclusive. The collection
dashboard is separate. See [CONTROL_PAGE.md](../CONTROL_PAGE.md) for teammates'
connection, branch, recording, power-off and station-process instructions.
