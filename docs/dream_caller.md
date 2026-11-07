# Independent arm Dream-1 caller

This local page calls Dream-1 through the station's installed Dream SDK and talks to
the new independent arm server over HTTP. It opens no CAN adapters or cameras of
its own. It currently targets the Metal arms served by `scripts/armserver.py`.

On the station, start the independent server for the arm(s) you want available:

```bash
cd ~/physical-ai-hacks
PYTHONPATH=src ~/dream-env/bin/python scripts/armserver.py --side both
```

Left listens on `127.0.0.1:8691`, right on `127.0.0.1:8692`. The center camera
must be available as `/overhead.jpg` on the selected listener. A `--side both`
launch shares its capture. A separate single-arm launch works too; do not use
`--no-overhead` for a listener used by this page.

In a second station terminal, run the page in the same Dream environment, using
the existing `~/.dreamlabs/credentials` or `DREAM_API_KEY`:

```bash
cd ~/physical-ai-hacks
PYTHONPATH=src ~/dream-env/bin/python scripts/dream_caller.py --port 8082
```

Forward the page to your laptop:

```bash
ssh -L 8082:localhost:8082 pah@<station-ip>
```

Open <http://localhost:8082>, choose **Left arm** or **Right arm**, and click
**Connect selected arm**. The page streams measured state and both live camera
views through that arm's listener. Enter a task and checkpoint (default
`dreamlabs/dream-1-beta`).

- **Call once · inspect** calls the real API and displays the selected arm's
  prediction, sending no motor commands.
- **Run selected arm** repeatedly calls Dream and plays the selected arm's
  joint/gripper chunks using `/move`.
- **Stop selected arm** cancels further playback and sends `/stop` to the selected
  listener. A pending API response is discarded for playback; it may still finish
  and display its prediction. Stop does not quit the server or release torque.
  The server's existing `/stop` holds measured joint positions and leaves its
  current gripper target unchanged.

Switching tabs changes only the view. Each arm has its own task, Dream session,
observation thread and playback thread; an explicitly started run continues while
you view the other arm. A slow model call for one does not block the other.

For each caller, the selected arm's six measured joints (radians) and measured
grip (0 closed, 1 open) are streamed into Dream. The other arm is always six zero
joints and zero grip, with a black JPEG in its wrist-camera slot. Dream's SDK
computes tool poses from those joints, including the dummy pose, and maps the
center image to the checkpoint's scene/overhead camera role. No opposite-arm
measurements, cameras, or cached context poses are read. The caller extracts only
its own `q_l_chunk` or `q_r_chunk`; all opposite-arm predictions are discarded.
Only explicitly connecting or running the second caller contacts its listener.

The page's JSON panel shows the pre-SDK observation; SDK IK adds tool poses and
adapts the fields for the model's embodiment. Metal gripper drive is also supplied
in millimetres for a Metal fine-tune. The API key stays in the station process.

Defaults are 10 observation ticks/s, 20 action rows/s and 320×180 camera frames.
Each returned chunk plays fully before the next request; observation streaming
continues during requests and playback. Adjust `--action-hz`, `--observation-hz`,
`--width`, and `--height` to match a checkpoint. Arm-server motion/rate behavior
continues to apply. Custom listeners can be set with `--left-url` and `--right-url`.
Errors appear on the page; an API call is not automatically retried.

Closing the page does not stop a run. Use its Stop button to send a stop command.
Exiting the caller process ends playback without sending `/quit`, homing,
releasing torque, or commanding the opposite arm.
