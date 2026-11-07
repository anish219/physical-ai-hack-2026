# Independent arm HTTP server

This replaces `/home/pah/pah/armserver.py`. It keeps the existing joint/gripper HTTP
API and Dream Metal driver, while each arm has its own state, HTTP listener, motor
loop and hardware ownership. It does not run inference, Oculus teleoperation, leader
teleoperation or recording.

On the NUC, run it in the existing Dream environment:

```bash
# Existing clients: left arm at the same URL and port as before
~/dream-env/bin/python ~/pah/armserver.py

# One arm server alongside a separate right-arm controller such as SO-101
~/dream-env/bin/python ~/pah/armserver.py --side left

# Both arms, independently addressed
~/dream-env/bin/python ~/pah/armserver.py --side both
```

| Arm | Base URL | Ownership |
| --- | --- | --- |
| Left | `http://127.0.0.1:8691` | shared `arms.lock`, exclusive `arms.left.lock` |
| Right | `http://127.0.0.1:8692` | shared `arms.lock`, exclusive `arms.right.lock` |

Default launch remains left-only. `--side right` starts only the right listener.
`--port` overrides a single arm's port, or the left port in Both mode;
`--right-port` overrides the right port in Both mode. Listeners remain loopback-only.

Each listener owns only its arm. Commands, `/stop`, `/quit`, startup failures and
motor-loop exceptions stay within that arm's worker. No arm mutex is held during
vendor I/O or camera capture. A slow left write therefore does not stop right
commands. A shared Python process is not isolation against interpreter/process
termination; separate single-arm processes remain an option.

## Existing API and behavior

Use the same paths and JSON bodies against the selected arm's base URL:

| Request | Existing behavior preserved |
| --- | --- |
| `GET /state` | `q`, `grip`, `goal`, `goal_grip`, `rate`, `moving`, `stalled`, `lag`, `stamp`, `cams`, `error` |
| `GET /wrist.jpg` | JPEG plus `X-Joints`, `X-Grip`, `X-Stamp`; 503 while no frame exists |
| `GET /overhead.jpg` | Same image response and metadata for this arm |
| `POST /move` | Optional `q` (six radians), `grip` (0–1), `rate` (rad/s), `stop_on_stall` |
| `POST /stop` | Set that arm's goal to its measured joints; gripper goal stays unchanged |
| `POST /quit` | Return that arm to its captured start, then vendor park/release and release ownership |

The original 30 Hz loop, proportional joint stepping, 0.6 rad/s rate cap, 1.5/s
gripper slew, 0.3 rad lag rule and optional 0.012 rad stall rule after the initial
ten samples are unchanged. Existing unknown-path responses are retained. No new
watchdogs, deadlines, homing targets, power controls or inferred calibration values
are introduced. Ctrl+C requests shutdown of the selected workers.

`--config` defaults to `~/.dreamlabs/station.yaml`, the existing Dream setup file,
not the collector's `configs/station.yaml`. Each side uses its own saved follower
configuration and `wrist_l`/`wrist_r` camera; the overhead camera is `scene`. The
ownership directory remains `~/.dreamlabs` even with another config-file path.

## Cameras and the existing share option

In Both mode, one capture thread opens the overhead camera and sends its JPEG to
both listeners, each with that arm's measured joints/gripper at capture time.
Quitting one listener leaves the other arm's wrist and overhead streams running.
Capture retains MJPEG, 1280×720, one-frame buffering, JPEG quality 85 and the existing
one-second reconnect interval. Camera threads read cached arm observations rather
than opening a second arm driver.

`--no-overhead` keeps its original meaning. Independent single-arm processes do
not share the in-process camera hub: explicitly give only one process the overhead
camera, using `--no-overhead` for the other.

`--share` retains the original explicit compatibility mode for an older
whole-station owner using the opposite arm: verify that this arm's CAN adapter and
wrist camera are free with `fuser`, then run wrist-only. The replacement also holds
its selected side lock in that mode. It never bypasses a same-side lock. With new
per-arm controllers, opposite-arm coexistence works without `--share`; when passed
alongside another process, `--share` retains wrist-only camera behavior.

## Installation and verification

The implementation is `src/dream_hackathon/armserver.py`. The small legacy entry
point is `scripts/armserver.py`; install it as `~/pah/armserver.py`. The NUC entry
point loads this repo from `~/physical-ai-hacks/src`. Back up the old script before replacement. Replacing files
does not restart an already-running process.

Hardware-free tests compare command traces, stall/lag decisions, state JSON,
response headers, paths and gripper handling directly with reference functions
copied from the original NUC server. A concurrent integration test deliberately
blocks left motor I/O and verifies right HTTP motion/stop/quit and lock release.
Linux tests cover coexistence with the opposite arm, same-arm exclusion and legacy
`--share` behavior. Running `--help`, importing the module, or reading its config
opens no hardware. A normal launch opens and energizes the selected arms, exactly
as the original server did.
