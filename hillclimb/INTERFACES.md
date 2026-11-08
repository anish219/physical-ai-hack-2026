# hillclimb interfaces

These contracts let four parts be built in parallel. Python 3.12, run on the station with `~/dream-env/bin/python` (it has the Dream client, numpy, opencv, mujoco). Use the standard library for HTTP (`http.server`, `urllib`); add no new dependencies unless that env already has them. Every part must degrade instead of crashing: a timeout, an error or a stale input becomes "unknown" or "no", never an exception that escapes.

## Layout
```
hillclimb/
  common/        config, camera and arm clients, keys, logging (shared; owner: left controller)
  gate/          USB gate service
  runner/        scripted tower runner (vendored tall_tower stack plus fixes)
  dream/         Dream stacking backend
  left/          left-arm controller (state machine) and HTTP endpoints for the main controller
  sim/           helpers to run everything against calib/station_sim (tests)
  docs/
```

## Station endpoints already there (read-only use, except /target and /move on the arm we own)
- Left arm server `http://127.0.0.1:8691`, right arm server `8692`: `GET /state`, `GET /overhead.jpg`, `GET /wrist.jpg`; `POST /target {"q":[6],"grip":g}` (direct); `POST /move`; `POST /stop`.
- Never post to the right arm, to 8081, 8084, 8697, 18081 or 18084. The station sim (`calib.station_sim`, from the pah-synth copy at `~/pah-synth`) serves the same API on its own port. Every part takes its base URLs from config, so tests point them at a sim.

## Config (`common/config.py`, env plus `hillclimb/config.json`)
`LEFT_URL`, `RIGHT_URL`, `OVERHEAD_FROM` (which arm server serves the overhead), `HILLCLIMB_PORT` (default 8770), `KEYS_FILE` (default `~/.config/hillclimb/keys.env`, holding `OPENAI_API_KEY`), `LOG_DIR`.

## USB gate service (`gate/`, mounted at `/gate/*` on the hillclimb server)
- `POST /gate/arm`: takes the reference at go. Stores the overhead reference frame, the occupied-socket baseline (from the team verifier if available) and the loose-cable count. It resets the latch.
- `GET /gate/state` returns:
```json
{"known": true, "open": false, "latched_at": null, "unplugged_count": 0, "plugged_count": 4,
 "gripper_released": false, "cable_on_table": false, "fully_inserted": false,
 "checks": {"socket_emptied": "no|yes|unknown", "jaw_released": "...", "fast_vlm": "...", "agentic_vlm": "...", "healthy": "..."},
 "last_reads": [{"t": 0, "kind": "fast|agentic", "answer": {}, "ms": 0}]}
```
  The `known` … `fully_inserted` fields match the main controller's `verify_usb` contract. `open` is the stricter latched gate from SPEC section 7.
- Background loop every 1 to 2 s: the fast overhead Terra check against the reference; on a yes, the agentic overhead check; the gate opens on two agentic yeses in a row with fresh frames, plus the socket and jaw checks. Overhead only votes. Timeouts count as "no".

## Left controller (`left/`, the hillclimb server's other routes)
- `POST /left/arm` (config: `backend: "scripted"|"dream"`, checkpoint, wording table, `max_goal`, `crunch_at_s`, `crunch_goal`, `freeze_s`, `round_s`)
- `POST /left/go` starts the round clock.
- `POST /left/stop`.
- `POST /left/estop` holds the arm.
- `GET /left/state`:
```json
{"state": "LOCKED|STACK|SETTLE_CHECK|REST|RECOVER|FREEZE|END|FAULT", "time_left_s": 0, "tower_blocks": 0, "tower_in": 0,
 "active_goal": 0, "stage2_passed": false, "last_step": {}, "backend": "scripted", "health": {}}
```
- The main controller's bindings:
  - `POST /left/stack_next` makes one placement attempt; it returns when done or failed: `{"ok": bool, "grew": bool, "tower_blocks": n, "why": str}`.
  - `POST /left/rebuild`.
  - `GET /left/verify_tower` returns `{"known":true,"block_count":n,"collapsed":false,"standing_free":true}`.
- The left arm never moves while `GET /gate/state` has `open:false`. The controller checks this before every command.

## Backends (both used by `left/`; same Python interface)
```python
class StackBackend:
    def prepare(self, scene) -> None        # planning or warm-up during the gate wait; must not touch blocks
    def place_next(self, scene, deadline_s) -> dict   # one block onto the tower: {"ok", "placed_id", "why", "timing"}
    def recover(self, scene, deadline_s) -> dict
    def retreat(self) -> None               # up-first move to the look/start pose
    def hold(self) -> None
```
- `runner/` implements it with the scripted tall_tower stack.
- `dream/` implements it with Dream checkpoint B and the harness rules: stop on held-then-released, commanded start pose, keep-out while the jaws are empty, up-first retreat, keep-warm.
- `scene` is a dict: tower blocks and top xyz, loose blocks (id, colour, xy, yaw, lying), all from the left controller's perception and records.

## Logs
JSON lines under `LOG_DIR`, one file per part; every read, decision and command, with a timestamp.
