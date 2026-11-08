# left: the left-arm stacking controller and the hillclimb server

## Run it (on the station)
```bash
cd <repo> && ~/dream-env/bin/python -m hillclimb.left.server --port 8770
```
- The status page is at `http://<station>:8770/`. It shows the three cameras at 2 fps, the gate checks, the left state machine, the timer, the tower and the score estimate.
- Settings come from `hillclimb/config.json`, overridden by the environment:
  - `LEFT_URL` and `RIGHT_URL`: the arm servers (default 8691 and 8692);
  - `OVERHEAD_FROM`: which arm server serves the overhead camera;
  - `KEYS_FILE`: holds `OPENAI_API_KEY`, never printed;
  - `LOG_DIR` and `STATE_FILE`.
- To test against a sim, start one with `python -m hillclimb.sim.launch --port 9131` and set `LEFT_URL=http://127.0.0.1:9131`.

## A round
1. `POST /left/arm` with the settings as JSON. Every field is optional:
```json
{"backend": "scripted", "checkpoint": "dream-1-ft:f9ed26ed-blocks-left-fresh500-20261108",
 "mode": "match", "drive": "auto", "round_s": 300, "max_goal": 9, "crunch_at_s": 90, "crunch_goal": 5, "freeze_s": 20}
```
   - `backend`: `scripted`, `dream` or `fake`.
   - `mode: "match"` locks every setting once go is pressed (409 until the round ends).
   - `drive: "auto"`: the controller places blocks itself. `drive: "external"`: the main controller calls `stack_next` for each block.
2. `POST /left/go` starts the 5-minute clock. The state goes to LOCKED; the backend's `prepare` runs (planning only).
3. **The left arm cannot move until the USB gate is open.** Every motion command goes through a guard that refuses it, and the guard never refuses a stop.
4. The states after that:
   - **STACK and SETTLE_CHECK:** the tower count goes up only on a confirmed placement.
   - **REST:** at the active goal. The crunch goal takes over at `crunch_at_s` left.
   - **RECOVER:** below the goal, after two agreeing "fell" looks or after a fault.
   - **FREEZE:** no new placement in the last `freeze_s`.
   - **END.**
5. `POST /left/stop` ends the round. `POST /left/estop` holds the arm at once.

A backend failure or exception becomes FAULT. The controller holds, waits `fault_wait_s`, then goes to RECOVER, and never crashes. The state is written to `STATE_FILE` every second; after a restart the round resumes on the same clock, with the same tower count, starting in RECOVER so it looks again first.

## How the main controller's bindings call it
Use HTTP operations, and set **Output object path** to `body`.

| Binding | Method and URL | Returns |
|---|---|---|
| `stack_next` | `POST http://127.0.0.1:8770/left/stack_next` | `{"ok", "grew", "tower_blocks", "why"}` |
| `rebuild` | `POST http://127.0.0.1:8770/left/rebuild` | `{"ok", "grew", "tower_blocks", "why"}` |
| `verify_tower` | `GET http://127.0.0.1:8770/left/verify_tower` | `{"known", "block_count", "collapsed", "standing_free"}` |
| `verify_usb` | `GET http://127.0.0.1:8770/gate/state` | the USB contract plus `open` (the latched strict gate) |

- For `stack_next` and `rebuild`, arm with `"drive": "external"` first. They block until the placement finishes, up to `step_timeout_s` plus 30 s, and they refuse while the gate is shut or inside the freeze window.

`GET /api/state` returns everything the status page shows: left state, gate state and health.

## Backends
`hillclimb.runner` (scripted) and `hillclimb.dream` (Dream) each expose `make_backend(settings: dict, arm: ArmClient)`. It returns an object with `prepare`, `place_next`, `recover`, `retreat` and `hold` (INTERFACES.md).
- `place_next` returns `{"ok", "grew", "verified", "placed_id", "why"}`.
- With `verified: true` the placement counts as it is. Otherwise Terra must agree that the tower grew by one and stands, unless `vision_confirm` is off.
- The arm object passed in already carries the gate guard, so use it for every motion.

## The gate module contract (hillclimb.gate)
The server mounts the gate's routes if `hillclimb.gate` exposes `http_handler(method, path, body) -> (status, dict)` for paths under `/gate/`. If it also has `start()`, that runs once at server start. The left controller reads the gate through that handler (`GET /gate/state`), or through `GATE_URL` when the gate runs as its own process. Anything missing reads as unknown, and the gate stays shut.

## Tests
```bash
LEFT_URL=http://127.0.0.1:9131 RIGHT_URL=http://127.0.0.1:9131 ~/dream-env/bin/python -m hillclimb.sim.test_round --server-port 9132
```
These use a fake gate and fake backends against a sim. They refuse to run if either URL points at a real arm.
