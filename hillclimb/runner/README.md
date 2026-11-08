# hillclimb.runner: the scripted tower runner

The left arm's scripted stacking path. It sees the table through the arm server's overhead and wrist cameras, plans a tower and builds it block by block with planned, checked moves. This is the code that recorded the Nov 7 and Nov 8 night stacking data on the real Metal arm, packaged for the harness.

## What is here
- `calib/`: the runner, vendored from the team's private working repo (pah-hackathon `calib/`, working tree of Nov 8 morning):
  - `tall_tower.py`, the tower runner;
  - `data_loop.py`, `opstack.py`, `arm_runner.py`: perception, block moves, arm API and records;
  - `tower_plan.py`, `tower_safety.py`, `safety.py`: kinematics, sag model and sweep checks;
  - `night_reach.py`, `tilt_helper.py`: the tilted grip for levels 8 and 9;
  - `night_build.py`, `night_move.py`: exact-order build and single moves;
  - `overhead_change.py`, `overhead_fit.py`, `fit_overhead.py`, `look_cover.py`, `cube_views.py`, `tagcube.py`: look and check helpers;
  - the calibration JSONs.
- `sim/physics.py` (plus `measured.json` and `servo_fit.json`) and `assets/dream/`: the arm model the planner's kinematics and sag use (needs `mujoco`, which `~/dream-env` has).
- `backend.py`: the `StackBackend` interface from `hillclimb/INTERFACES.md`:
  - `prepare()`: looks plus the whole plan, no block touched;
  - `place_next()`: one more block on the tower, or the tilted grip at levels 8 and 9 when helper spots are given;
  - `recover()`, `retreat()`, `hold()`.

  Every call returns a dict, and nothing raises out of it.
- `simclock.py`: the station sim's clock.
- `sim_test.py`: the backend test against the station sim.

Imports were rewritten to `hillclimb.runner.*`. Runs, records and status go to `~/hillclimb_runs` (set `HILLCLIMB_RUNS` to change it). The laptop-only parts are removed: tunnels, copying datasets back over the network, and an optional outside vision check of weak green or purple fits, now a no-op. Every collision, reach, sweep, carry-floor and knock refusal is unchanged.

## Fixes included
| Fix | What it does |
|---|---|
| Overhead camera | `cam_mid.json` is the Nov 8 refit: 13 px median miss on real frames, against 78 px for the old file |
| F1 | The wrist ignores the impossible table fit "inside the block under" |
| F2 | The overhead may only say "fallen" up to 2 high (parallax) |
| F3 | A weak re-look within 2 cm of where the block was is accepted |
| F4 | A place that the standing check confirmed is not dropped for a fit just under the bar |
| G1 | Top look: support and standing checks from almost straight above the tower |
| G2 | `OVERHEAD_CAM` setting, default the Nov 8 refit |
| G3 | The whole-sequence planner never stacks a colour on the same colour |
| G4 | The place aim corrects for the block's measured offset in the jaws |
| G5 | Pick order: blocks only reachable at a low carry go first |
| G6 | `tallest --stage`: a staging pass into a reachable grid first |
| G7, G8 | `tallest tbrun`: replays an outside planner's plan through the runner's own moves, with a fallback. Sim only; it needs pah-synth's builder |
| G9 | A weak after-place fit within 2 cm of the target is accepted |
| `TALL_KNOWN_BIG` | Piece sizes per table, as a JSON override: `{"green": false, "orange": false}` for all-small cubes. **Set it to the venue's real pieces**, because a wrong entry forces the wrong shape on every block of that colour |
| `GATE_FILE` | The first pick or push waits until the file exists (the USB gate). Looks and planning run before it. `place_next()` refuses at once while the file is missing |
| `BIAS_FILES` | The per-level place correction also learns from these record files (capped at 8 mm) |
| Tilt helper | Levels 8 and 9 tilted 40 deg from a staged 2-block helper stack, with an arm-link clearance check and a safe way out of the tilt |

## Running it against the station sim
```bash
cd ~/pah-synth && MUJOCO_GL=egl SIM_DR=plain SIM_RENDER_SCALE=0.5 SIM_OVERHEAD=nov8 \
  ~/dream-env/bin/python -m calib.station_sim --port 9111 --speed 2 --mode real --layout LAYOUT.json --photoreal 10
cd <checkout> && RUNNER_SIM_PORT=9111 TALL_KNOWN_BIG='{"green": false, "orange": false}' PYTHONPATH=. \
  ~/dream-env/bin/python -m hillclimb.runner.sim_test --levels 8
```
Use a sim port that is none of the real station's ports; the runner refuses one that is. Without `RUNNER_SIM_PORT` the backend drives the real left arm server (`LEFT_URL`, default `http://127.0.0.1:8691`), so run it there only with the arm owner present.

## Known limits
- **One continuous run from a clear table to 9 has not been shown yet.**
  - Shown in the sim: 6 high cleanly with fingers down.
  - Shown in the realistic sim: levels 8 and 9 from a tidy 7 with pre-staged helpers (9 standing in 2 of 3 runs).
- **9 needs more blocks than fit in reach at once.** About 11 blocks are needed (7 fingers down, plus 2 helper bases, plus 2 tilted), so blocks must be fed in during the build.
- **7 fingers down and the tilted 8 and 9 need one tower spot where both plan.** The backend does not pick that spot for the tilt yet: give `helper_spots` and build at a spot checked for both.
- **Pace:** about 15 to 17 s per confirmed block. A re-place costs about 10 s.
