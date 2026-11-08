# hillclimb.dream: the Dream stacking backend

The left arm's `StackBackend` (see `../INTERFACES.md`) run by a Dream-1 fine-tune. The default checkpoint is B (`dream-1-ft:f9ed26ed-blocks-left-fresh500-20261108`). The controller loads it with:

```python
from hillclimb.dream import make_backend
backend = make_backend(settings_dict, arm_client)   # arm_client: hillclimb.common.arm.ArmClient
```

`make_backend` starts opening and warming the policy at once. `place_next` answers `policy warming` until it is ready.

## What a step does (`place_next(scene, deadline_s)`)
1. **Picks a loose block and writes the prompt** in the checkpoint's own wording (`hillclimb.common.wording`: `real` for B and C, `synth` for fine-tune #2).
   - If the scene carries block positions (`loose`, `top`), it uses them.
   - Otherwise Terra reads the current frames and lists the tower top's colour and the loose blocks it can name. Only names the table can make unambiguous are sent.
   - Colours B misses most in the sim (purple, yellow) go last.
2. **Goes to the training start pose,** using the commanded joints `(0.0, -1.78, 1.80, -1.456, 0.02, 0.0)` with the jaws at 0.80. That's the median of row 0 of the action column over B's scripted stack, move and unstack episodes. It goes up first if it isn't already there.
3. **Runs the policy** until the jaw has held for at least 1.5 s, then let go with the jaws open, then 2 s more. The policy never stops on its own.
4. **Stops and retreats up first:** rise until the fingertips clear the tower top plus a block by 6 cm, cross, then join the start pose. It opens the jaws only outside the tower column.
5. **Returns** `{"ok", "grew", "verified", "placed_id", "prompt", "why", "timing"}`.
   - `ok`: the jaw cycle completed.
   - `grew`: the release happened over the tower, when the tower's position is known.
   - `verified` is always false, so the controller confirms the growth with Terra.

## Safety rules in the loop
- **Keep-out:** while the jaws hold nothing, any row that would put the tool point or a fingertip inside the tower's column (footprint plus 2 cm, up to the top plus 3 cm) is refused, and the arm keeps its last target.
  - The column comes from `scene["tower_xy"]`, or `settings["tower_xy"]`, or else the first release position.
  - With no tower position there is no keep-out.
- **Rows go through the controller's `ArmClient`,** so a closed gate refuses every motion.
- **Dream refusals never escape:**
  - `pool_warming`, capacity and rate-limit refusals set mode `warming` and retry after `retry_after_s`;
  - an expired session restarts the episode;
  - timeouts and network errors retry;
  - only a refused key or an unknown checkpoint gives mode `error`.
- **Keep-warm:** while idle, one inspect-only call every 30 s, with its rows never played.

## Playback
- Our own loop (`loop.py`) on the `dream` library; the team's `dream_caller` is not used or edited.
- Observations come from the arm server's `/state`, `/overhead.jpg` and `/wrist.jpg`, at 10 Hz and 320x180, in the checkpoint's embodiment slots.
- Rows are played to `/target` at 20 Hz. Each chunk plays from its row 0, as in the experiments that succeeded; the next chunk is fetched about 1.5 s before the current one runs out, so there's no pause.

## Kinematics
- `geometry.py` uses the calibrated arm model in `HILLCLIMB_CALIB_ROOT` (default `~/pah-synth`): `calib/tower_plan.py`, `calib/arm_left.json`, `calib/sag_left.json` and `sim/servo_fit.json`.
- **Model joints** = measured + `dq`; **commands** = `sag.command(q_model) - dq`.
- If the model can't load, the retreat falls back to a plain joint move to the start pose, and the keep-out is off.

## Sim check (station)
```bash
cd <repo>; ~/dream-env/bin/python -m hillclimb.dream.sim_trial all --steps level1,level2,missed2 --sim http://127.0.0.1:9121
```
- The sim must be running:
```bash
cd ~/pah-synth && MUJOCO_GL=egl SIM_DR=plain SIM_RENDER_SCALE=0.5 SIM_OVERHEAD=nov8 ~/dream-env/bin/python -m calib.station_sim --port 9121 --speed 1 --mode real --layout LAYOUT.json --photoreal 10
```
- Each step builds a tidy tower through `/reset`, runs one placement and prints the truth before and after.
- `--terra` drops the positions so the Terra scene read is used. It needs `OPENAI_API_KEY` in the keys file.
