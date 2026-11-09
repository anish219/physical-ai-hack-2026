# Run sheet: the left arm on the real station

Every step that moves the arm needs the arm owner at the station with a hand on the e-stop.

## 0. Before anything moves (at the station)
- **The left arm:** powered, at its rest pose, nothing in the gripper.
- **The team controller (8081):** no input selected on the left arm (Dream, SO-101 and Sequences all off).
- **The blocks:**
  - 18 to 30 cm out from the left arm's base, within 14.5 cm either side;
  - about 10 cm apart, flat, none touching;
  - nothing in the way of the arm's rest pose.
- **The pieces:** `TALL_KNOWN_BIG` must match the real pieces.
  - The default treats green and one orange as the big 83 mm bricks (the Nov 8 table).
  - All small cubes: `TALL_KNOWN_BIG='{"green": false, "orange": false}'`.

## 1. Start the server (on the station)
```bash
cd ~/hillclimb-repo && tmux new -d -s hillclimb "OVERHEAD_FROM=left TALL_KNOWN_BIG='{}' ~/dream-env/bin/python -m hillclimb.left.server --port 8770 2>&1 | tee -a ~/hillclimb.log"
```
Set `TALL_KNOWN_BIG` to match the pieces (step 0).

## 2. Watch it (on a laptop)
```bash
ssh -N -L 8770:127.0.0.1:8770 pah@nuck-1-pah
```
Open http://localhost:8770. It shows the three cameras, the gate lights, the state, the timer, the tower and the score.

## 3. Check the USB strip crop (once per camera position)
```bash
curl -s -X POST localhost:8770/gate/calibrate -d '{}'
```
- Look at `gate_strip_preview.jpg` in the log folder.
- If the box is wrong, post the right one: `{"box": [x0, y0, x1, y1], "apply": true}`, in overhead pixels.

## 4. First real test: stacking only, practice mode, small goal
This bypasses the USB gate. It's allowed only in practice mode, and it's logged.
```bash
curl -s -X POST localhost:8770/left/arm -d '{"backend":"scripted","mode":"practice","drive":"auto","gate_bypass":true,"round_s":300,"max_goal":3,"crunch_goal":3,"freeze_s":20}'
curl -s -X POST localhost:8770/left/go -d '{}'
```
- **Stop at any time:** `curl -s -X POST localhost:8770/left/stop`
- **E-stop (hold the arm at once):** `curl -s -X POST localhost:8770/left/estop`

## 5. Full round, with the USB arm working (match mode)
```bash
curl -s -X POST localhost:8770/left/arm -d '{"backend":"scripted","mode":"match","drive":"auto","round_s":300,"max_goal":6,"crunch_at_s":90,"crunch_goal":5,"freeze_s":20}'
curl -s -X POST localhost:8770/left/go -d '{}'     # also arms the USB gate: it takes its reference photo now
```
- **While the gate is shut:** the left arm only looks and plans. It never touches a block.
- **The gate opens** when an unplugged cable is seen resting on the table twice in a row, from the overhead camera, after a jaw release.

## 6. From the main controller (Full Auto)
- Arm with `"drive": "external"`.
- Point the bindings at `POST /left/stack_next`, `POST /left/rebuild`, `GET /left/verify_tower` and `GET /gate/state`.
- Set **Output object path** to `body`.
- The details are in `left/README.md`.

## Dream instead of scripted
Use `"backend":"dream"` with `"checkpoint"`, defaulting to B. It's only safe up to about level 4 to 6 (see `docs/SPIKES.md`).
