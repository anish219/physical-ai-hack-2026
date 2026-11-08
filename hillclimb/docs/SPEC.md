# Hill Climb harness: spec

Draft 2, Nov 8 2026, about 8:50 AM (spike results folded in). For review by Isaiah. Submissions are due at 3 PM today.

**Team integration (Nov 8, 10:50):** the team runs one main controller with a Full Auto button that starts both arms' controllers at once. This harness is the left arm's controller and the shared USB gate, exposed as endpoints that the main controller's orchestration bindings call (`stack_next`, `rebuild`, `verify_usb`, `verify_tower`). It does not replace the main controller.

This spec says **what** the harness must do. The design doc will cover **how**: architecture, stack, languages and integrations. It follows once this spec is agreed. Nothing here is built yet.

## 1. Purpose
A scripted competition harness for the PAH Hill Climb on the MakerMods Metal Arm station (NUCK-1).
- It replaces running each round by hand.
- It runs both arms through the relay, keeps score, and shows everything live.
- It must never refuse, crash or stall. When anything is unsure, it holds still and keeps the gates shut: that costs time, never a disqualification.

## 2. The rules it plays by (official rubric v4)
- **Match:** 1v1, 5 minutes, both teams at once on matched stations. After "go" only the policy moves the robot: no teleop, no keyboard, no voice, no touching.
- **Stop:** a designated team member may call "stop" at any time to end the run.
- **Stage 1, USB:** unplug cables that start plugged into the hub. Plug in only the loose cables lying on the table, into free ports. An unplugged cable may not go back in.
  - Pass bar: at least 1 cable unplugged.
- **Stage 2, stack:** about 10 stackable shapes.
  - Pass bar: a structure at least 4 in tall, free-standing 3 s after the gripper lets go.
- **Stage 3, balls:** shared golf balls; a black 8-ball in your bin loses the match. No pass bar.
- **Gates:** the gripper may touch a later stage's objects only after the current stage's pass bar is met. Touching early is a DQ.

| Event | Points |
|---|---|
| First cable unplugged | +10 |
| Each further cable unplugged | +5 |
| Each loose cable plugged into a free port (up to 4) | +20 |
| Stage 2 tower > 4 in | +20 |
| Each full inch of tower at the end, if Stage 2 passed | +3 |
| Taller tower (bracket matches only, ties pay nothing) | +10 |
| Each ball in our bin at the end | +5 |
| LiveKit shown to the referee before the match | +10 |
| Dimensional shown to the referee before the match | +5 |

**Team rule, stricter than the rubric:** the left arm may not touch any block until the right arm has unplugged a cable **and that cable is resting on the table**. This gate can never open wrongly.

## 3. Scope
**v1 (must have for the match):**
- Both arms running at once.
- The USB gate (section 7) and the left stacking state machine (section 8).
- The score estimate.
- The dashboard, with plain HTTP streaming.
- The station sim and recorded-data tests.
- Fault handling (section 13).

**v2 (if time allows):** LiveKit streaming (+10 points); loose-cable plug-ins as a right-arm skill.

**v3:** a thin DimOS wrapper (+5 points); the ball stage.

**Out of scope:**
- Training or fine-tuning.
- Editing the team's code.
- Running anything on the real arms without the arm owner.

## 4. People and modes
- **Operator (Isaiah):** sets up and arms a round. Every setting is chosen before go.
- **Stop caller:** the designated team member who calls "stop". The harness tells them when it is a good moment (section 10).
- **Practice mode:**
  - Any setting may change at any time.
  - Manual buttons exist, for example "USB confirmed" and "skip to stacking".
  - Every manual action is marked as practice in the log.
- **Match mode:**
  - At go, every setting locks: policies, prompts, goals, timers and calibration.
  - After go, the dashboard is read-only except **Stop round** and **E-stop (hold both arms)**.
  - There are no override buttons in match mode at all.

## 5. Round lifecycle
1. **Idle:** arms at home and holding. No Dream calls (or warm-keep only, section 11).
2. **Setup:** pick the policy, checkpoint and prompts per state, the goals and the timers. Draw or confirm the strip crop.
3. **Clear state / Start (arm the round):** runs the pre-go checklist. The round can only be armed when every check passes or is explicitly waived in practice mode.
   - Both arm servers answer, and the team controller on 8081 has no input on either arm.
   - All three cameras deliver fresh frames, under 0.5 s old.
   - Both Dream policies are open and warm: no NO_CAPACITY.
   - The vision service answers a test read in under 3 s.
   - The USB socket verifier has its baseline: which sockets start occupied.
   - The reference photos are taken from the overhead camera, and the loose cables and blocks are counted.
   - The settings snapshot is written to disk and locked.
4. **Go:** the round clock starts (default 5:00), both state machines start, and the state is locked.
5. **Ended:** by stop, time up or E-stop. Both arms hold where they are and Dream calls stop. The final score estimate and the full log are saved.
6. **Clear state:** wipes the round and returns to Idle. Only from Ended, or from Idle in practice mode.

## 6. Right arm (USB): state model and state machine

### 6.1 State model (always shown)
- **Per socket (IDs 0 to 3, as the verifier numbers them):** occupied-at-start yes or no, and now one of occupied / empty / unknown.
- **Per cable:**
  - its origin: started plugged, or started loose;
  - where it is now: plugged / held / resting on the table / unknown.
- **Counters:**
  - unplugs from the hub, which only ever count up;
  - loose cables plugged in, which also only count up;
  - illegal re-plug attempts, which should stay 0.
- **Jaw:** measured width, and holding yes or no from the width. A USB plug reads about 4 to 6 mm, closed on nothing about 1.5 mm, open over 10 mm.

### 6.2 State machine
`HOME → UNPLUG → RELEASE_ON_TABLE → VERIFY → (next UNPLUG | PLUG_IN [v2] | DONE)`
- **UNPLUG:** Dream (usb-unplug checkpoint) with the prompt for the next occupied socket.
- **RELEASE_ON_TABLE:** the prompt or skill sets the cable down clear of the strip and the block area, then opens the jaw.
- **VERIFY:**
  - Runs while the arm keeps working. The arm does not park, and may already be moving to the next socket.
  - The gate checks (section 7) wait for frames in which the arm no longer hides the strip.
  - Confirmed: the counters change.
  - Not confirmed within 20 s: retry UNPLUG on the same socket.
- **DONE:** no occupied socket is left (or no plug-in skill), so the arm parks at home and Dream calls stop.
- **Workspace rule:** the right arm never enters the block area or the ball area, in any state.

## 7. The USB gate (Stage 1 pass for the left arm)
The gate opens only when **all four** checks hold at the same time. It then **latches** for the rest of the round. The right arm does not need to park.
1. **Socket emptied:** the team's geometry verifier has latched that an initially occupied socket is now empty.
2. **Released:** the measured jaw width shows a hold, then an opening, and the jaw is open now.
3. **Seen unplugged and on the table, twice in a row:** the fast check and then the agentic check, both from the **fixed overhead camera**.
   - **Fast check:** GPT-6.1 Terra, reasoning off, about 1.9 s. Asked: "since the reference, has exactly one more cable come out of the strip and is it lying on the table, released?". Input: the zoomed strip crop plus the full view.
   - **Agentic check:** runs only after a fast "yes". Terra may ask for a zoomed crop, an earlier frame, or the frame 1 to 2 s later, and must end with a verdict within 15 s.
   - **The yes must repeat** on a second agentic check with fresh frames, about 2 s later.
4. **Healthy:** no input to checks 1 to 3 is stale, timed out or failed. A timeout or a missing verdict counts as "no".

Evidence (see SPIKES.md; from 51 unplug and 37 insertion recordings):
- **Fast check, overhead:** 0 false "go"s in 562 certain-"no" reads.
- **Agentic check, overhead tools only:** 0 false "go"s in 326 single checks.
- **Combined:** 0 of 281 false "go"s; caught 36 of 48.
- **Wrist camera:** the fast check gave 4 false "go"s in 281, because the moving camera fakes a missing cable. Wrist tools added nothing to the agentic check's safety and lowered its catch rate.

So **only the fixed overhead camera votes on this gate.** The dashboard still shows the wrist. A missed "go" costs a few seconds; a false one would be a DQ.

## 8. Left arm (stacking): state machine

### 8.1 Settings (locked at go)
- **max_goal:** blocks, for example 9.
- **crunch_at:** time left at which the crunch goal takes over, for example 1:30.
- **crunch_goal:** blocks, for example 5. Never above max_goal.
- **freeze_window:** seconds before the end in which no new placement starts. Default 20 s.
- **Piece catalog:** the height of each piece type, used for inches.

### 8.2 Derived values (always shown)
- **time_left.**
- **tower_height:** in blocks and in full inches.
- **active_goal:** max_goal, or crunch_goal when time_left ≤ crunch_at.
- **stage2_passed:** latched once the tower is at least 4 in and stands 3 s after a release.

### 8.3 States
`LOCKED → STACK → SETTLE_CHECK → (STACK | REST) ; REST → RECOVER → STACK ; any → FREEZE → END`
- **LOCKED:** before the USB gate opens.
  - The arm holds at home and **every motion command to the left arm is refused**.
  - Dream for the left arm is off or warm-keep only.
- **STACK:** one placement cycle: Dream (stacking checkpoint) or a scripted skill, with the prompt for the next block.
- **SETTLE_CHECK:** after a release, the arm backs off to a look pose and waits 3 s.
  - The vision model checks "the stack grew by one and stands".
  - A confirmed placement adds one block.
  - An unconfirmed one does not count. The tower height never goes up on a guess.
- **REST:** tower_height ≥ active_goal. The arm goes home and stops calling Dream, because more blocks only risk the tower.
- **RECOVER:** tower_height < active_goal at any time, including during REST. That can mean a fall, a knock, or the crunch goal dropping below the current height.
  - The arm restacks, starting from the largest standing part.
- **FREEZE:** inside the freeze window. No new placement starts.
  - A placement already in progress finishes and releases.
  - Then the arm goes home.
- **Never:** the left arm never touches the strip, cables or balls.

### 8.4 Tower height bookkeeping
- Height counts **confirmed placements**. Absolute vision reads are only a cross-check within one block; the spike showed they are poor above 4 blocks.
- A suspected fall triggers a look-again read, never an instant restack. A suspected fall is a confirmed shrink, or an overhead change in the tower area.

## 9. Policies and prompts
- **Policy types:**
  - **Dream checkpoint:** id plus a prompt per state.
  - **Scripted skill:** for example the tower builder.
  - **Hold:** do nothing.
- **Per arm, per state:** a policy and a prompt, picked from lists on the dashboard and saved as a named preset.
- **Prompt changes:** a state change that changes the prompt starts a new Dream episode (reset_episode).
- **Allowed scripted skills:** a referee should confirm whether a scripted skill counts as "the policy". The harness supports both, so this is a per-run choice.

### 9.1 Prompt wording: one table per checkpoint, never retyped
- **Fine-tune #2** (`...-realsynth500-20261108`): COMMANDS and QUALIFIERS imported from pah-synth `sim/synth.py` at commit 5fb8a79. For example "stack the {x} on top of the {y}", "put the {x} on top of the tower", qualifiers "nearest the robot's base" and "on the robot's left".
- **Checkpoints B and C:** their training wording, from the station's `~/ft1/wording_B_C.txt`. For example "stack the green block on top of the blue block", "lying on its side" with no comma, no size words.
- **Checks before any prompt is sent:**
  - A block name that matches more than one candidate is rejected; the director picks again.
  - Orientation ("lying on its side") comes from perception, never from the director's guess.

### 9.2 The director (Terra), in both stacking paths
- **It picks the next block and writes the prompt,** in about 2 s. It may pick only from the candidate list that perception or the executor gives it, so it can never name a block that isn't there or can't be reached. Sticks, balls and cables are never blocks.
- **It judges each step from a look pose with the arm backed off.** In the spikes it never called a failure a success. But a block sitting squarely on another can hide the one below, so a "failed" verdict gets a zoomed look before any retry.
- **It may overrule the scripted executor's after-place failure calls**, but never its refusals before a move. It never starts a recovery on its own.

### 9.3 Dream path rules (from the sim spikes)
- **Stop the policy as soon as a placed-and-released check passes** (1 to 2 Hz). The policy never stops by itself: it re-grips the placed block or starts another.
- **Start every prompt from the training start pose** (the real episodes' median: hand about 21 cm over the table, jaws 0.8). Use the **commanded** joints, never the measured ones. Measured joints already include the sag, so sending them puts the hand about 5 degrees low, and the sim sags a second time. The same mistake caused the 33 mm settle onto a tower at 02:50.
- **Keep the tower spot clear of the start pose.** At the start pose the hand sits about 20 cm over the first tower spot, so a tower 5 or more high there gets hit.
- **Count height, not the named block.** A step succeeds when the tower grew by one and stands, and nothing was taken off it, whichever block landed. The named block only decides the next prompt.
- **Every move back to the start pose goes up first.** Open the jaws, rise straight up to the tower top plus about 6 cm, then cross away from the tower. Never cut through the tower's space: a direct retreat knocked a 4-high tower down in the sim.
- **Keep the policy warm all round.** Cold start measured 2.5 to 17 minutes, and B went cold after about 4 minutes idle.
- **The scripted path is the fallback, and it's on by default.**
  - Checkpoint B landed the named block on the named target in 5 of 12 prompts, and stacked in 3 of 5 for a second director.
  - It cannot yet place onto a stack of 2 or more.
  - C, and the team's original stacking checkpoint, stacked 0 times in the sim.

### 9.4 Scripted path (the real-arm runner, `tall_tower.py`)
- **The real arm must use the Nov 8 overhead refit** (`cam_mid_nov8.json`, median miss 13 px against 78 px for the current file), or a venue refit under about 20 px.
- **The perception-rule fixes F1 to F4 are applied** (see SPIKES.md).
- **Support and standing checks look straight down with the wrist over the tower top.**
- **Levels 8 to 10 use the 40 degree tilted grip from a helper stack.**
- **Every collision, reach, sweep and knock refusal stays.**
- **The runner starts at "go" but holds its first pick until the USB gate opens.** It does its looks, tower-spot search and planning during the wait, which saves 46 to 56 s per round. The gate is a file or flag the harness sets, and the runner must never touch a block before it.
- **Realistic pace:** about 15 to 17 s per confirmed block, with motion at the arm server's 0.6 rad/s joint cap. A re-place costs about 10 s, so first-try accuracy matters more than raw speed.

## 10. Score estimate
Shown live, with each part and how sure it is:
- USB: 10 + 5 × (unplugs − 1) if unplugs ≥ 1, plus 20 × plug-ins.
- Tower: +20 if stage2_passed, plus 3 × the full inches of the confirmed tower height.
- Bonus: LiveKit and Dimensional, if enabled and shown to the referee (a toggle set before go).
- Bracket +10 and balls: shown as "not counted", since we cannot see the other team.

**"Call stop now" banner.** It appears when all of these are true:
- stage2_passed;
- tower_height ≥ active_goal and REST has held for 5 s;
- the right arm is DONE.

It also says how many points stopping locks in.

## 11. Dream usage
- **Streams:** two independent Dream sessions, left and right, running at once.
- **No calls while idle:** an arm in HOME, REST, LOCKED or DONE makes no Dream calls.
- **Warm-keep (optional, per arm):** run the policy with a do-nothing prompt, discard its output, and never send its rows to the arm. This avoids the 3 to 5 minute cold start.
- **NO_CAPACITY or pool_warming:** the arm holds, the harness retries after `retry_after_s`, and the dashboard shows "warming". The round continues.

## 12. Dashboard
- **Cameras:** overhead, left wrist and right wrist, live at low FPS (2 to 5).
- **The two state machines:** the current state highlighted, the time in that state, and the last 10 transitions.
- **State models:** sockets and cables (right), tower height and goals (left), the round clock, and the active goal with the reason it is active.
- **Gate panel:** each of the five checks as a light, with its latest evidence. That means the verifier result, jaw width, park pose, and the two vision reads with their photos.
- **Score panel** and the "call stop now" banner.
- **Policy, checkpoint and prompt pickers:** editable in Idle or Setup, locked after go.
- **Health:**
  - arm servers, cameras (frame age), Dream (latency, warming), vision service (latency, last error);
  - disk space on the station.
- **Event log:** every read, decision and command, with times.
- **Look:** cleaner and better looking than the team's existing page, kept to one screen.

## 13. Never crash, never stall, never refuse
| Failure | Behaviour |
|---|---|
| Arm server down or slow | That arm holds (no new commands); retry with backoff; the round continues on the other arm |
| Camera frame stale (> 0.5 s) or missing | No Dream rows are played for that arm; gates and checks read "no" |
| Dream error, timeout or NO_CAPACITY | Arm holds; retry; shown as warming or error |
| Dream session expired | Drop the pending rows, open a new episode, ramp in |
| Vision timeout (> 3 s), error, unparseable answer | Counts as "no"; retry on the next tick |
| Socket verifier unavailable | Gate check 1 fails, so the gate stays shut |
| Team controller takes an arm (8081 input set) | That arm holds and shows "arm busy"; the harness never fights it |
| Harness process crash | A supervisor restarts it; the round resumes from the saved state and the wall clock; arms hold meanwhile |
| Disk nearly full | Logging of frames drops first; decisions keep running |
| A setting is invalid at Start | Arming is refused with a clear reason; never a half-armed round |

- **Every loop has a timeout,** and no single failure can stop the round clock or the dashboard.
- **All model calls are off the motion thread.** The arm never waits on a vision read to keep holding.

## 14. Integration boundaries
- **The team's code** (`~/physical-ai-hacks` on the station) is read-only. The harness talks to the arm servers on 8691 (left) and 8692 (right) through their existing `/state`, `/target`, `/overhead.jpg` and `/wrist.jpg` endpoints. It may import the team's modules, but never edit them.
- **Ports never touched:** the arm servers are only used through those endpoints. 8084, 8081 (read-only status checks only), 8697, 18081 and 18084 are left alone.
- **The socket verifier** (`~/usb-geometry-verifier-20261107`) is used as it is.
- **API keys** are read from a local env file on the machine that runs the vision checks. They are never committed.
- **The real arms** are run only with the arm owner present.

## 15. Testing and acceptance
| Test | Pass bar |
|---|---|
| **Gate stress set** (281 certain "no", 48 "go", from the 88 USB episodes) | 0 false "go" from the gate as built (all five checks); at least 80% caught |
| **Left arm on the station sim** (`calib/station_sim.py --photoreal`, speaks the arm server API including `/target`) | A full mock round: LOCKED until a scripted gate opens, stacks to the goal, rests, recovers from an injected knock, honours the crunch goal and the freeze window |
| **Right arm on recorded data** (replayed frames and jaw traces as a fake arm server) | Counters and socket states match the labels; no illegal transitions |
| **Fault injection** (kill an arm server, freeze a camera, fake NO_CAPACITY, make the vision service time out, kill the harness mid-round) | The round never crashes or stalls; the gates stay shut; the clock and dashboard keep running |
| **Lock check** | After go, no setting can change and no manual command reaches an arm, apart from Stop and E-stop |
| **Spike done bar (Isaiah):** 9 high in the sim, both scripted and with a trained policy under the harness | Both reach 9 standing, by truth, repeatably |
| **Dry run on the real station** with the arm owner | The pre-go checklist passes; both arms run; the dashboard is live |

## 16. Open questions
0. **Repeated plug-in and unplug cycles (being confirmed with the judges).**
   - Joe's orchestration doc says the team got judge approval for repeated plug-in, release and unplug cycles.
   - Rubric v4 says an unplugged cable may not go back in.
   - The right arm's plan (section 6) follows the PDF until the judges confirm. If they allow it, the right arm loops plug-in and unplug for points.
1. **Which page is "Joe's dashboard"?** The unified control page on 8081, or another one? It sets the style bar.
2. **Scripted skills:** do they count as "the policy" for the referee? This affects whether the tower builder is allowed in the match.
3. **Where the harness runs:** decided: on the station, next to the arms.
4. **The competition hub:** is it the same strip and camera placement as our recordings? If not, the overhead crop and the verifier baseline must be redone at the venue.
5. **Defaults to confirm:**
   - freeze window 20 s;
   - crunch goal 5 at 1:30 left;
   - max goal 9;
   - 3 verify tries before re-trying an unplug;
   - balls out of v1;
   - plug-ins in v2.
6. **The piece catalog:** which shapes and heights are at the station? The rubric says "various stackable shapes, about 10 per station".
