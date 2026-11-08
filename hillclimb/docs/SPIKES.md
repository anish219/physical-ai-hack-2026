# Hill Climb harness: experiment results (Nov 8, 4 AM to 11 AM)

These are the experiments behind `SPEC.md`. All were run against recorded station data or the station sim; nothing here ran on the real arms.

## 1. USB gate: "a cable is unplugged and resting on the table"
A false "go" lets the left arm touch blocks too early, which is a DQ, so this gate is tuned for zero false "go"s first and catch rate second.

- **Test set:**
  - every recorded unplug episode (51) and USB-C insertion episode (37);
  - a reference frame from the episode start against a later "now" frame;
  - 281 moments that are certainly "no" (the cable is still plugged, or held in the jaw, per the recorded jaw width);
  - 48 moments that are "go" (the cable released and lying on the table).
- **Results:**

| Setup | False "go" | Caught |
|---|---|---|
| GPT-6.1 Terra, reasoning off, wrist + overhead frames | 4 of 281 | 47 of 48 |
| **Terra, overhead camera only (zoomed strip crop + full view), 2 runs** | **0 of 562** | 43 to 44 of 48 |
| Terra agentic check, overhead tools only (it may ask for zooms and earlier or later frames) | 0 of 326 single checks | |
| Fast overhead check, then agentic overhead check, both must say yes | **0 of 281** | 36 of 48 |
| Fireworks-hosted fast models (Kimi K3 Fast, Qwen 3.8 Max, Ember-1, others), same question | 9 to 15 of 18 on a smaller set | |

- **Why the wrist camera fails:** it moves with the arm. A finger hides a plugged cable, or the view angle changes, and the model reads that as a cable gone. So **only the fixed overhead camera votes on the gate.**
- **Timing and cost:** the fast check takes about 1.9 s median and $0.0015 per call; the agentic check about 3.6 s median, up to about 10 s, and about $0.004.
- **Other checks:** the team's geometry socket verifier (non-neural, 0 false openings on 50 episodes) and the measured jaw release are the other required checks in the gate.

## 2. Terra as the director (sim scenes with known truth)
- **Next-block instructions in the training-label style:** valid in 23 of 24 scenes. It says "nothing left" correctly when no loose cube remains. It needs a "sticks, balls and cables are not blocks" rule.
- **Tower height from frames:** within one block nearly always, but exact only 40 to 67% of the time. So the harness keeps its own count of confirmed placements, and Terra's height is only a cross-check.
- **Judging a step:** it never called a failed step a success. But a block sitting squarely on another can hide the lower one, so a "failed" verdict gets a second look before any retry.
- **Speed:** about 1.7 to 2 s per decision. A big model directing the same checkpoints took 10 to 15 s per decision with the same outcomes.

## 3. Dream stacking checkpoints in the station sim
| Checkpoint | Training data | Result in the sim |
|---|---|---|
| stacking-left-500 (Nov 7) | 27 teleop stacks | 0 stacks in 18 trials |
| stack-left-scripted500 (C) | 21 scripted stacks | 0 of 18 prompts |
| **blocks-left-fresh500 (B)** | 87 real episodes (47 stacks, 16 unstacks, 24 moves) | **Places reliably up to level 4; up to level 6 with a keep-out** |

B, placing onto a tidy pre-built tower, with the harness rules:

| Level placed | Tower grew by one |
|---|---|
| 1 | 4 of 6 |
| 2 | 3 of 4 |
| 3 | 3 of 3 |
| 4 | 2 of 3 |
| 5 | 0 of 9 |
| 6, with a keep-out around the tower | 2 of 3 |

Harness rules these runs found:
- **Stop the policy on a confirmed place:** the jaw held for 1.5 s, then let go, then a 2 s wait. B never stops on its own: it re-grips the placed block.
- **Start every prompt from the training start pose,** using the commanded joints, not the measured ones (the measured ones sag a second time, about 5 degrees).
- **Move back up first:** rise straight up past the tower top, then across. A direct move knocked a 4-high tower.
- **Keep a no-go zone around the tower while the jaws are empty,** and never open the jaws inside the tower's space.
- **Keep the policy warm:** cold start was 2.5 to 17 minutes, and it went cold after about 4 minutes idle.
- **Count height, not colour:** a step succeeds when the tower grew by one and stands.
- **Speed:** playing rows at 30 Hz instead of 20 Hz gave no clear gain.

Fine-tune #2 adds 175 synthetic tower placements (levels 2 to 9). It was still training at the time of writing, so it isn't tested here.

## 4. Scripted tower runner (`tall_tower.py`) in the sim
- **Overhead camera file:** the file the real runner used missed real block positions by a median 78 px (3% within 30 px); the Nov 8 refit misses by 13 px (93%). This caused most of last night's "not standing" refusals on the real arm.
- **Perception-rule fixes F1 to F4** (the wrist ignores impossible table fits; the overhead may only say "fallen" up to 2 high; weak re-looks are accepted near the expected spot; a block a standing check just confirmed isn't dropped): the runner went from 4 to 5 high, then **6 high clean** with a straight-down top look, aim correction and a no-same-colour rule.
- **Levels 8 and 9:** a 40 degree tilted grip from a 2-block helper stack, with an arm-link clearance check and a safe way out of the tilt.
  - **With true positions:** level 8 in 5 of 5 runs, level 9 in 4 of 4.
  - **In the realistic sim** (25 mm sag, the runner's own perception): level 8 in 4 of 4, level 9 in 2 of 3, **9 standing in 2 of 3 runs** from a tidy 7.
- **Still open:** one continuous run from a clear table to 9.
  - 7 fingers down and the tilted 8 and 9 need one tower spot where both plan.
  - About 11 blocks don't fit in reach at once, so blocks must be fed in during the build.
- **Pace:** about 15 to 17 s per confirmed block, with motion capped by the arm server at 0.6 rad/s per joint. A re-place costs about 10 s.
- **Planning during the USB wait:** saves 46 to 56 s per round; the runner holds its first pick until the gate opens.

## 5. Sim versus real
- **Block perception:** the runner's block finder performs about the same on sim and real frames (95% of loose blocks found in the sim, about 5 mm error). The earlier weak green came from the runner's piece list calling green the big brick. **At the venue, set the piece list to the real pieces.**
- **Wrist camera model:** accurate. Driving the sim with measured joints double-sags the arm.
- **Lift sag:** the sim's is about 20 to 28% more than the real arm's.
- **Placement accuracy:** matches the real arm up to level 3. Above that the sim has no real data.

## 6. Prompt wording
- **Fine-tune #2:** uses the wording in pah-synth `sim/synth.py` (COMMANDS and QUALIFIERS).
- **Checkpoints B and C:** use their own training wording.
- **The harness:** keeps one wording table per checkpoint and never retypes the strings.
