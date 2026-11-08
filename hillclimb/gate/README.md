# gate: the USB gate

Decides when the left arm may touch blocks: a USB cable has been pulled out of the strip **and** is lying on the table, released. A false "go" disqualifies the team, so every doubt keeps the gate shut. SPEC section 7 has the rule; SPIKES section 1 has the evidence.

## How it decides
After `POST /gate/arm` (at go), a background loop runs. All of these must hold at the same moment; the gate then latches open for the round.
1. **Vision, overhead camera only.**
   - Every 2 s, a fast GPT-6.1 Terra read (reasoning off, JSON) of reference versus now, on the strip crop plus the full view.
   - On a yes, an agentic check: Terra may ask for up to 4 more frames or zooms from 3 s before to 2 s after now, within a 15 s budget, then gives a verdict.
   - After one agentic yes, a second agentic check on a frame at least 1.5 s newer. Two in a row are needed.
   - Any "no", error, timeout or missing verdict resets the streak.
2. **Jaw:** the right jaw held a plug (normalized grip 0.03 to 0.085 for 0.5 s) and then opened (over 0.10 for 0.3 s) since the gate was armed. The release is remembered, because the right arm may already be on the next cable. Closing on nothing after a hold (a slipped plug) cancels it.
3. **Socket verifier (optional):** the team's geometry verifier (`~/usb-geometry-verifier-20261107`) is loaded read-only when present.
   - Its latched proof is "yes"; anything else is "unknown".
   - With `SOCKET_REQUIRED=false` (the default), "unknown" does not block. It never says a plain "no".
4. **Healthy:** overhead frames and right-arm state are fresh. Stale input counts as "no".

The wrist camera never votes: it moves with the arm, and in testing that faked "missing" cables.

## Routes
| Route | What it does |
|---|---|
| `POST /gate/arm` | Takes the reference frame and starts the round. `{"strip_box": [x0,y0,x1,y1]}` (optional) |
| `GET /gate/state` | `open`, `armed`, `latched_at`, the main controller's `verify_usb` fields (`known`, `unplugged_count`, `plugged_count`, `gripper_released`, `cable_on_table`, `fully_inserted`), every check, the last 10 reads |
| `POST /gate/calibrate` | `{"box": [...]}` sets the strip crop. `{}` asks the automatic finder for a suggestion and writes `gate_strip_preview.jpg` (green: in use, orange: suggestion); add `"apply": true` to use it |
| `GET /gate/health` | Frame and state freshness, the verifier, whether a key is present (never the key) |
| `POST /gate/stop` | Stops the loop |

Mounted by the hillclimb server through `hillclimb.gate.http_handler(method, path, body)`, with `hillclimb.gate.start()` at server start. Standalone for tests: `python -m hillclimb.gate.serve --port 8771 --left-url ... --right-url ...`.

## State on disk
`LOG_DIR/gate_state.json` and `gate_ref.jpg` hold the round: armed, open, the latch time, the crop, the reference frame and the jaw releases.
- A restart restores them exactly, so it can neither close an open gate nor open a shut one.
- Only `POST /gate/arm` starts a new round.
- The vision streak is not kept, so after a restart a shut gate needs two fresh agentic yeses again.

## Settings
Environment, then `hillclimb/config.json` (top level or a `"gate"` section), then defaults in `settings.py`.
- **Shared names:** `LEFT_URL`, `RIGHT_URL`, `OVERHEAD_FROM`, `KEYS_FILE`, `LOG_DIR`, `VLM_MODEL`.
- **Gate-only names:** `STRIP_BOX`, `FAST_*`, `AGENTIC_*`, `JAW_*`, `SOCKET_*`, `GATE_*`.
- The key is `OPENAI_API_KEY`, from the environment or `KEYS_FILE`. It is never printed or logged.

## Tests
| Test | Command | Result (Nov 8) |
|---|---|---|
| Latch and jaw logic, fake reads | `python -m hillclimb.gate.tests.test_core` | 20 of 20 pass (Mac and station) |
| Recorded USB episodes through the real checks (281 certain "no", 48 "go") | `python -m hillclimb.gate.tests.replay_stress VIDS labels4_certain.json grip10hz.json out.jsonl` | **0 false "go"**; fast check yes on 0 of 281 "no", 44 of 48 "go"; first agentic yes on 0 "no", 30 of 44 "go"; fast p50 1.8 s, p90 2.1 s; agentic p50 4.7 s, max 8.3 s |
| Live loop against a station sim, including the sim dying and two restarts | `python -m hillclimb.gate.tests.live_sim --sim-url ... --kill-cmd ...` | 17 of 17 pass, on the Mac and on the station |

The replay's "caught" count (4 of 48) understates a live gate. The recordings end 1 to 2 s after the release, so the confirming frame still shows the gripper over the cable, and the check rightly says "not clearly released". Live, the loop keeps checking as the arm moves away.

## Before a match
1. Check the strip crop: `POST /gate/calibrate {}` and look at `gate_strip_preview.jpg`. If the green box doesn't hold the strip and its cables, set it with `{"box": [...]}`. The automatic finder was right on only 2 of 10 recorded frames, so it is never applied without `"apply": true`.
2. Check `GET /gate/health`: frames and state fresh, key present.
3. `POST /gate/arm` at go, with the table as it starts.
