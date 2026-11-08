# Right-arm handoff: independent USB strategy

This is a design and implementation handoff only. No new runner, launcher or GUI
has been implemented for this split. Existing controller behavior is unchanged.

## Requested architecture

The right and left arms will have independent control workers. One headless launch
command starts both workers with the same Go time and match identity. Starting
both workers simultaneously does **not** start stacking early: the left worker
waits for the right worker's verified first USB removal. After that handoff, right
continues its own strategy without waiting for left stacking actions or tower
observations. Right does not command left, and left does not command right.

Keep the existing strategy and its explicit 300-second match duration. Both arms
use elapsed time from the original Go, including attempts and retries. A retry,
first-removal handoff or tower collapse does not restart the clock. Preserve the
existing explicit end-of-match behavior for each arm's own work; do not introduce
additional watchdogs, power changes, automatic recovery actions or hidden cutoffs.

## Right-arm sequence

| State | Action / check | Next state |
| --- | --- | --- |
| Await Go | Receive the shared run identity and Go time. | Pull all |
| Pull all | Complete one recorded bundle-pull attempt, using the CV-aligned implementation where bound. | Inspect first removal |
| Inspect first removal | Inspect the four original hub ports. Require a known scene and at least one verified removal. Attempt completion alone is insufficient. | Unknown or zero: Pull all. At least one: publish the first-removal gate once, then Inspect hub. |
| Inspect hub | Inspect original-port occupancy. | Unknown: wait the configured observation interval and inspect again. Fewer than four removed: Clear hub. All four removed: Insert. |
| Clear hub | Execute one window of the trained unplug policy. | Inspect hub |
| Insert | Execute the USB insertion recording, including letting go. | Verify insertion |
| Verify insertion | Require known scene, fully inserted USB, and released gripper. | Any condition unmet: Insert. All met: Return to start. |
| Return to start | Complete the recorded move to the unplug policy's starting pose. | Cycle unplug |
| Cycle unplug | Call the trained unplug policy to unplug, place the USB on the table, and let go. | Verify unplug |
| Verify unplug | Require known scene, no occupied original hub ports, cable on the table, and released gripper. | Any condition unmet: Cycle unplug. All met: Insert. |

Repeat insertion and unplugging for the remainder of the match. There are no ball
operations and no optional reinsertion during the initial removal/clearing stage.
Once the original hub positions are clear, pursue the repeated USB scoring cycle.
The repeated-cycle strategy follows the team's reported judge approval.

These retry paths handle an unsuccessful physical attempt reported by verification.
Additional handling of execution exceptions, provider errors or crashed scripts
has not been specified; do not silently invent recovery movements or change phases.

## Coordination contract with the left-arm owner

The launcher provides one unique run identity and one shared Go time to both
workers. The first-removal handoff contains that run identity, a positive gate,
the observed removal count, the observation time, and references to the supporting
verification evidence. Choose the transport with the left-arm owner; the existing
implementation does not yet define an IPC service or new launcher protocol.

Publish the gate only after a known removal count of at least one. The left worker
must not act on an unknown result, a completed pull recording, a stale gate from a
previous run, or a model's prediction that a removal should have occurred. Publish
once for the current run; later scene changes do not revoke that initial pass.
Right continues immediately into hub clearing and does not wait for an
acknowledgment or a tower pass before proceeding with its existing strategy.

Left retains its current tower strategy: stack toward the editable cap of seven,
rebuild after collapse, and from 240 seconds after Go hold when the current tower
contains **more than five** blocks. A standing count of five can become six; six
and seven hold in that final minute. Collapse still returns to rebuilding. Left
owns all tower observation, stacking, rebuilding and late-time decisions.

## Verification contract

Keep operation completion separate from physical success. Deterministic evidence
can establish that a replay/API action completed and provide measured detector
results; combine it with the VLM scene assessment in the verifier. An API success
response or elapsed action window alone does not prove USB manipulation succeeded.
Real-camera verification accuracy still needs validation.

| Field | Meaning |
| --- | --- |
| `known` | All fields needed for this decision are observable and resolved. Unknown never passes a gate. |
| `unplugged_count` | Four minus occupied original hub ports; do not count unrelated loose cables. |
| `plugged_count` | Number of occupied original hub ports. |
| `fully_inserted` | The cycle USB is fully seated, rather than held near the port. |
| `gripper_released` | The arm has let go of the USB. |
| `cable_on_table` | The unplugged cycle cable is resting on the table. |

Use a fresh inspection after each attempt. During the repeated unplug cycle,
preserve the current map's check of `plugged_count == 0` together with table and
release evidence. Counts and release observations must describe the same scene,
not a mixture of results retained from previous attempts.

## Bindings to supply or update

Keep these names stable so recordings, policies, HTTP adapters and verification
providers can be exchanged without rewriting the state transitions.

| Binding | Available / still needed |
| --- | --- |
| `pull_all` | A prepared CV-aligned replay exists at `scripts/replay_usb_bundle.py`, with assets in `recordings/usb-bundle-pull/`. Its plan is marked prepared, not executed. The competition map still has a dummy binding. |
| `verify_usb` | Supply the automatic deterministic/VLM verifier returning the scene fields above. The map has a dummy observer returning unknown and a planned VLM configuration. |
| `unplug_policy` | Supply the trained right-arm checkpoint and task/API configuration. The planned map uses a three-second action window per attempt; it remains editable. |
| `plug_usb` | Supply the insertion recording, including release. Currently a dummy binding. |
| `return_usb_start` | Supply the return-to-policy-start recording. Currently a dummy binding. |

The CV-aligned pull must use the full bundle-pull implementation when selected.
Generic playback of its JSONL trajectory does not perform CV alignment. The full
implementation approaches the taught checkpoint, aligns the strip/wire features,
corrects the remaining pull trajectory using the calibrated arm model, and retains
the recorded release location. See [USB bundle pull](usb_bundle_pull.md) for the
existing entry point, calibration inputs, parameters and execution semantics.

## Existing interfaces and source of truth

- Canonical strategy: `src/dream_hackathon/demo_strategy.py`, especially node IDs
  `pull_all`, `first_usb_view`, `first_usb_gate`, `clear_view`, `clear_policy`,
  `plug_attempt`, `plug_view`, `return_start`, `cycle_unplug`, and `unplug_view`.
- Exported editable map: `configs/demo_strategy.json`.
- Existing execution adapters: `src/dream_hackathon/orchestration.py`; these cover
  recordings, scripts, HTTP, VLM and Dream operations. They are not a new headless
  dual-worker launcher.
- Right-arm listener: `http://127.0.0.1:8692`; existing interfaces include `/state`,
  `/target`, `/stop`, `/overhead.jpg` and `/wrist.jpg`. A headless right worker should
  use its own listener and cameras, without requiring the controller GUI.
- Existing single-arm model transport: `StationCaller` in
  `src/dream_hackathon/dream_caller.py`; select right and supply only the right
  endpoint. Reuse the native joint-checkpoint adapter for compatible trained models.
- Station checkout: `/home/pah/physical-ai-hacks`. Existing station config and
  credentials remain there. Team assets and latest rubric are backed up at
  `/home/pah/laptop-handoff-20261108/`.
- `src/dream_hackathon/demo_master.py` describes the older ball strategy. Do not
  use it as the strategy source for this handoff.

## Acceptance checks for the implementing teammate

1. One headless launch starts both workers on the same match clock; neither
   worker depends on a browser or GUI selection.
2. Zero removals and unknown first inspections repeat the bundle pull and never
   start left stacking. A verified removal publishes exactly one current-run gate.
3. Right continues clearing independently while left stacks; each worker's
   commands, API clients and explicit stop/end actions target only its own arm.
4. Unknown clearing observations wait and re-inspect; fewer than four removals
   calls the unplug policy; four removals starts insertion attempts.
5. Insertion repeats until seated and released. Returning to start completes
   before the unplug policy is invoked.
6. Unplugging repeats until unplugged, on the table and released, then returns to
   insertion. No ball task or new recovery movement is introduced.
7. The shared elapsed clock survives retries and the first-removal handoff. Left's
   final-minute decisions use time from Go, not time since stacking began.
8. Substitute fake operations and scene results to test the transitions before
   checking actual recordings, model bindings and camera verification separately.

Existing relevant tests are `tests/test_demo_strategy.py`,
`tests/test_orchestration.py`, `tests/test_bundle_pull.py`,
`tests/test_model_selection.py` and `tests/test_dream_joint_finetune.py`.
The independent-worker launch and IPC behavior above still needs implementation
and its own verification; previous controller tests do not establish that it exists.
