# hillclimb

The left arm's stacking controller and the shared USB gate for the Hill Climb relay, built to plug into the main controller's Full Auto orchestration. The main controller's bindings (`stack_next`, `rebuild`, `verify_usb`, `verify_tower`) call these as endpoints. Nothing in the existing controller code is changed.

- `docs/SPEC.md`: what the harness does: rounds, gates, state machines, scoring, failure handling.
- `docs/SPIKES.md`: the experiments behind the spec.

Status: spec and experiment results only. Code follows on this branch.
