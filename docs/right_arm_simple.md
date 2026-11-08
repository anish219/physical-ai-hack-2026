# Simple right-arm demo

Single robot entry point, from `/home/pah/physical-ai-hacks`:

```sh
PYTHONPATH=src .venv/bin/python -u scripts/run_right_arm_simple.py
```

The UI may supply `--run-id`, `--go-time` (Unix seconds), `--run-dir`, and
`--config`. Go time and run identity should match the left worker's launch.
Only the right endpoint is used. This simple version does not publish the
older `first-removal.json` handoff; the UI/left owner must use their own gate.

Sequence: reset to configured unplug start → continuous unplug checkpoint →
three fresh CV observations confirming all four ports open over at least .4 seconds
→ stop unplug inference and wait for its call to unwind → reset → continuous
insertion checkpoint. There is no fixed per-policy action window. Error, stale
arm measurement, 12-second physical stall (editable `--stall-seconds`), operator
stop or the original 300-second match limit stops the run. The insertion phase
has no automatic success verifier or next cycle in this simplified version.

Use the existing `configs/right_arm_controller.json`. For a fresh installation,
copy the example once and preserve any existing station-specific config. The
simple script ignores the older `verify_usb` placeholder; it uses the V2 CV
module and the separately installed geometry package directly. Reset tolerance
is .06 rad, based on actual station tracking error and operator pose review.

Config `usb_cv_calibration` selects `configs/usb-scene-live-20261108.npz`.
`--cv-calibration` overrides it. This adds empty-port references from an
operator-confirmed live frame while keeping original occupied templates and
decision thresholds. It addresses live empty-port template mismatch, especially
ports0and3. The original deployed verifier installation is unchanged. Calibration
is specific to the current physical setup; do not assume qualification after a
camera/adapter change.

Status: `<run-dir>/right-status.json`; event history: `events.jsonl`. Each status
contains phase and current checkpoint. `--debug-cv` additionally saves `cv.jsonl`
with measurements and capture provenance, plus periodic camera snapshots.
Stop uses SIGINT/SIGTERM and requests right-arm stop without powering it down.

## Station validation on November 8

Old integration log never saw four open ports. With the supplemental calibration:
15/15 stationary all-empty frames detected all four open;10/10 frames with USB-C
port0occupied correctly reported occupied/open/open/open, with zero false clearances.
Initial all-plugged saved frames remained occupied. Seven released fixture cases
were inspected offline; no non-all-open case became all-open. The existing44unit
checks passed. This is limited live validation, not general verifier qualification.

Run `right-transition-1794170275` completed real unplug → CV four-open confirmation
→ reset → start of `dream-1-ft:f9ed26ed-usb-manip-gen-cos1000-20261108`. The insertion
request was refused by Dream with DA-4000, citing an electrical hazard from the
power-strip scene. No insertion action rows were executed. Script exited and the
arm was verified stationary (enabled, no arm error). Resolve that service rejection
before claiming insertion or the entire demo has succeeded.
