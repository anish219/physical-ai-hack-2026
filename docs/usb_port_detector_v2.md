# USB port observations V2

`dream_hackathon.usb_port_detector.UsbPortDetector` reuses the tested
`GeometryOnlyVerifier` CV core and its calibration from the separate installed
package. The original `UsbGeometryVerifier` runtime and deployment remain untouched.
This version returns current observations; the right-arm controller owns baseline
comparison, temporal confirmation, handoff publication and retry decisions.

## Observation contract

Call `observe(rgb_scene=..., rgb_wrist=..., stamps=..., now=...)` with RGB uint8
640×360 arrays and capture metadata (`camera`, `frame_id`, `captured_at`, `run_id`).
`now` and capture times must use the same monotonic clock. The existing HTTP helper
converts actual same-host `X-Stamp` camera capture times to this clock and derives
capture IDs from that timestamp and the original JPEG. It does not substitute
request time for capture time.

Result fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | 2 |
| `run_id` | Current observation run |
| `ports` | Four `{port_id, state}` entries; calibrated physical IDs 0–3 |
| `state` | `open`, `occupied`, or `unknown` |
| `occupancy_known` | All four port states are resolved |
| `occupied_count`, `open_count` | Exact counts, or JSON null if any port is unknown |
| `unknown_count` | Number of unresolved ports |
| `captures` | Actual supplied capture identity/timestamp metadata |
| `detection` | Current CV measurements, camera registration and per-port evidence |
| `reason` | Current observation/metadata outcome |
| `capabilities` | Explicitly identifies the limits of port occupancy observations |

Occluded or ambiguous ports remain unknown. Valid views that disagree about a
port make that port unknown. A view without valid adapter geometry cannot supply
port evidence. An individually resolved port is preserved even when exact total
counts are unknown. A later image can report a previously open port as occupied:
there is no success latch. Repeated/reused capture IDs, out-of-order, future/stale,
wrong-run, or skewed capture metadata cannot produce known occupancy.

An open port alone is not a verified removal: the controller must compare it with
the actual occupied baseline for this run. An occupied port does not establish
full seating. This adapter does not return `stage_passed`, full-insertion success,
gripper-release success, or cable-on-table success. It must not be substituted
unchanged for the older controller's entire `verify_usb` response.

## Read-only test on NUCK

Once these new files have been deployed, run from the controller checkout:

```sh
PYTHONPATH=src .venv/bin/python scripts/inspect_usb_ports.py --iterations 20
```

For an explicitly overhead-only check when the wrist stream is unavailable:

```sh
PYTHONPATH=src .venv/bin/python scripts/inspect_usb_ports.py \
  --wrist-url '' --iterations 20
```

The default CV import directory is:
`/home/pah/usb-geometry-verifier-20261107/usb-geometry-verifier-v1/scripts`.
`--scripts-path` overrides it. Calibration/assets stay in that installation.
The script only GETs existing camera JPEGs, prints JSON and never opens camera
devices, powers arms, sends motion commands or grants stage permission. A configured
camera transport failure reports unknown; it does not silently remove that view.

## Controller integration boundary

The controller still needs a phase-aware observation adapter:

- First removal: compare observed ports against the run's occupied baseline,
  confirm the transition over genuine fresh captures, then publish the gate once.
- Hub clearing: use current known port counts, with unknown waiting for observation.
- Insertion: add independent evidence for full seating and gripper release.
- Cycle unplug: add independent evidence for cable resting on the table and release.

The existing controller's `known` applies to all its requested completion fields.
V2 deliberately exposes `occupancy_known` so missing completion observations cannot
be mistaken for success. No controller hookup, main push, robot deployment, live
unplug-transition qualification or motion was performed by this V2 change.

## Validation

Twelve V2 tests exercise current/reversed occupancy, partial views, view conflicts,
metadata rejection, wrong image dimensions and detector failure; all32existing
controller tests also pass. Seven real image fixture cases exactly match the
released CV package's expected per-port results. That is offline compatibility
validation; the unfinished live unplug-transition test remains necessary.
