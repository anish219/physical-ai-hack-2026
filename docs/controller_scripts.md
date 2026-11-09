# Controller START / STOP scripts

The station controller supports per-arm Script and HTTP inputs. Saving or loading
an entry does not start it. The header START launches the loaded left and right
entries; pressing it again stops them.

For the current two-arm setup, select Script and save these entries:

| Arm | Script entry |
| --- | --- |
| Left | `hillclimb/left/match.py:start` |
| Right | `scripts/start_right_arm_simple.py:start` |

The left adapter calls the hillclimb master at `http://127.0.0.1:8770`:
`POST /left/arm`, then `POST /left/go`. STOP calls `POST /left/stop`.
If the service is not running, the adapter starts it. The hillclimb service
sources were brought over from branch `hillclimb` at `fdab660`.

The right adapter launches `scripts/run_right_arm_simple.py` in the station
Python environment with a run ID, Go time, and `/tmp/relay/<run-id>` output
directory. It uses `configs/right_arm_controller.json` and its configured
policy checkpoints and start poses. Its sequence resets, unplugs until the CV
detector sees four open ports, resets, and runs insertion. STOP sends SIGTERM
to that worker. The station-specific configuration remains on the NUC.

The left Script entry and its HTTP input at port 8770 use the same master;
they are alternative ways to call it.

Validation:

```sh
PYTHONPATH=src:. python -m pytest tests/test_both_controller_scripts.py \
  tests/test_sequences.py tests/test_hillclimb_match.py \
  tests/test_unified_control.py tests/test_right_arm_controller.py -q
```

The combined integration test uses fake subprocess and HTTP operations; it
verifies START/STOP dispatch without moving hardware.
