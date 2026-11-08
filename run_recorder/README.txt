INSTALL FROM A CLEAN CHECKOUT (Linux NUC only)

From the repository root, using the existing station Python 3.12/3.13:
  ~/dream-env/bin/python run_recorder/setup.py --host 10.103.0.132
  bash run_recorder/start.sh
  cd run_recorder && .venv/bin/python -m unittest -v test_recorder
Requires tmux and ionice (already available on the station). Setup creates its
own environment, downloads pinned official SDKs, generates local credentials,
and writes the supplied LAN address to station.json. It never starts robots.
Choose the current NUC address if it differs from the example.
The deployed installation below remains independent of this checkout; copying
source or merging main does not restart any running service.

LEONARD — LIVEKIT + DIMENSIONAL RUN RECORDER

Deployed independently at /home/pah/run-recorder on the NUC.
No live controller code, dependencies, camera ownership, or robot controls changed.

OPEN
Judge / spectator: http://10.103.0.132:8791/ (same LAN, read-only, no SSH)
Operator: http://localhost:8791/ with the tunnel below running:
  ssh -N -L 8791:127.0.0.1:8791 nuc
The operator browser also connects to 10.103.0.132:7880 for LiveKit.
If an existing 8791 tunnel is already open, reuse it.

ONE WINDOW FOR BOTH SPONSORS
1. For saved history, start a manual rehearsal recording or use the normal controller's big START;
   a new match is detected automatically within approximately two seconds.
2. Click Broadcast scene in the recorder page. Recording is optional for live video.
   Keep that operator page open.
3. Show the LiveKit connected status, received track ID, and live scene video.
4. Below it, show the Dimensional saved images/events counters.
5. Click an arm-stage event or scrub the timeline; saved images and robot state
   are retrieved through dimOS. Play reviews saved samples only.
6. Stop & save ends a manual recording. Automatic match recordings finish when
   the controller reports match_stopped_at. Stopping recording never stops robots.
Individual Start this arm runs can be captured with manual recording.

Suggested explanation to judges:
"LiveKit carries our live spectator scene. Dimensional stores timestamped robot
observations and arm execution events, so we can review what happened in a run."

PERFORMANCE AND FAILURE BEHAVIOR
- Separate process with Linux SCHED_IDLE, niceness 15, and idle-class disk I/O.
- Existing encoded JPEGs fetched directly from arm workers, at most 1 Hz/camera.
  No camera open, resize, decode, encode or control command on the NUC recorder.
- Controller state 0.5 Hz; stage feed 0.5 Hz while recording.
- LiveKit reuses the in-memory scene sample. While not recording, live preview
  samples only the scene camera at 1 Hz and stops within 3 seconds of no requests. Operator browser encodes 640x360,
  <=1 fps VP8 capped at 180 kbit/s; no additional robot camera fetches.
- Queue capped at 24 entries and each response at 1 MiB. Overflow drops samples.
- Feed requests time out after 250 ms. Stale frames >5 s are skipped.
- dimOS writes occur only in a dedicated writer; SQLite WAL and batched commits.
- 900 seconds or 512 MiB payload per run; stop if free disk falls below 2 GiB.
- Recorder web requests capped at 8 concurrent threads and 2 s socket timeout.
- Storage trouble stops recording; it has no route to the motion-control loop.
- No automatic deletion. Runs remain in runs/<run-id>/run.db.
- Up to one second of uncommitted samples may be lost in a hard process kill.
- Recorded images are sparse observations, not full-rate training trajectories.
- Camera timestamps come from X-Stamp; state samples use observation time.
  Polling is not a lossless or perfectly synchronized event capture.
- Automatic capture begins after match detection, so the first ~2 seconds can
  be missed; start recording manually before a match if that context is needed.

MEASURED 2026-11-08
During 20 seconds of an existing live station run, with all three cameras saved
and a LiveKit broadcast active: recorder 0.80% of one CPU core / 48.2 MiB RSS;
LiveKit 0.35% / 97.9 MiB. Controller GET median 4.39 ms, p95 11.11 ms, 0/20
failures; zero recorder drops, fetch failures or slow fetches. An earlier sample
before these services started measured 4.14 ms median / 6.54 ms p95. These are
short observations under changing station workload, not a controlled full-match
performance guarantee. performance-live.json contains the live measurement.

SDK PROVENANCE
Official dimOS 0.0.15 source distribution, SHA-256 verified against PyPI metadata.
Unmodified dimos.memory.blobstore.sqlite.SqliteBlobStore handles every payload
write AND playback read. Custom observations table indexes timestamps and stage
labels. This is a small adapter to dimOS storage, not its full blueprint recorder.
Official LiveKit server 1.13.11 and JS client 2.23.0, downloaded into .runtime.
versions.json records source URLs and hashes. Upstream licenses remain with source.
Local generated LiveKit credentials stay in livekit-secret.json and livekit.yaml;
never include these files in Git or distribute them to judges. Viewers receive
short-lived subscribe-only room tokens. Only localhost operator connections can
start/stop recording or obtain a publish-capable token.

OPERATIONS
Start/recover: ssh nuc 'bash /home/pah/run-recorder/start.sh'
Sessions: run-recorder and run-recorder-livekit in tmux.
Logs: /home/pah/run-recorder/{recorder,livekit}.log
Graceful recorder shutdown (recording only):
  ssh nuc 'tmux send-keys -t run-recorder:0.0 C-c'
Stop video relay server (no robot effect):
  ssh nuc 'tmux send-keys -t run-recorder-livekit:0.0 C-c'
The recorder launch script is idempotent. It does not restart the controller.
Private-network viewers need connectivity to NUC TCP 8791, 7880, 7881 and UDP
7882. No cloud service is involved. A localhost-only page tunnel does not tunnel
WebRTC media; the operator browser currently reaches the NUC over the LAN.

VALIDATION
  cd /home/pah/run-recorder
  .venv/bin/python -m unittest -v test_recorder
14 tests passed: real dimOS roundtrip and concurrent reads, automatic match
lifecycle and image/event capture, queue overflow, stale images, disk/duration
limits, shutdown flush, cross-run isolation, telemetry allowlist and read-only
viewer / publish-token restrictions. Ruff check and format passed.
Browser verified received video decoding and three saved images read from dimOS.
No robot motion was started by implementation or validation.
