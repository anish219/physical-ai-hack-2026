# Local sponsor bonus demo

Two real integrations, in a separate process and environment:

- **LiveKit:** a locally hosted LiveKit server carries a browser screen-share
  video track from a publisher to a spectator through WebRTC.
- **Dimensional:** the spectator sends received JPEG frames every five seconds
  and room/track events to the local recorder. The official dimOS
  `dimos.memory.blobstore.sqlite.SqliteBlobStore` writes and reads the payloads.
  The page shows counts, the database path, and a saved image read back from it.

The only connection to the project is the operator selecting its existing
dashboard tab in the browser's share picker. There are no project imports,
controller API requests, camera opens, robot connections, or main dependency
changes. This demo has its own `.venv`, downloaded SDKs, ports, and recordings.
It is not started by any existing project entry point.

## Setup (Windows, Python 3.12)

From the repository root:

```powershell
python bonus_demo/setup_demo.py
```

Setup downloads pinned official releases into `bonus_demo/.runtime` and installs
the recorder's small Python dependency set in `bonus_demo/.venv`. The full dimOS
source distribution is imported directly, without compiling its unrelated native
navigation extension or installing the robot/AI dependency stack. No upstream
source is modified. Versions and source URLs are saved in `.runtime/versions.json`.
Upstream licenses are included with the downloads.

## Run

```powershell
.\bonus_demo\start.ps1
```

1. Open <http://127.0.0.1:8790/> in Chrome or Edge.
2. Click **Share dashboard tab** and choose your existing robot dashboard tab.
3. Click **Open spectator**. It receives the video through LiveKit.
4. Wait five seconds. The Dimensional panel shows recorded frames and an image
   retrieved from the SQLite database.
5. Show the referee the spectator, both integration labels, saved-frame count,
   and database. This demonstrates actual video transport and recording.

The HTTP app uses `127.0.0.1:8790`; LiveKit uses its local development ports,
including `127.0.0.1:7880`. No cloud account, keys, CDN, deployment, or outbound
application requests are used after setup. The LiveKit dev credentials are
`devkey` / `secret`. The spectator's LiveKit token cannot publish tracks or data.
The demo doesn't send commands to the policy, before or during a run.

**Stop sharing** ends the video track. Ctrl+C in the terminal ends the demo and
the LiveKit child process it launched. Recordings remain under
`bonus_demo/recordings/<UTC timestamp>/bonus.db`; the server log is beside it.

Recorded JPEGs are in `frames_blob`, event JSON in `events_blob`. This uses
Dimensional's blob-store API directly, not the full dimOS blueprint recorder.

No existing project files need to change or ship for this demo. Keep it on the
laptop; choosing the robot dashboard tab works even if that page comes through
your existing SSH tunnel. Referee acceptance is still their call: the rubric
only says the system must use each sponsor and show it before the match.

## Verification

```powershell
.\bonus_demo\.venv\Scripts\python.exe -m unittest discover -s bonus_demo -p "test_*.py" -v
```

References: [LiveKit local server](https://docs.livekit.io/transport/self-hosting/local/),
[screen sharing](https://docs.livekit.io/transport/media/screenshare/),
[Dimensional source](https://github.com/dimensionalOS/dimos).
