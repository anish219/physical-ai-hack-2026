#!/usr/bin/env bash
set -euo pipefail
recorder_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$recorder_root"
recorder_host="$(.venv/bin/python -c 'import json; print(json.load(open("station.json"))["livekit_host"])')"
if ! tmux has-session -t =run-recorder-livekit 2>/dev/null; then
  printf -v livekit_cmd 'cd %q && exec ionice -c 3 nice -n 15 .runtime/livekit/livekit-server --config livekit.yaml >> livekit.log 2>&1' "$recorder_root"
  tmux new-session -d -s run-recorder-livekit "$livekit_cmd"
fi
if ! tmux has-session -t =run-recorder 2>/dev/null; then
  printf -v recorder_cmd 'cd %q && exec ionice -c 3 .venv/bin/python -u recorder.py --livekit-host %q >> recorder.log 2>&1' "$recorder_root" "$recorder_host"
  tmux new-session -d -s run-recorder "$recorder_cmd"
fi
printf 'Judge view: http://%s:8791/\nOperator: forward port 8791 over SSH and open http://localhost:8791/\n' "$recorder_host"
