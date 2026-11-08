#!/bin/bash
# hc.sh ensure|start|stop|status: the one way to run the left hillclimb server (port 8770) from ~/hillclimb-repo.
#   ensure  server up and running the newest code on disk (restarts a stale one only when no round is running)
#   start   ensure, then arm a match round and go (optional JSON settings as $2)
#   stop    stop the round
#   status  version and state
R=$HOME/hillclimb-repo; P=${HC_PORT:-8770}; U=http://127.0.0.1:$P
newest() { find "$R/hillclimb" \( -name '*.py' -o -name '*.html' \) -not -path '*/runs/*' -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1; }
version() { local v; v=$(curl -s -m 1 $U/api/version); echo "$v" | grep -q '"started"' && echo "$v"; }
field() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)" 2>/dev/null; }
serve() {
  tmux kill-session -t hillclimb$P 2>/dev/null
  for i in $(seq 25); do ss -ltn | grep -q ":$P " || break; sleep 0.2; done
  ss -ltn | grep -q ":$P " && { echo "port $P still held by another process: not starting a second server"; return 1; }
  tmux new-session -d -s hillclimb$P "HC_PORT=$P bash $R/hc_start.sh"
  for i in $(seq 50); do version >/dev/null && { echo "server up (pid $(version | field '["pid"]'))"; return 0; }; sleep 0.2; done
  echo "server did not come up; see ~/hillclimb.log"; return 1
}
ensure() {
  v=$(version)
  if [ -z "$v" ]; then
    if ss -ltn | grep -q ":$P "; then echo "port $P is held by another process (an old in-process copy): restart that process once"; return 1; fi
    serve; return
  fi
  started=$(echo "$v" | field '["started"]' | cut -d. -f1)
  [ "$(newest)" -le "$started" ] && { echo "server current"; return 0; }
  st=$(curl -s -m 2 $U/api/state | field '["left"]["state"]')
  case "$st" in
    IDLE|END) echo "code changed since the server started: restarting"; serve ;;
    *) echo "code changed, but a round is $st: not restarting until it ends"; return 0 ;;
  esac
}
post() { curl -s -m 30 -XPOST $U$1 -H 'Content-Type: application/json' -d "${2:-{\}}"; echo; }
case "$1" in
  ensure) ensure ;;
  start) ensure || exit 1
         post /left/arm "${2:-{\"backend\":\"scripted\",\"mode\":\"match\",\"drive\":\"auto\",\"round_s\":300,\"max_goal\":6,\"crunch_at_s\":90,\"crunch_goal\":5,\"freeze_s\":20\}}"
         post /left/go ;;
  stop) post /left/stop ;;
  status) version; echo; curl -s -m 2 $U/api/state | field '["left"]["state"]' ;;
  *) echo "usage: hc.sh ensure|start|stop|status"; exit 2 ;;
esac
