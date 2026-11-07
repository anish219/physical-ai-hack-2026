"""The SO-101 leader's Track button and translation-scale slider."""

import http.server
import json
import math
import threading


class Controls:
    def __init__(self, scale: float, side: str, dry_run: bool = False, grip_torque_nm: float = 0.5) -> None:
        self._lock = threading.Lock()
        self._state = dict(tracking=False, scale=scale, side=side, dry_run=dry_run, revision=0,
                           arm_open=False, recording=False, recording_requested=False, frames=0,
                           saved_episodes=0, dataset_dir="", power_off=False, killing=False, message="",
                           home_requested=False, homing=False, recording_error="", reject_requested=False,
                           reject_kind="",
                           grip_torque_nm=grip_torque_nm, grip_revision=0, grip_applied_torque_nm=None,
                           grip_measured_torque_nm=None, grip_holding=False, grip_error="")

    def snapshot(self) -> dict:
        with self._lock:
            return self._state.copy()

    def change(self, values: dict) -> dict:
        updates = {}
        if "tracking" in values:
            if not isinstance(values["tracking"], bool):
                raise ValueError("tracking must be true or false")
            updates["tracking"] = values["tracking"]
        if "scale" in values:
            scale = values["scale"]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
                raise ValueError("scale must be a finite number")
            updates["scale"] = float(scale)
        if "recording_requested" in values:
            if not isinstance(values["recording_requested"], bool):
                raise ValueError("recording_requested must be true or false")
            updates["recording_requested"] = values["recording_requested"]
        if "grip_torque_nm" in values:
            value = values["grip_torque_nm"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.1 <= value <= 2.0:
                raise ValueError("Grip holding torque must be between 0.1 and 2.0 N·m")
            updates["grip_torque_nm"] = float(value)
        with self._lock:
            if updates.get("recording_requested") and (not self._state["arm_open"] or self._state["dry_run"]):
                raise ValueError("Track to open the arm before recording; recording requires live mode")
            if any(self._state[key] != value for key, value in updates.items()):
                if any(self._state[key] != value for key, value in updates.items() if key in ("tracking", "scale")):
                    self._state["revision"] += 1
                if "grip_torque_nm" in updates and updates["grip_torque_nm"] != self._state["grip_torque_nm"]:
                    self._state["grip_revision"] += 1
                self._state.update(updates)
                if updates.get("tracking"):
                    self._state.update(home_requested=False, homing=False)
            return self._state.copy()

    def request_reject(self) -> dict:
        with self._lock:
            if not (self._state["recording"] or self._state["recording_requested"] or self._state["saved_episodes"]):
                raise ValueError("No example to reject")
            current = self._state["recording"] or self._state["recording_requested"]
            self._state.update(recording_requested=False, reject_requested=True,
                               reject_kind="current" if current else "saved")
            return self._state.copy()

    def request_home(self) -> dict:
        with self._lock:
            if not self._state["arm_open"]:
                raise ValueError("Track to open the arm before returning")
            if self._state["recording"] or self._state["recording_requested"]:
                raise ValueError("Save the recording before returning, as with the existing Home control")
            self._state.update(tracking=False, home_requested=True, homing=True,
                               message="Returning to the captured startup pose")
            self._state["revision"] += 1
            return self._state.copy()

    def report(self, **values) -> None:
        with self._lock:
            self._state.update(values)


class LeaderDashboard:
    def __init__(self, controls: Controls, port: int = 8081, on_kill=None) -> None:
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, code, data, content_type):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path == "/":
                    self.reply(200, PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/api/state":
                    self.reply(200, json.dumps(controls.snapshot()).encode(), "application/json")
                else:
                    self.reply(404, b"Not found", "text/plain")

            def do_POST(self):  # noqa: N802
                if self.path == "/api/kill":
                    controls.change({"tracking": False, "recording_requested": False})
                    controls.report(home_requested=False, homing=False)
                    side = controls.snapshot()["side"]
                    controls.report(killing=True, message=f"Disabling {side} arm…")
                    try:
                        if on_kill is None:
                            raise RuntimeError("motor power control is not connected")
                        problems = on_kill()
                        controls.report(power_off=not problems, arm_open=False, killing=False,
                                        grip_holding=False, grip_measured_torque_nm=None,
                                        grip_applied_torque_nm=None,
                                        message="; ".join(problems) if problems
                                        else f"{side.title()} arm: motor torque off")
                    except Exception as error:
                        controls.report(killing=False, message=f"Kill arm failed: {error}")
                        self.reply(500, json.dumps(controls.snapshot()).encode(), "application/json")
                        return
                    self.reply(200, json.dumps(controls.snapshot()).encode(), "application/json")
                    return
                if self.path == "/api/home":
                    try:
                        state = controls.request_home()
                    except ValueError as error:
                        self.reply(400, json.dumps({"error": str(error)}).encode(), "application/json")
                        return
                    self.reply(200, json.dumps(state).encode(), "application/json")
                    return
                if self.path == "/api/reject":
                    try:
                        state = controls.request_reject()
                    except ValueError as error:
                        self.reply(400, json.dumps({"error": str(error)}).encode(), "application/json")
                        return
                    self.reply(200, json.dumps(state).encode(), "application/json")
                    return
                if self.path != "/api/control":
                    self.reply(404, b"Not found", "text/plain")
                    return
                try:
                    values = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    if not isinstance(values, dict):
                        raise ValueError("expected an object")
                    result = controls.change(values)
                except (ValueError, TypeError) as error:
                    self.reply(400, json.dumps({"error": str(error)}).encode(), "application/json")
                    return
                self.reply(200, json.dumps(result).encode(), "application/json")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="so101-dashboard", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 · Arm control</title><style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#edf0f3;background:#101315}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:28px}
main{width:min(100%,420px)}header{display:flex;align-items:center;justify-content:space-between;margin-bottom:40px}
.eyebrow{font-size:11px;letter-spacing:.16em;color:#939ca5;text-transform:uppercase}
.mode{font-size:12px;color:#a9b4bd;border:1px solid #343b40;border-radius:20px;padding:6px 10px}
h1{font-size:32px;letter-spacing:-.035em;margin:0 0 8px;font-weight:550}
p{color:#939ca5;font-size:14px;line-height:1.6;margin:0 0 30px}
button{width:100%;border:0;border-radius:14px;padding:22px;background:#b5f2cb;color:#10281a;
font-family:inherit;font-size:18px;font-weight:600;cursor:pointer;transition:background .15s}
button[data-tracking=true]{background:#30383f;color:#f1f3f5;border:1px solid #56616a}
.tracking-buttons{display:grid;grid-template-columns:1fr 1fr;gap:12px}
#untrack{background:#30383f;color:#f1f3f5;border:1px solid #56616a}
#tracking-state{display:inline-block;padding:7px 11px;margin-bottom:18px;border-radius:20px;
font-size:12px;font-weight:650;letter-spacing:.08em;background:#30383f;color:#c5cbd0}
#tracking-state[data-tracked=true]{background:#b5f2cb;color:#10281a}
#reject{margin-top:12px;background:transparent;color:#ffb4ac;border:1px solid #764641;padding:14px;font-size:14px}
.feedback{font-size:12px;color:#a9b4bd;margin-top:12px;line-height:1.6}
#record{margin-top:12px;background:#252c31;color:#edf0f3;border:1px solid #414b53;padding:16px;font-size:15px}
#record[data-recording=true]{color:#ffb4ac;border-color:#b34f48}
#home{margin-top:12px;background:#252c31;color:#edf0f3;border:1px solid #414b53;padding:16px;font-size:15px}
#kill{margin-top:24px;background:#b52f32;color:#fff;padding:17px;font-size:15px;letter-spacing:.025em}
button:disabled{opacity:.45;cursor:default}
button:hover{filter:brightness(1.08)}
button:focus-visible,input:focus-visible{outline:2px solid #b5f2cb;outline-offset:5px}
.scale{margin-top:36px;padding-top:28px;border-top:1px solid #30363b}
.label{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;font-size:14px}
output{font-variant-numeric:tabular-nums;color:#b5f2cb;font-size:18px}
input{width:100%;accent-color:#b5f2cb;cursor:pointer}
.ends{display:flex;justify-content:space-between;color:#737e87;font-size:11px;margin-top:9px}
#message{min-height:22px;font-size:12px;color:#b6bec5;margin-top:24px;white-space:pre-line}
</style></head><body><main>
<header><span class="eyebrow">SO-101 / Metal</span><span class="mode" id="side">Right arm</span></header>
<h1>Leader control</h1><p id="hint">Untracked. Position the leader, then track.</p>
<div id="tracking-state" role="status" aria-live="polite">UNTRACKED</div>
<div class="tracking-buttons"><button id="track" type="button">Track</button>
<button id="untrack" type="button">Untrack</button></div>
<button id="record" type="button" aria-pressed="false">Record example</button>
<button id="reject" type="button">Reject example</button>
<button id="home" type="button">Return to 0</button>
<div class="scale"><label class="label" for="scale"><span>Translation scale</span>
<output id="value">1.00×</output></label>
<input id="scale" type="range" min="0" max="3" step="0.05" value="1" aria-label="Translation scale">
<div class="ends"><span>0×</span><span>3×</span></div></div>
<div class="scale"><label class="label" for="grip-torque"><span>Grip holding torque</span>
<output id="grip-value">0.50 N·m</output></label>
<input id="grip-torque" type="range" min="0.1" max="2" step="0.05" value="0.5"
aria-label="Grip holding torque">
<div class="ends"><span>0.1 N·m</span><span>2.0 N·m</span></div>
<div class="feedback" id="grip-feedback">Motor torque feedback available after Track.</div></div>
<button id="kill" type="button">Kill arm</button>
<div id="message" role="status"></div>
</main><script>
const button=document.getElementById('track'),slider=document.getElementById('scale');
const record=document.getElementById('record'),kill=document.getElementById('kill');
const home=document.getElementById('home');
const untrack=document.getElementById('untrack'),reject=document.getElementById('reject');
const grip=document.getElementById('grip-torque'),indicator=document.getElementById('tracking-state');
let tracking=false,recording=false,dragging=false,gripDragging=false,chain=Promise.resolve();
function render(s){tracking=s.tracking;
indicator.textContent=tracking?(s.arm_open?'TRACKED':'STARTING…'):'UNTRACKED';
indicator.dataset.tracked=tracking&&s.arm_open;
document.getElementById('side').textContent=(s.side==='right'?'Right':'Left')+' arm'+(s.dry_run?' · dry run':'');
document.getElementById('hint').textContent=s.power_off?
(s.side==='right'?'Right':'Left')+' arm: motor torque off. Track reopens this arm.':
s.homing?'Returning to the captured startup pose.':
tracking?'Tracking leader motion.':'Untracked. Position the leader, then track.';
recording=s.recording_requested;record.textContent=recording?'Stop & save example':'Record example';
record.dataset.recording=recording;record.setAttribute('aria-pressed',String(recording));
record.disabled=!s.arm_open||s.dry_run;kill.textContent=s.killing?'Disabling arm…':'Kill arm';
home.textContent=s.homing?'Returning…':'Return to 0';
home.disabled=!s.arm_open||s.homing||s.recording||s.recording_requested;
reject.textContent=s.recording||s.recording_requested?'Reject current example':'Reject last saved example';
reject.disabled=!(s.recording||s.recording_requested||s.saved_episodes);
if(!gripDragging){grip.value=s.grip_torque_nm;
document.getElementById('grip-value').textContent=Number(s.grip_torque_nm).toFixed(2)+' N·m';}
const measured=s.grip_measured_torque_nm==null?'—':Math.abs(s.grip_measured_torque_nm).toFixed(2)+' N·m';
const applied=s.grip_applied_torque_nm==null?'pending Track':Number(s.grip_applied_torque_nm).toFixed(2)+' N·m';
document.getElementById('grip-feedback').textContent='Applied hold: '+applied+' · Measured motor torque: '+measured+
(s.grip_holding?' · HOLDING':'');
document.getElementById('message').textContent=s.recording?'Recording · '+s.frames+' frames':
[s.message,s.recording_error?'Recording: '+s.recording_error:'',s.grip_error].filter(Boolean).join('\\n');
if(!dragging){slider.value=s.scale;document.getElementById('value').textContent=Number(s.scale).toFixed(2)+'×';}}
function send(values,path='/api/control'){chain=chain.catch(()=>{}).then(async()=>{
const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify(values)});
const s=await r.json();if(!r.ok)throw Error(s.error);render(s);
}).catch(e=>{document.getElementById('message').textContent=e.message;});return chain;}
button.onclick=()=>send({tracking:true});
untrack.onclick=()=>send({tracking:false});
record.onclick=()=>send({recording_requested:!recording});
reject.onclick=()=>send({},'/api/reject');
home.onclick=()=>send({},'/api/home');
kill.onclick=async()=>{try{const r=await fetch('/api/kill',{method:'POST'});const s=await r.json();render(s);}
catch(e){document.getElementById('message').textContent=e.message;}};
slider.oninput=()=>{dragging=true;document.getElementById('value').textContent=Number(slider.value).toFixed(2)+'×';};
slider.onchange=()=>{const scale=Number(slider.value);dragging=false;send({scale});};
grip.oninput=()=>{gripDragging=true;
document.getElementById('grip-value').textContent=Number(grip.value).toFixed(2)+' N·m';};
grip.onchange=()=>{const torque=Number(grip.value);gripDragging=false;send({grip_torque_nm:torque});};
async function poll(){try{const r=await fetch('/api/state');if(!r.ok)throw Error('Connection unavailable');
render(await r.json());}
catch(e){document.getElementById('message').textContent=e.message;}}
poll();setInterval(poll,500);
</script></body></html>"""
