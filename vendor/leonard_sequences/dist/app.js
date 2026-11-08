import { sources } from './adapters.js';
import { bridgeEnabled, bridgeRequest } from './bridge-client.js';
const $ = id => document.getElementById(id);
const sequenceKey = 'leonard.arm-sequences.v1';
const id = () => crypto.randomUUID();
const defaultSequences = () => ({
  left: { phases: [] },
  right: { phases: [{ id: id(), title: 'Unplug USB', steps: [{ id: id(), type: 'dream', label: 'Unplug the USB', prompt: 'Unplug the USB from the port, then move it clear of the workspace.', modelId: 'dreamlabs/dream-1-beta', durationSeconds: 20 }] }] }
});
function loadSequences() {
  try {
    const saved = JSON.parse(localStorage.getItem(sequenceKey));
    if (saved?.left?.phases && saved?.right?.phases) {
      for (const side of ['left','right']) for (const phase of saved[side].phases) for (const step of phase.steps || []) delete step.state;
      return saved;
    }
  } catch {}
  return defaultSequences();
}
let sequences = loadSequences();
const runs = { left: null, right: null };
const armState = {};
let camerasFresh = false;
const entries = [];
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[char]));
function saveSequences() { localStorage.setItem(sequenceKey, JSON.stringify(sequences)); }
function log(message) {
  entries.unshift({ time: new Date().toISOString(), message }); entries.length = Math.min(entries.length, 200);
  $('log').replaceChildren(...entries.map(entry => { const li = document.createElement('li'); const time = document.createElement('time'); time.textContent = new Date(entry.time).toLocaleTimeString([], { hour12: false }); li.append(time, entry.message); return li; }));
}
function phaseMarkup(phase, side, phaseIndex) {
  const steps = (phase.steps || []).map((step, stepIndex) => `<div class="subphase" data-step-id="${esc(step.id)}">
    <div class="subphase-rail"><span class="subphase-bulb ${step.state === 'complete' ? 'complete' : step.state === 'active' ? 'active' : ''}"></span><span>SUB ${String(stepIndex+1).padStart(2,'0')}</span></div>
    <div class="subphase-fields"><div class="field-row"><input data-field="label" aria-label="Subphase name" value="${esc(step.label)}" placeholder="Subphase name"><select data-field="type" aria-label="Subphase type"><option value="dream" ${step.type === 'dream' ? 'selected' : ''} ${side === 'left' ? 'disabled' : ''}>Dream task prompt</option><option value="wait" ${step.type === 'wait' ? 'selected' : ''}>Wait / checkpoint</option></select><button class="icon-button" data-action="remove-step" title="Remove subphase" aria-label="Remove subphase">×</button></div>
    ${step.type === 'dream' ? `<textarea data-field="prompt" aria-label="Dream task prompt" placeholder="Tell Dream what the right arm should do">${esc(step.prompt)}</textarea><div class="field-row model-row"><label>Model <input data-field="modelId" value="${esc(step.modelId || 'dreamlabs/dream-1-beta')}" placeholder="Dream model ID"></label><label>Action window (s) <input data-field="durationSeconds" type="number" min="1" max="300" value="${esc(step.durationSeconds || 20)}"></label></div><div class="subphase-note">Dream receives all 3 views and both arm states; only right-arm targets run. The window advances by time, not verified success. The base model also needs calibrated Metal arm geometry on the NUC.</div>` : `<div class="field-row"><label>Wait (seconds) <input data-field="durationSeconds" type="number" min="0" max="300" value="${esc(step.durationSeconds ?? 2)}"></label><span class="subphase-note">Checkpoint only · sends no robot command</span></div>`}</div>
  </div>`).join('');
  return `<article class="phase-card" data-phase-id="${esc(phase.id)}"><header class="phase-head"><span class="phase-number">${String(phaseIndex+1).padStart(2,'0')}</span><input data-field="title" aria-label="Phase title" value="${esc(phase.title)}" placeholder="Phase name"><button class="icon-button" data-action="remove-phase" title="Remove phase" aria-label="Remove phase">×</button></header><div class="subphases">${steps || '<div class="empty-steps">No subphases yet. Add a prompt or checkpoint.</div>'}</div><button class="add-subphase" data-action="add-step">＋ Add subphase</button></article>`;
}
function renderSequences(force = false) {
  const focused = document.activeElement;
  // Bridge status polls every 500 ms. Keep the active editor node alive while
  // the user types; replacing the sequence markup would drop focus and caret.
  if (!force && $('sequences').contains(focused) && ['INPUT', 'TEXTAREA', 'SELECT'].includes(focused.tagName)) return;
  $('sequences').innerHTML = ['left','right'].map(side => {
    const sequence = sequences[side], running = !!runs[side];
    const progress = runs[side]?.label ? `· ${esc(runs[side].label)}` : '';
    const readiness = canRunRight() ? 'Dream ready' : bridgeEnabled ? 'Needs right control + fresh views' : 'Bridge offline';
    return `<section class="sequence-card ${side}" data-side="${side}"><div class="sequence-head"><div><div class="sequence-kicker">${side === 'left' ? 'ARM 01' : 'ARM 02'}</div><h3>${side === 'left' ? 'Left arm' : 'Right arm'}</h3></div><span class="sequence-run-state ${running ? 'running' : ''}">${running ? `Running${progress}` : side === 'right' ? readiness : 'No actions'}</span><button class="run-sequence ${running ? 'stop' : ''}" data-action="${running ? 'stop-sequence' : 'run-sequence'}" ${running ? '' : side === 'right' && !canRunRight() ? 'disabled title="Take dashboard control and wait for fresh cameras and feedback from both arms"' : ''}>${running ? '■ Stop' : '▶ Run sequence'}</button></div><div class="phase-list">${sequence.phases.map((phase,index) => phaseMarkup(phase,side,index)).join('') || `<div class="empty-sequence">${side === 'left' ? 'This arm can stay idle while the right arm runs.' : 'Add a phase, then give it a Dream task prompt.'}</div>`}</div><button class="add-phase" data-action="add-phase">＋ Add phase</button></section>`;
  }).join('');
}
function canRunRight() { const arm = armState.right, left = armState.left; return bridgeEnabled && camerasFresh && arm?.connected && arm?.controlled && arm?.powered && arm?.fresh && arm?.owner !== 'leader' && left?.powered && left?.fresh; }
function newStep(side) { return side === 'right' ? { id:id(), type:'dream', label:'Dream task', prompt:'', modelId:'dreamlabs/dream-1-beta', durationSeconds:20 } : { id:id(), type:'wait', label:'Idle checkpoint', durationSeconds:2 }; }
function stopWait(ms, side, token) { return new Promise(resolve => { const started = Date.now(); const tick = () => { if (runs[side] !== token || Date.now()-started >= ms) resolve(); else setTimeout(tick,100); }; tick(); }); }
async function stopSequence(side) {
  const token = runs[side]; if (!token) return;
  token.cancelled = true;
  if (side === 'right' && token.dreamActive) { try { await bridgeRequest('dream/stop','POST',{}); } catch (error) { log(`Dream stop · ${error.message}`); } }
  runs[side] = null; renderSequences(); $('status').textContent = 'Ready'; log(`${side} arm sequence stopped`);
}
async function pollDream(side, token) {
  while (runs[side] === token && !token.cancelled) {
    const status = await bridgeRequest('dream/status');
    token.label = status.prompt ? status.prompt.slice(0,40) : 'Dream action'; renderSequences();
    if (status.state === 'complete') return;
    if (status.state === 'error' || status.state === 'stopped') throw new Error(status.error || 'Dream action stopped');
    await new Promise(resolve => setTimeout(resolve,400));
  }
  throw new Error('Sequence stopped');
}
async function runSequence(side) {
  if (runs[side]) return;
  if (side === 'right' && !canRunRight()) { log('Take control of the powered right arm before starting its sequence'); return; }
  const token = { cancelled:false, dreamActive:false, label:'Starting' }; runs[side] = token;
  renderSequences(); $('status').textContent = `${side === 'left' ? 'Left' : 'Right'} running`;
  log(`${side} arm sequence started`);
  try {
    for (let p=0; p<sequences[side].phases.length; p++) {
      const phase = sequences[side].phases[p];
      for (const step of phase.steps || []) {
        if (runs[side] !== token || token.cancelled) throw new Error('Sequence stopped');
        token.label = `${phase.title} · ${step.label}`; step.state = 'active'; renderSequences(); saveSequences();
        if (step.type === 'wait') await stopWait(Math.max(0,Number(step.durationSeconds)||0)*1000,side,token);
        else {
          if (side !== 'right') throw new Error('Dream actions are currently enabled for the right arm only');
          if (!canRunRight()) throw new Error('Right-arm control or fresh telemetry was lost');
          token.dreamActive = true;
          await bridgeRequest('dream/start','POST',{ prompt:step.prompt, checkpoint:step.modelId || 'dreamlabs/dream-1-beta', durationSeconds:Number(step.durationSeconds)||20 });
          await pollDream(side,token); token.dreamActive = false;
        }
        if (runs[side] !== token || token.cancelled) throw new Error('Sequence stopped');
        step.state = 'complete'; saveSequences(); renderSequences();
        log(`${side} · ${phase.title} / ${step.label} window complete`);
      }
    }
    if (runs[side] === token) { runs[side] = null; $('status').textContent = 'Ready'; log(`${side} arm sequence finished`); }
  } catch (error) {
    if (token.dreamActive) { try { await bridgeRequest('dream/stop','POST',{}); } catch {} }
    if (runs[side] === token) { runs[side] = null; $('status').textContent = 'Ready'; log(`${side} sequence · ${error.message}`); }
  } finally { if (runs[side] !== token) { for (const phase of sequences[side].phases) for (const step of phase.steps || []) if (step.state === 'active') delete step.state; } renderSequences(); }
}
function locateField(target) {
  const card = target.closest('.sequence-card'), phaseCard = target.closest('.phase-card'), stepCard = target.closest('.subphase');
  if (!card) return null;
  const side = card.dataset.side, phase = sequences[side].phases.find(item => item.id === phaseCard?.dataset.phaseId), step = phase?.steps?.find(item => item.id === stepCard?.dataset.stepId);
  return { side, phase, step };
}
document.addEventListener('input', event => {
  const found = locateField(event.target); if (!found) return;
  const field = event.target.dataset.field; if (!field) return;
  let value = event.target.value;
  if (field === 'durationSeconds') value = Math.min(300,Math.max(0,Number(value)||0));
  if (found.step) found.step[field] = value;
  else if (found.phase) found.phase[field] = value;
  saveSequences();
});
document.addEventListener('change', event => {
  if (event.target.dataset.field !== 'type') return;
  const found=locateField(event.target); if (!found?.step) return;
  found.step.type=event.target.value; saveSequences(); renderSequences(true);
});
document.addEventListener('click', event => {
  const button=event.target.closest('[data-action]'); if(!button)return;
  const card=button.closest('.sequence-card'), side=card?.dataset.side, found=locateField(button);
  switch(button.dataset.action){
    case 'add-phase': sequences[side].phases.push({id:id(),title:`Phase ${sequences[side].phases.length+1}`,steps:[]}); break;
    case 'add-step': found.phase.steps.push(newStep(side)); break;
    case 'remove-phase': sequences[side].phases=sequences[side].phases.filter(item=>item.id!==found.phase.id); break;
    case 'remove-step': found.phase.steps=found.phase.steps.filter(item=>item.id!==found.step.id); break;
    case 'run-sequence': void runSequence(side); return;
    case 'stop-sequence': void stopSequence(side); return;
    default:return;
  }
  saveSequences(); renderSequences();
});
  renderSequences();

$('cameras').innerHTML = [{ id: 0, label: 'Left arm · wrist' }, { id: 2, label: 'Workspace · overview' }, { id: 1, label: 'Right arm · wrist' }].map(({ id, label }) => `<div class="camera" data-camera-id="${id}"><div class="camera-view"><span class="camera-icon">⌑</span><span>Awaiting camera feed</span></div><div class="camera-label">${label}<span>CAM 0${id+1}</span></div></div>`).join('');
function renderMotors(motors) {
  $('arms').innerHTML = ['arm01', 'arm02'].map((key, i) => {
    const positions = motors[key];
    return `<div class="arm-row"><b>${i === 0 ? 'Left arm' : 'Right arm'}</b><div class="joint-values">${Array.isArray(positions) ? positions.map((value,j) => `<span class="joint"><span>J${j+1}</span><output aria-label="Arm ${i+1} joint ${j+1}">${Number(value).toFixed(1)}°</output></span>`).join('') : '<span class="small">Not connected</span>'}</div></div>`;
  }).join('');
}
if (!bridgeEnabled) renderMotors(await sources.readMotorPositions()); else renderMotors({});
if (bridgeEnabled) {
  document.querySelector('footer span').textContent = 'NUC bridge · left arm is observation-only for Dream sequences';
  const button = document.createElement('button'); button.textContent = 'Connect dashboard';
  const message = $('bridge-message');
  $('arms').parentElement.prepend(button);
  const cameraPlayers = {}, armPanels = {};
  for (const side of ['left','right']) {
    const controls=document.createElement('div'); controls.className='right-controls'; controls.hidden=true;
    controls.innerHTML=`<b>${side==='left'?'Left':'Right'} arm</b><span class="arm-owner">Read-only</span><button data-action="acquire">Take control</button><button data-action="release" hidden>Disconnect</button><div class="owned-controls" hidden><button data-action="on">Power on</button><button data-action="off">Power off · torque release</button><button data-action="stop">Hold position</button><label>Jog <select aria-label="${side} arm joint">${[1,2,3,4,5,6].map(n=>`<option value="${n-1}">J${n}</option>`).join('')}</select></label><span class="step-toggles" role="group" aria-label="${side} arm jog step"><button data-step="1" aria-pressed="true">1°</button><button data-step="5" aria-pressed="false">5°</button></span><button data-action="minus">−1°</button><button data-action="plus">+1°</button></div>`;
    $('arms').parentElement.append(controls);
    const panel={controls,busy:false,step:1};armPanels[side]=panel;
    controls.onclick=async event=>{
      const target=event.target.closest('button');if(!target||target.disabled)return;
      if(target.dataset.step){panel.step=Number(target.dataset.step);for(const toggle of controls.querySelectorAll('[data-step]'))toggle.setAttribute('aria-pressed',String(Number(toggle.dataset.step)===panel.step));controls.querySelector('[data-action="minus"]').textContent=`−${panel.step}°`;controls.querySelector('[data-action="plus"]').textContent=`+${panel.step}°`;return;}
      const name=target.dataset.action;if(!name||panel.busy)return;
      panel.busy=true;for(const control of controls.querySelectorAll('[data-action]'))control.disabled=true;
      const path=`arms/${side}/`;
      try{
        if(name==='on'||name==='off')await bridgeRequest(path+'power','POST',{enabled:name==='on'});
        else if(['acquire','release'].includes(name))await bridgeRequest(path+name,'POST',name==='acquire'?{takeover:target.dataset.takeover==='true'}:{});
        else if(name==='stop')await bridgeRequest(path+'stop','POST',{});
        else await bridgeRequest(path+'jog','POST',{joint:Number(controls.querySelector('select').value),deltaDegrees:(name==='plus'?1:-1)*panel.step});
        log(`${side} arm · ${name} request accepted`);
      }catch(error){log(error.message);controls.querySelector('.arm-owner').textContent=error.message;}
      finally{panel.busy=false;void updateBridge();}
    };
  }
  let connected=false,busy=false;
  button.onclick=async()=>{button.disabled=true;try{const result=await bridgeRequest(connected?'disconnect':'connect','POST');connected=result.connected;log(`Dashboard ${connected?'connected':'disconnected'} · ${result.mode}`);}catch(error){log(error.message);message.textContent=error.message;}finally{button.disabled=false;button.textContent=connected?'Disconnect dashboard':'Connect dashboard';if(!connected)renderMotors({});}};
  async function updateBridge(){
    if(busy)return;busy=true;
    try{
      const status=await bridgeRequest('status');connected=status.connected;button.textContent=connected?'Disconnect dashboard':'Connect dashboard';button.hidden=status.controlsAvailable;
      camerasFresh=['wrist_l','overhead','wrist_r'].every(role=>status.video?.[role]?.fresh===true);
      for(const [side,panel] of Object.entries(armPanels)){
        const {controls}=panel,arm=status.arms?.[side];if(!arm)continue;
        armState[side]={...arm,connected,fresh:arm.fresh};
        controls.hidden=!status.controlsAvailable;
        controls.querySelector('.arm-owner').textContent=arm.owner==='leader'?'Leader controller · takeover stops teleoperation':arm.owner==='unknown'?'Controller unavailable · read-only':arm.controlled?'Dashboard control':'Read-only';
        const acquire=controls.querySelector('[data-action="acquire"]');acquire.hidden=arm.controlled;acquire.textContent=arm.owner==='leader'?'Take over from leader':'Take control';acquire.dataset.takeover=String(arm.owner==='leader');acquire.disabled=panel.busy||!connected||!['available','leader'].includes(arm.owner);
        const release=controls.querySelector('[data-action="release"]');release.hidden=!arm.controlled;release.disabled=panel.busy;
        controls.querySelector('.owned-controls').hidden=!arm.controlled;
        for(const control of controls.querySelectorAll('.owned-controls [data-action]')){const name=control.dataset.action;control.disabled=panel.busy||!connected||!arm.controlled||(name==='on'?arm.powered:!arm.powered||!arm.fresh);}
      }
      renderSequences();
      for(const [id,role] of [[0,'wrist_l'],[2,'overhead'],[1,'wrist_r']]){
        const camera=document.querySelector(`[data-camera-id="${id}"]`),health=status.video?.[role];if(!health)continue;
        camera.querySelector('.camera-label span').textContent=`CAM 0${id+1} · ${Math.round(health.fps)} fps`;
        const description=camera.querySelector('.camera-view span:last-of-type');
        if(description&&!health.fresh){const powered=role==='wrist_l'?status.arms?.left?.powered:role==='wrist_r'?status.arms?.right?.powered:status.arms?.left?.powered||status.arms?.right?.powered;description.textContent=powered?'Reconnecting camera…':'Arm service off · camera paused';}
        const img=camera.querySelector('img');if(img)img.hidden=!health.fresh;if(!health.fresh)cameraPlayers[role]?.recover();for(const child of camera.querySelectorAll('.camera-view span'))child.hidden=health.fresh;
      }
      if(connected){renderMotors(await sources.readMotorPositions());message.textContent=`Left ${status.arms?.left?.powered?'on':'off'} · Right ${status.armPowered?'on':'off'} · Take control per arm to enable commands`;}
      else{renderMotors({});message.textContent='Dashboard disconnected';}
    }catch(error){renderMotors({});message.textContent=`Bridge unavailable: ${error.message}`;}
    finally{busy=false;}
  }
  void updateBridge();setInterval(updateBridge,500);
  for(const [id,role] of [[0,'wrist_l'],[2,'overhead'],[1,'wrist_r']]){
    const view=document.querySelector(`[data-camera-id="${id}"] .camera-view`),img=document.createElement('img');img.alt=`${role} camera feed`;img.hidden=true;view.append(img);
    let reconnect,lastAttempt=0;
    const show=available=>{img.hidden=!available;for(const child of view.querySelectorAll('span'))child.hidden=available;};
    const openStream=()=>{clearTimeout(reconnect);lastAttempt=Date.now();img.src=`/api/cameras/${role}/stream.mjpg?t=${lastAttempt}`;};
    img.onload=()=>show(true);img.onerror=()=>{show(false);clearTimeout(reconnect);reconnect=setTimeout(openStream,1000);};
    cameraPlayers[role]={recover(){if(Date.now()-lastAttempt>3000)openStream();}};
    window.addEventListener('pagehide',()=>{clearTimeout(reconnect);img.removeAttribute('src');});window.addEventListener('pageshow',event=>{if(event.persisted)openStream();});openStream();
  }
} else {
  $('bridge-message').textContent='Connect to the NUC bridge to run sequences on the arms';
}
log(`Sequence editor ready · Dream ${bridgeEnabled?'bridge available':'not connected'}`);
