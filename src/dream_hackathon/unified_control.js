const $ = id => document.getElementById(id);
const selections = ['left', 'right', 'both'];
const clientId = crypto.randomUUID();
let station = {}, dragging = false, gripDragging = {};
const sharedDrafts = {}, pending = {}, inFlight = {}, drains = {}, versions = {};

for (const selection of selections) {
  const panel = $('api-' + selection + '-panel');
  panel.append($('api-panel-template').content.cloneNode(true));
  for (const element of panel.querySelectorAll('[data-field]')) {
    element.id = 'api-' + selection + '-' + element.dataset.field;
  }
  for (const label of panel.querySelectorAll('[data-for]')) {
    label.htmlFor = 'api-' + selection + '-' + label.dataset.for;
  }
  $('api-' + selection + '-title').textContent = 'Dream API · ' +
    (selection === 'both' ? 'Both arms' : selection === 'left' ? 'Left arm' : 'Right arm');
  sharedDrafts[selection] = {task: '', checkpoint: 'dreamlabs/dream-1-beta'};
  pending[selection] = {}; inFlight[selection] = {}; versions[selection] = -1;
  for (const field of ['task', 'checkpoint']) {
    $('api-' + selection + '-' + field).oninput = event => {
      pending[selection][field] = event.target.value;
      flushDraft(selection);
    };
  }
  for (const name of ['connect', 'call', 'run', 'stop']) {
    $('api-' + selection + '-' + name).onclick = async () => {
      await flushDraft(selection);
      await action('/api/dream/' + selection + '/' + name);
    };
  }
}

function syncDraftFields(selection) {
  for (const field of ['task', 'checkpoint']) {
    if (field in pending[selection] || field in inFlight[selection]) continue;
    const element = $('api-' + selection + '-' + field);
    const value = sharedDrafts[selection][field];
    if (element.value === value) continue;
    const start = element.selectionStart, end = element.selectionEnd;
    element.value = value;
    if (element === document.activeElement && start !== null) {
      element.setSelectionRange(Math.min(start, value.length), Math.min(end, value.length));
    }
  }
}

function receiveDraft(selection, fields, revision) {
  if (!(selection in sharedDrafts) || revision < versions[selection]) return;
  versions[selection] = revision;
  for (const field of ['task', 'checkpoint']) {
    if (field in fields) sharedDrafts[selection][field] = fields[field];
  }
  syncDraftFields(selection);
}

function flushDraft(selection) {
  if (drains[selection]) return drains[selection];
  if (!Object.keys(pending[selection]).length) return Promise.resolve();
  drains[selection] = (async () => {
    while (Object.keys(pending[selection]).length) {
      const fields = pending[selection]; pending[selection] = {}; inFlight[selection] = fields;
      const response = await fetch('/api/dream/' + selection + '/draft', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...fields, client_id: clientId})
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error);
      receiveDraft(selection, result.fields, result.revision);
      inFlight[selection] = {};
      syncDraftFields(selection);
    }
  })().catch(error => {$('message').textContent = error.message;})
    .finally(() => {drains[selection] = null;});
  return drains[selection];
}

async function action(path, body = {}) {
  try {
    const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)});
    const result = await response.json();
    if (!response.ok) throw Error(result.error);
    $('message').textContent = ''; render(result); return result;
  } catch (error) { $('message').textContent = error.message; return null; }
}

function renderSelections() {
  const inputs = station.inputs || {}, both = station.combined_input;
  for (const side of ['left', 'right']) {
    for (const source of ['dream', 'so101']) {
      $(side + '-input-' + source).setAttribute('aria-pressed', String(!both && inputs[side] === source));
    }
  }
  $('both-input-dream').setAttribute('aria-pressed', String(both === 'dream'));
  $('so101-panel').hidden = !Object.values(inputs).includes('so101');
  for (const selection of selections) {
    $('api-' + selection + '-panel').hidden = selection === 'both' ? both !== 'dream' :
      !!both || inputs[selection] !== 'dream';
  }
  const count = document.querySelectorAll('#input-panels > article:not([hidden])').length;
  $('input-panels').hidden = count === 0;
  $('input-panels').classList.toggle('single', count === 1);
}

function render(state) {
  station = state;
  const arms = state.arms || {};
  for (const side of ['left', 'right']) {
    const arm = arms[side] || {};
    $(side + '-power-state').textContent = arm.arm_open ? 'ON' : arm.requested_power ? 'STARTING' : 'OFF';
    $(side + '-power-state').dataset.on = !!arm.arm_open;
    $(side + '-power').dataset.on = !!arm.arm_open;
    $(side + '-power').textContent = arm.arm_open ? 'Turn off / kill' : 'Turn on';
    $(side + '-home').textContent = arm.homing ? 'Returning…' : 'Return to 0';
    $(side + '-home').disabled = !!arm.homing; $(side + '-zero').disabled = !arm.arm_open;
    $(side + '-zero-note').textContent = arm.zero_joints ? 'Saved zero · retained across Kill and power-on' :
      'First power-on captures the initial zero';
    $(side + '-status').textContent = arm.message || '';
    if (!gripDragging[side] && arm.grip_torque_nm != null) {
      $(side + '-grip-torque').value = arm.grip_torque_nm;
      $(side + '-grip-value').textContent = Number(arm.grip_torque_nm).toFixed(2) + ' N·m';
    }
    const applied = arm.grip_applied_torque_nm == null ? 'pending power-on' : Number(arm.grip_applied_torque_nm).toFixed(2) + ' N·m';
    const measured = arm.grip_measured_torque_nm == null ? '—' : Math.abs(arm.grip_measured_torque_nm).toFixed(2) + ' N·m';
    $(side + '-grip-feedback').textContent = 'Applied: ' + applied + ' · Measured: ' + measured + (arm.grip_holding ? ' · HOLDING' : '');
  }
  for (const selection of selections) {
    const draft = (state.api_drafts || {})[selection];
    if (draft) receiveDraft(selection, draft, draft.revision);
    const api = (state.dream || {})[selection] || {};
    $('api-' + selection + '-run').disabled = !!api.busy;
    $('api-' + selection + '-call').disabled = !!api.busy;
    $('api-' + selection + '-status').textContent = (api.mode || 'idle') + ' · ' + (api.calls || 0) +
      ' calls · ' + (api.commands || 0) + ' rows sent';
    $('api-' + selection + '-error').textContent = api.error || api.observation_error || '';
    $('api-' + selection + '-result').textContent = JSON.stringify({observation: api.observation, rows: api.rows}, null, 2);
  }
  const assigned = !!state.assigned_side;
  $('side').textContent = assigned ? state.assigned_side + ' arm' : 'Not assigned';
  $('tracking-state').textContent = state.tracking ? 'TRACKED' : 'UNTRACKED';
  $('tracking-state').dataset.on = !!state.tracking;
  for (const id of ['track', 'untrack', 'scale']) $(id).disabled = !assigned;
  $('hint').textContent = 'Track captures fresh leader and arm anchors. Untrack holds position.';
  if (!dragging) { $('scale').value = state.scale; $('value').textContent = Number(state.scale).toFixed(2) + '×'; }
  $('record').disabled = !assigned || !state.arm_open;
  $('record').textContent = state.recording_requested ? 'Stop & save example' : 'Record example';
  $('reject').disabled = !assigned || !(state.recording || state.recording_requested || state.saved_episodes);
  $('reject').textContent = state.recording || state.recording_requested ? 'Reject current example' : 'Reject last saved example';
  $('record-status').textContent = [state.recording ? 'Recording · ' + state.frames + ' frames' : state.saved_episodes + ' saved examples',
    state.dataset_dir, state.recording_error].filter(Boolean).join(' · ');
  renderSelections();
}

for (const side of ['left', 'right']) {
  $(side + '-power').onclick = () => action('/api/arms/' + side + '/power', {enabled: !station.arms[side].arm_open});
  $(side + '-home').onclick = () => action('/api/arms/' + side + '/home');
  $(side + '-zero').onclick = () => action('/api/arms/' + side + '/zero');
  $(side + '-grip-torque').oninput = () => {gripDragging[side] = true;
    $(side + '-grip-value').textContent = Number($(side + '-grip-torque').value).toFixed(2) + ' N·m';};
  $(side + '-grip-torque').onchange = async () => {
    await action('/api/arms/' + side + '/grip-torque', {torque_nm: Number($(side + '-grip-torque').value)});
    gripDragging[side] = false; render(station);
  };
  for (const source of ['dream', 'so101']) {
    $(side + '-input-' + source).onclick = () => action('/api/arms/' + side + '/input', {
      source: !station.combined_input && station.inputs[side] === source ? null : source
    });
  }
}
$('both-input-dream').onclick = () => action('/api/both/input', {source: station.combined_input === 'dream' ? null : 'dream'});
$('track').onclick = () => action('/api/control', {tracking: true});
$('untrack').onclick = () => action('/api/control', {tracking: false});
$('record').onclick = () => action('/api/control', {recording_requested: !station.recording_requested});
$('reject').onclick = () => action('/api/reject');
$('scale').oninput = () => {dragging = true; $('value').textContent = Number($('scale').value).toFixed(2) + '×';};
$('scale').onchange = () => {dragging = false; action('/api/control', {scale: Number($('scale').value)});};

const events = new EventSource('/api/events');
events.onmessage = message => {
  const event = JSON.parse(message.data);
  if (event.type === 'draft') receiveDraft(event.selection, event.fields, event.revision);
  if (event.type === 'drafts') {
    for (const selection of selections) {
      const draft = event.api_drafts[selection]; receiveDraft(selection, draft, draft.revision);
    }
  }
  if (event.type === 'selection') {
    station.inputs = event.inputs; station.combined_input = event.combined_input; renderSelections();
  }
};
async function poll() {
  try {
    const response = await fetch('/api/state'); if (!response.ok) throw Error('Connection unavailable');
    render(await response.json()); $('connection').textContent = 'Connected';
  } catch (error) { $('connection').textContent = 'Disconnected'; }
}
function cameras() {
  for (const [id, role] of [['cam-left', 'left'], ['cam-top', 'overhead'], ['cam-right', 'right']]) {
    const image = $(id); if (image.dataset.loading === 'true') continue; image.dataset.loading = 'true';
    image.onload = image.onerror = () => {image.dataset.loading = 'false';};
    image.src = '/api/camera/' + role + '.jpg?t=' + Date.now();
  }
}
poll(); setInterval(poll, 500); cameras(); setInterval(cameras, 500);
