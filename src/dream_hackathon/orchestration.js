// Native draggable flow editor. The station owns execution; browser tabs only edit/view it.
(() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch =>
    ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[ch]));
  const clone = value => JSON.parse(JSON.stringify(value));
  const uid = () => crypto.randomUUID();
  const ports = {start:['next'], trigger:['next','timeout'], action:['next','error','timeout'],
    condition:['true','false'], loop:['body','done'], fork:['next'], join:['next'], end:[]};
  const labels = {start:'Go', trigger:'Trigger', action:'Operation', condition:'Condition', loop:'Loop',
    fork:'Parallel', join:'Join', end:'End'};
  const drafts = {}, saved = {}, dirty = {}, versions = {}, live = {};
  let selection = 'both', selected = null, selectedEdge = null, connecting = null, zoom = 1, drag = null;
  let editingJson = false;
  const graph = () => drafts[selection] || {nodes:[], edges:[]};
  const node = () => graph().nodes.find(item => item.id === selected);
  const defaults = operation => ({
    script: {operation:'script', argv:['python', 'your_script.py']},
    http: {operation:'http', method:'POST', url:'http://127.0.0.1:8000/execute', headers:{}, body:{}},
    dream: {operation:'dream', selection, task:'Describe the operation', checkpoint:'dreamlabs/dream-1-beta',
      run:true, durationSeconds:10},
    vlm: {operation:'vlm', provider:'openai', model:'gpt-6.1-terra',
      imageUrl:location.origin + '/api/camera/overhead.jpg', question:'Is the requested task visibly complete?',
      apiKeyEnv:'OPENAI_API_KEY'},
    emit: {operation:'emit', name:'event_name', value:true},
    set: {operation:'set', name:'variable_name', value:true}
  })[operation];
  const conditionDefault = kind => ({
    delay:{kind, seconds:1}, elapsed:{kind, seconds:10},
    script_end:{kind, name:'operation_node_id', count:1},
    signal:{kind, name:'gate_pass', path:'', op:'eq', value:true},
    vlm:{kind, name:'vlm_result', path:'verdict', op:'eq', value:'yes'},
    result:{kind, name:'operation_node_id', path:'verdict', op:'eq', value:'yes'},
    variable:{kind, name:'variable_name', path:'', op:'eq', value:true},
    all:{kind, conditions:[{kind:'signal', name:'deterministic_pass', value:true},
      {kind:'vlm', name:'vlm_result', path:'verdict', value:'yes'}]},
    any:{kind, conditions:[{kind:'delay', seconds:5}, {kind:'signal', name:'continue', value:true}]},
    not:{kind, condition:{kind:'signal', name:'finished', value:true}}
  })[kind];

  function tab(name) {
    $('control-tab').hidden = name !== 'control'; $('flow-tab').hidden = name !== 'flow';
    for (const key of ['control', 'flow']) $('tab-' + key).setAttribute('aria-selected', String(name === key));
  }
  $('tab-control').onclick = () => tab('control');
  $('tab-flow').onclick = () => {tab('flow'); draw();};
  document.querySelector('.system-tabs').onkeydown = event => {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
    const name = $('flow-tab').hidden ? 'flow' : 'control'; tab(name); $('tab-' + name).focus();
  };

  function changed() {
    dirty[selection] = true;
    $('flow-save-status').textContent = 'Unsaved edits · apply on the next Go';
  }

  function summary(item) {
    const c = item.config;
    if (item.type === 'action') return c.operation + '\n' +
      (c.task || c.question || c.url || (c.argv || []).join(' ') || c.name || '');
    if (item.type === 'trigger' || item.type === 'condition') return describeCondition(c.condition);
    if (item.type === 'loop') return c.mode === 'while' ? 'While ' + describeCondition(c.condition) :
      (c.count ?? 1) + ' repetitions · wire body back here';
    if (item.type === 'fork') return 'Start every connected next branch';
    if (item.type === 'join') return 'Wait for every incoming branch';
    if (item.type === 'start') return 'Starts when Go is pressed';
    return 'This branch finishes here';
  }
  function describeCondition(c = {}) {
    if (c.kind === 'delay') return 'Wait ' + c.seconds + 's after activation';
    if (c.kind === 'elapsed') return 'At ' + c.seconds + 's after Go';
    if (c.kind === 'script_end') return 'After script ' + c.name + ' ends';
    if (['all','any','not'].includes(c.kind)) return c.kind.toUpperCase() + ' · nested conditions';
    return [c.kind, c.name, c.path, c.op || 'eq', JSON.stringify(c.value)].filter(Boolean).join(' ');
  }

  function drawEdges() {
    const nodes = Object.fromEntries(graph().nodes.map(item => [item.id, item]));
    $('flow-edges').innerHTML = '<defs><marker id="flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:#7e98a8;stroke:none"/></marker></defs>' +
      graph().edges.map(edge => {
        const a = nodes[edge.source], b = nodes[edge.target]; if (!a || !b) return '';
        const outputs = ports[a.type], index = outputs.indexOf(edge.port);
        const x1 = a.x + 5 + 220 * (index + .5) / outputs.length, y1 = a.y + 140;
        const x2 = b.x + 115, y2 = b.y - 6, bend = Math.max(60, Math.abs(y2 - y1) / 2);
        return `<path data-edge="${esc(edge.id)}" class="${edge.id === selectedEdge ? 'selected' : ''}" d="M${x1},${y1} C${x1},${y1 + bend} ${x2},${y2 - bend} ${x2},${y2}" marker-end="url(#flow-arrow)"><title>${esc(a.label + ' / ' + edge.port + ' → ' + b.label)}</title></path>`;
      }).join('');
  }

  function draw() {
    $('flow-nodes').innerHTML = graph().nodes.map(item => `<article class="flow-node ${selected === item.id ? 'selected' : ''}" data-node="${esc(item.id)}" data-type="${item.type}" data-status="${esc(live[selection]?.nodes?.[item.id]?.status || '')}" style="left:${item.x}px;top:${item.y}px">
      ${item.type === 'start' ? '' : `<button class="flow-in" data-input="${esc(item.id)}" aria-label="Connect input to ${esc(item.label)}">●</button>`}
      <header data-drag="${esc(item.id)}"><small>${labels[item.type]}</small><strong>${esc(item.label)}</strong></header><div class="flow-summary">${esc(summary(item))}</div>
      <div class="flow-ports">${ports[item.type].map(port => `<button data-output="${esc(item.id)}" data-port="${port}" class="${connecting?.source === item.id && connecting?.port === port ? 'pending' : ''}" aria-label="${esc(item.label)} output ${port}">${port} ●</button>`).join('')}</div></article>`).join('');
    drawEdges();
    $('flow-connect-hint').textContent = connecting ? 'Choose a block’s input port to connect. Escape cancels.' :
      'Connect an output port to an input port. Drag block headers to arrange.';
  }

  function field(label, key, value, type='text') {
    return `<label>${esc(label)}<input type="${type}" data-config="${key}" value="${esc(value)}" ${type === 'number' ? 'min="0" step="any"' : ''}></label>`;
  }
  function textField(label, key, value, json=false) {
    return `<label>${esc(label)}<textarea data-config="${key}" ${json ? 'data-json="true"' : ''} rows="3">${esc(json ? JSON.stringify(value, null, 2) : value)}</textarea></label>`;
  }
  function selectField(label, key, value, options) {
    return `<label>${esc(label)}<select data-config="${key}">${options.map(option =>
      `<option value="${esc(option[0])}" ${option[0] === value ? 'selected' : ''}>${esc(option[1])}</option>`).join('')}</select></label>`;
  }
  function conditionFields(c) {
    let html = selectField('Trigger source', 'condition.kind', c.kind, [
      ['delay','Timer after block activation'], ['elapsed','Elapsed time since Go'], ['script_end','Script ended'],
      ['vlm','VLM input signal'], ['signal','Input signal'], ['result','Operation result'], ['variable','Variable'],
      ['all','All conditions (AND)'], ['any','Any condition (OR)'], ['not','Negate condition']]);
    if (['delay','elapsed'].includes(c.kind)) html += field('Seconds', 'condition.seconds', c.seconds, 'number');
    else if (c.kind === 'script_end') html += field('Script block ID', 'condition.name', c.name) +
      field('Completed execution count', 'condition.count', c.count ?? 1, 'number');
    else if (['all','any','not'].includes(c.kind)) html += '<p>Edit nested conditions in Configuration JSON below.</p>';
    else html += field(c.kind === 'result' ? 'Operation block ID' : 'Input / variable name', 'condition.name', c.name) +
      field('Value path (dot notation)', 'condition.path', c.path || '') +
      selectField('Comparison', 'condition.op', c.op || 'eq', ['eq','ne','gt','gte','lt','lte','contains','exists'].map(x => [x,x])) +
      textField('Expected value (JSON)', 'condition.value', c.value ?? true, true);
    return html;
  }

  function inspector() {
    const item = node(); editingJson = false;
    if (selectedEdge) {
      $('flow-inspector').innerHTML = '<h3>Connection</h3><p>Output → input</p><button id="flow-delete-edge" class="flow-danger">Delete connection</button>';
      $('flow-delete-edge').onclick = () => {graph().edges = graph().edges.filter(edge => edge.id !== selectedEdge);
        selectedEdge = null; changed(); draw(); inspector();};
      return;
    }
    if (!item) {$('flow-inspector').innerHTML = '<h3>Block settings</h3><p>Select a block or connection.</p>'; return;}
    const c = item.config;
    let html = `<h3>${labels[item.type]}</h3><p>Block ID: <code>${esc(item.id)}</code></p><label>Label<input id="flow-node-label" type="text" value="${esc(item.label)}"></label>`;
    if (item.type === 'trigger' || item.type === 'condition') html += conditionFields(c.condition);
    if (item.type === 'trigger') html += field('Time limit, seconds (blank = no limit)', 'timeoutSeconds', c.timeoutSeconds ?? '', 'number');
    if (item.type === 'loop') {
      html += selectField('Loop mode', 'mode', c.mode || 'count', [['count','Repeat N times'], ['while','While condition matches']]);
      html += c.mode === 'while' ? conditionFields(c.condition) : field('Repetitions', 'count', c.count ?? 1, 'number');
      html += '<p>Connect body to an operation chain and its last output back to this Loop. Connect done to the next phase.</p>';
    }
    if (item.type === 'action') {
      html += selectField('Execute', 'operation', c.operation, [['script','Hard script'], ['http','HTTP API'],
        ['dream','Dream API'], ['vlm','VLM verification'], ['emit','Emit signal'], ['set','Set variable']]);
      if (c.operation === 'script') html += textField('Command arguments (JSON array)', 'argv', c.argv, true) + field('Working directory', 'cwd', c.cwd || '');
      if (c.operation === 'http') html += field('URL', 'url', c.url) +
        selectField('Method', 'method', c.method || 'POST', ['GET','POST','PUT','PATCH','DELETE'].map(x => [x,x])) +
        textField('Headers (JSON)', 'headers', c.headers || {}, true) + textField('Body (JSON; null for no body)', 'body', c.body ?? null, true);
      if (c.operation === 'dream') html += selectField('Arms', 'selection', c.selection || selection,
        (selection === 'both' ? selections : [selection]).map(x => [x,x])) +
        textField('Task', 'task', c.task) + field('Checkpoint', 'checkpoint', c.checkpoint) +
        selectField('Execution', 'run', c.run === false ? 'false' : 'true', [['true','Play action window'],['false','Call once, inspect only']]) +
        field('Playback window (seconds)', 'durationSeconds', c.durationSeconds ?? 10, 'number');
      if (c.operation === 'vlm') html += selectField('Provider', 'provider', c.provider, [['openai','OpenAI'],['ollama','Ollama']]) +
        field('Model', 'model', c.model) + textField('Verification question', 'question', c.question) +
        field('Camera / image URL', 'imageUrl', c.imageUrl || '') + field('Or local image path', 'imagePath', c.imagePath || '') +
        field('API key environment variable', 'apiKeyEnv', c.apiKeyEnv || 'OPENAI_API_KEY') +
        field('Provider base URL (blank = default)', 'url', c.url || '');
      if (['emit','set'].includes(c.operation)) html += field('Name', 'name', c.name) + textField('Value (JSON)', 'value', c.value, true);
      if (['script','http','vlm'].includes(c.operation)) html += field('Time limit, seconds (optional)', 'timeoutSeconds', c.timeoutSeconds ?? '', 'number');
      html += '<p>Values can reference <code>{{results.block_id.verdict}}</code>, <code>{{signals.name}}</code>, <code>{{variables.name}}</code>, or <code>{{env.KEY}}</code>.</p>';
    }
    html += `<details><summary>Configuration JSON · nested conditions</summary><textarea id="flow-config-json" rows="12">${esc(JSON.stringify(c, null, 2))}</textarea><button id="flow-config-apply">Apply configuration</button></details>
      <p id="flow-field-error" class="error"></p><button id="flow-delete-node" class="flow-danger">Delete block</button>`;
    $('flow-inspector').innerHTML = html;
    $('flow-node-label').oninput = event => {item.label = event.target.value; changed(); draw();};
    $('flow-config-json').oninput = () => {editingJson = true;};
    $('flow-config-apply').onclick = () => {
      try {const value = JSON.parse($('flow-config-json').value); if (!value || Array.isArray(value) || typeof value !== 'object') throw Error('Use a configuration object');
        item.config = value; changed(); draw(); inspector();} catch (error) {$('flow-field-error').textContent = error.message;}
    };
    $('flow-delete-node').onclick = () => {graph().nodes = graph().nodes.filter(n => n.id !== item.id);
      graph().edges = graph().edges.filter(edge => edge.source !== item.id && edge.target !== item.id);
      selected = null; changed(); draw(); inspector();};
    $('flow-inspector').onchange = event => {
      const key = event.target.dataset.config; if (!key) return;
      try {
        let value = event.target.value;
        if (event.target.dataset.json) value = JSON.parse(value);
        else if (event.target.type === 'number') value = value === '' ? null : Number(value);
        if (key === 'operation') item.config = defaults(value);
        else if (key === 'condition.kind') item.config.condition = conditionDefault(value);
        else if (key === 'mode') {item.config.mode = value; if (value === 'while') item.config.condition ||= conditionDefault('signal');}
        else {
          if (key === 'run') value = value === 'true';
          const path = key.split('.'); let target = item.config;
          for (const part of path.slice(0,-1)) target = target[part];
          target[path.at(-1)] = value;
        }
        changed(); draw(); inspector();
      } catch (error) {$('flow-field-error').textContent = error.message;}
    };
  }

  $('flow-nodes').onclick = event => {
    if (drag?.moved) return;
    const output = event.target.closest('[data-output]');
    if (output) {connecting = {source:output.dataset.output, port:output.dataset.port}; draw(); return;}
    const input = event.target.closest('[data-input]');
    if (input && connecting) {
      graph().edges.push({id:uid(), ...connecting, target:input.dataset.input});
      connecting = null; changed(); draw(); return;
    }
    const block = event.target.closest('[data-node]'); if (!block) return;
    selected = block.dataset.node; selectedEdge = null; draw(); inspector();
  };
  $('flow-edges').onclick = event => {
    const edge = event.target.closest('[data-edge]'); if (!edge) return;
    selectedEdge = edge.dataset.edge; selected = null; drawEdges(); inspector();
  };
  document.addEventListener('keydown', event => {if (event.key === 'Escape') {connecting = null; draw();}});
  $('flow-nodes').onpointerdown = event => {
    const header = event.target.closest('[data-drag]'); if (!header) return;
    const item = graph().nodes.find(n => n.id === header.dataset.drag);
    selected = item.id; selectedEdge = null;
    drag = {item, startX:event.clientX, startY:event.clientY, x:item.x, y:item.y, moved:false};
    header.setPointerCapture(event.pointerId);
    header.onpointermove = move => {
      if (!drag) return;
      const dx = (move.clientX - drag.startX) / zoom, dy = (move.clientY - drag.startY) / zoom;
      drag.moved ||= Math.abs(dx) + Math.abs(dy) > 3;
      if (!drag.moved) return;
      item.x = Math.max(0, Math.min(2350, drag.x + dx)); item.y = Math.max(20, Math.min(1610, drag.y + dy));
      header.parentElement.style.left = item.x + 'px'; header.parentElement.style.top = item.y + 'px'; drawEdges();
    };
    header.onpointerup = () => {if (drag?.moved) changed(); header.onpointermove = null;
      setTimeout(() => {drag = null; draw(); inspector();}, 0);};
  };

  document.querySelectorAll('[data-add-block]').forEach(button => {button.onclick = () => {
    const type = button.dataset.addBlock;
    const config = type === 'action' ? defaults('emit') : ['trigger','condition'].includes(type) ?
      {condition:conditionDefault('delay')} : type === 'loop' ? {mode:'count', count:3} : {};
    const item = {id:type + '_' + uid().slice(0,8), type, label:labels[type],
      x:60 + $('flow-viewport').scrollLeft / zoom, y:80 + $('flow-viewport').scrollTop / zoom + (graph().nodes.length % 3) * 170, config};
    graph().nodes.push(item); selected = item.id; selectedEdge = null; changed(); draw(); inspector();
  };});

  function switchSelection(value) {
    selection = value; selected = null; selectedEdge = null; connecting = null;
    $('flow-selection').value = value; draw(); inspector(); runtime();
    $('flow-save-status').textContent = dirty[value] ? 'Unsaved edits · apply on the next Go' : 'Saved flow';
  }
  $('flow-selection').onchange = event => switchSelection(event.target.value);
  async function save(which = selection) {
    if (!drafts[which]) return true;
    if (which === selection && editingJson) {$('message').textContent = 'Apply configuration JSON before saving'; return false;}
    try {
      const definition = clone(drafts[which]);
      const response = await fetch('/api/flows/' + which + '/save', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({definition, revision:versions[which]})});
      const result = await response.json(); if (!response.ok) throw Error(result.error);
      const unchanged = JSON.stringify(drafts[which]) === JSON.stringify(definition);
      versions[which] = result.flows[which].revision;
      if (unchanged) dirty[which] = false;
      window.renderFlowState(result.flows);
      if (which === selection) $('flow-save-status').textContent = unchanged ? 'Saved · shared with connected browsers' : 'Unsaved newer edits';
      return unchanged;
    } catch (error) {$('message').textContent = error.message; return false;}
  }
  $('flow-save').onclick = () => save();
  $('flow-reload').onclick = () => {if (!saved[selection]) return; drafts[selection] = clone(saved[selection]); dirty[selection] = false; switchSelection(selection);};
  $('flow-export').onclick = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(graph(), null, 2)], {type:'application/json'}));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'flow-' + selection + '.json'; anchor.click(); URL.revokeObjectURL(url);
  };
  $('flow-import').onclick = () => $('flow-import-file').click();
  $('flow-import-file').onchange = async event => {
    try {const value = JSON.parse(await event.target.files[0].text()); if (!Array.isArray(value.nodes) || !Array.isArray(value.edges)) throw Error('Expected nodes and edges arrays');
      drafts[selection] = value; selected = null; selectedEdge = null; changed(); draw(); inspector();
    } catch (error) {$('message').textContent = error.message;} finally {event.target.value = '';}
  };
  $('flow-example').onclick = () => {
    drafts[selection] = {nodes:[
      {id:'go', type:'start', label:'Go', x:80,y:80,config:{}},
      {id:'repeat',type:'loop',label:'Repeat three times',x:400,y:80,config:{mode:'count',count:3}},
      {id:'operation',type:'action',label:'Replace with script or API',x:740,y:310,config:{operation:'emit',name:'iteration',value:true}},
      {id:'end',type:'end',label:'Finished',x:740,y:80,config:{}}
    ],edges:[
      {id:uid(),source:'go',port:'next',target:'repeat'}, {id:uid(),source:'repeat',port:'body',target:'operation'},
      {id:uid(),source:'operation',port:'next',target:'repeat'}, {id:uid(),source:'repeat',port:'done',target:'end'}
    ]}; selected=null; selectedEdge=null; changed(); draw(); inspector();
  };
  function setZoom(value) {zoom = Math.max(.4, Math.min(1.5, value)); $('flow-canvas').style.transform = `scale(${zoom})`;
    $('flow-zoom-label').textContent = Math.round(zoom * 100) + '%';}
  $('flow-zoom-in').onclick = () => setZoom(zoom + .1); $('flow-zoom-out').onclick = () => setZoom(zoom - .1);
  $('flow-signal-send').onclick = async () => {
    try {const value = JSON.parse($('flow-signal-value').value); await action('/api/flows/' + selection + '/signal',
      {name:$('flow-signal-name').value, value});} catch (error) {$('message').textContent = error.message;}
  };

  function runtime() {
    const state = live[selection] || {};
    $('flow-run-status').textContent = (state.state || 'idle') + ' · ' + Number(state.elapsedSeconds || 0).toFixed(1) +
      's since Go · ' + (state.error || 'Go / Stop is on the Controller tab under Full Auto.');
    $('flow-context').textContent = JSON.stringify(state.context || {}, null, 2);
    for (const el of document.querySelectorAll('#flow-nodes [data-node]')) el.dataset.status = state.nodes?.[el.dataset.node]?.status || '';
    for (const which of selections) {
      const run = live[which] || {};
      $('auto-' + which + '-status').textContent = (run.state || 'idle') + ' · ' + Number(run.elapsedSeconds || 0).toFixed(1) + 's since Go';
      $('auto-' + which + '-error').textContent = run.error || '';
      $('auto-' + which + '-go').disabled = !!run.busy;
    }
  }
  for (const which of selections) {
    const panel = $('auto-' + which + '-panel'); panel.append($('auto-panel-template').content.cloneNode(true));
    for (const el of panel.querySelectorAll('[data-auto-field]')) el.id = 'auto-' + which + '-' + el.dataset.autoField;
    $('auto-' + which + '-title').textContent = 'Full Auto · ' + which;
    $('auto-' + which + '-edit').onclick = () => {switchSelection(which); tab('flow');};
    $('auto-' + which + '-go').onclick = async () => {
      if (dirty[which] && !await save(which)) return;
      await action('/api/flows/' + which + '/go');
    };
    $('auto-' + which + '-stop').onclick = () => action('/api/flows/' + which + '/stop');
  }
  window.renderFlowState = states => {
    if (!states) return;
    let redraw = false;
    for (const which of selections) {
      const state = states[which]; if (!state) continue;
      const prior = versions[which] ?? -1;
      if (state.revision < prior) continue;
      live[which] = state; saved[which] = clone(state.definition);
      if (!dirty[which] && (!drafts[which] || state.revision !== prior)) {
        drafts[which] = clone(state.definition); if (which === selection) redraw = true;
      }
      // Keep the edit's base revision until explicitly reloaded after a concurrent change.
      if (!dirty[which] || state.revision === prior) versions[which] = state.revision;
    }
    if (redraw && !drag) {draw(); inspector();}
    runtime();
  };
  if (station.flows) window.renderFlowState(station.flows);
})();
