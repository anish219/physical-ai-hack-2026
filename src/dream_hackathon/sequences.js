// Phase/subphase editor adapted from the preserved Leonard sequence editor.
(() => {
  const definitions = {}, latest = {}, pending = {}, saving = {}, revisions = {};
  const clone = value => JSON.parse(JSON.stringify(value));
  const id = () => crypto.randomUUID();
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char =>
    ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const element = (selection, field) => $('sequence-' + selection + '-' + field);

  function markup(definition) {
    return definition.phases.map((phase, phaseIndex) => `<section class="sequence-phase" data-phase="${escape(phase.id)}">
      <div class="sequence-row"><span>${phaseIndex + 1}.</span><input type="text" data-field="title" aria-label="Phase title" value="${escape(phase.title)}">
      <button data-action="remove-phase" aria-label="Remove phase">×</button></div>
      ${(phase.steps || []).map(step => `<div class="sequence-step" data-step="${escape(step.id)}">
      <div class="sequence-row"><input type="text" data-field="label" aria-label="Subphase name" value="${escape(step.label)}">
      <select data-field="type" aria-label="Subphase type"><option value="dream" ${step.type === 'dream' ? 'selected' : ''}>Dream task</option>
      <option value="wait" ${step.type === 'wait' ? 'selected' : ''}>Wait / checkpoint</option></select>
      <button data-action="remove-step" aria-label="Remove subphase">×</button></div>
      ${step.type === 'dream' ? `<textarea data-field="prompt" aria-label="Dream task prompt" placeholder="Describe this subphase">${escape(step.prompt)}</textarea>
      <div class="sequence-options"><label>Checkpoint<input type="text" data-field="modelId" aria-label="Sequence checkpoint" value="${escape(step.modelId)}"></label>
      <label>Window (s)<input type="number" data-field="durationSeconds" aria-label="Action window seconds" min="1" max="300" step="0.1" value="${escape(step.durationSeconds)}"></label></div>` :
      `<label>Wait (seconds)<input type="number" data-field="durationSeconds" aria-label="Wait seconds" min="0" max="300" step="0.1" value="${escape(step.durationSeconds)}"></label>`}
      </div>`).join('')}
      <button data-action="add-step">+ Add subphase</button></section>`).join('') || '<p>No phases yet. Add a phase to begin.</p>';
  }

  function structure(definition) {
    return JSON.stringify(definition.phases.map(phase => [phase.id, phase.steps.map(step => [step.id, step.type])]));
  }

  function redraw(selection, previous) {
    const editor = element(selection, 'editor'), definition = definitions[selection];
    if (previous && structure(previous) === structure(definition)) {
      // Keep focused text nodes and caret positions during shared edits.
      for (const node of editor.querySelectorAll('[data-field]')) {
        const phaseId = node.closest('[data-phase]').dataset.phase;
        const stepId = node.closest('[data-step]')?.dataset.step;
        const phase = definition.phases.find(value => value.id === phaseId);
        const target = stepId ? phase.steps.find(value => value.id === stepId) : phase;
        const value = String(target[node.dataset.field] ?? '');
        if (node.value === value) continue;
        const start = node.selectionStart, end = node.selectionEnd;
        node.value = value;
        if (node === document.activeElement && start != null) {
          node.setSelectionRange(Math.min(start, value.length), Math.min(end, value.length));
        }
      }
    } else editor.innerHTML = markup(definition);
  }

  function applySaved(selection) {
    if (saving[selection] || pending[selection] || !latest[selection]) return;
    const previous = definitions[selection];
    definitions[selection] = clone(latest[selection].definition);
    redraw(selection, previous);
  }

  async function save(selection) {
    pending[selection] = clone(definitions[selection]);
    if (saving[selection]) return saving[selection];
    let failed = false;
    saving[selection] = (async () => {
      while (pending[selection]) {
        const definition = pending[selection]; pending[selection] = null;
        element(selection, 'saved').textContent = 'Saving shared sequence…';
        const response = await fetch('/api/sequences/' + selection + '/save', {
          method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({definition})
        });
        const result = await response.json();
        if (!response.ok) throw Error(result.error);
        window.renderSequenceState(result.sequences);
        element(selection, 'saved').textContent = 'Saved · shared with everyone connected';
      }
      return true;
    })().catch(error => {failed = true; element(selection, 'saved').textContent = error.message; return false;})
      .finally(() => {saving[selection] = null; if (!failed) applySaved(selection);});
    return saving[selection];
  }

  for (const selection of selections) {
    const panel = $('sequence-' + selection + '-panel');
    panel.append($('sequence-panel-template').content.cloneNode(true));
    for (const node of panel.querySelectorAll('[data-sequence-field]')) {
      node.id = 'sequence-' + selection + '-' + node.dataset.sequenceField;
    }
    element(selection, 'title').textContent = 'Sequences · ' +
      (selection === 'both' ? 'Both arms' : selection === 'left' ? 'Left arm' : 'Right arm');
    definitions[selection] = {phases: []}; revisions[selection] = -1;
    element(selection, 'editor').innerHTML = markup(definitions[selection]);
    element(selection, 'add-phase').onclick = () => {
      definitions[selection].phases.push({id:id(), title:'Phase ' + (definitions[selection].phases.length + 1), steps:[]});
      redraw(selection); save(selection);
    };
    panel.addEventListener('input', event => {
      const field = event.target.dataset.field;
      if (!field || field === 'type') return;
      const phase = definitions[selection].phases.find(value => value.id === event.target.closest('[data-phase]').dataset.phase);
      const stepId = event.target.closest('[data-step]')?.dataset.step;
      const target = stepId ? phase.steps.find(value => value.id === stepId) : phase;
      target[field] = field === 'durationSeconds' ? Number(event.target.value) : event.target.value;
      save(selection);
    });
    panel.addEventListener('change', event => {
      if (event.target.dataset.field !== 'type') return;
      const phase = definitions[selection].phases.find(value => value.id === event.target.closest('[data-phase]').dataset.phase);
      const step = phase.steps.find(value => value.id === event.target.closest('[data-step]').dataset.step);
      step.type = event.target.value;
      if (step.type === 'dream' && step.durationSeconds < 1) step.durationSeconds = 20;
      redraw(selection); save(selection);
    });
    panel.addEventListener('click', event => {
      const button = event.target.closest('[data-action]'); if (!button) return;
      const phase = definitions[selection].phases.find(value => value.id === button.closest('[data-phase]')?.dataset.phase);
      if (button.dataset.action === 'remove-phase') {
        definitions[selection].phases = definitions[selection].phases.filter(value => value.id !== phase.id);
      } else if (button.dataset.action === 'add-step') {
        phase.steps.push({id:id(), type:'dream', label:'Dream task', prompt:'', modelId:'dreamlabs/dream-1-beta', durationSeconds:20});
      } else if (button.dataset.action === 'remove-step') {
        phase.steps = phase.steps.filter(value => value.id !== button.closest('[data-step]').dataset.step);
      }
      redraw(selection); save(selection);
    });
    element(selection, 'run').onclick = async () => {
      if (await save(selection) === false) return;
      await action('/api/sequences/' + selection + '/run');
    };
    element(selection, 'stop').onclick = () => action('/api/sequences/' + selection + '/stop');
  }

  window.renderSequenceState = states => {
    for (const selection of selections) {
      const state = states?.[selection]; if (!state) continue;
      if (state.revision > revisions[selection]) {
        revisions[selection] = state.revision; latest[selection] = state; applySaved(selection);
      }
      element(selection, 'run').disabled = !!state.busy;
      element(selection, 'status').textContent = [state.state || 'idle', state.phase_title, state.step_label,
        state.running ? Number(state.elapsedSeconds).toFixed(1) + ' / ' + state.durationSeconds + ' s' : ''].filter(Boolean).join(' · ');
      element(selection, 'error').textContent = state.error || '';
      for (const [phaseIndex, phase] of [...element(selection, 'editor').querySelectorAll('[data-phase]')].entries()) {
        for (const [stepIndex, step] of [...phase.querySelectorAll('[data-step]')].entries()) {
          step.classList.toggle('active', !!state.running && phaseIndex === state.phase_index && stepIndex === state.step_index);
          step.classList.toggle('complete', state.state === 'complete' || state.phase_index > phaseIndex ||
            (state.phase_index === phaseIndex && state.step_index > stepIndex));
        }
      }
    }
  };
  if (station.sequences) window.renderSequenceState(station.sequences);
})();
