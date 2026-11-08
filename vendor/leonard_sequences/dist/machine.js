// Pure orchestration layer. Adapters own transport, cancellation and hardware execution.
export const challenges = [
  { name: 'Unplug USB', description: 'Grasp, extract, release', phases: ['Locate & grasp', 'Extract USB', 'Release'], models: ['usb-grasp-v1', 'usb-extract-v1', 'usb-release-v1'], conditions: ['usb_grasped', 'usb_unplugged', 'usb_released'] },
  { name: 'Stack blocks', description: 'Pick, align, stack', phases: ['Pick block', 'Align block', 'Place block'], models: ['blocks-pick-v1', 'blocks-align-v1', 'blocks-place-v1'], conditions: ['block_grasped', 'block_aligned', 'block_stacked'] },
  { name: 'Sort colored balls', description: 'Identify, pick, sort', phases: ['Identify color', 'Pick ball', 'Place in basket'], models: ['balls-color-v1', 'balls-pick-v1', 'balls-sort-v1'], conditions: ['ball_color_identified', 'ball_grasped', 'ball_in_correct_basket'] }
];

export class Orchestrator {
  constructor(adapters, { onChange = () => {}, log = () => {}, now = () => performance.now(), confirmations = 2, maxAttempts = 3 } = {}) {
    this.adapters = adapters; this.onChange = onChange; this.log = log; this.now = now;
    this.confirmations = confirmations; this.maxAttempts = maxAttempts; this.epoch = 0;
    this.state = { task: 0, phase: 0, status: 'Idle', elapsed: 0, totalElapsed: 0, attempt: 1, done: new Set(), checkpoint: null, verification: null, confirmations: 0 };
    this.totalStartedAt = null; this.runningSince = null; this.seen = new Set();
  }
  current() { return challenges[this.state.task]; }
  elapsed() { return this.state.elapsed + (this.runningSince === null ? 0 : this.now() - this.runningSince); }
  totalElapsed() { return this.state.totalElapsed + (this.totalStartedAt === null ? 0 : this.now() - this.totalStartedAt); }
  finishRun() { this.state.totalElapsed = this.totalElapsed(); this.totalStartedAt = null; }
  emit() { this.onChange(this.state); }
  valid(epoch) { return epoch === this.epoch && this.state.status === 'Running'; }
  freeze(status) { this.state.elapsed = this.elapsed(); this.runningSince = null; this.state.status = status; this.epoch++; this.state.confirmations = 0; this.emit(); }
  async hold(status, reason) {
    this.freeze('Stopping'); const epoch = this.epoch;
    try { await this.adapters.stopCommands({ reason }); if (epoch !== this.epoch) return false; this.state.status = status; this.emit(); return true; }
    catch (error) { this.state.status = 'Blocked'; this.log(`Unable to confirm command stop: ${error.message}`); this.emit(); return false; }
  }
  async start() {
    if (!['Idle', 'Paused'].includes(this.state.status)) return;
    if (!this.current().models[this.state.phase]) { this.log('Assign a model before starting'); return; }
    this.epoch++; this.seen.clear(); this.state.confirmations = 0;
    if (this.totalStartedAt === null) this.totalStartedAt = this.now();
    this.state.status = 'Running'; this.runningSince = this.now(); this.log(`Running · ${this.current().phases[this.state.phase]}`); this.emit();
  }
  async pause() { if (this.state.status === 'Running') await this.hold('Paused', 'Operator pause'); }
  async snapshot() {
    const [cameras, motors] = await Promise.all([this.adapters.readCameras(), this.adapters.readMotorPositions()]);
    return { cameras, motors, cycleTimeMs: Math.round(this.elapsed()), task: this.state.task, phase: this.state.phase, modelId: this.current().models[this.state.phase], condition: this.current().conditions[this.state.phase], attempt: this.state.attempt };
  }
  // Run independently from monitorTick: inference must never block observation checks.
  async controlTick() {
    if (this.controlBusy || this.state.status !== 'Running') return;
    this.controlBusy = true; const epoch = this.epoch;
    try {
      const observation = await this.snapshot(); if (!this.valid(epoch)) return;
      const prediction = await this.adapters.infer(observation); if (!this.valid(epoch)) return;
      await this.adapters.sendCommands(prediction.commands, { epoch, task: observation.task, phase: observation.phase });
    } catch (error) { if (this.valid(epoch)) await this.hold('Paused', `Control error: ${error.message}`); }
    finally { this.controlBusy = false; }
  }
  async monitorTick() {
    if (this.monitorBusy || this.state.status !== 'Running') return;
    this.monitorBusy = true; const epoch = this.epoch;
    try {
      const observation = await this.snapshot(); if (!this.valid(epoch)) return;
      const failure = await this.adapters.evaluateFailure(observation); if (!this.valid(epoch)) return;
      if (failure) { await this.fail(failure.reason || String(failure), observation); return; }
      const result = await this.adapters.verifyCondition(observation); if (!this.valid(epoch)) return;
      const wellFormed = result && result.condition === observation.condition && ['yes','no','uncertain'].includes(result.result);
      this.state.verification = wellFormed ? result : { condition: observation.condition, result: 'uncertain', evidence: 'Invalid verifier response' };
      // A "no" means the success predicate is not met, not that an attempt failed.
      const fresh = wellFormed && typeof result.observationId === 'string' && !this.seen.has(result.observationId);
      if (fresh) this.seen.add(result.observationId);
      if (!wellFormed || result.result !== 'yes') this.state.confirmations = 0;
      else if (fresh) this.state.confirmations++;
      this.emit();
      if (this.state.confirmations >= this.confirmations) await this.advance();
    } catch (error) {
      if (this.valid(epoch)) { this.state.confirmations = 0; this.state.verification = { condition: this.current().conditions[this.state.phase], result: 'uncertain', evidence: error.message }; await this.hold('Paused', 'Monitor unavailable'); }
    } finally { this.monitorBusy = false; }
  }
  async advance() {
    if (!await this.hold('Transitioning', 'Verified phase completion')) return;
    const s = this.state;
    this.log(`Verified · ${this.current().conditions[s.phase]}`);
    if (s.phase < this.current().phases.length - 1) s.phase++;
    else { s.done.add(s.task); if (s.task === challenges.length - 1) { s.status = 'Complete'; this.finishRun(); this.emit(); return; } s.task++; s.phase = 0; s.elapsed = 0; }
    s.attempt = 1; s.checkpoint = null; s.verification = null; s.status = 'Idle'; await this.start();
  }
  async fail(reason, observation = null) {
    if (!['Running','Paused'].includes(this.state.status)) return;
    if (!await this.hold('Stopping', reason)) return;
    this.state.checkpoint = observation || { task: this.state.task, phase: this.state.phase, modelId: this.current().models[this.state.phase], cycleTimeMs: this.elapsed() };
    this.log(`Failure · ${reason}`);
    if (this.state.attempt >= this.maxAttempts) { this.state.status = 'Blocked'; this.log('Attempt limit reached'); this.emit(); return; }
    this.state.status = 'Recovering'; const epoch = this.epoch; this.emit();
    try {
      const result = await this.adapters.recover(structuredClone(this.state.checkpoint)); if (epoch !== this.epoch) return;
      const verified = result?.completed === true && await this.adapters.verifyRecovery(await this.snapshot(), result);
      if (epoch !== this.epoch) return;
      if (verified !== true) { this.state.status = 'Recovery required'; this.log('Recovery awaits verification'); this.emit(); return; }
      this.state.attempt++; this.state.verification = null; this.state.status = 'Idle'; this.log('Recovery verified · retrying phase'); await this.start();
    } catch (error) { if (epoch === this.epoch) { this.state.status = 'Recovery required'; this.log(`Recovery error: ${error.message}`); this.emit(); } }
  }
  async reset() {
    if (!await this.hold('Resetting', 'Reset run')) return;
    Object.assign(this.state, { task: 0, phase: 0, elapsed: 0, totalElapsed: 0, attempt: 1, status: 'Idle', done: new Set(), checkpoint: null, verification: null, confirmations: 0 });
    this.totalStartedAt = null; this.seen.clear(); this.log('Run reset · model routes retained'); this.emit();
  }
}
