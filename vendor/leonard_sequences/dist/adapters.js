import { bridgeEnabled, bridgeRequest, cameraObservations } from './bridge-client.js';
// ?bridge=1 uses the local NUC bridge for observations only.
export const sources = {
  cameras: ['Arm 01 · wrist', 'Arm 02 · wrist', 'Workspace · overview'],
  async readCameras() { return bridgeEnabled ? cameraObservations() : this.cameras.map((label, id) => ({ id, label, frame: null, timestamp: null })); },
  async readMotorPositions() { return bridgeEnabled ? bridgeRequest('telemetry') : { arm01: [0, -35, 65, 0, 45, 0], arm02: [0, -35, 65, 0, 45, 0], timestamp: null }; },
  async infer(context) { if (bridgeEnabled) throw new Error('Observation-only bridge: Dream control is not connected'); return { simulated: true, modelId: context.modelId, commands: [] }; },
  async sendCommands(commands, _context) { if (bridgeEnabled) throw new Error('Movement disabled'); return { simulated: true, accepted: false, commands }; },
  // Resolve only when queued actions are cancelled and both arms are held.
  async stopCommands(_context) { return { simulated: true }; },
  // One narrow image predicate per phase. Replace with a call through your backend;
  // never put a VLM API key in this frontend. observationId identifies the source
  // image, not the request, so repeated analysis of one frame cannot confirm success.
  async verifyCondition(observation) {
    return { condition: observation.condition, result: 'uncertain', observationId: null, evidence: 'Image verifier not connected' };
  },
  // Implement criteria later; caller supplies a synchronized observation snapshot.
  evaluateFailure(_observation) { return null; },
  async recover(checkpoint) { return { simulated: true, completed: false, checkpoint }; },
  async verifyRecovery(_observation, _recovery) { return false; }
};
