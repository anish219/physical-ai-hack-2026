// The bridge serves this app on the NUC; an SSH tunnel keeps it same-origin.
const requestedMode = new URLSearchParams(location.search).get('bridge');
async function detectBridge() {
  if (requestedMode === '0') return false;
  if (requestedMode === '1') return true;
  try {
    const response = await fetch('/api/status', { cache: 'no-store', signal: AbortSignal.timeout(2000) });
    if (!response.ok) return false;
    const status = await response.json();
    return status.side === 'right' && ['mock', 'hardware', 'existing-service'].includes(status.mode);
  } catch { return false; }
}
export const bridgeEnabled = await detectBridge();
export async function bridgeRequest(path, method = 'GET', body) {
  const response = await fetch(`/api/${path}`, {
    method, body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store', signal: AbortSignal.timeout(5000),
    headers: method === 'POST' ? { 'X-Leonard-Client': '1', 'Content-Type': 'application/json' } : {}
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Bridge HTTP ${response.status}`);
  return payload;
}

export async function cameraObservations() {
  return Promise.all(['wrist_l', 'wrist_r', 'overhead'].map(async (role, id) => {
    const url = `/api/cameras/${role}/frame.jpg`;
    try {
      const response = await fetch(url, { cache: 'no-store', signal: AbortSignal.timeout(3000) });
      if (!response.ok) throw new Error('Camera unavailable');
      const blob = await response.blob();
      const frame = await new Promise((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result);
        reader.onerror = reject; reader.readAsDataURL(blob);
      });
      return { id, role, frame, timestamp: Number(response.headers.get('X-Capture-Timestamp')),
        observationId: response.headers.get('X-Observation-Id') };
    } catch (error) { return { id, role, frame: null, timestamp: null, error: error.message }; }
  }));
}
