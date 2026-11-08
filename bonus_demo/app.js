/* Standalone clients: no robot/controller endpoints or commands. */
const viewer = new URLSearchParams(location.search).has('viewer');
const $ = id => document.getElementById(id);
const room = new LivekitClient.Room({ adaptiveStream: false, dynacast: false });
let recordingTimer;
let lastSavedFrame;

function showError(error) {
  $('error').textContent = String(error.message || error);
}

async function event(type, details = {}) {
  const data = { type, role: viewer ? 'viewer' : 'publisher', ...details };
  $('log').textContent = JSON.stringify(data) + '\n' + $('log').textContent;
  const response = await fetch('/api/event', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error(`Dimensional event write failed: ${response.status}`);
}

async function connect() {
  if (room.state === LivekitClient.ConnectionState.Connected) return;
  const role = viewer ? 'viewer' : 'publisher';
  const response = await fetch(`/api/token?role=${role}`);
  const config = await response.json();
  await room.connect(config.url, config.token);
  await event('room_connected', { identity: room.localParticipant.identity, room: config.room });
}

async function saveReceivedFrame() {
  const video = $('video');
  if (!video.videoWidth || video.readyState < 2) return;
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const jpeg = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.75));
  const response = await fetch('/api/frame', {
    method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: jpeg,
  });
  if (!response.ok) throw new Error(`Dimensional frame write failed: ${response.status}`);
}

room.on(LivekitClient.RoomEvent.ConnectionStateChanged, state => {
  $('status').textContent = `${viewer ? 'Spectator' : 'Publisher'} · ${state}`;
});
room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
  if (track.kind !== LivekitClient.Track.Kind.Video) return;
  track.attach($('video'));
  $('media').textContent = `Receiving ${publication.trackSid} from ${participant.identity}`;
  clearInterval(recordingTimer);
  recordingTimer = setInterval(() => saveReceivedFrame().catch(showError), 5000);
  event('video_received', { track: publication.trackSid, publisher: participant.identity }).catch(showError);
});
room.on(LivekitClient.RoomEvent.TrackUnsubscribed, track => {
  if (track.kind !== LivekitClient.Track.Kind.Video) return;
  track.detach($('video'));
  clearInterval(recordingTimer);
  $('media').textContent = 'Publisher stopped sharing.';
  event('video_ended').catch(showError);
});
room.on(LivekitClient.RoomEvent.LocalTrackUnpublished, publication => {
  $('share').disabled = false;
  $('stop').disabled = true;
  $('media').textContent = 'Sharing ended.';
  event('sharing_ended', { track: publication.trackSid }).catch(showError);
});

$('share').onclick = async () => {
  $('error').textContent = '';
  $('share').disabled = true;
  try {
    await connect();
    const publication = await room.localParticipant.setScreenShareEnabled(true, { audio: false });
    if (publication) {
      publication.track.attach($('video'));
      $('stop').disabled = false;
      $('media').textContent = `Publishing ${publication.trackSid} · open the spectator to view and record it.`;
      await event('sharing_started', { track: publication.trackSid });
    } else {
      $('share').disabled = false;
    }
  } catch (error) {
    $('share').disabled = false;
    showError(error);
  }
};
$('stop').onclick = () => room.localParticipant.setScreenShareEnabled(false).catch(showError);

async function refreshRecording() {
  const response = await fetch('/api/status');
  const data = await response.json();
  $('recording').textContent = `${data.events} events · ${data.frames} received frames saved`;
  $('database').textContent = data.database;
  if (data.latest_frame_id && data.latest_frame_id !== lastSavedFrame) {
    $('saved').src = `/api/latest.jpg?id=${data.latest_frame_id}`;
    $('saved-link').hidden = false;
    lastSavedFrame = data.latest_frame_id;
  }
}

$('publisher-controls').hidden = viewer;
$('description').textContent = viewer
  ? 'Spectator: receiving the shared tab through LiveKit and recording received frames with Dimensional.'
  : 'Share the existing robot dashboard tab, then open the spectator. Both services run on this computer.';
if (viewer) connect().catch(showError);
refreshRecording().catch(showError);
setInterval(() => refreshRecording().catch(showError), 2000);
