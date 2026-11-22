import {requestJSON, waitForAnalysis, speedLabel, validateResult} from './api-client.js';

const $ = id => document.getElementById(id);
const screens = {upload: $('uploadScreen'), processing: $('processingScreen'), case: $('caseScreen')};
const player = $('audioPlayer');
const API = window.ACOUSTICASE_API_BASE || window.location.origin;
const assetURL = path => new URL(path, API).href;
const TYPE_LABEL = {measurement: 'Measurement', model_suggestion: 'Model suggestion', conditional_inference: 'Conditional inference'};
const COLORS = {measurement: '#d47b4e', model_suggestion: '#8270ce', conditional_inference: '#36957a'};
let currentCase = null, waveform = null, track = null, soundScores = null;
let activeRequest = null, selectedEvent = null, view = [0, 1], zoomed = false, pendingSeek = null, toastTimer;

function formatTime(value) {
  const tenths = Math.round(Math.max(0, Number.isFinite(value) ? value : 0) * 10);
  return `${String(Math.floor(tenths / 600)).padStart(2, '0')}:${((tenths % 600) / 10).toFixed(1).padStart(4, '0')}`;
}
function switchScreen(name) {
  Object.entries(screens).forEach(([key, element]) => element.classList.toggle('hidden', key !== name));
  window.scrollTo(0, 0);
}
function showToast(message) {
  clearTimeout(toastTimer); $('toast').textContent = message; $('toast').classList.add('show');
  toastTimer = setTimeout(() => { $('toast').classList.remove('show'); $('toast').textContent = ''; }, 5000);
}
function setBusy(busy) { for (const id of ['audioInput', 'openDemoButton', 'openSimulationButton']) $(id).disabled = busy; }
function showError(error) {
  switchScreen('upload'); $('uploadError').textContent = error.message || 'Could not connect to the analysis server.';
  $('uploadError').classList.remove('hidden'); $('statusText').textContent = 'Analysis needs attention';
}
function newRecording() {
  activeRequest?.abort(); activeRequest = null; setBusy(false); player.pause();
  player.removeAttribute('src'); player.load(); pendingSeek = null; currentCase = null;
  for (const id of ['simulatedOption', 'stationaryOption', 'constantToneOption']) $(id).checked = false;
  clearTimeout(toastTimer); $('toast').classList.remove('show'); $('toast').textContent = '';
  $('audioInput').value = ''; $('uploadError').classList.add('hidden');
  switchScreen('upload'); $('statusText').textContent = 'Ready for a recording';
}

async function analyze(file, options, controller) {
  if (!file.size) throw new Error('Choose a non-empty recording.');
  if (file.size > 100 * 1024 * 1024) throw new Error('Choose a recording smaller than 100 MiB.');
  const signal = controller.signal;
  switchScreen('processing'); $('processingTitle').textContent = 'Uploading recording…';
  $('processingDescription').textContent = 'Preparing the recording for analysis.'; $('progressBar').style.width = '0%';
  $('statusText').textContent = 'Analyzing recording';
  const form = new FormData(); form.append('file', file); form.append('options', JSON.stringify(options));
  const job = await requestJSON(assetURL('/analyses'), {method: 'POST', body: form, signal});
  const complete = await waitForAnalysis(job, id => requestJSON(assetURL(`/analyses/${id}`), {signal}), {
    signal, onStatus: status => {
      $('processingTitle').textContent = status.stage || 'Analyzing recording…';
      $('processingDescription').textContent = status.status === 'queued' ? 'Waiting for the current analysis to finish.' : 'Measuring audio and preparing linked evidence.';
      $('progressBar').style.width = `${Math.max(0, Math.min(100, (status.progress || 0) * 100))}%`;
    }
  });
  const result = validateResult(await requestJSON(assetURL(complete.results_url), {signal}));
  const keys = ['waveform_url', 'track_url', 'sound_scores_url'];
  const loaded = await Promise.allSettled(keys.map(key => result.assets[key] ? requestJSON(assetURL(result.assets[key]), {signal}) : Promise.resolve(null)));
  if (signal.aborted || activeRequest !== controller) return;
  [waveform, track, soundScores] = loaded.map(value => value.status === 'fulfilled' ? value.value : null);
  const warnings = loaded.flatMap((value, i) => value.status === 'rejected' ? [`Could not load ${keys[i].replace('_url', '')} visualization.`] : []);
  openCase(result, warnings);
}
async function startUpload(file, demoName = null) {
  activeRequest?.abort(); player.pause(); pendingSeek = null;
  const controller = new AbortController(); activeRequest = controller;
  $('uploadError').classList.add('hidden'); setBusy(true);
  try {
    let options = {is_simulated: $('simulatedOption').checked, assume_stationary_microphone: $('stationaryOption').checked, assume_constant_tone: $('constantToneOption').checked};
    if (demoName) {
      switchScreen('processing'); $('processingTitle').textContent = 'Loading example…';
      $('processingDescription').textContent = 'This example will run through the same analysis as an uploaded recording.'; $('progressBar').style.width = '0%';
      const response = await fetch(assetURL(`/demo-audio/${demoName}`), {signal: controller.signal});
      if (!response.ok) throw new Error('The demo recording is unavailable on this server.');
      file = new File([await response.blob()], demoName, {type: 'audio/wav'});
      const knownSimulation = demoName === 'simulated-pass-by.wav';
      options = {is_simulated: true, assume_stationary_microphone: knownSimulation, assume_constant_tone: knownSimulation};
    }
    await analyze(file, options, controller);
  } catch (error) { if (error.name !== 'AbortError' && activeRequest === controller) showError(error); }
  finally { if (activeRequest === controller) setBusy(false); $('audioInput').value = ''; }
}

function fillList(id, items) { $(id).replaceChildren(...items.map(text => { const li = document.createElement('li'); li.textContent = text; return li; })); }
function openCase(result, visualWarnings) {
  currentCase = result; selectedEvent = null; zoomed = false; view = [0, result.recording.duration_s];
  $('recordingName').textContent = result.recording.filename; $('caseId').textContent = result.analysis_id;
  $('recordingDuration').textContent = formatTime(result.recording.duration_s); $('totalTime').textContent = formatTime(result.recording.duration_s);
  $('audioSeek').max = result.recording.duration_s; $('simulationBadge').classList.toggle('hidden', !result.recording.is_simulated);
  $('statusText').textContent = result.recording.is_simulated ? 'Simulated / constructed case' : 'Analysis complete';
  $('filterButton').value = 'all'; $('zoomButton').textContent = 'Zoom to event';
  $('spectrogram').src = assetURL(result.assets.spectrogram_url); $('originalDownload').href = assetURL(result.assets.original_url);
  player.src = assetURL(result.assets.audio_url); player.load();
  $('soundScoresPanel').classList.toggle('hidden', !soundScores);
  $('soundGroupSelect').replaceChildren(...(soundScores?.groups || []).map(group => {
    const option = document.createElement('option'); option.value = group.key; option.textContent = group.label; return option;
  }));
  const warnings = [...(result.warnings || []), ...visualWarnings];
  if (result.recording.original_audio_start_s) warnings.push(`For original video timestamps, add ${result.recording.original_audio_start_s.toFixed(3)} seconds to analysis times.`);
  fillList('warningsList', warnings.length ? warnings : ['No processing warnings were reported.']); fillList('limitationsList', result.limitations || []);
  switchScreen('case'); renderEvents(); renderMotion();
  if (result.events.length) selectEvent(result.events[0], false, false);
  else {
    $('findingSummary').textContent = 'No events met the analysis thresholds. This does not establish that no relevant sound occurred.';
    $('evidenceType').textContent = 'No finding'; $('evidenceDetails').textContent = ''; $('openFinding').disabled = true; $('zoomButton').disabled = true;
  }
  updatePosition(0); drawAll();
}
function renderEvents() {
  const filter = $('filterButton').value, events = currentCase.events.filter(e => filter === 'all' || e.evidence_type === filter);
  $('eventCount').textContent = `${events.length} / ${currentCase.events.length}`; $('noEvents').classList.toggle('hidden', events.length > 0);
  $('noEvents').textContent = currentCase.events.length ? 'No events match this filter.' : 'No events met the analysis thresholds.';
  $('eventList').replaceChildren(...events.map(event => {
    const card = document.createElement('button'); card.className = 'event-card'; card.type = 'button'; card.dataset.id = event.id;
    card.classList.toggle('active', event.id === selectedEvent?.id); card.setAttribute('aria-pressed', String(event.id === selectedEvent?.id));
    const dot = document.createElement('span'); dot.className = 'event-color'; dot.style.background = COLORS[event.evidence_type];
    const text = document.createElement('span'), name = document.createElement('span'), meta = document.createElement('span');
    name.className = 'event-name'; name.textContent = event.label;
    meta.className = 'event-meta'; meta.textContent = `${TYPE_LABEL[event.evidence_type]} · ${event.id}`; text.append(name, meta);
    const time = document.createElement('span'); time.className = 'event-time'; time.textContent = `${formatTime(event.start_s)}–${formatTime(event.end_s)}`;
    card.append(dot, text, time); card.addEventListener('click', () => selectEvent(event)); return card;
  })); renderMarkers();
}
function renderMarkers() {
  $('eventMarkers').replaceChildren(); if (!currentCase) return;
  const filter = $('filterButton').value;
  for (const event of currentCase.events) {
    if ((filter !== 'all' && event.evidence_type !== filter) || event.peak_s < view[0] || event.peak_s > view[1]) continue;
    const marker = document.createElement('button'); marker.className = 'marker'; marker.type = 'button'; marker.dataset.id = event.id;
    marker.style.left = `${(event.peak_s - view[0]) / (view[1] - view[0]) * 100}%`; marker.style.setProperty('--marker-color', COLORS[event.evidence_type]);
    marker.classList.toggle('active', event.id === selectedEvent?.id); marker.setAttribute('aria-label', `Play ${event.label} at ${formatTime(event.start_s)}`);
    marker.title = `${event.label} · ${formatTime(event.peak_s)}`; marker.addEventListener('click', () => selectEvent(event)); $('eventMarkers').append(marker);
  }
  const start = Math.max(view[0], selectedEvent?.start_s ?? 0), end = Math.min(view[1], selectedEvent?.end_s ?? 0);
  $('selectedRange').classList.toggle('hidden', !selectedEvent || end <= start);
  $('selectedRange').style.left = `${(start - view[0]) / (view[1] - view[0]) * 100}%`;
  $('selectedRange').style.width = `${Math.max(0, end - start) / (view[1] - view[0]) * 100}%`;
}
function selectEvent(event, play = true, move = true) {
  selectedEvent = event; $('openFinding').disabled = false; $('zoomButton').disabled = false;
  const findings = currentCase.findings.filter(f => f.evidence_ids.includes(event.id));
  $('findingSummary').textContent = findings.length ? findings.map(f => f.text).join('\n') : event.explanation;
  $('evidenceType').textContent = TYPE_LABEL[event.evidence_type]; const score = event.measurements?.model_score;
  $('evidenceDetails').textContent = `Evidence ${event.id} · ${formatTime(event.start_s)}–${formatTime(event.end_s)}` + (Number.isFinite(score) ? ` · Uncalibrated model score ${score.toFixed(3)}` : '');
  if (soundScores?.groups.some(g => g.key === event.measurements?.sound_group)) { $('soundGroupSelect').value = event.measurements.sound_group; drawSoundScores(); }
  if (zoomed) setEventZoom(); renderEvents(); if (move) seek(event.start_s, play);
}
function seek(time, play = false) {
  if (!currentCase) return; const target = Math.max(0, Math.min(currentCase.recording.duration_s, time));
  if (player.readyState >= 1) player.currentTime = target; else pendingSeek = target;
  updatePosition(target); if (play) player.play().catch(() => showToast('Press Play to hear the selected evidence.'));
}
function updatePosition(time) {
  if (!currentCase) return; const clamped = Math.max(0, Math.min(currentCase.recording.duration_s, time || 0));
  $('currentTime').textContent = formatTime(clamped); $('audioSeek').value = clamped;
  $('playhead').style.left = `${(clamped - view[0]) / (view[1] - view[0]) * 100}%`; $('playhead').classList.toggle('hidden', clamped < view[0] || clamped > view[1]);
}

function canvasContext(id) {
  const canvas = $(id), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext('2d'); context.scale(ratio, ratio); return [context, rect.width, rect.height];
}
function axis(id, start, end) {
  $(id).replaceChildren(...Array.from({length: 5}, (_, i) => { const span = document.createElement('span'); span.textContent = formatTime(start + (end - start) * i / 4); return span; }));
}
function drawWaveform() {
  const [c, w, h] = canvasContext('waveform'); c.strokeStyle = '#d8e2de'; c.beginPath(); c.moveTo(0, h/2); c.lineTo(w, h/2); c.stroke();
  axis('waveformAxis', ...view); renderMarkers(); updatePosition(player.currentTime);
  if (!waveform?.start_s?.length) { c.fillStyle = '#667174'; c.fillText('Waveform unavailable', 16, 25); return; }
  const amplitude = Math.max(1, ...waveform.max.map(Math.abs), ...waveform.min.map(Math.abs));
  c.strokeStyle = '#4d9c87'; c.lineWidth = Math.max(1, w / waveform.start_s.length); c.beginPath();
  waveform.start_s.forEach((start, i) => {
    const time = (start + waveform.end_s[i]) / 2; if (time < view[0] || time > view[1]) return;
    const x = (time - view[0]) / (view[1] - view[0]) * w;
    c.moveTo(x, h/2 - waveform.max[i] / amplitude * h*.45); c.lineTo(x, h/2 - waveform.min[i] / amplitude * h*.45);
  }); c.stroke();
}
function drawSoundScores() {
  if (!soundScores) return; const group = soundScores.groups.find(g => g.key === $('soundGroupSelect').value); if (!group) return;
  const [c, w, h] = canvasContext('soundScores'), duration = currentCase.recording.duration_s, y = score => h - 12 - score * (h - 24);
  c.strokeStyle = '#c6cbc7'; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(0, y(group.onset_threshold)); c.lineTo(w, y(group.onset_threshold)); c.stroke();
  c.setLineDash([]); c.strokeStyle = '#8270ce'; c.lineWidth = 2; c.beginPath();
  group.scores.forEach((score, i) => { const x = (soundScores.start_s[i] + soundScores.end_s[i]) / 2 / duration * w; i ? c.lineTo(x, y(score)) : c.moveTo(x, y(score)); });
  c.stroke(); axis('soundScoresAxis', 0, duration);
}
function drawFrequency() {
  const [c, w, h] = canvasContext('frequencyChart');
  if (!track?.time_s?.length || !track.supported.some(Boolean)) { c.fillStyle = '#667174'; c.fillText('No supported frequency track', 8, h/2); return; }
  const fit = currentCase.motion.fit, measured = track.frequency_hz.filter((_, i) => track.supported[i]);
  const allValues = [...measured, ...(fit?.predicted_frequency_hz || [])].filter(Number.isFinite);
  const low = Math.min(...allValues), high = Math.max(...allValues), span = Math.max(20, high - low), t0 = track.time_s[0], t1 = track.time_s.at(-1);
  const plot = (times, values, support, color, dashed) => {
    c.strokeStyle = color; c.lineWidth = 2; c.setLineDash(dashed ? [4, 4] : []); c.beginPath(); let connected = false;
    times.forEach((time, i) => {
      if (!Number.isFinite(values[i]) || (support && !support[i])) { connected = false; return; }
      const x = (time - t0) / Math.max(.001, t1 - t0) * w, y = h - 10 - (values[i] - low) / span * (h - 20);
      connected ? c.lineTo(x, y) : c.moveTo(x, y); connected = true;
    }); c.stroke(); c.setLineDash([]);
  };
  plot(track.time_s, track.frequency_hz, track.supported, '#d47b4e', false);
  if (fit) plot(fit.time_s, fit.predicted_frequency_hz, null, currentCase.motion.status === 'insufficient_evidence' ? '#909995' : '#8270ce', true);
}
function renderMotion() {
  const motion = currentCase.motion, supported = track?.frequency_hz?.filter((_, i) => track.supported[i]) || [];
  $('frequencyRange').textContent = supported.length ? `${Math.min(...supported).toFixed(0)}–${Math.max(...supported).toFixed(0)} Hz` : 'No supported tone';
  $('frequencyWindow').textContent = track ? `${formatTime(track.start_s)}–${formatTime(track.end_s)}` : 'Track unavailable';
  $('fitLegend').textContent = motion.fit ? `Orange: measured ridge · Dashed: ${motion.status === 'insufficient_evidence' ? 'rejected fit' : 'candidate fit'}` : 'Orange: supported measured ridge';
  $('closestApproach').textContent = Number.isFinite(motion.received_closest_approach_s) ? `${motion.received_closest_approach_s.toFixed(2)} s · candidate` : 'Unavailable';
  $('speedEstimate').textContent = speedLabel(motion);
  $('motionStatus').textContent = {conditional_fit: 'Conditional fit', assumptions_required: 'Assumptions required', insufficient_evidence: 'Insufficient evidence'}[motion.status];
  fillList('motionReasons', motion.reasons || []); $('motionCaveat').textContent = motion.caveat;
}
function drawAll() {
  if (!currentCase || screens.case.classList.contains('hidden')) return;
  requestAnimationFrame(() => { if (currentCase) { drawWaveform(); drawFrequency(); drawSoundScores(); } });
}
function setEventZoom() {
  if (!selectedEvent) return; const margin = Math.max(.5, (selectedEvent.end_s - selectedEvent.start_s) / 2);
  view = [Math.max(0, selectedEvent.start_s - margin), Math.min(currentCase.recording.duration_s, selectedEvent.end_s + margin)]; drawAll();
}

$('openDemoButton').addEventListener('click', () => startUpload(null, 'constructed-sound-scene.wav'));
$('openSimulationButton').addEventListener('click', () => startUpload(null, 'simulated-pass-by.wav'));
$('audioInput').addEventListener('change', e => { const file = e.target.files[0]; if (file) startUpload(file); });
const dropzone = $('dropzone'); dropzone.tabIndex = 0;
dropzone.addEventListener('keydown', e => { if (['Enter', ' '].includes(e.key)) { e.preventDefault(); $('audioInput').click(); } });
for (const type of ['dragenter', 'dragover']) dropzone.addEventListener(type, e => { e.preventDefault(); dropzone.classList.add('dragover'); });
for (const type of ['dragleave', 'drop']) dropzone.addEventListener(type, e => { e.preventDefault(); dropzone.classList.remove('dragover'); });
dropzone.addEventListener('drop', e => { const file = e.dataTransfer.files[0]; if (file && !$('audioInput').disabled) startUpload(file); });
$('playButton').addEventListener('click', () => { if (player.paused) player.play().catch(() => showToast('Audio playback failed. Try loading this recording again.')); else player.pause(); });
player.addEventListener('loadedmetadata', () => { if (pendingSeek !== null) { player.currentTime = pendingSeek; pendingSeek = null; } });
player.addEventListener('play', () => { $('playButton').textContent = '❚❚'; $('playButton').setAttribute('aria-label', 'Pause recording'); });
player.addEventListener('pause', () => { $('playButton').textContent = '▶'; $('playButton').setAttribute('aria-label', 'Play recording'); });
player.addEventListener('timeupdate', () => updatePosition(player.currentTime));
player.addEventListener('error', () => { if (currentCase) showToast('The analysis audio could not be loaded.'); });
$('audioSeek').addEventListener('input', e => seek(Number(e.target.value)));
$('openFinding').addEventListener('click', () => { if (selectedEvent) seek(selectedEvent.start_s, true); });
$('zoomButton').addEventListener('click', () => {
  if (!selectedEvent) return; zoomed = !zoomed; $('zoomButton').textContent = zoomed ? 'Show full recording' : 'Zoom to event';
  if (zoomed) setEventZoom(); else { view = [0, currentCase.recording.duration_s]; drawAll(); }
});
$('filterButton').addEventListener('change', () => {
  if (!currentCase) return;
  const visible = currentCase.events.filter(e => $('filterButton').value === 'all' || e.evidence_type === $('filterButton').value);
  if (visible.length && !visible.some(e => e.id === selectedEvent?.id)) selectEvent(visible[0], false, false);
  else renderEvents();
});
$('soundGroupSelect').addEventListener('change', drawSoundScores);
$('newCaseButton').addEventListener('click', newRecording);
document.querySelector('.brand').addEventListener('click', e => { e.preventDefault(); newRecording(); });
$('spectrogram').addEventListener('error', () => { if (currentCase) showToast('The measured spectrogram is unavailable.'); });
$('reportButton').addEventListener('click', async () => {
  const result = currentCase; if (!result) return;
  try {
    const response = await fetch(assetURL(result.assets.report_url)); if (!response.ok) throw new Error('Report download failed.');
    const url = URL.createObjectURL(await response.blob()), link = document.createElement('a');
    link.href = url; link.download = `AcoustiCase-${result.analysis_id}-report.md`; document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000); showToast('Evidence report downloaded.');
  } catch (error) { showToast(error.message); }
});
window.addEventListener('resize', drawAll);
requestJSON(assetURL('/health')).then(health => {
  if (!currentCase && !activeRequest) $('statusText').textContent = health.classifier_available ? 'Ready · Sound model available' : 'Ready · Signal analysis only';
}).catch(() => { if (!currentCase && !activeRequest) $('statusText').textContent = 'Analysis server unavailable'; });
