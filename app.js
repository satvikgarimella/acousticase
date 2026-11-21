const DEMO_DURATION = 18.4;
const caseData = {
  id: 'REC-DEMO-001', filename: 'downtown-passby.wav', duration: DEMO_DURATION,
  events: [
    { id: 'EVT-001', start: 4.2, end: 4.9, label: 'Vehicle pass-by', type: 'Tonal movement', color: '#ff9b6c' },
    { id: 'EVT-002', start: 10.8, end: 11.2, label: 'Car horn', type: 'Sound classifier', color: '#8b7ceb' },
    { id: 'EVT-003', start: 14.4, end: 15.8, label: 'Background vehicle', type: 'Sound classifier', color: '#54bfa0' }
  ]
};
const $ = (id) => document.getElementById(id);
const screens = { upload: $('uploadScreen'), processing: $('processingScreen'), case: $('caseScreen') };
const player = $('audioPlayer');
let currentCase = { ...caseData };
let demoTimer;

function formatTime(time) { const m = Math.floor(time / 60); const s = (time % 60).toFixed(1).padStart(4, '0'); return `${String(m).padStart(2, '0')}:${s}`; }
function switchScreen(name) { Object.entries(screens).forEach(([key, el]) => el.classList.toggle('hidden', key !== name)); }
function showToast(message) { const toast = $('toast'); toast.textContent = message; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2600); }

function startProcessing(file) {
  switchScreen('processing');
  const stages = [
    ['Reading audio…', 'Keeping the original file intact while preparing an analysis copy.', 25],
    ['Mapping the waveform…', 'Creating a visual record of changes in sound over time.', 52],
    ['Finding notable moments…', 'Checking for sudden sounds and recognizable audio events.', 76],
    ['Preparing evidence links…', 'Connecting every finding to a precise time in the recording.', 100]
  ];
  let i = 0;
  clearInterval(demoTimer);
  const go = () => {
    const [title, description, progress] = stages[i];
    $('processingTitle').textContent = title; $('processingDescription').textContent = description; $('progressBar').style.width = `${progress}%`;
    if (++i === stages.length) { clearInterval(demoTimer); setTimeout(() => openCase(file), 700); }
  };
  go(); demoTimer = setInterval(go, 850);
}
function openCase(file) {
  if (file) { currentCase = { ...caseData, filename: file.name, id: `REC-${Math.random().toString(36).slice(2, 8).toUpperCase()}` }; $('uploadTime').textContent = 'just now'; }
  $('recordingName').textContent = currentCase.filename; $('caseId').textContent = currentCase.id;
  $('recordingDuration').textContent = formatTime(currentCase.duration); $('totalTime').textContent = formatTime(currentCase.duration);
  $('audioSeek').max = currentCase.duration;
  renderEvents(); renderMarkers(); drawAll(); switchScreen('case');
}
function renderEvents() {
  const list = $('eventList'); list.innerHTML = '';
  currentCase.events.forEach(event => {
    const card = document.createElement('button'); card.className = 'event-card'; card.type = 'button'; card.dataset.id = event.id;
    card.innerHTML = `<span class="event-color" style="background:${event.color}"></span><span><span class="event-name">${event.label}</span><span class="event-meta">${event.type} · Evidence linked</span></span><span class="event-time">${formatTime(event.start).slice(3)}–${formatTime(event.end).slice(3)}</span>`;
    card.addEventListener('click', () => selectEvent(event)); list.appendChild(card);
  });
  $('eventCount').textContent = currentCase.events.length;
}
function renderMarkers() {
  const holder = $('eventMarkers'); holder.innerHTML = '';
  currentCase.events.forEach(event => { const marker = document.createElement('button'); marker.className = 'marker'; marker.style.left = `${event.start / currentCase.duration * 100}%`; marker.style.background = event.color; marker.dataset.id = event.id; marker.dataset.label = event.label; marker.setAttribute('aria-label', `Seek to ${event.label}`); marker.addEventListener('click', () => selectEvent(event)); holder.appendChild(marker); });
}
function selectEvent(event) {
  seek(event.start); document.querySelectorAll('.event-card,.marker').forEach(x => x.classList.toggle('active', x.dataset.id === event.id));
  $('findingSummary').textContent = event.id === 'EVT-001' ? 'A tonal source decreases in frequency while passing the microphone, consistent with a moving vehicle.' : `${event.label} was detected from ${formatTime(event.start).slice(3)} to ${formatTime(event.end).slice(3)}. The audio and chart location are linked as evidence.`;
  showToast(`Evidence selected: ${event.label} at ${formatTime(event.start).slice(3)}`);
}
function seek(time) {
  if (player.src) player.currentTime = time;
  updatePosition(time);
}
function updatePosition(time) {
  const clamped = Math.min(currentCase.duration, Math.max(0, time || 0)); const percentage = clamped / currentCase.duration * 100;
  $('currentTime').textContent = formatTime(clamped); $('audioSeek').value = clamped;
  $('playhead').style.left = `${percentage}%`; $('spectrogramPlayhead').style.left = `${percentage}%`;
}
function drawWaveform() {
  const canvas = $('waveform'), rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; canvas.width = rect.width * dpr; canvas.height = rect.height * dpr; const c = canvas.getContext('2d'); c.scale(dpr,dpr); const w=rect.width,h=rect.height; c.clearRect(0,0,w,h); c.strokeStyle='#b8e8dc'; c.lineWidth=1.5; c.beginPath();
  for(let x=0;x<w;x+=2){const t=x/w; let amp=(Math.sin(x*.15)*.15+Math.sin(x*.041)*.18+.13); amp+=Math.exp(-Math.pow((t-.25)*12,2))*.75+Math.exp(-Math.pow((t-.60)*26,2))*.42+Math.exp(-Math.pow((t-.82)*11,2))*.48; const y=h/2+Math.sin(x*.63)*amp*h*.37; x?c.lineTo(x,y):c.moveTo(x,y)} c.stroke(); c.strokeStyle='#d8e2de'; c.beginPath(); c.moveTo(0,h/2);c.lineTo(w,h/2);c.stroke();
}
function drawSpectrogram() {
  const canvas=$('spectrogram'), rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);const w=rect.width,h=rect.height;const bg=c.createLinearGradient(0,0,0,h);bg.addColorStop(0,'#243f48');bg.addColorStop(1,'#10242d');c.fillStyle=bg;c.fillRect(0,0,w,h);
  for(let x=0;x<w;x+=4){for(let y=0;y<h;y+=4){let strength=(Math.sin(x*.11+y*.19)+1)*.06; const t=x/w; strength+=Math.exp(-Math.pow((y/h-(.78-t*.48))*17,2))*Math.exp(-Math.pow((t-.25)*7,2))*.8; strength+=Math.exp(-Math.pow((t-.60)*20,2))*Math.exp(-Math.pow((y/h-.55)*13,2))*.65; if(strength>.14){c.fillStyle=`rgba(${Math.round(115+strength*150)},${Math.round(100+strength*130)},${Math.round(170-strength*100)},${Math.min(.85,strength)})`;c.fillRect(x,y,4,4)}}}
}
function drawFrequency(){const canvas=$('frequencyChart'),r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;const c=canvas.getContext('2d');c.scale(d,d);const w=r.width,h=r.height;c.strokeStyle='#e2e9e6';c.lineWidth=1;for(let y=16;y<h;y+=30){c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}c.strokeStyle='#ff9b6c';c.lineWidth=3;c.beginPath();for(let x=0;x<w;x+=3){const t=x/w;const y=17+t*65+Math.sin(t*13)*4; x?c.lineTo(x,y):c.moveTo(x,y)}c.stroke();c.fillStyle='#ff9b6c';c.beginPath();c.arc(w*.52,17+.52*65+Math.sin(.52*13)*4,4,0,Math.PI*2);c.fill()}
function drawAll(){requestAnimationFrame(()=>{drawWaveform();drawSpectrogram();drawFrequency()})}

$('openDemoButton').addEventListener('click',()=>openCase());
$('audioInput').addEventListener('change',(e)=>{const file=e.target.files[0];if(file){player.src=URL.createObjectURL(file);startProcessing(file)}});
const dropzone=$('dropzone');['dragenter','dragover'].forEach(type=>dropzone.addEventListener(type,e=>{e.preventDefault();dropzone.classList.add('dragover')}));['dragleave','drop'].forEach(type=>dropzone.addEventListener(type,e=>{e.preventDefault();dropzone.classList.remove('dragover')}));dropzone.addEventListener('drop',e=>{const file=e.dataTransfer.files[0];if(file&&file.type.startsWith('audio/')){player.src=URL.createObjectURL(file);startProcessing(file)}else showToast('Please choose an audio file.');});
$('playButton').addEventListener('click',()=>{if(player.src){player.paused?player.play():player.pause()}else{showToast('Demo playback is simulated—upload an audio file to play it.');$('playButton').textContent=$('playButton').textContent==='▶'?'❚❚':'▶'}});
player.addEventListener('play',()=>$('playButton').textContent='❚❚');player.addEventListener('pause',()=>$('playButton').textContent='▶');player.addEventListener('timeupdate',()=>updatePosition(player.currentTime));player.addEventListener('ended',()=>updatePosition(0));
$('audioSeek').addEventListener('input',e=>seek(+e.target.value));$('openFinding').addEventListener('click',()=>selectEvent(currentCase.events[0]));$('zoomButton').addEventListener('click',()=>selectEvent(currentCase.events[0]));
$('filterButton').addEventListener('click',()=>showToast('All evidence types are currently shown.'));$('newCaseButton').addEventListener('click',()=>{player.pause();switchScreen('upload');});
$('reportButton').addEventListener('click',()=>{const report=`ACOUSTICASE EVIDENCE REPORT\n\nRecording: ${currentCase.filename}\nCase ID: ${currentCase.id}\nDuration: ${formatTime(currentCase.duration)}\n\nDETECTED EVENTS\n${currentCase.events.map(e=>`- ${e.id}: ${e.label} (${formatTime(e.start).slice(3)}–${formatTime(e.end).slice(3)}), ${e.type}`).join('\n')}\n\nFINDING\nA tonal source decreases in frequency during EVT-001, consistent with a moving vehicle.\n\nLIMITATION\nSpeed cannot be determined reliably from this recording alone.`;const blob=new Blob([report],{type:'text/plain'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`AcoustiCase-${currentCase.id}-report.txt`;a.click();URL.revokeObjectURL(url);showToast('Evidence report downloaded.');});
window.addEventListener('resize',drawAll);
