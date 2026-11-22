export async function requestJSON(url, options = {}) {
  const response = await fetch(url, options);
  let body;
  try { body = await response.json(); } catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body));
  return body;
}

export function pause(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('Cancelled', 'AbortError'));
    const done = () => { signal?.removeEventListener('abort', abort); resolve(); };
    const timer = setTimeout(done, ms);
    const abort = () => { clearTimeout(timer); reject(new DOMException('Cancelled', 'AbortError')); };
    signal?.addEventListener('abort', abort, {once: true});
  });
}

export async function waitForAnalysis(initial, fetchStatus, {signal, onStatus = () => {}, intervalMs = 750, timeoutMs = 300000} = {}) {
  const deadline = Date.now() + timeoutMs;
  let state = initial;
  while (true) {
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    onStatus(state);
    if (state.status === 'complete') return state;
    if (state.status === 'failed') throw new Error(state.error || 'The recording could not be analyzed.');
    if (!['queued', 'running'].includes(state.status)) throw new Error('Unexpected analysis status.');
    if (Date.now() >= deadline) throw new Error('Analysis is taking longer than expected. Upload this recording again to retrieve its current result.');
    await pause(intervalMs, signal);
    state = await fetchStatus(state.analysis_id);
  }
}

export function speedLabel(motion) {
  if (motion?.status === 'conditional_fit' && Number.isFinite(motion.speed_km_h)) return `${motion.speed_km_h.toFixed(1)} km/h · conditional`;
  return motion?.status === 'assumptions_required' ? 'Assumptions required' : 'Insufficient evidence';
}

export function validateResult(result) {
  if (result.status !== 'complete' || !Number.isFinite(result.recording?.duration_s) || result.recording.duration_s <= 0
      || !Array.isArray(result.events) || !Array.isArray(result.findings) || !result.assets?.audio_url) throw new Error('The server returned an incomplete analysis result.');
  const ids = new Set(result.events.map(event => event.id));
  for (const event of result.events) {
    if (![event.start_s, event.end_s, event.peak_s].every(Number.isFinite) || event.start_s < 0
        || event.start_s > event.peak_s || event.peak_s > event.end_s || event.end_s > result.recording.duration_s + .001) throw new Error('The result contains invalid event timestamps.');
  }
  if (result.findings.some(f => !Array.isArray(f.evidence_ids) || f.evidence_ids.some(id => !ids.has(id)))) throw new Error('The result contains a broken evidence reference.');
  return result;
}
