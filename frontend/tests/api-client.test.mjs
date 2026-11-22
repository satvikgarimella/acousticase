import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {requestJSON, pause, waitForAnalysis, speedLabel, validateResult} from '../api-client.js';

test('a cached completed job does not poll', async () => {
  const state = {status: 'complete', analysis_id: 'cached'};
  assert.equal(await waitForAnalysis(state, () => { throw new Error('unexpected poll'); }), state);
});

test('queued and running jobs poll until completion', async () => {
  const statuses = [], remaining = ['running', 'complete'];
  const result = await waitForAnalysis({status: 'queued', analysis_id: 'a'}, async id => {
    assert.equal(id, 'a'); return {status: remaining.shift(), analysis_id: id};
  }, {intervalMs: 0, onStatus: state => statuses.push(state.status)});
  assert.equal(result.status, 'complete'); assert.deepEqual(statuses, ['queued', 'running', 'complete']);
});

test('failed jobs show the backend error and stop', async () => {
  await assert.rejects(waitForAnalysis({status: 'failed', error: 'No audio stream'}, () => {}), /No audio stream/);
});

test('cancelling a request interrupts its polling delay', async () => {
  const controller = new AbortController(); const waiting = pause(60000, controller.signal); controller.abort();
  await assert.rejects(waiting, {name: 'AbortError'});
});

test('timeouts and invalid job states do not loop forever', async () => {
  await assert.rejects(waitForAnalysis({status: 'queued'}, () => {}, {timeoutMs: 0}), /longer than expected/);
  await assert.rejects(waitForAnalysis({status: 'unknown'}, () => {}), /Unexpected analysis status/);
});

test('speed stays unavailable until the backend returns a conditional estimate', () => {
  assert.equal(speedLabel({status: 'insufficient_evidence', speed_km_h: null}), 'Insufficient evidence');
  assert.equal(speedLabel({status: 'assumptions_required', speed_km_h: null}), 'Assumptions required');
  assert.equal(speedLabel({status: 'conditional_fit', speed_km_h: null}), 'Insufficient evidence');
  assert.equal(speedLabel({status: 'conditional_fit', speed_km_h: 72.008}), '72.0 km/h · conditional');
});

test('both checked-in real API payloads validate', async () => {
  for (const name of ['example-result.json', 'example-sound-result.json']) {
    const data = JSON.parse(await readFile(new URL(`../../backend/examples/${name}`, import.meta.url), 'utf8'));
    assert.equal(validateResult(data), data);
  }
});

test('broken evidence links and out-of-bounds events are rejected', async () => {
  const result = JSON.parse(await readFile(new URL('../../backend/examples/example-result.json', import.meta.url), 'utf8'));
  const broken = structuredClone(result); broken.findings[0].evidence_ids = ['missing'];
  assert.throws(() => validateResult(broken), /broken evidence/);
  result.events[0].end_s = result.recording.duration_s + 1;
  assert.throws(() => validateResult(result), /invalid event timestamps/);
});

test('HTTP errors preserve useful messages', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({detail: 'Queue full'}), {status: 429});
    await assert.rejects(requestJSON('http://localhost/analyses'), /Queue full/);
    globalThis.fetch = async () => new Response('not json', {status: 502});
    await assert.rejects(requestJSON('http://localhost/analyses'), /unreadable response/);
  } finally { globalThis.fetch = original; }
});
