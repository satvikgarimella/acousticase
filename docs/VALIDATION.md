# Validation record

Local run on 2026-11-21, Python 3.12.14 / macOS arm64.

## Automated tests

25 tests passed after the ML update. Negative controls: stationary tone, white noise, silence, upward chirp, and downward linear chirp. No speed was reported for those controls. API tests cover original-byte preservation, duration/size limits, caching, invalid input, and evidence references. Additional tests cover label mapping, score hysteresis, distinct/simultaneous events, clipping timestamps, explicit event caps, profile cache hashes, source-split leakage, evaluation counts, and portable model inference.

## Independent synthetic pass-by checks

Audio was synthesized from source emission samples and their propagation arrival times, separately from the estimator frequency equation. The assumptions are true by construction. This checks implementation; it does not establish accuracy on real cars.

| True speed (m/s) | Estimated (m/s) | True received CPA (s) | Estimated CPA (s) | Fit RMSE (Hz) |
| --- | --- | --- | --- | --- |
| 12.000 | 12.002 | 7.544 | 7.544 | 0.004 |
| 20.000 | 20.002 | 7.573 | 7.573 | 0.005 |
| 35.000 | 35.004 | 7.602 | 7.602 | 0.009 |

## Real-file smoke test

The supplied screen recording decoded and processed successfully (106.58 seconds of audio). YAMNet's top class was speech; motion returned insufficient_evidence. Processing took 3.409 seconds in this local run. This is a decode/classifier/integration check on narration, not a vehicle-speed benchmark.

## Remaining evaluation

No held-out real vehicle recordings with independently measured speed have been evaluated. No collision-cause inference, multiple-source tracking, calibrated probabilities, or generative report model is implemented. Docker deployment has not been exercised. See [the ML model card](ML_MODEL_CARD.md) for the new sound-classification benchmark and its separate limits.

## ML update

320 ESC-50 recordings were split into 192 training, 64 validation, and 64 test clips using the dataset's source-separated folds. The trained eight-class linear head labeled 61/64 test clips correctly, versus 58/64 for a grouped YAMNet baseline. It is a separate experimental clip classifier, not the API's event detector.

For four target sound groups, the updated event detector's clip-presence macro F1 was 0.734 versus 0.675 originally. Correct detections rose from 18/32 to 24/32; false positives rose from 1 to 4 across 224 negative class/clip decisions. Glass recall declined to 1/8. No event-boundary accuracy was measured.

Live model checks passed: 135-patch chunked/whole-waveform equivalence on 65 seconds, five short-clip boundary cases, silence with zero events, and real-model API upload/poll/results/score asset/evidence links. The full reproducible report, per-class results, limits, and sources are in [ML_MODEL_CARD.md](ML_MODEL_CARD.md).

Both demos also passed against a live HTTP server on pipeline 0.2.0. The synthetic speed remained 72.008 km/h. The constructed 25-second sound collage produced engine, horn, glass, and siren suggestions, with no speed estimate; processing took 0.410 seconds after model warm-up in this local run. This curated example is not included in evaluation metrics.

## Live HTTP smoke test

Upload, status polling, results, waveform/track assets, spectrogram, report, and OpenAPI docs passed against localhost. Synthetic 72 km/h demo estimate: 72.008 km/h; classifier status: enabled; processing: 11.512 seconds (cold start included).
