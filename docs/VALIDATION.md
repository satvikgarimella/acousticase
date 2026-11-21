# Validation record

Local run on 2026-11-21, Python 3.12.14 / macOS arm64.

## Automated tests

15 tests passed. Negative controls: stationary tone, white noise, silence, upward chirp, and downward linear chirp. No speed was reported for those controls. API tests cover original-byte preservation, duration/size limits, caching, invalid input, and evidence references.

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

No held-out real vehicle recordings with independently measured speed have been evaluated. No collision-cause inference, multiple-source tracking, calibrated probabilities, or generative report model is implemented. Docker deployment has not been exercised.

## Live HTTP smoke test

Upload, status polling, results, waveform/track assets, spectrogram, report, and OpenAPI docs passed against localhost. Synthetic 72 km/h demo estimate: 72.008 km/h; classifier status: enabled; processing: 11.512 seconds (cold start included).
