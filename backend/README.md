# AcoustiCase — Satvik's backend

A working CPU-first prototype for audio event investigation. Arjun owns the frontend; the integration contract is in [FRONTEND_HANDOFF.md](../docs/FRONTEND_HANDOFF.md).

## Run on this machine

```bash
bash run.sh
```

Open http://127.0.0.1:8000/app/ for the integrated investigation interface or http://127.0.0.1:8000/docs for the interactive API client. Health and classifier availability: http://127.0.0.1:8000/health. The server mounts the sibling `frontend/` directory when present; a backend-only installation retains the API landing page.

The supplied launcher uses the backend's `.venv`. Set it up using the installation steps below. No paid API key is required. The report is generated deterministically from measurements and sound-model suggestions, not by an LLM.

## Install on another machine

Python 3.11+ (tested with Python 3.12), macOS/Linux recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
bash run.sh
```

PyAV supplies the media decoder; a separate ffmpeg executable is not required. Core analysis works without TensorFlow. Optional sound classification:

```bash
pip install -e '.[classifier]'
python scripts/download_yamnet.py
bash run.sh
```

The model is downloaded only by this setup command. Upload processing uses local files and does not contact model services. The launcher enables a local `models/yamnet` model automatically. On other launch methods, set `ACOUSTICASE_YAMNET_DIR` to that folder. First inference takes longer while TensorFlow and the model load.

The downloader uses the official Google-hosted `yamnet.h5` and checks its SHA-256. The small upstream architecture files and class map are vendored with their Apache 2.0 license. `requirements-tested.txt` records the exact direct dependency versions used for this build, including classifier and testing dependencies.

## What is implemented

- WAV/MP3/FLAC/OGG/M4A/MP4 and other formats supported by the installed PyAV build; the first audio stream is analyzed.
- Original-byte preservation, SHA-256, source audio timestamp offset, mono 16 kHz analysis/playback copy, duration/size checks, timestamp-discontinuity rejection, clipping warning.
- Downsampled waveform envelope, full-clip spectrogram PNG, rapid-energy transient measurements.
- Optional pretrained Google YAMNet sound labels as **model suggestions**, with uncalibrated scores and approximate time windows.
- Grouped sound events, separate onset/continuation thresholds, short-gap consolidation, broad-label suppression, and an optional frame-score asset for the frontend.
- Reproducible real-audio evaluation and a separately trained experimental clip classifier. See [ML model card](../docs/ML_MODEL_CARD.md) for the 320-clip split, 61/64 test classification result, detector trade-offs, and limits.
- One continuous tonal ridge in a selected window, a propagation-delay-aware constant-speed Doppler fit, comparison with steady-tone and linear-drift baselines, and explicit unsupported outcomes.
- Evidence IDs, deterministic report findings, Markdown report, asynchronous job status, persistent disk results, content/options/model-version caching, and a bounded single-worker queue.
- OpenAPI documentation and a TypeScript contract for Arjun.

## First example

```bash
python scripts/generate_demo.py
curl -X POST http://127.0.0.1:8000/analyses \
  -F 'file=@examples/simulated-pass-by.wav' \
  -F 'options={"is_simulated":true,"assume_constant_tone":true,"assume_stationary_microphone":true,"use_classifier":false}'
```

Use the returned `analysis_id` to poll `GET /analyses/{id}`, then fetch `GET /analyses/{id}/results` once status is `complete`. Returned asset URLs are relative to the backend origin. Tests and generated audio are explicitly simulated; they are not a benchmark of real-car accuracy.

For a richer sound timeline, upload `examples/constructed-sound-scene.wav` with options `{"is_simulated":true}`. It is a curated collage of separate engine, horn, glass, and siren recordings, not an actual incident. Keep motion assumptions false. Attribution and source-excerpt spans accompany the audio. Sound recognition remains experimental, especially for glass breaking and mixtures.

## ML development

The app's event detector uses grouped YAMNet scores. The trained eight-class head is a separate short-clip experiment and is not used to infer event timestamps. To reproduce training and evaluation, install `pip install -e '.[ml,test]'`, run `python scripts/download_eval.py`, `python scripts/ml_benchmark.py extract`, then `python scripts/ml_benchmark.py run`. See the model card for details and limitations. Raw data and features stay in ignored `data/`.

## Motion controls and interpretation

The default tracking window is the first 30 seconds. Set `tracking_start_s` and `tracking_end_s` for a window of up to 60 seconds elsewhere. Set `frequency_min_hz`/`frequency_max_hz` to help isolate one source. Whole-file event detection still processes the full clip.

`assume_constant_tone` and `assume_stationary_microphone` default to **false**. These are operator assertions, not properties the backend can independently verify. Setting them true does not turn an unsuitable recording into a valid measurement.

| Motion status | Meaning |
| --- | --- |
| `insufficient_evidence` | No stable ridge or fit checks failed. Speed and closest-approach output are null. A diagnostic rejected fit may still be included. |
| `assumptions_required` | A curve passes numerical checks, but the two operator assertions are absent. Speed remains null. |
| `conditional_fit` | Curve checks passed and both assumptions were asserted. Speed is conditional on the model; it is not verified vehicle speed. |

The pass-by model fits source frequency, speed, distance/speed, and the arrival time of sound emitted at closest approach. It accounts for travel time through stationary air. It does not identify a road, geographic location, compass heading, source identity, or accident cause. A changing engine tone can imitate the curve. A good residual is not a calibrated confidence probability.

The original bytes are retained. All analysis timestamps align with `playback.wav`, starting at the first decoded audio sample. For the original video timeline add `recording.original_audio_start_s`. Multiple tracks, discontinuities, unusual container edits, and moving microphones require additional analysis.

## Limits and deployment

- 100 MiB per upload, 300 seconds per recording, one analysis worker, at most four queued/running jobs.
- Local trusted hackathon prototype: no authentication, authorization, automatic retention/deletion, or durable multi-process task broker. Do not expose sensitive uploads on a public endpoint. CORS is not authentication.
- Run one Uvicorn process. Restarted pending jobs become failed and can be retried by uploading again. Complete results survive restarts.
- Default CORS origins cover localhost/127.0.0.1 ports 3000, 5173, and 8501. Set `ACOUSTICASE_CORS_ORIGINS` to a comma-separated allowlist for other frontend origins.
- Data is stored under `data/`; set `ACOUSTICASE_DATA_DIR` to override. No cloud deployment or paid resources are provisioned.
- Optional Docker build: `docker build -t acousticase .`; local run: `docker run --rm -p 127.0.0.1:8000:8000 -v "$(pwd)/data:/app/data" acousticase`. This image has core analysis only; TensorFlow/model weights are not included. Docker is an optional recipe, not part of the verified local runtime.

## Test

```bash
python -m pytest -q
```

With the server running, `python scripts/smoke_test.py` exercises a real HTTP upload and refreshes Arjun's example payloads. See [VALIDATION.md](../docs/VALIDATION.md) for observed results and evaluation gaps.

Tests cover independently synthesized moving sources, source-speed recovery tolerance, pass-by timing, assumption gating, stationary tone/noise/silence/chirp negative controls, transient timing, API upload/poll/results/assets, original preservation, caching, evidence references, invalid inputs, and size/duration limits.

Real recordings still need a measured ground-truth evaluation before claims of real-world speed accuracy. The included known-speed simulator is useful for implementation checks, not sufficient scientific validation.

## Attribution

- Google YAMNet, AudioSet, and TensorFlow: https://www.tensorflow.org/hub/tutorials/yamnet and https://github.com/tensorflow/models/tree/master/research/audioset/yamnet (retain upstream model/license terms).
- SciPy signal processing/optimization, NumPy, PyAV, SoundFile, Matplotlib, FastAPI, and Pydantic power the implementation.
- Inspired by the acoustic Doppler discussion and `mjshiggins/dopplertrack`; this backend's tracking/fitting code is a new, limited prototype and does not contain that project's flight-search pipeline.
