# AcoustiCase

An experimental audio investigation workspace for Hack Western 13. Upload a recording, inspect timestamped acoustic events, and trace each finding back to the recording and its measurements.

**Team:** Satvik — analysis, backend, and validation. Arjun — frontend, evidence presentation, and demo.

## Repository layout

```text
backend/     Python/FastAPI engine, ML evaluation, tests, API types, and labeled demo examples
frontend/    Arjun's investigation interface, connected to the analysis API
docs/        Frontend integration handoff and validation results
```

## Start the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
bash run.sh
```

Open http://127.0.0.1:8000/app/ for the integrated app or http://127.0.0.1:8000/docs for the API. The backend serves the interface on the same origin; no frontend build or second server is needed. The core pipeline needs no paid model API key. Optional local sound classification:

```bash
pip install -e '.[classifier]'
python scripts/download_yamnet.py
bash run.sh
```

See [backend setup and API details](backend/README.md), [Arjun's integration handoff](docs/FRONTEND_HANDOFF.md), and [validation results](docs/VALIDATION.md).

## Current implementation

The backend includes audio/video decoding, waveform/spectrogram generation, transient detection, grouped YAMNet sound events and score timelines, single-tone tracking, conditional Doppler fitting, asynchronous job status, result caching, and reports grounded in evidence IDs. A separately trained experimental classifier, real-audio benchmark, and reproducible training scripts are included. See [the ML model card](docs/ML_MODEL_CARD.md) for measured results and limitations.

The frontend supports real uploads, processing status, aligned audio playback, event selection and filtering, waveform zoom, measured spectrograms, sound scores, conditional motion results, and backend report downloads. Its built-in examples use the same processing API. Simulated inputs are labeled, and real-upload motion assumptions default to unchecked.

Speed estimates are experimental and conditional on a fixed emitted tone, a stationary microphone, and a simple straight-line pass. Validation currently includes synthetic known-speed audio and a real narration-file smoke test, not a real-car accuracy benchmark. The app does not establish accident causation. Reports currently use structured evidence directly, without an LLM.

## Working together

- Keep `main` demo-ready.
- Satvik: use branches such as `satvik/backend` for analysis and API work.
- Arjun: use branches such as `arjun/frontend` for the investigation interface.
- Merge through pull requests and agree on API changes together.
- The API types and example response live in `backend/frontend-types.ts` and `backend/examples/example-result.json`.
- A `localhost` backend is reachable only from the same computer. A shared server is a separate deployment step.

## Files kept local

Uploaded recordings, generated analysis results, downloaded model weights, raw evaluation audio, environment files, credentials, virtual environments, and caches are ignored by Git. A small trained experimental head and evaluation reports are checked in under `backend/evaluation/artifacts/`.

Audio examples include synthetic signals and an explicitly constructed collage of ESC-50 field recordings. The collage is curated from training examples, not an actual incident or an evaluation clip. Its attribution is in `backend/examples/CONSTRUCTED-SCENE-ATTRIBUTION.md`; the upstream CC BY-NC dataset license is preserved in `backend/evaluation/ESC50-LICENSE`.

The bundled Google YAMNet architecture and class-map files retain their upstream Apache 2.0 license and attribution under `backend/src/acousticase/vendor/yamnet/`.
