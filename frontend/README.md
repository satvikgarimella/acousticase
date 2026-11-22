# Arjun's frontend

Arjun's plain HTML/CSS/JavaScript interface is connected to Satvik's FastAPI backend. No framework build or dependency installation is required to run it.

Read [the integration handoff](../docs/FRONTEND_HANDOFF.md) first. Backend API types are in [frontend-types.ts](../backend/frontend-types.ts), and [example-result.json](../backend/examples/example-result.json) provides a real response from the simulated demo.

## Run

Start the backend with `bash run.sh` from `backend/`, then open http://127.0.0.1:8000/app/. The API serves this directory and the two allowlisted demo recordings. All requests use the same origin. The backend-only Docker recipe does not include this sibling frontend directory; use the documented local setup for the full app.

The interface uploads multipart files, polls actual status, loads returned assets, and plays the analysis audio aligned with event timestamps. Waveforms, spectrograms, frequency tracks, findings, and downloaded Markdown reports come from analysis results. There are no simulated chart values in the running app.

The constructed-scene demo sets `is_simulated=true` with motion assumptions false. The known-speed example is a generated fixed-tone pass-by with a stationary microphone and enables those assumptions only for that example. New recordings reset all three checkboxes. Real uploads require explicit operator assertions before a conditional speed is displayed. Null speed is never rendered as zero.

`sample-api-response.json` is Arjun's original illustrative design reference. The running interface does not load it. Current integration examples and types are under `backend/`.

## Checks

Run `node --test tests/*.test.mjs` (or `npm test`) from this directory. Tests cover cached and failed jobs, polling cancellation/timeouts, backend payloads, evidence references, timestamps, HTTP errors, and conditional speed labels. See `../docs/FRONTEND_INTEGRATION.md` for browser verification.

Satvik owns scientific analysis and the API; Arjun owns interaction, visualization, report presentation, and the demo.
