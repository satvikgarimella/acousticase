# Frontend integration verification

Arjun's original interface was pushed to the separate `master` branch at commit `0dfc950`. The integration branch merges that history and relocates the interface into `frontend/`, preserving his authorship. Its visual layout is retained while the illustrative data flow is replaced with the real API.

## Run the complete app

From `backend/`, use the existing environment and run `bash run.sh`. Open http://127.0.0.1:8000/app/. API documentation remains at `/docs`. The server mounts the sibling frontend folder, so no separate frontend server or package installation is required. `ACOUSTICASE_PORT=8001 bash run.sh` selects another local port; the UI automatically uses its own origin.

Run one backend process at a time against a data directory. This remains a trusted local prototype, not a public authenticated deployment. The existing backend-only Docker recipe omits the sibling frontend; use the local setup for this integrated demo.

## Connected behavior

- Actual multipart file upload, cached/queued/running/failed/complete status handling, bounded polling, and cancellation of client-side waits.
- Playback uses the analysis audio with the same timebase as evidence. Selecting an event seeks to its start and requests playback.
- Waveforms use returned min/max samples. Spectrograms use the generated image. Frequency curves use supported ridge samples and the candidate fit; rejected fits are identified.
- Events and findings use actual evidence references, evidence-type labels, filtering, and waveform zoom. No hard-coded confidence category is shown.
- Optional sound-score history displays one selected group and its onset threshold, explicitly labeled as uncalibrated.
- Motion estimates preserve unsupported/null results. The displayed value is conditional only when the backend reports `conditional_fit`.
- New recordings reset simulated/constructed, stationary-microphone, and fixed-tone assertions. The dedicated synthetic demo supplies its known construction assumptions; the constructed-scene demo supplies neither motion assumption.
- Report downloads retrieve the backend Markdown file. Original audio remains separately downloadable. Warnings, limitations, and simulated labels are visible.
- Example recordings are served from an explicit allowlist, including the constructed scene's attribution. The frontend static mount does not expose the repository root.

## Verification performed

Local checks on 2026-11-21:

| Check | Observed result |
| --- | --- |
| Backend suite | 26 tests passed |
| Frontend Node suite | 9 tests passed |
| Constructed-scene browser run | Actual 25-second case; engine, horn, glass, and siren suggestions |
| Event filter | Four model suggestions shown out of 13 total events |
| Linked horn selection | Playback sought to 6.72 seconds; matching finding and evidence ID displayed |
| Zoom | Selected event interval displayed with updated time axis |
| Report download | Browser-downloaded report matched backend bytes exactly |
| Silence file upload | Zero events, disabled event controls, no supported tone, insufficient speed evidence |
| Synthetic file upload without motion assertions | “Assumptions required”; no numeric speed |
| Known-speed synthetic demo | “72.0 km/h · conditional” |
| New-recording assertions | All three previously checked inputs reset to unchecked |
| Browser console inspection | No captured errors or warnings at the inspected checkpoint |

Browser checks used the in-app browser's current narrow viewport. Other browser engines and desktop/mobile breakpoints were not exhaustively tested. File decoding/size limits and invalid jobs are covered by backend tests; frontend HTTP-error handling is also tested. This integration does not change the scientific limitations in `ML_MODEL_CARD.md`.

`frontend/sample-api-response.json` is the original illustrative design reference and is not loaded by the application. Use `backend/examples/example-result.json`, `example-sound-result.json`, and `frontend-types.ts` for the actual contract.
