import json
import time
from pathlib import Path

from .audio import SAMPLE_RATE, decode_audio, waveform_envelope
from .classifier import YamnetClassifier
from .dsp import fit_motion, save_spectrogram, track_tone, transient_events
from .models import AnalysisOptions, AnalysisResult, Event, Finding

PIPELINE_VERSION = "0.1.0"
LIMITATIONS = [
    "Experimental acoustic analysis, not validated accident reconstruction or forensic certification.",
    "No spatial path, compass direction, source identity, or accident causation is determined.",
    "Pitch changes can be caused by changing RPM/source pitch, echoes, multiple sources, or microphone motion.",
    "Conditional speeds assume a constant emitted tone, stationary microphone, constant-speed straight-line pass, and the configured sound speed.",
    "Model scores and fit residuals are not calibrated confidence probabilities. Missing detections do not establish that an event did not occur.",
    "All displayed timestamps refer to the first decoded audio sample in playback.wav. Add original_audio_start_s to align with the source media timestamps.",
]


def write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_analysis(job_dir: Path, analysis_id: str, metadata: dict, options: AnalysisOptions,
                 classifier: YamnetClassifier, max_duration_s: float, progress) -> AnalysisResult:
    began = time.monotonic()
    progress("Decoding audio", 0.1)
    decoded = decode_audio(job_dir/"original", job_dir/"playback.wav", max_duration_s)
    x = decoded.samples
    duration = len(x)/SAMPLE_RATE
    warnings = list(decoded.warnings)
    progress("Generating waveform and spectrogram", 0.25)
    write_json(job_dir/"waveform.json", waveform_envelope(x))
    save_spectrogram(x, job_dir/"spectrogram.png")
    progress("Detecting transient events", 0.4)
    events = transient_events(x)
    progress("Classifying candidate sounds", 0.5)
    try:
        suggestions, classifier_info = classifier.classify(x, options.use_classifier)
        events.extend(suggestions)
    except Exception as exc:
        classifier_info = {"status": "failed", "model": "Google YAMNet v1", "reason": f"Local classifier failed ({type(exc).__name__}); signal measurements are still available."}
        warnings.append(classifier_info["reason"])
    if classifier_info["status"] == "disabled":
        warnings.append("Sound classification is disabled. Transients are measured, but no vehicle or other sound-class labels are inferred.")
    progress("Tracking a tonal ridge", 0.65)
    track, tracking_warnings = track_tone(x, options)
    warnings.extend(tracking_warnings)
    progress("Testing the pass-by model", 0.8)
    motion = fit_motion(track, options)
    write_json(job_dir/"track.json", track)
    if track["support_fraction"] >= 0.65:
        events.append({
            "id": "tone-1", "start_s": track["start_s"], "end_s": track["end_s"],
            "peak_s": (track["start_s"]+track["end_s"])/2,
            "kind": "tonal_track", "label": "Tracked tonal component", "evidence_type": "measurement",
            "measurements": {"support_fraction": track["support_fraction"]},
            "explanation": "A persistent narrow spectral ridge was tracked. This alone does not identify a vehicle or demonstrate motion.",
        })
    if motion["status"] == "conditional_fit":
        tcpa = motion["received_closest_approach_s"]
        events.append({
            "id": "pass-by-1", "start_s": track["start_s"], "end_s": track["end_s"], "peak_s": tcpa,
            "kind": "possible_pass_by", "label": "Conditional pass-by fit", "evidence_type": "conditional_inference",
            "measurements": {"speed_m_s": motion["speed_m_s"], "speed_km_h": motion["speed_km_h"], "received_closest_approach_s": tcpa, "fit_rmse_hz": motion["fit"]["rmse_hz"]},
            "explanation": "The curve fits a simple Doppler model under user-asserted assumptions. The timestamp is arrival of the sound emitted at closest approach.",
        })
    events.sort(key=lambda e: (e["start_s"], e["id"]))
    findings = []
    for event in events:
        if event["kind"] == "transient":
            text = f"A sudden acoustic transient was measured at {event['peak_s']:.2f} s. Its cause is undetermined."
        elif event["kind"] == "sound_class":
            text = f"The sound classifier suggests {event['measurements']['class_label'].lower()} between {event['start_s']:.2f} and {event['end_s']:.2f} s; this is unverified."
        elif event["kind"] == "possible_pass_by":
            text = f"Under the stated assumptions, the fitted speed is {motion['speed_km_h']:.1f} km/h and the received closest-approach time is {motion['received_closest_approach_s']:.2f} s."
        else:
            text = f"A tonal component was tracked between {event['start_s']:.2f} and {event['end_s']:.2f} s; the source is unidentified."
        findings.append(Finding(id=f"finding-{len(findings)+1}", text=text, evidence_ids=[event["id"]], evidence_type=event["evidence_type"]))
    base = f"/analyses/{analysis_id}/assets"
    result = AnalysisResult(
        pipeline_version=PIPELINE_VERSION, analysis_id=analysis_id,
        recording={**metadata, "duration_s": duration, "analysis_sample_rate_hz": SAMPLE_RATE,
                   "source_sample_rate_hz": decoded.source_sample_rate, "source_channels": decoded.source_channels,
                   "original_audio_start_s": decoded.original_audio_start_s,
                   "timebase": "seconds_from_first_decoded_audio_sample", "is_simulated": options.is_simulated},
        options=options, events=[Event(**e) for e in events], motion=motion, findings=findings,
        classifier=classifier_info,
        assets={"audio_url": f"{base}/playback.wav", "waveform_url": f"{base}/waveform.json",
                "spectrogram_url": f"{base}/spectrogram.png", "track_url": f"{base}/track.json",
                "report_url": f"{base}/report.md", "original_url": f"{base}/original"},
        warnings=warnings, limitations=LIMITATIONS, processing_seconds=round(time.monotonic()-began, 3),
    )
    progress("Writing evidence report", 0.95)
    report = ["# AcoustiCase investigation report", "", "Experimental acoustic analysis. Generated from structured measurements and model suggestions; no generative model is used.", "",
              f"Analysis ID: {analysis_id}", f"Duration: {duration:.2f} seconds", f"Simulated recording: {options.is_simulated}", "",
              "## Findings", ""]
    if not findings:
        report.append("No events met the configured thresholds. This does not establish that no relevant event occurred.")
    for finding in findings:
        report.append(f"- **{finding.evidence_type}** — {finding.text} Evidence IDs: {', '.join(finding.evidence_ids)}.")
    report.extend(["", "## Motion estimate", "", f"Status: {motion['status']}"])
    report.extend(f"- {reason}" for reason in motion["reasons"])
    report.extend(["", "## Limitations", ""] + [f"- {item}" for item in LIMITATIONS])
    report.extend(["", "## Processing warnings", ""] + [f"- {item}" for item in warnings])
    (job_dir/"report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    write_json(job_dir/"result.json", result.model_dump(mode="json"))
    return result
