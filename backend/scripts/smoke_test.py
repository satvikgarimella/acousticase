"""Exercise a running server and save Arjun's example API payloads."""
import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    examples = Path(__file__).resolve().parents[1]/"examples"
    base = args.base.rstrip("/")
    with httpx.Client(timeout=60) as client:
        health = client.get(base+"/health")
        health.raise_for_status()
        print("health", health.json(), flush=True)
        with (examples/"simulated-pass-by.wav").open("rb") as audio:
            response = client.post(base+"/analyses", files={"file": ("simulated-pass-by.wav", audio, "audio/wav")},
                                   data={"options": json.dumps({"is_simulated": True, "assume_constant_tone": True, "assume_stationary_microphone": True})})
        response.raise_for_status()
        job = response.json()
        print("submitted", job["analysis_id"], flush=True)
        for _ in range(90):
            status_response = client.get(base+"/analyses/"+job["analysis_id"])
            status_response.raise_for_status()
            state = status_response.json()
            if state["status"] in {"complete", "failed"}:
                break
            time.sleep(1)
        if state["status"] != "complete":
            raise RuntimeError(state)
        results_response = client.get(base+state["results_url"])
        results_response.raise_for_status()
        result = results_response.json()
        assert result["recording"]["is_simulated"] is True
        assert result["motion"]["status"] == "conditional_fit", result["motion"]
        assert abs(result["motion"]["speed_km_h"]-72) < 6
        assert any(event["kind"] == "transient" and 12.49 < event["peak_s"] < 12.57 for event in result["events"])
        (examples/"example-status.json").write_text(json.dumps(state, indent=2)+"\n")
        (examples/"example-result.json").write_text(json.dumps(result, indent=2)+"\n")
        for key in ["waveform_url", "track_url"]:
            payload = client.get(base+result["assets"][key])
            payload.raise_for_status()
            (examples/(key.replace("_url", "")+".json")).write_text(payload.text.rstrip()+"\n")
        image = client.get(base+result["assets"]["spectrogram_url"])
        image.raise_for_status()
        (examples/"spectrogram.png").write_bytes(image.content)
        report = client.get(base+result["assets"]["report_url"])
        report.raise_for_status()
        (examples/"example-report.md").write_text(report.text.rstrip()+"\n")
        if "sound_scores_url" in result["assets"]:
            scores = client.get(base+result["assets"]["sound_scores_url"])
            scores.raise_for_status()
            (examples/"sound-scores.json").write_text(scores.text+"\n")
        schema = client.get(base+"/openapi.json")
        schema.raise_for_status()
        (examples.parent/"openapi.json").write_text(json.dumps(schema.json(), indent=2)+"\n")
        assert client.get(base+"/docs").status_code == 200
        print(json.dumps({"speed_km_h": result["motion"]["speed_km_h"],
                          "received_cpa_s": result["motion"]["received_closest_approach_s"],
                          "fit_rmse_hz": result["motion"]["fit"]["rmse_hz"],
                          "event_count": len(result["events"]), "classifier_status": result["classifier"]["status"],
                          "runtime_s": result["processing_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
