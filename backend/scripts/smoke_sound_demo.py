"""Upload the labeled constructed scene and save a frontend integration example."""
import argparse
import json
from pathlib import Path
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    examples = Path(__file__).resolve().parents[1] / "examples"
    with httpx.Client(base_url=args.base.rstrip("/"), timeout=60) as client:
        with (examples / "constructed-sound-scene.wav").open("rb") as audio:
            response = client.post("/analyses", files={"file": ("constructed-sound-scene.wav", audio, "audio/wav")},
                                   data={"options": json.dumps({"is_simulated": True})})
        response.raise_for_status()
        job = response.json()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            status_response = client.get(f"/analyses/{job['analysis_id']}")
            status_response.raise_for_status()
            status = status_response.json()
            if status["status"] in {"complete", "failed"}:
                break
            time.sleep(.5)
        assert status["status"] == "complete", status
        response = client.get(status["results_url"])
        response.raise_for_status()
        result = response.json()
        assert result["recording"]["is_simulated"] is True
        assert result["motion"]["speed_m_s"] is None
        assert result["classifier"]["status"] == "enabled"
        groups = {e["measurements"].get("sound_group") for e in result["events"] if e["kind"] == "sound_class"}
        assert {"engine", "horn", "glass", "siren"} <= groups, groups
        ids = {e["id"] for e in result["events"]}
        assert all(set(f["evidence_ids"]) <= ids for f in result["findings"])
        for url in result["assets"].values():
            client.get(url).raise_for_status()
        (examples / "example-sound-result.json").write_text(json.dumps(result, indent=2)+"\n")
        timeline = client.get(result["assets"]["sound_scores_url"])
        timeline.raise_for_status()
        (examples / "example-scene-scores.json").write_text(json.dumps(timeline.json(), indent=2)+"\n")
        print(json.dumps({"analysis_id": result["analysis_id"], "groups": sorted(groups),
                          "processing_seconds": result["processing_seconds"], "speed_m_s": result["motion"]["speed_m_s"]}, indent=2))


if __name__ == "__main__":
    main()
