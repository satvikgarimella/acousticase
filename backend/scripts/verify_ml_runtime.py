"""Optional real-model checks: bounded inference, controls, and API evidence assets."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import tensorflow as tf
from fastapi.testclient import TestClient

from acousticase.api import create_app
from acousticase.audio import decode_audio
from acousticase.classifier import YamnetClassifier
from acousticase.config import Settings


def main():
    classifier = YamnetClassifier(ROOT / "models/yamnet")
    # Cross two chunk boundaries using non-silent content. Compare with the
    # upstream whole-waveform calculation, including the final padded patch.
    samples = np.random.default_rng(47).normal(0, .025, 16000*65+137).astype(np.float32)
    scores, embeddings = classifier.infer(samples)
    reference_scores, reference_embeddings, _ = classifier.model(tf.constant(samples))
    np.testing.assert_allclose(scores, reference_scores.numpy(), atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(embeddings, reference_embeddings.numpy(), atol=2e-4, rtol=2e-4)
    for length in (4000, 15600, 15601, 23280, 23281):
        short = samples[:length]
        actual, _ = classifier.infer(short)
        expected, _, _ = classifier.model(tf.constant(short))
        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=2e-4)
    events, _ = classifier.classify(np.zeros(16000*5, np.float32))
    assert not events, events
    sample = sorted((ROOT / "data/ml/esc50/audio").glob("1-*-42.wav"))[0]
    with tempfile.TemporaryDirectory() as temp:
        decoded = decode_audio(sample, Path(temp)/"playback.wav", 6)
        expected_events, _ = classifier.classify(decoded.samples)
        settings = Settings(data_dir=Path(temp)/"api", model_dir=ROOT/"models/yamnet")
        with TestClient(create_app(settings)) as client:
            assert client.get("/health").json()["pipeline_version"] == "0.2.0"
            response = client.post("/analyses", files={"file": (sample.name, sample.read_bytes())})
            assert response.status_code == 202, response.text
            job = response.json()
            deadline = time.monotonic()+60
            while time.monotonic() < deadline:
                status = client.get(f"/analyses/{job['analysis_id']}").json()
                if status["status"] in {"complete", "failed"}:
                    break
                time.sleep(.05)
            assert status["status"] == "complete", status
            result = client.get(status["results_url"]).json()
            assert result["classifier"]["status"] == "enabled"
            assert [e for e in result["events"] if e["kind"] == "sound_class"] == expected_events
            assert result["motion"]["speed_m_s"] is None
            for url in result["assets"].values():
                assert client.get(url).status_code == 200
            timeline = client.get(result["assets"]["sound_scores_url"]).json()
            assert len(timeline["groups"]) == 12
            assert all(len(g["scores"]) == len(timeline["start_s"]) for g in timeline["groups"])
            ids = {e["id"] for e in result["events"]}
            assert all(set(f["evidence_ids"]) <= ids for f in result["findings"])
    print(json.dumps({"status": "passed", "whole_waveform_equivalence_frames": len(scores),
        "short_clip_boundaries": 5, "silence_events": len(events), "live_api_asset_checks": "passed"}, indent=2))


if __name__ == "__main__":
    main()
