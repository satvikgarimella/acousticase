import hashlib
import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from acousticase.api import create_app
from acousticase.config import Settings


def wav_bytes(seconds=2, rate=22050, channels=1):
    data = np.zeros((int(seconds*rate), channels), dtype=np.float32)
    target = io.BytesIO()
    sf.write(target, data, rate, format="WAV")
    return target.getvalue()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, model_dir=None))) as c:
        yield c


def wait_for_job(client, job_id):
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        status = client.get(f"/analyses/{job_id}").json()
        if status["status"] in {"complete", "failed"}:
            return status
        time.sleep(0.03)
    pytest.fail("Analysis timed out")


def test_upload_poll_result_assets_and_content_cache(client):
    original = wav_bytes(channels=2)
    uploaded = client.post("/analyses", files={"file": ("clip.wav", original)})
    assert uploaded.status_code == 202
    job_id = uploaded.json()["analysis_id"]
    status = wait_for_job(client, job_id)
    assert status["status"] == "complete", status
    response = client.get(status["results_url"])
    assert response.status_code == 200
    result = response.json()
    assert result["recording"]["duration_s"] == pytest.approx(2, abs=0.001)
    assert result["recording"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert result["motion"]["speed_m_s"] is None
    assert result["classifier"]["status"] == "disabled"
    for url in result["assets"].values():
        assert client.get(url).status_code == 200
    assert client.get(result["assets"]["original_url"]).content == original
    repeated = client.post("/analyses", files={"file": ("clip.wav", original)})
    assert repeated.json()["analysis_id"] == job_id
    assert repeated.json()["status"] == "complete"
    assert client.get(f"/analyses/{job_id}/assets/metadata.json").status_code == 404


def test_invalid_options_and_empty_upload(client):
    assert client.post("/analyses", files={"file": ("empty.wav", b"")}).status_code == 400
    assert client.post("/analyses", files={"file": ("x.wav", wav_bytes())},
                       data={"options": '{"frequency_min_hz": 800,"frequency_max_hz": 400}'}).status_code == 422
    assert client.post("/analyses", files={"file": ("x.wav", wav_bytes())}, data={"options": '{"tracking_start_s":NaN}'}).status_code == 422


def test_bad_file_becomes_failed_job(client):
    response = client.post("/analyses", files={"file": ("bad.wav", b"not an audio file")})
    job_id = response.json()["analysis_id"]
    assert wait_for_job(client, job_id)["status"] == "failed"
    assert client.get(f"/analyses/{job_id}/results").status_code == 409


def test_duration_and_size_limits(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_duration_s=1, max_upload_bytes=200000))) as client:
        response = client.post("/analyses", files={"file": ("long.wav", wav_bytes(2))})
        status = wait_for_job(client, response.json()["analysis_id"])
        assert status["status"] == "failed"
        assert "limit" in status["error"]
        assert client.post("/analyses", files={"file": ("large.wav", b"0"*200001)}).status_code == 413


def test_real_event_evidence_links(client):
    t = np.arange(16000*3)/16000
    x = np.zeros_like(t)
    x[16000:16080] = np.hanning(80)*0.8
    buffer = io.BytesIO()
    sf.write(buffer, x, 16000, format="WAV")
    response = client.post("/analyses", files={"file": ("pulse.wav", buffer.getvalue())})
    status = wait_for_job(client, response.json()["analysis_id"])
    result = client.get(status["results_url"]).json()
    ids = {e["id"] for e in result["events"]}
    assert result["findings"]
    assert all(set(f["evidence_ids"]) <= ids for f in result["findings"])


def test_integrated_frontend_and_allowlisted_demos(client):
    root = client.get("/")
    assert root.status_code == 200 and str(root.url).endswith("/app/")
    assert 'type="module" src="app.js"' in root.text
    assert client.get("/app/api-client.js").status_code == 200
    demo = client.get("/demo-audio/constructed-sound-scene.wav")
    assert demo.status_code == 200 and demo.content[:4] == b"RIFF"
    assert client.get("/demo-audio/CONSTRUCTED-SCENE-ATTRIBUTION.md").status_code == 200
    assert client.get("/demo-audio/README.md").status_code == 404
    assert client.get("/app/%2e%2e/backend/pyproject.toml").status_code == 404
