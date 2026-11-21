import hashlib
import json
import logging
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import ValidationError

from .classifier import YamnetClassifier
from .config import Settings
from .models import AnalysisOptions, AnalysisResult, JobStatus
from .pipeline import PIPELINE_VERSION, run_analysis, write_json

logger = logging.getLogger(__name__)
ASSETS = {"playback.wav": "audio/wav", "waveform.json": "application/json",
          "spectrogram.png": "image/png", "track.json": "application/json",
          "report.md": "text/markdown", "original": "application/octet-stream"}


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.classifier = YamnetClassifier(settings.model_dir)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio")
        self.capacity = threading.BoundedSemaphore(settings.max_pending_jobs)
        self.lock = threading.Lock()
        for status_path in self.root.glob("*/status.json"):
            status = json.loads(status_path.read_text())
            if status["status"] in {"running", "queued"}:
                status.update(status="failed", stage="Interrupted", error="Server restarted before analysis completed. Re-upload to retry.")
                write_json(status_path, status)

    def folder(self, analysis_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", analysis_id):
            raise HTTPException(404, "Analysis not found.")
        folder = self.root/analysis_id
        if not (folder/"status.json").exists():
            raise HTTPException(404, "Analysis not found.")
        return folder

    def status(self, analysis_id):
        return json.loads((self.folder(analysis_id)/"status.json").read_text())

    def set_status(self, folder, analysis_id, status, stage, progress, error=None):
        payload = JobStatus(analysis_id=analysis_id, status=status, stage=stage, progress=progress,
                            error=error, results_url=f"/analyses/{analysis_id}/results")
        write_json(folder/"status.json", payload.model_dump())

    def process(self, folder, analysis_id, metadata, options):
        try:
            run_analysis(folder, analysis_id, metadata, options, self.classifier, self.settings.max_duration_s,
                         lambda stage, progress: self.set_status(folder, analysis_id, "running", stage, progress))
            self.set_status(folder, analysis_id, "complete", "Complete", 1)
        except Exception as exc:
            logger.exception("Analysis %s failed", analysis_id)
            message = str(exc) if isinstance(exc, ValueError) else f"Could not process this recording ({type(exc).__name__}). See server logs."
            self.set_status(folder, analysis_id, "failed", "Failed", 1, message)
        finally:
            self.capacity.release()


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = Runtime(config)
        yield
        app.state.runtime.executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title="AcoustiCase — Satvik's backend", version=PIPELINE_VERSION,
                  description="Local experimental audio investigation API. Single worker; no authentication. See README before deployment.", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing():
        return '<!doctype html><title>AcoustiCase backend</title><body style="font:18px system-ui;max-width:680px;margin:80px auto"><h1>AcoustiCase backend</h1><p>Audio processing, evidence events, and conditional Doppler analysis.</p><p><a href="/docs">Open interactive API documentation</a></p><p><a href="/health">Check server and classifier status</a></p><p>Arjun can build the investigation interface against this API.</p></body>'

    @app.get("/health")
    def health():
        return {"status": "ok", "pipeline_version": PIPELINE_VERSION,
                "classifier_available": app.state.runtime.classifier.available,
                "max_duration_s": config.max_duration_s, "max_upload_bytes": config.max_upload_bytes}

    @app.post("/analyses", response_model=JobStatus, status_code=202)
    async def analyze(file: UploadFile = File(...), options: str = Form("{}")):
        try:
            parsed = AnalysisOptions.model_validate_json(options)
        except ValidationError as exc:
            await file.close()
            errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
            raise HTTPException(422, errors) from exc
        runtime = app.state.runtime
        staging = runtime.root/f"upload-{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with staging.open("wb") as output:
                while chunk := await file.read(1024*1024):
                    size += len(chunk)
                    if size > config.max_upload_bytes:
                        raise HTTPException(413, "Upload exceeds the configured size limit.")
                    digest.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise HTTPException(400, "Empty upload.")
            content_hash = digest.hexdigest()
            cache_key = f"{content_hash}|{parsed.model_dump_json()}|{PIPELINE_VERSION}|{runtime.classifier.cache_tag}"
            analysis_id = hashlib.sha256(cache_key.encode()).hexdigest()[:32]
            folder = runtime.root/analysis_id
            with runtime.lock:
                if (folder/"status.json").exists():
                    status = runtime.status(analysis_id)
                    if status["status"] != "failed":
                        return status
                if not runtime.capacity.acquire(blocking=False):
                    raise HTTPException(429, "Processing queue is full. Retry after a current analysis finishes.", headers={"Retry-After": "5"})
                try:
                    folder.mkdir(exist_ok=True)
                    staging.replace(folder/"original")
                    name = Path((file.filename or "recording").replace("\\", "/")).name[:200]
                    metadata = {"filename": name, "sha256": content_hash, "size_bytes": size}
                    write_json(folder/"metadata.json", metadata)
                    runtime.set_status(folder, analysis_id, "queued", "Queued", 0)
                    runtime.executor.submit(runtime.process, folder, analysis_id, metadata, parsed)
                except Exception:
                    runtime.capacity.release()
                    raise
            return runtime.status(analysis_id)
        finally:
            staging.unlink(missing_ok=True)
            await file.close()

    @app.get("/analyses/{analysis_id}", response_model=JobStatus)
    def get_status(analysis_id: str):
        return app.state.runtime.status(analysis_id)

    @app.get("/analyses/{analysis_id}/results", response_model=AnalysisResult)
    def get_results(analysis_id: str):
        runtime = app.state.runtime
        status = runtime.status(analysis_id)
        if status["status"] != "complete":
            raise HTTPException(409, {"status": status["status"], "error": status["error"]})
        return json.loads((runtime.folder(analysis_id)/"result.json").read_text())

    @app.get("/analyses/{analysis_id}/assets/{name}")
    def get_asset(analysis_id: str, name: str):
        runtime = app.state.runtime
        if name not in ASSETS:
            raise HTTPException(404, "Asset not found.")
        folder = runtime.folder(analysis_id)
        if runtime.status(analysis_id)["status"] != "complete":
            raise HTTPException(409, "Analysis has not completed.")
        path = folder/name
        if not path.is_file():
            raise HTTPException(404, "Asset not found.")
        return FileResponse(path, media_type=ASSETS[name], filename="original-upload" if name == "original" else None)

    return app


app = create_app()
