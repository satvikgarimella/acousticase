from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tracking_start_s: float = Field(default=0, ge=0)
    tracking_end_s: float | None = Field(default=None, gt=0)
    frequency_min_hz: float = Field(default=80, ge=30, le=3000)
    frequency_max_hz: float = Field(default=1500, ge=50, le=4000)
    speed_of_sound_m_s: float = Field(default=343, ge=300, le=380)
    assume_constant_tone: bool = False
    assume_stationary_microphone: bool = False
    is_simulated: bool = False
    use_classifier: bool = True

    @model_validator(mode="after")
    def valid_ranges(self):
        if self.frequency_max_hz - self.frequency_min_hz < 20:
            raise ValueError("Frequency range must be at least 20 Hz wide.")
        if self.tracking_end_s is not None:
            if self.tracking_end_s <= self.tracking_start_s:
                raise ValueError("Tracking end must follow tracking start.")
            if self.tracking_end_s - self.tracking_start_s > 60:
                raise ValueError("Select at most 60 seconds for tonal tracking.")
        return self


class Event(BaseModel):
    id: str
    start_s: float
    end_s: float
    peak_s: float
    kind: Literal["transient", "sound_class", "tonal_track", "possible_pass_by"]
    label: str
    evidence_type: Literal["measurement", "model_suggestion", "conditional_inference"]
    measurements: dict = Field(default_factory=dict)
    explanation: str


class Finding(BaseModel):
    id: str
    text: str
    evidence_ids: list[str]
    evidence_type: Literal["measurement", "model_suggestion", "conditional_inference"]


class JobStatus(BaseModel):
    analysis_id: str
    status: Literal["queued", "running", "complete", "failed"]
    stage: str
    progress: float = Field(ge=0, le=1)
    error: str | None = None
    results_url: str


class AnalysisResult(BaseModel):
    schema_version: str = "1.0"
    pipeline_version: str
    analysis_id: str
    status: Literal["complete"] = "complete"
    recording: dict
    options: AnalysisOptions
    events: list[Event]
    motion: dict
    findings: list[Finding]
    classifier: dict
    assets: dict[str, str]
    warnings: list[str]
    limitations: list[str]
    processing_seconds: float
