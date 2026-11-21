export type EvidenceType = 'measurement' | 'model_suggestion' | 'conditional_inference';
export interface JobStatus {
  analysis_id: string;
  status: 'queued' | 'running' | 'complete' | 'failed';
  stage: string;
  progress: number; // 0..1; processing progress, not confidence
  error: string | null;
  results_url: string;
}
export interface EvidenceEvent {
  id: string;
  start_s: number;
  end_s: number;
  peak_s: number;
  kind: 'transient' | 'sound_class' | 'tonal_track' | 'possible_pass_by';
  label: string;
  evidence_type: EvidenceType;
  measurements: Record<string, unknown>;
  explanation: string;
}
export interface AnalysisResult {
  schema_version: string;
  pipeline_version: string;
  analysis_id: string;
  status: 'complete';
  recording: {
    filename: string; duration_s: number; sha256: string; size_bytes: number;
    source_sample_rate_hz: number; analysis_sample_rate_hz: number;
    source_channels: number; original_audio_start_s: number;
    timebase: string; is_simulated: boolean;
  };
  options: Record<string, unknown>;
  events: EvidenceEvent[];
  motion: {
    status: 'insufficient_evidence' | 'assumptions_required' | 'conditional_fit';
    speed_m_s: number | null; speed_km_h: number | null;
    received_closest_approach_s: number | null;
    fit: null | {
      time_s: number[]; predicted_frequency_hz: number[];
      rmse_hz: number; constant_baseline_rmse_hz: number;
      linear_baseline_rmse_hz: number; emitted_frequency_hz: number;
      jacobian_condition: number | null;
    };
    reasons: string[]; assumptions: Record<string, unknown>; caveat: string;
  };
  findings: Array<{ id: string; text: string; evidence_ids: string[]; evidence_type: EvidenceType }>;
  classifier: {status: 'enabled' | 'disabled' | 'failed'; model: string | null; [key: string]: unknown};
  assets: Record<'audio_url' | 'waveform_url' | 'spectrogram_url' | 'track_url' | 'report_url' | 'original_url', string> & {sound_scores_url?: string};
  warnings: string[]; limitations: string[]; processing_seconds: number;
}

export interface SoundScoreTimeline {
  timebase: string;
  window_s: number;
  hop_s: number;
  score_meaning: string;
  start_s: number[];
  end_s: number[];
  groups: Array<{key: string; label: string; onset_threshold: number; scores: number[]}>;
}
