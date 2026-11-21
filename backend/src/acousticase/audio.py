from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


@dataclass
class DecodedAudio:
    samples: np.ndarray
    source_sample_rate: int
    source_channels: int
    original_audio_start_s: float
    warnings: list[str]


def decode_audio(path: Path, output: Path, max_duration_s: float) -> DecodedAudio:
    """Decode the first audio stream without silently squeezing timestamp gaps."""
    chunks = []
    count = 0
    first_time = None
    previous_end = None
    source_rate = 0
    source_channels = 0
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("The uploaded file has no audio stream.")
        stream = container.streams.audio[0]
        if stream.duration is not None and stream.time_base is not None:
            if float(stream.duration * stream.time_base) > max_duration_s + 0.1:
                raise ValueError(f"Audio exceeds the {max_duration_s:g}-second limit. Upload a shorter clip.")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            source_rate = frame.sample_rate
            source_channels = len(frame.layout.channels)
            frame_time = float(frame.time) if frame.time is not None else None
            if frame_time is not None:
                if first_time is None:
                    first_time = frame_time
                if previous_end is not None and abs(frame_time - previous_end) > 0.1:
                    raise ValueError("Audio timestamps contain a gap or discontinuity over 100 ms; export a continuous clip first.")
                previous_end = frame_time + frame.samples / frame.sample_rate
            for out_frame in resampler.resample(frame):
                data = out_frame.to_ndarray().reshape(-1).astype(np.float32)
                count += len(data)
                if count > int(max_duration_s * SAMPLE_RATE):
                    raise ValueError(f"Audio exceeds the {max_duration_s:g}-second limit. Upload a shorter clip.")
                chunks.append(data)
        for out_frame in resampler.resample(None):
            chunks.append(out_frame.to_ndarray().reshape(-1).astype(np.float32))
    if not chunks:
        raise ValueError("No audio samples could be decoded.")
    samples = np.concatenate(chunks)
    if len(samples) < SAMPLE_RATE // 4:
        raise ValueError("Upload at least 0.25 seconds of audio.")
    if len(samples) > max_duration_s * SAMPLE_RATE:
        raise ValueError(f"Audio exceeds the {max_duration_s:g}-second limit.")
    if not np.isfinite(samples).all():
        raise ValueError("Decoded audio contains invalid sample values.")
    warnings = []
    if np.mean(np.abs(samples) >= 0.999) > 0.001:
        warnings.append("Clipping detected; clipped waveforms can distort event and frequency measurements.")
    if source_channels > 1:
        warnings.append("Channels were mixed to mono for analysis; stereo direction and spatial paths are not estimated.")
    # FLOAT WAV preserves the decoded mono values; the source upload is kept separately.
    sf.write(output, samples, SAMPLE_RATE, subtype="FLOAT")
    return DecodedAudio(samples, source_rate, source_channels, first_time or 0, warnings)


def waveform_envelope(samples: np.ndarray, bins: int = 1800) -> dict:
    edges = np.linspace(0, len(samples), min(bins, len(samples)) + 1, dtype=int)
    return {
        "timebase": "seconds_from_first_decoded_audio_sample",
        "start_s": [float(x / SAMPLE_RATE) for x in edges[:-1]],
        "end_s": [float(x / SAMPLE_RATE) for x in edges[1:]],
        "min": [float(np.min(samples[a:b])) for a, b in zip(edges[:-1], edges[1:])],
        "max": [float(np.max(samples[a:b])) for a, b in zip(edges[:-1], edges[1:])],
    }
