import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import chirp

from acousticase.dsp import fit_motion, track_tone, transient_events
from acousticase.models import AnalysisOptions

spec = importlib.util.spec_from_file_location("demo", Path(__file__).parents[1]/"scripts/generate_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.mark.parametrize("speed,distance,frequency", [(12, 15, 500), (20, 25, 600), (35, 35, 850)])
def test_recovers_simulated_motion_from_emission_time_audio(speed, distance, frequency):
    audio, truth = demo.simulate_pass_by(speed=speed, distance=distance, frequency=frequency, transient=False)
    options = AnalysisOptions(assume_constant_tone=True, assume_stationary_microphone=True)
    track, _ = track_tone(audio, options)
    result = fit_motion(track, options)
    assert result["status"] == "conditional_fit", result
    assert result["speed_m_s"] == pytest.approx(speed, rel=0.08)
    assert result["received_closest_approach_s"] == pytest.approx(truth["received_closest_approach_s"], abs=0.2)


def test_never_exposes_speed_without_asserted_assumptions():
    audio, _ = demo.simulate_pass_by(transient=False)
    options = AnalysisOptions()
    track, _ = track_tone(audio, options)
    result = fit_motion(track, options)
    assert result["status"] == "assumptions_required"
    assert result["speed_m_s"] is None


@pytest.mark.parametrize("kind", ["stationary", "noise", "silence", "rising_chirp", "falling_linear_chirp"])
def test_negative_controls_do_not_produce_speed(kind):
    t = np.arange(16000*8)/16000
    audio = {
        "stationary": 0.25*np.sin(2*np.pi*600*t),
        "noise": np.random.default_rng(3).normal(0, 0.1, len(t)),
        "silence": np.zeros_like(t),
        "rising_chirp": 0.25*chirp(t, 400, 8, 800),
        "falling_linear_chirp": 0.25*chirp(t, 800, 8, 400),
    }[kind].astype(np.float32)
    options = AnalysisOptions(assume_constant_tone=True, assume_stationary_microphone=True)
    track, _ = track_tone(audio, options)
    result = fit_motion(track, options)
    assert result["status"] == "insufficient_evidence", result
    assert result["speed_m_s"] is None


def test_transient_time_matches_inserted_burst():
    audio, _ = demo.simulate_pass_by()
    events = transient_events(audio)
    assert any(12.49 <= event["peak_s"] <= 12.57 for event in events), events
    assert all(event["label"] == "Sudden acoustic transient" for event in events)
