"""Generate explicitly simulated audio using emission-time sampling.

This simulator constructs source emission samples and interpolates their
arrival times; it does not call the estimator's frequency model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def simulate_pass_by(speed=20.0, distance=25.0, duration=16.0, cpa_emission=7.5, frequency=600.0,
                     sample_rate=16000, seed=42, transient=True):
    rng = np.random.default_rng(seed)
    received = np.arange(round(duration*sample_rate))/sample_rate
    emitted = np.arange(-2*sample_rate, round((duration+1)*sample_rate)*2)/(2*sample_rate)
    ranges = np.sqrt(distance**2+(speed*(emitted-cpa_emission))**2)
    arrival = emitted+ranges/343.0
    phase = 2*np.pi*frequency*emitted
    source = (0.35*np.sin(phase)+0.05*np.sin(2*phase))*distance/ranges
    audio = np.interp(received, arrival, source) + rng.normal(0, 0.0015, len(received))
    transient_time = 12.5
    if transient and duration > transient_time+0.1:
        n = int(0.06*sample_rate)
        burst = rng.normal(0, 0.32, n)*np.hanning(n)
        start = int(transient_time*sample_rate)
        audio[start:start+n] += burst
    return audio.astype(np.float32), {
        "is_simulated": True, "source": "generated constant-tone straight pass; not a real vehicle recording",
        "speed_m_s": speed, "speed_km_h": speed*3.6, "closest_distance_m": distance,
        "emitted_frequency_hz": frequency, "geometric_closest_approach_emission_s": cpa_emission,
        "received_closest_approach_s": cpa_emission+distance/343,
        "transient_start_s": transient_time if transient else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audio, truth = simulate_pass_by()
    sf.write(args.output/"simulated-pass-by.wav", audio, 16000)
    (args.output/"simulated-pass-by.truth.json").write_text(json.dumps(truth, indent=2)+"\n")
    t = np.arange(16000*8)/16000
    stationary = (0.25*np.sin(2*np.pi*600*t)).astype(np.float32)
    sf.write(args.output/"stationary-tone.wav", stationary, 16000)
    sf.write(args.output/"silence.wav", np.zeros(16000*3), 16000)
    print(f"Created simulated pass-by, stationary-tone, and silence examples in {args.output}.")


if __name__ == "__main__":
    main()
