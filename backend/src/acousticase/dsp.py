from pathlib import Path

import numpy as np
from scipy import ndimage, signal
from scipy.optimize import least_squares

from .audio import SAMPLE_RATE
from .models import AnalysisOptions


def transient_events(x: np.ndarray) -> list[dict]:
    frame, hop = 320, 160
    if len(x) < frame or np.max(np.abs(x)) < 1e-6:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    rise = np.maximum(0, db - np.r_[db[0], db[:-1]])
    peaks, _ = signal.find_peaks(rise, height=6, prominence=4, distance=15)
    peaks = [i for i in peaks if np.max(db[i:i+6]) > max(-65, float(np.median(db)) + 6)]
    events = []
    for i in sorted(sorted(peaks, key=lambda i: rise[i], reverse=True)[:30]):
        a, b = max(0, i * hop - frame), min(len(x), i * hop + 2 * frame)
        peak = (a + int(np.argmax(np.abs(x[a:b])))) / SAMPLE_RATE
        events.append({
            "id": f"transient-{len(events)+1}", "start_s": max(0, peak - 0.08),
            "end_s": min(len(x) / SAMPLE_RATE, peak + 0.12), "peak_s": peak,
            "kind": "transient", "label": "Sudden acoustic transient",
            "evidence_type": "measurement",
            "measurements": {"rms_rise_db": round(float(rise[i]), 2), "peak_dbfs": round(float(20*np.log10(max(np.max(np.abs(x[a:b])), 1e-8))), 2)},
            "explanation": "A rapid increase in recorded signal energy. Its cause is not determined; this does not establish a collision.",
        })
    return events


def save_spectrogram(x: np.ndarray, path: Path) -> dict:
    # Import lazily so health checks and schema generation stay lightweight.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    f, t, z = signal.stft(x, SAMPLE_RATE, nperseg=min(1024, len(x)), noverlap=min(768, len(x)//2), boundary=None)
    db = 20 * np.log10(np.maximum(np.abs(z), 1e-8))
    stride = max(1, int(np.ceil(len(t) / 1000)))
    figure = Figure(figsize=(12, 4), dpi=120, facecolor="#101622")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.set_facecolor("#101622")
    vmax = max(-80, float(np.percentile(db, 99.5)))
    mesh = ax.pcolormesh(t[::stride], f, db[:, ::stride], shading="auto", cmap="magma", vmin=vmax-65, vmax=vmax)
    ax.set(xlabel="Time from audio start (s)", ylabel="Frequency (Hz)", ylim=(0, 4000), xlim=(0, len(x)/SAMPLE_RATE))
    ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    bar = figure.colorbar(mesh, ax=ax, label="STFT magnitude (dB, uncalibrated)")
    bar.ax.tick_params(colors="#cbd5e1")
    bar.ax.yaxis.label.set_color("#cbd5e1")
    figure.tight_layout()
    figure.savefig(path)
    return {"frequency_min_hz": 0, "frequency_max_hz": 4000, "duration_s": len(x)/SAMPLE_RATE}


def observed_frequency(t: np.ndarray, f0: float, speed: float, tau: float, received_cpa: float, c: float = 343) -> np.ndarray:
    """Constant-speed straight pass with emission-to-reception propagation delay.

    tau = closest distance / speed. received_cpa is when sound emitted at
    geometric closest approach reaches the microphone, not the emission time.
    """
    distance = speed * tau
    tr = np.asarray(t) - (received_cpa - distance / c)
    emission = (c*c*tr - np.sqrt(c*c*speed*speed*tr*tr + (c*c-speed*speed)*distance*distance)) / (c*c-speed*speed)
    radial = speed*speed*emission / np.sqrt(distance*distance + speed*speed*emission*emission)
    return f0 / (1 + radial / c)


def track_tone(x: np.ndarray, options: AnalysisOptions) -> tuple[dict, list[str]]:
    duration = len(x) / SAMPLE_RATE
    start = options.tracking_start_s
    end = options.tracking_end_s if options.tracking_end_s is not None else min(duration, start+30)
    if start >= duration or end > duration + 0.01:
        raise ValueError("Tracking window lies outside the recording.")
    end = min(end, duration)
    warnings = []
    if options.tracking_end_s is None and end < duration:
        warnings.append("Tonal tracking covers only the first 30 seconds of the selected start. Submit a new analysis with a different window to inspect later audio.")
    segment = x[round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)]
    empty = {"start_s": start, "end_s": end, "time_s": [], "frequency_hz": [], "local_contrast_db": [], "support_fraction": 0.0}
    if len(segment) < 4096 or np.max(np.abs(segment)) < 1e-6:
        return empty, warnings
    freqs, times, z = signal.stft(segment, SAMPLE_RATE, nperseg=4096, noverlap=3776, nfft=8192, boundary=None, padded=False)
    band = (freqs >= options.frequency_min_hz) & (freqs <= options.frequency_max_hz)
    freqs = freqs[band]
    db = 20*np.log10(np.maximum(np.abs(z[band]), 1e-9))
    # Local spectral contrast rewards a narrow ridge rather than broadband energy.
    contrast = db - ndimage.median_filter(db, size=(31, 1), mode="nearest")
    emission = np.clip((contrast - 6)/6, -2, 5)
    nfreq, ntime = emission.shape
    previous = emission[:, 0].copy()
    parents = np.empty((ntime, nfreq), dtype=np.int16)
    idx = np.arange(nfreq)
    offsets = np.arange(-6, 7)
    predecessor = idx[:, None] + offsets[None, :]
    valid = (predecessor >= 0) & (predecessor < nfreq)
    predecessor = np.clip(predecessor, 0, nfreq-1)
    for j in range(1, ntime):
        alternatives = previous[predecessor] - 0.3*np.abs(offsets)[None, :]
        alternatives[~valid] = -np.inf
        choice = np.argmax(alternatives, axis=1)
        parents[j] = predecessor[idx, choice]
        previous = alternatives[idx, choice] + emission[:, j]
        previous -= np.max(previous)
    ridge = np.empty(ntime, dtype=int)
    ridge[-1] = int(np.argmax(previous))
    for j in range(ntime-1, 0, -1):
        ridge[j-1] = parents[j, ridge[j]]
    # Dynamic-programming penalties can choose a shoulder of a broad peak.
    # Recenter within the same local ridge before sub-bin interpolation.
    neighborhoods = np.clip(ridge[:, None] + np.arange(-4, 5)[None, :], 0, nfreq-1)
    ridge = neighborhoods[np.arange(ntime), np.argmax(db[neighborhoods, np.arange(ntime)[:, None]], axis=1)]
    left, right = np.maximum(0, ridge-1), np.minimum(nfreq-1, ridge+1)
    columns = np.arange(ntime)
    a, b, d = db[left, columns], db[ridge, columns], db[right, columns]
    denominator = a-2*b+d
    delta = np.zeros(ntime)
    safe = np.abs(denominator) > 1e-6
    delta[safe] = np.clip(0.5*(a[safe]-d[safe])/denominator[safe], -0.5, 0.5)
    frequency = freqs[ridge] + delta*(freqs[1]-freqs[0])
    frequency = ndimage.median_filter(frequency, size=5, mode="nearest")
    local_contrast = contrast[ridge, columns]
    # Contrast alone does not make a near-silent tone meaningful.
    supported = (local_contrast >= 8) & (db[ridge, columns] >= -65)
    return {
        "start_s": start, "end_s": end,
        "time_s": (times+start).tolist(), "frequency_hz": frequency.tolist(),
        "local_contrast_db": local_contrast.tolist(), "supported": supported.tolist(),
        "support_fraction": float(np.mean(supported)),
    }, warnings


def fit_motion(track: dict, options: AnalysisOptions) -> dict:
    result = {
        "status": "insufficient_evidence", "speed_m_s": None, "speed_km_h": None,
        "received_closest_approach_s": None, "fit": None, "reasons": [],
        "assumptions": {
            "constant_emitted_tone_asserted": options.assume_constant_tone,
            "stationary_microphone_asserted": options.assume_stationary_microphone,
            "model": "single constant-speed straight-line source; stationary air; fixed emitted frequency",
            "speed_of_sound_m_s": options.speed_of_sound_m_s,
        },
        "caveat": "A matching curve does not prove motion: a changing source pitch can imitate Doppler. No spatial direction, vehicle identity, or accident cause is inferred.",
    }
    t = np.asarray(track["time_s"])
    f = np.asarray(track["frequency_hz"])
    supported = np.asarray(track.get("supported", []), dtype=bool)
    if len(t) < 30 or track["support_fraction"] < 0.65:
        result["reasons"].append("No sufficiently persistent, narrow tonal ridge in this window.")
        return result
    t, f = t[supported], f[supported]
    if t[-1]-t[0] < 3:
        result["reasons"].append("At least three seconds of supported tone are required.")
        return result
    if np.percentile(f, 90)-np.percentile(f, 10) < max(8, 0.03*np.median(f)):
        result["reasons"].append("The tone is too steady to support a pass-by fit.")
        return result
    stride = max(1, int(np.ceil(len(t)/600)))
    t, f = t[::stride], f[::stride]
    low = [max(20, np.min(f)*0.7), 2, 0.04, t[0]+0.5]
    high = [np.max(f)*1.3, 100, 30, t[-1]-0.5]
    best = None
    for speed, tau in [(15, 0.4), (35, 1.5), (70, 4), (25, 8)]:
        initial = [float(np.median(f)), speed, tau, float(np.median(t))]
        fitted = least_squares(lambda p: observed_frequency(t, *p, options.speed_of_sound_m_s)-f,
                               initial, bounds=(low, high), loss="soft_l1", f_scale=2,
                               x_scale="jac", max_nfev=300)
        if best is None or fitted.cost < best.cost:
            best = fitted
    f0, speed, tau, tcpa = [float(v) for v in best.x]
    predicted = observed_frequency(t, *best.x, options.speed_of_sound_m_s)
    rmse = float(np.sqrt(np.mean((predicted-f)**2)))
    constant_rmse = float(np.sqrt(np.mean((f-np.mean(f))**2)))
    linear = np.polyval(np.polyfit(t-t[0], f, 1), t-t[0])
    linear_rmse = float(np.sqrt(np.mean((f-linear)**2)))
    normalized_jac = best.jac / np.maximum(np.linalg.norm(best.jac, axis=0), 1e-12)
    condition = float(np.linalg.cond(normalized_jac))
    reasons = []
    if rmse > max(3, 0.015*f0):
        reasons.append("The observed ridge does not closely follow the pass-by model.")
    if rmse > constant_rmse*0.4 or rmse > max(0.5, linear_rmse*0.8):
        reasons.append("The model does not improve enough over a steady tone or simple pitch drift.")
    if min(tcpa-t[0], t[-1]-tcpa) < max(1.0, 1.5*tau):
        reasons.append("The window does not sufficiently cover both sides of the proposed pass-by.")
    if condition > 1e4 or not np.isfinite(condition):
        reasons.append("The fitted parameters are not identifiable from this window.")
    if speed > 98 or speed < 2.1 or tau < 0.045 or tau > 29:
        reasons.append("The fit reaches a parameter bound; its speed is not reliable.")
    all_t = np.asarray(track["time_s"])
    result["fit"] = {
        "rmse_hz": rmse, "constant_baseline_rmse_hz": constant_rmse,
        "linear_baseline_rmse_hz": linear_rmse, "jacobian_condition": condition if np.isfinite(condition) else None,
        "emitted_frequency_hz": f0,
        "time_s": all_t.tolist(),
        "predicted_frequency_hz": observed_frequency(all_t, *best.x, options.speed_of_sound_m_s).tolist(),
    }
    result["reasons"] = reasons
    if reasons:
        return result
    result["received_closest_approach_s"] = tcpa
    if not (options.assume_constant_tone and options.assume_stationary_microphone):
        result["status"] = "assumptions_required"
        result["reasons"] = ["A curve fits, but both the fixed source tone and stationary microphone assumptions must be asserted before a conditional speed is shown."]
        return result
    result.update(status="conditional_fit", speed_m_s=speed, speed_km_h=speed*3.6)
    return result
