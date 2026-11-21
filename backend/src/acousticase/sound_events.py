"""Turn uncalibrated YAMNet frame scores into conservative, grouped suggestions."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

WINDOW_S = 0.96
HOP_S = 0.48
PROFILE_PATH = Path(__file__).with_name("detection_profile.json")


@dataclass(frozen=True)
class SoundGroup:
    key: str
    label: str
    source_labels: tuple[str, ...]
    brief: bool = False


GROUPS = (
    SoundGroup("horn", "vehicle horn", ("Vehicle horn, car horn, honking", "Air horn, truck horn"), True),
    SoundGroup("siren", "siren", ("Siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Civil defense siren")),
    SoundGroup("engine", "engine sound", ("Engine", "Engine starting", "Idling", "Accelerating, revving, vroom", "Light engine (high frequency)", "Medium engine (mid frequency)", "Heavy engine (low frequency)")),
    SoundGroup("tire_squeal", "tire squeal or skidding sound", ("Tire squeal", "Skidding"), True),
    SoundGroup("glass", "glass breaking sound", ("Shatter",), True),
    SoundGroup("impact", "impact-like sound", ("Bang", "Thump, thud", "Smash, crash"), True),
    SoundGroup("vehicle", "vehicle or traffic sound", ("Vehicle", "Motor vehicle (road)", "Car", "Car passing by", "Motorcycle", "Truck", "Bus", "Traffic noise, roadway noise")),
    SoundGroup("speech", "speech", ("Speech", "Child speech, kid speaking")),
    SoundGroup("footsteps", "footsteps", ("Walk, footsteps",), True),
    SoundGroup("dog", "dog vocalization", ("Dog", "Bark", "Yip", "Bow-wow"), True),
    SoundGroup("rain", "rain", ("Rain", "Rain on surface", "Raindrop")),
    SoundGroup("wind", "wind or microphone wind noise", ("Wind", "Wind noise (microphone)")),
)
GROUP_BY_KEY = {g.key: g for g in GROUPS}


def load_profile(path: Path = PROFILE_PATH):
    profile = json.loads(path.read_text())
    if set(profile["onset_thresholds"]) != set(GROUP_BY_KEY):
        raise ValueError("Detection profile must define every sound group.")
    if not all(np.isfinite(v) and 0 < v <= 1 for v in profile["onset_thresholds"].values()):
        raise ValueError("Detection thresholds must be finite and in (0, 1].")
    return profile


def profile_hash(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:16]


def group_scores(scores, labels):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Expected finite frame scores with one column per label.")
    indices = {name: i for i, name in enumerate(labels)}
    missing = {label for group in GROUPS for label in group.source_labels if label not in indices}
    if missing:
        raise ValueError(f"Unexpected YAMNet class map; missing labels: {sorted(missing)}")
    return {g.key: scores[:, [indices[name] for name in g.source_labels]].max(axis=1) for g in GROUPS}


def active_regions(values, threshold, brief=False):
    """Hysteresis: weak continuation requires a strong seed; bridge one weak frame."""
    values = np.asarray(values)
    active = values >= threshold * 0.55
    if len(active) > 2:
        active[1:-1] |= active[:-2] & active[2:]
    edges = np.diff(np.r_[False, active, False].astype(int))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        strong_count = int(np.count_nonzero(values[a:b] >= threshold))
        # Brief sounds can occupy one patch; sustained sounds need persistence,
        # unless a single patch is especially strong. These are heuristics.
        if strong_count >= 2 or (strong_count and (brief or values[a:b].max() >= max(0.65, threshold))):
            yield int(a), int(b)


def events_from_scores(scores, labels, duration_s, profile=None, max_events=100):
    profile = profile if profile is not None else load_profile()
    grouped = group_scores(scores, labels)
    indices = {name: i for i, name in enumerate(labels)}
    events = []
    for group in GROUPS:
        values = grouped[group.key]
        threshold = profile["onset_thresholds"][group.key]
        for a, b in active_regions(values, threshold, group.brief):
            start = a * HOP_S
            end = min(duration_s, (b - 1) * HOP_S + WINDOW_S)
            if end <= start:
                continue
            source = sorted(
                ({"label": name, "peak_model_score": float(np.max(scores[a:b, indices[name]]))}
                 for name in group.source_labels), key=lambda x: x["peak_model_score"], reverse=True)
            events.append({
                "id": "", "start_s": float(start), "end_s": float(end),
                "peak_s": float(min(end, (a + np.argmax(values[a:b])) * HOP_S + WINDOW_S / 2)),
                "kind": "sound_class", "label": f"Possible {group.label}", "evidence_type": "model_suggestion",
                "measurements": {"class_label": group.label, "sound_group": group.key,
                                 "model_score": float(values[a:b].max()), "mean_model_score": float(values[a:b].mean()),
                                 "onset_threshold": threshold, "active_frames": int(np.count_nonzero(values[a:b] >= threshold)),
                                 "source_labels": source, "timing_resolution_s": HOP_S},
                "explanation": "Grouped YAMNet suggestion with approximate window boundaries. Scores are uncalibrated; the label does not establish source identity, a collision, or its cause.",
            })
    # Suppress a broad vehicle label only when a more specific sound covers most
    # of that interval. Keep distinct simultaneous sound groups.
    specific = [e for e in events if e["measurements"]["sound_group"] in {"engine", "horn", "siren", "tire_squeal"}]
    visible = []
    for event in events:
        duplicate = event["measurements"]["sound_group"] == "vehicle" and any(
            max(0, min(event["end_s"], s["end_s"]) - max(event["start_s"], s["start_s"]))
            / (event["end_s"] - event["start_s"]) >= 0.8 for s in specific)
        if not duplicate:
            visible.append(event)
    suppressed = len(events) - len(visible)
    omitted = max(0, len(visible) - max_events)
    visible = sorted(sorted(visible, key=lambda e: (-e["measurements"]["model_score"], e["start_s"], e["label"]))[:max_events],
                     key=lambda e: (e["start_s"], e["label"]))
    for i, event in enumerate(visible, 1):
        event["id"] = f"class-{i}"
    return visible, {"suppressed_broad_labels": suppressed, "omitted_event_count": omitted}


def score_timeline(scores, labels, duration_s, profile):
    grouped = group_scores(scores, labels)
    starts = np.arange(len(scores)) * HOP_S
    valid = starts < duration_s
    return {
        "timebase": "seconds_from_first_decoded_audio_sample", "window_s": WINDOW_S, "hop_s": HOP_S,
        "score_meaning": "Uncalibrated group maximum of related YAMNet scores; not event probabilities.",
        "start_s": starts[valid].tolist(), "end_s": np.minimum(duration_s, starts[valid] + WINDOW_S).tolist(),
        "groups": [{"key": g.key, "label": g.label, "onset_threshold": profile["onset_thresholds"][g.key],
                    "scores": np.round(grouped[g.key][valid], 5).tolist()} for g in GROUPS],
    }
