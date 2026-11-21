"""Reproducible evaluation helpers. Clip labels do not supply event timestamps."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .sound_events import GROUP_BY_KEY, active_regions, group_scores

CATEGORY_GROUP = {"car_horn": "horn", "siren": "siren", "engine": "engine", "glass_breaking": "glass",
                  "rain": "rain", "wind": "wind", "footsteps": "footsteps", "dog": "dog"}
TARGET_GROUPS = ("horn", "siren", "engine", "glass")
LEGACY_LABELS = {
    "Vehicle", "Car", "Car passing by", "Engine", "Engine starting", "Engine accelerating, revving, vroom",
    "Idling", "Motorcycle", "Truck", "Bus", "Traffic noise, roadway noise", "Tire squeal",
    "Vehicle horn, car horn, honking", "Siren", "Police car (siren)", "Ambulance (siren)",
    "Fire engine, fire truck (siren)", "Skidding", "Crash", "Bang", "Thump, thud", "Shatter", "Speech",
}


def class_labels():
    path = Path(__file__).with_name("vendor") / "yamnet/yamnet_class_map.csv"
    with path.open(newline="") as stream:
        return [row["display_name"] for row in csv.DictReader(stream)]


def validate_split(rows, config):
    """Reject clip/source leakage and incomplete partitions before model fitting."""
    partitions = {key: set(config[f"{key}_folds"]) for key in ("train", "validation", "test")}
    if any(partitions[a] & partitions[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Fold partitions overlap.")
    seen_files, source_partitions = set(), {}
    for row in rows:
        if row["filename"] in seen_files:
            raise ValueError("Duplicate clip in evaluation manifest.")
        seen_files.add(row["filename"])
        part = [p for p, folds in partitions.items() if int(row["fold"]) in folds]
        if len(part) != 1:
            raise ValueError("Each clip must belong to exactly one partition.")
        previous = source_partitions.setdefault(row["src_file"], part[0])
        if previous != part[0]:
            raise ValueError("Fragments of one source recording cross partitions.")
    for name, folds in partitions.items():
        present = {r["category"] for r in rows if int(r["fold"]) in folds}
        if present != set(config["categories"]):
            raise ValueError(f"Missing or unexpected categories in {name} split.")
    return partitions


def binary_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def detects(values, group, threshold):
    return next(active_regions(values, threshold, GROUP_BY_KEY[group].brief), None) is not None


def legacy_detects(scores, labels, group):
    """Exactly the v0.1 threshold/persistence policy, before its 30-event cap."""
    for label in GROUP_BY_KEY[group].source_labels:
        if label not in LEGACY_LABELS:
            continue
        values = scores[:, labels.index(label)]
        active = values >= 0.35
        edges = np.diff(np.r_[False, active, False].astype(int))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= 2 or values[a:b].max() >= 0.6:
                return True
    return False


def feature_digest(manifest):
    payload = {"revision": manifest["revision"], "clips": [{k: r[k] for k in ("filename", "sha256", "fold", "category", "src_file")} for r in manifest["clips"]]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class LinearAudioHead:
    """Experimental clip-level transfer classifier; no arbitrary pickle loading."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as model:
            self.mean = model["mean"]
            self.scale = model["scale"]
            self.weights = model["weights"]
            self.bias = model["bias"]
            self.classes = model["classes"].tolist()

    def predict(self, embeddings):
        pooled = np.r_[embeddings.mean(axis=0), embeddings.std(axis=0)]
        z = (pooled - self.mean) / self.scale
        logits = z @ self.weights.T + self.bias
        exp = np.exp(logits - logits.max())
        scores = exp / exp.sum()
        return sorted([{"label": name, "model_score": float(score)} for name, score in zip(self.classes, scores)],
                      key=lambda x: x["model_score"], reverse=True)
