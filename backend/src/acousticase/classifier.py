import csv
import importlib.util
from pathlib import Path

import numpy as np

RELEVANT = {
    "Vehicle", "Car", "Car passing by", "Engine", "Engine starting", "Engine accelerating, revving, vroom",
    "Idling", "Motorcycle", "Truck", "Bus", "Traffic noise, roadway noise", "Tire squeal",
    "Vehicle horn, car horn, honking", "Siren", "Police car (siren)", "Ambulance (siren)",
    "Fire engine, fire truck (siren)", "Skidding", "Crash", "Bang", "Thump, thud", "Shatter", "Speech",
}


class YamnetClassifier:
    def __init__(self, model_dir: Path | None):
        self.model_dir = model_dir
        self.model = None
        self.labels = []
        has_weights = bool(model_dir and ((model_dir / "saved_model.pb").exists() or (model_dir / "yamnet.h5").exists()))
        self.available = bool(has_weights and importlib.util.find_spec("tensorflow") and importlib.util.find_spec("tf_keras"))

    @property
    def cache_tag(self):
        return "yamnet-google-v1" if self.available else "no-classifier"

    def classify(self, samples: np.ndarray, enabled: bool = True) -> tuple[list[dict], dict]:
        if not enabled or not self.available:
            return [], {"status": "disabled", "model": None, "reason": "Disabled by options or local YAMNet weights/runtime are not installed."}
        import tensorflow as tf
        if self.model is None:
            if (self.model_dir / "yamnet.h5").exists():
                from .vendor.yamnet import params, yamnet
                self.model = yamnet.yamnet_frames_model(params.Params())
                self.model.load_weights(str(self.model_dir / "yamnet.h5"))
                class_map = Path(yamnet.__file__).with_name("yamnet_class_map.csv")
            else:
                self.model = tf.saved_model.load(str(self.model_dir))
                class_map = Path(self.model.class_map_path().numpy().decode())
            with open(class_map, newline="") as handle:
                self.labels = [row["display_name"] for row in csv.DictReader(handle)]
        scores, _, _ = self.model(tf.convert_to_tensor(np.clip(samples, -1, 1), dtype=tf.float32))
        scores = scores.numpy()
        means = scores.mean(axis=0)
        top = np.argsort(means)[-5:][::-1]
        info = {
            "status": "enabled", "model": "Google YAMNet v1 / AudioSet",
            "top_classes": [{"label": self.labels[i], "mean_model_score": float(means[i])} for i in top],
            "score_meaning": "Uncalibrated model scores, not probabilities that a real-world event occurred.",
            "window_s": 0.96, "hop_s": 0.48,
        }
        duration = len(samples)/16000
        candidates = []
        for i, label in enumerate(self.labels):
            if label not in RELEVANT:
                continue
            active = scores[:, i] >= 0.35
            edges = np.diff(np.r_[False, active, False].astype(int))
            for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                peak_score = float(scores[a:b, i].max())
                if b-a < 2 and peak_score < 0.6:
                    continue
                start, end = float(a*0.48), min(duration, float((b-1)*0.48+0.96))
                if end <= start:
                    continue
                candidates.append({
                    "id": "", "start_s": start, "end_s": end,
                    "peak_s": min(duration, float((a+np.argmax(scores[a:b, i]))*0.48+0.48)),
                    "kind": "sound_class", "label": f"Possible {label.lower()}",
                    "evidence_type": "model_suggestion",
                    "measurements": {"class_label": label, "model_score": peak_score},
                    "explanation": "YAMNet sound-class suggestion. Scores are uncalibrated and do not establish identity, cause, or a collision.",
                })
        candidates = sorted(sorted(candidates, key=lambda e: e["measurements"]["model_score"], reverse=True)[:30], key=lambda e: e["start_s"])
        for i, event in enumerate(candidates):
            event["id"] = f"class-{i+1}"
        return candidates, info
