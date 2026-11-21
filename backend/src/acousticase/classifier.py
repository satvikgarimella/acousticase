import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from .sound_events import events_from_scores, load_profile, profile_hash, score_timeline


class YamnetClassifier:
    def __init__(self, model_dir: Path | None):
        self.model_dir = model_dir
        self.model = None
        self.labels = []
        self.profile = load_profile()
        has_weights = bool(model_dir and ((model_dir / "saved_model.pb").exists() or (model_dir / "yamnet.h5").exists()))
        self.available = bool(has_weights and importlib.util.find_spec("tensorflow") and importlib.util.find_spec("tf_keras"))

    @property
    def cache_tag(self):
        return f"yamnet-google-v1:{profile_hash(self.profile)}" if self.available else "no-classifier"

    def infer(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return pretrained scores and embeddings; no fitting or network requests."""
        if not self.available:
            raise RuntimeError("Install classifier dependencies and local YAMNet weights first.")
        if samples.ndim != 1 or len(samples) == 0 or not np.isfinite(samples).all():
            raise ValueError("Expected non-empty finite mono audio at 16 kHz.")
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
        # Bound convolution activation memory. Each block starts at a YAMNet
        # patch boundary and includes the exact right context for 64 patches.
        # A patch spans 0.96 s of spectrogram plus 0.015 s of STFT context.
        hop, window, block_frames = 7680, 15600, 64
        frame_count = 1 + max(0, int(np.ceil((len(samples) - window) / hop)))
        score_blocks, embedding_blocks = [], []
        for frame_start in range(0, frame_count, block_frames):
            count = min(block_frames, frame_count - frame_start)
            begin = frame_start * hop
            end = begin + (count - 1) * hop + window
            scores, embeddings, _ = self.model(tf.convert_to_tensor(np.clip(samples[begin:end], -1, 1), dtype=tf.float32))
            score_blocks.append(scores.numpy()[:count])
            embedding_blocks.append(embeddings.numpy()[:count])
        return np.concatenate(score_blocks), np.concatenate(embedding_blocks)

    def classify(self, samples: np.ndarray, enabled: bool = True,
                 score_output: Path | None = None) -> tuple[list[dict], dict]:
        if not enabled or not self.available:
            return [], {"status": "disabled", "model": None, "reason": "Disabled by options or local YAMNet weights/runtime are not installed."}
        scores, _ = self.infer(samples)
        means = scores.mean(axis=0)
        top = np.argsort(means)[-5:][::-1]
        info = {
            "status": "enabled", "model": "Google YAMNet v1 / AudioSet",
            "top_classes": [{"label": self.labels[i], "mean_model_score": float(means[i])} for i in top],
            "score_meaning": "Uncalibrated model scores, not probabilities that a real-world event occurred.",
            "window_s": 0.96, "hop_s": 0.48,
            "detector_version": self.profile["version"], "detector_profile_hash": profile_hash(self.profile),
            "timing_caveat": "Patch-level suggestions have approximate boundaries; no event-onset accuracy has been established.",
        }
        duration = len(samples)/16000
        candidates, diagnostics = events_from_scores(scores, self.labels, duration, self.profile)
        info.update(diagnostics)
        if score_output is not None:
            score_output.write_text(json.dumps(score_timeline(scores, self.labels, duration, self.profile), allow_nan=False), encoding="utf-8")
        return candidates, info
