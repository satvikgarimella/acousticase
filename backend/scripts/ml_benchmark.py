"""Extract frozen YAMNet features, train a small head, and compare held-out results."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from acousticase.audio import decode_audio
from acousticase.classifier import YamnetClassifier
from acousticase.evaluation import (CATEGORY_GROUP, TARGET_GROUPS, LinearAudioHead, binary_metrics,
    class_labels, detects, feature_digest, legacy_detects, validate_split)
from acousticase.sound_events import group_scores, load_profile


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def extract(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    validate_split(manifest["clips"], manifest)
    args.features.mkdir(parents=True, exist_ok=True)
    weights_sha = hashlib.sha256((args.model / "yamnet.h5").read_bytes()).hexdigest()
    if weights_sha != "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744":
        raise ValueError("Benchmark requires the pinned official YAMNet weights.")
    classifier = YamnetClassifier(args.model)
    started = time.monotonic()
    for i, row in enumerate(manifest["clips"], 1):
        path = args.data / "audio" / row["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Audio checksum mismatch: {row['filename']}")
        output = args.features / (path.stem + ".npz")
        if output.exists():
            with np.load(output, allow_pickle=False) as cached:
                if str(cached["audio_sha256"]) == row["sha256"] and str(cached["extractor_version"]) == "yamnet-google-v1-16k-mean-std-v1":
                    continue
        with tempfile.TemporaryDirectory() as temp:
            decoded = decode_audio(path, Path(temp) / "playback.wav", 6)
        scores, embeddings = classifier.infer(decoded.samples)
        np.savez_compressed(output, scores=scores, embeddings=embeddings, audio_sha256=row["sha256"],
                            extractor_version="yamnet-google-v1-16k-mean-std-v1")
        if i % 20 == 0:
            print(f"Extracted {i}/{len(manifest['clips'])} recordings ({time.monotonic()-started:.1f}s)", flush=True)
    save_json(args.features / "provenance.json", {"manifest_sha256": feature_digest(manifest),
        "extractor": "Google YAMNet v1, frozen", "base_model_sha256": weights_sha,
        "pooling": "mean and standard deviation over frames",
        "feature_dimension": 2048, "seconds": round(time.monotonic()-started, 3)})
    print("Feature extraction complete.", flush=True)


def run(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import sklearn

    manifest = json.loads((args.data / "manifest.json").read_text())
    rows = manifest["clips"]
    partitions = validate_split(rows, manifest)
    provenance = json.loads((args.features / "provenance.json").read_text())
    if provenance["manifest_sha256"] != feature_digest(manifest):
        raise ValueError("Features belong to a different dataset manifest; re-extract.")
    labels = class_labels()
    vectors, all_scores, all_embeddings = [], [], []
    for row in rows:
        with np.load(args.features / (Path(row["filename"]).stem + ".npz"), allow_pickle=False) as value:
            if str(value["audio_sha256"]) != row["sha256"]:
                raise ValueError("Feature/audio checksum mismatch.")
            embeddings = value["embeddings"]
            vectors.append(np.r_[embeddings.mean(axis=0), embeddings.std(axis=0)])
            all_embeddings.append(embeddings)
            all_scores.append(value["scores"])
    x = np.asarray(vectors)
    y = np.asarray([r["category"] for r in rows])
    split = {name: np.asarray([int(r["fold"]) in folds for r in rows]) for name, folds in partitions.items()}
    grouped = [group_scores(s, labels) for s in all_scores]
    classes = sorted(manifest["categories"])
    baseline = np.asarray([classes[int(np.argmax([g[CATEGORY_GROUP[c]].mean() for c in classes]))] for g in grouped])
    trials, best = [], None
    for c in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=7))
        model.fit(x[split["train"]], y[split["train"]])
        predicted = model.predict(x[split["validation"]])
        score = f1_score(y[split["validation"]], predicted, labels=classes, average="macro", zero_division=0)
        trials.append({"C": c, "validation_macro_f1": float(score)})
        if best is None or score > best[0]:
            best = (score, model, c)
    model = best[1]
    args.output.mkdir(parents=True, exist_ok=True)
    scaler, head = model.steps[0][1], model.steps[1][1]
    model_path = args.output / "experimental-clip-head.npz"
    np.savez_compressed(model_path, mean=scaler.mean_, scale=scaler.scale_, weights=head.coef_,
                        bias=head.intercept_, classes=head.classes_)
    # Verify the portable runtime reproduces sklearn before reporting results.
    exported = LinearAudioHead(model_path)
    assert all(exported.predict(all_embeddings[i])[0]["label"] == model.predict(x[i:i+1])[0] for i in range(len(rows)))

    initial = load_profile(ROOT / "evaluation/initial-detection-profile.json")
    tuned = copy.deepcopy(initial)
    tuned["version"] = "esc50-dev-v1"
    tuned["provenance"] = "Four target thresholds selected on ESC-50 subset fold 4 using clip-label F0.5; folds 1-3 train the separate head, fold 5 held out. Other thresholds remain heuristics. Not probability calibration."
    threshold_trials = {}
    for key in TARGET_GROUPS:
        truth = [CATEGORY_GROUP[r["category"]] == key for r, keep in zip(rows, split["validation"]) if keep]
        scores = [g[key] for g, keep in zip(grouped, split["validation"]) if keep]
        candidates = []
        for threshold in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
            metrics = binary_metrics(truth, [detects(s, key, threshold) for s in scores])
            p, r = metrics["precision"], metrics["recall"]
            fhalf = 1.25*p*r/(0.25*p+r) if p+r else 0.0
            candidates.append({"threshold": threshold, "f0_5": fhalf, **metrics})
        # Prefer precision and then the higher threshold on exact ties.
        winner = max(candidates, key=lambda r: (r["f0_5"], r["precision"], r["threshold"]))
        tuned["onset_thresholds"][key] = winner["threshold"]
        threshold_trials[key] = candidates
    save_json(args.output / "candidate-detection-profile.json", tuned)

    report = {"dataset": {k: manifest[k] for k in ("repository", "revision", "categories", "train_folds", "validation_folds", "test_folds")},
              "manifest_sha256": feature_digest(manifest), "feature_provenance": provenance,
              "split_counts": {name: int(mask.sum()) for name, mask in split.items()},
              "training": {"method": "Frozen YAMNet mean/std embeddings + StandardScaler + multinomial logistic regression",
                           "scikit_learn_version": sklearn.__version__, "trials": trials, "selected_C": best[2],
                           "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()},
              "closed_set_classification": {}, "clip_presence_detection": {}, "threshold_selection": threshold_trials,
              "limits": ["Small eight-class subset, not a full ESC-50 benchmark or a traffic deployment benchmark.",
                         "Clip labels are weak labels: secondary sounds may occur. No event timestamp accuracy is measured.",
                         "YAMNet pretraining overlap with this public dataset has not been audited.",
                         "The custom head chooses among eight known classes; unknown sounds and mixtures are not validated.",
                         "No accident-cause, tire-squeal, collision, or real-speed accuracy is established.",
                         "Raw data and derived model are for this noncommercial research prototype; retain ESC-50 license/attribution."]}
    for name in ("validation", "test"):
        mask = split[name]
        report["closed_set_classification"][name] = {}
        for method, predicted in (("pretrained_grouped_baseline", baseline[mask]), ("trained_linear_head", model.predict(x[mask]))):
            report["closed_set_classification"][name][method] = {
                "accuracy": float(accuracy_score(y[mask], predicted)),
                "macro_f1": float(f1_score(y[mask], predicted, labels=classes, average="macro", zero_division=0)),
                "classes": classes, "confusion_matrix": confusion_matrix(y[mask], predicted, labels=classes).tolist(),
                "per_class": classification_report(y[mask], predicted, labels=classes, output_dict=True, zero_division=0)}
        report["clip_presence_detection"][name] = {}
        for method in ("legacy_v0_1", "grouped_initial", "grouped_tuned"):
            per_class = {}
            for key in TARGET_GROUPS:
                truth, predicted = [], []
                for i in np.flatnonzero(mask):
                    truth.append(CATEGORY_GROUP[rows[i]["category"]] == key)
                    if method == "legacy_v0_1":
                        predicted.append(legacy_detects(all_scores[i], labels, key))
                    else:
                        profile = initial if method == "grouped_initial" else tuned
                        predicted.append(detects(grouped[i][key], key, profile["onset_thresholds"][key]))
                per_class[key] = binary_metrics(truth, predicted)
            report["clip_presence_detection"][name][method] = {"per_class": per_class,
                "macro_f1": float(np.mean([r["f1"] for r in per_class.values()])),
                "total_false_positives": sum(r["fp"] for r in per_class.values()),
                "total_true_positives": sum(r["tp"] for r in per_class.values())}
    predictions = []
    predicted = model.predict(x)
    for i, row in enumerate(rows):
        if split["test"][i]:
            predictions.append({"filename": row["filename"], "true_label": row["category"],
                                "baseline_label": str(baseline[i]), "trained_label": str(predicted[i])})
    save_json(args.output / "test-predictions.json", predictions)
    save_json(args.output / "benchmark.json", report)
    save_json(args.output / "dataset-manifest.json", manifest)
    print(json.dumps({"split_counts": report["split_counts"],
        "test_classification": {k: {s: v[s] for s in ("accuracy", "macro_f1")} for k, v in report["closed_set_classification"]["test"].items()},
        "validation_detection": report["clip_presence_detection"]["validation"],
        "test_detection": report["clip_presence_detection"]["test"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["extract", "run"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/ml/esc50")
    parser.add_argument("--features", type=Path, default=ROOT / "data/ml/features")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yamnet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ml/results")
    args = parser.parse_args()
    {"extract": extract, "run": run}[args.action](args)


if __name__ == "__main__":
    main()
