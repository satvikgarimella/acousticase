"""Try the experimental eight-class head on a short, single-sound recording."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

from acousticase.audio import decode_audio
from acousticase.classifier import YamnetClassifier
from acousticase.evaluation import LinearAudioHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--yamnet", type=Path, default=ROOT / "models/yamnet")
    parser.add_argument("--head", type=Path, default=ROOT / "evaluation/artifacts/experimental-clip-head.npz")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        decoded = decode_audio(args.audio, Path(temp) / "playback.wav", 6)
    _, embeddings = YamnetClassifier(args.yamnet).infer(decoded.samples)
    predictions = LinearAudioHead(args.head).predict(embeddings)
    print(json.dumps({"mode": "experimental_closed_set_clip_classification", "predictions": predictions,
        "caveat": "Evaluated on five-second clips from eight categories. Always chooses among those categories; unknown sounds, mixtures, timestamps, and causal conclusions are not validated. Scores are not calibrated probabilities."}, indent=2))


if __name__ == "__main__":
    main()
