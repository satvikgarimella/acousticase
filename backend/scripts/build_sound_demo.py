"""Build an explicitly constructed demo from clearly recognized training clips."""
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import soundfile as sf

from acousticase.audio import SAMPLE_RATE, decode_audio
from acousticase.evaluation import CATEGORY_GROUP, class_labels
from acousticase.sound_events import group_scores


def main():
    dataset = ROOT / "data/ml/esc50"
    manifest = json.loads((dataset / "manifest.json").read_text())
    labels = class_labels()
    audio = [np.zeros(SAMPLE_RATE, np.float32)]
    spans, offset = [], 1.0
    for category in ("engine", "car_horn", "glass_breaking", "siren"):
        candidates = []
        for row in manifest["clips"]:
            if row["category"] != category or int(row["fold"]) not in manifest["train_folds"]:
                continue
            with np.load(ROOT / "data/ml/features" / (Path(row["filename"]).stem + ".npz"), allow_pickle=False) as value:
                strength = group_scores(value["scores"], labels)[CATEGORY_GROUP[category]].mean()
            candidates.append((float(strength), row["filename"], row))
        row = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][2]
        with tempfile.TemporaryDirectory() as temp:
            samples = decode_audio(dataset / "audio" / row["filename"], Path(temp)/"playback.wav", 6).samples
        # Short fades avoid a hard waveform edge being mistaken for an impact.
        fade = np.linspace(0, 1, 160, dtype=np.float32)
        samples[:160] *= fade
        samples[-160:] *= fade[::-1]
        audio.extend([samples, np.zeros(SAMPLE_RATE, np.float32)])
        duration = len(samples) / SAMPLE_RATE
        spans.append({"category": category, "excerpt_start_s": offset, "excerpt_end_s": offset+duration,
                      "source_file": row["filename"], "source_sha256": row["sha256"], "source_url": row["url"],
                      "freesound_id": row["src_file"], "fold": int(row["fold"])})
        offset += duration + 1
    target = ROOT / "examples/constructed-sound-scene.wav"
    sf.write(target, np.concatenate(audio), SAMPLE_RATE, subtype="PCM_16")
    truth = {"is_simulated": True, "description": "Constructed collage of separate field recordings. Not a recording of a real incident.",
             "selection": "Highest mean target-group score in training folds per class; curated for demonstration, not evaluation.",
             "timing_note": "Spans locate inserted source excerpts, not manually annotated sound onsets.",
             "modifications": "Resampled to mono 16 kHz; 10 ms edge fades; one-second silent gaps; PCM16 export.",
             "source_revision": manifest["revision"], "license": "ESC-50 CC BY-NC 3.0; see ../evaluation/ESC50-LICENSE for all source attributions.",
             "excerpts": spans}
    target.with_suffix(".json").write_text(json.dumps(truth, indent=2)+"\n")
    license_lines = (dataset / "LICENSE").read_text().splitlines()
    attribution = ["# Constructed sound scene attribution", "", truth["description"], "", truth["selection"], "", truth["modifications"], "",
                   "Source: [ESC-50 by Karol J. Piczak](https://github.com/karolpiczak/ESC-50). Dataset license: [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/).", "",
                   "The complete upstream license and source credits are retained in `../evaluation/ESC50-LICENSE`.", "", "## Included excerpts", ""]
    for span in spans:
        prefix = "- [" + span["source_file"].rsplit("-", 1)[0] + ".ogg]"
        matches = [line for line in license_lines if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Could not resolve attribution for {span['source_file']}")
        attribution.extend([f"{span['excerpt_start_s']:.1f}–{span['excerpt_end_s']:.1f} s: {span['category']}", "", matches[0], ""])
    (ROOT / "examples/CONSTRUCTED-SCENE-ATTRIBUTION.md").write_text("\n".join(attribution))
    print(json.dumps(truth, indent=2))


if __name__ == "__main__":
    main()
