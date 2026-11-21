"""Download only a pinned ESC-50 subset, retaining source metadata and attribution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def fetch(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ml/esc50")
    args = parser.parse_args()
    config = json.loads((ROOT / "evaluation/esc50_subset.json").read_text())
    base = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{config['revision']}"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audio").mkdir(exist_ok=True)
    for local, remote in [("metadata.csv", "meta/esc50.csv"), ("LICENSE", "LICENSE"),
                          ("README.upstream.md", "README.md")]:
        (args.output / local).write_bytes(fetch(f"{base}/{remote}"))
    rows = [r for r in csv.DictReader(io.StringIO((args.output / "metadata.csv").read_text()))
            if r["category"] in config["categories"]]
    previous = {}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("revision") == config["revision"]:
            previous = {r["filename"]: r["sha256"] for r in manifest["clips"]}

    def download(row):
        name = row["filename"]
        if not re.fullmatch(r"[1-5]-\d+-[A-Z]-\d+\.wav", name):
            raise ValueError(f"Unexpected dataset filename: {name}")
        path = args.output / "audio" / name
        content = path.read_bytes() if path.exists() else b""
        if not content or hashlib.sha256(content).hexdigest() != previous.get(name):
            content = fetch(f"{base}/audio/{name}")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        return {**row, "sha256": hashlib.sha256(content).hexdigest(), "url": f"{base}/audio/{name}"}

    clips = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(download, rows), 1):
            clips.append(row)
            if i % 40 == 0:
                print(f"Downloaded/verified {i}/{len(rows)} clips", flush=True)
    manifest_path.write_text(json.dumps({**config, "license_file": "LICENSE", "clips": clips}, indent=2) + "\n")
    print(f"Ready: {len(clips)} clips. Raw audio stays in ignored data/. Preserve LICENSE with any copies.")


if __name__ == "__main__":
    main()
