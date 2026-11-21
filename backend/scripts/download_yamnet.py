"""Fetch pinned official Google YAMNet weights once, before accepting uploads."""
import argparse
import hashlib
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
SHA256 = "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/yamnet"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/"yamnet.h5"
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
            print(f"Verified existing model: {target.resolve()}")
            return
        raise RuntimeError("An unexpected weights file already exists; no files were overwritten.")
    import requests
    temporary = args.output/"yamnet.h5.part"
    try:
        with requests.get(MODEL_URL, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            total = 0
            digest = hashlib.sha256()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024*1024):
                    total += len(chunk)
                    if total > 32*1024*1024:
                        raise RuntimeError("Model download exceeds the expected size limit.")
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("Model checksum differs from the verified YAMNet release.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"YAMNet ready at {target.resolve()}")
    print(f"Source: {MODEL_URL}")


if __name__ == "__main__":
    main()
