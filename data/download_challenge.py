"""Download a pinned revision of an official challenge dataset from the Hugging Face Hub and verify it.

Every file is checked against the Hub's own hashes (LFS sha256, or the git blob sha1 for small
files). Downloads resume from ``.partial`` files and never go through a proxy.

    python data/download_challenge.py --which train --dest data/challenge-v1.1.0 --include data/diii_d_train/
    python data/download_challenge.py --which final --dest data/challenge-final
    python data/download_challenge.py --which final --dest /tmp/sample --include data/mast_private_test/ --limit 2
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)

DATASETS = {
    # DIII-D training shots (and the public-test inputs) of release v1.1.0
    "train": ("Sophelio/fusion-equilibrium-challenge", "1e280905b85f2a6fdde7e06fca8cf3a1edf447cb"),
    # Final-phase test inputs (diii_d_private_test, mast_private_test)
    "final": ("ganatma/fusion-equilibrium-challenge-final", "965040c12b814f33c8e7209f6b1799932247e793"),
}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def manifest(endpoint, repo, revision):
    url = f"{endpoint}/api/datasets/{repo}/revision/{revision}?blobs=true"
    with OPENER.open(url, timeout=120) as response:
        value = json.load(response)
    if value["sha"] != revision:
        raise RuntimeError(f"Hub returned revision {value['sha']}, expected {revision}")
    return value


def verified(path: Path, entry) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    lfs = entry.get("lfs")
    h = hashlib.sha256() if lfs else hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest() == (lfs["sha256"] if lfs else entry["blobId"])


def fetch(endpoint, repo, revision, dest: Path, entry, attempts=5):
    target = dest / entry["rfilename"]
    if verified(target, entry):
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    url = f"{endpoint}/datasets/{repo}/resolve/{revision}/{entry['rfilename']}"
    for attempt in range(attempts):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            if start > entry["size"]:
                partial.unlink(); start = 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
            with OPENER.open(request, timeout=240) as response, partial.open("ab" if start else "wb") as out:
                if start and response.status != 206:  # server ignored the range: restart
                    out.truncate(0)
                for block in iter(lambda: response.read(8 << 20), b""):
                    out.write(block)
            if verified(partial, entry):
                partial.replace(target)
                return "downloaded"
            partial.unlink()
        except OSError as error:
            print(json.dumps({"retry": entry["rfilename"], "attempt": attempt + 1, "error": str(error)[:200]}), flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed after {attempts} attempts: {entry['rfilename']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=sorted(DATASETS), required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--include", nargs="*", default=[], help="only files under these path prefixes")
    p.add_argument("--limit", type=int, default=0, help="only the first N matching files (smoke test)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    a = p.parse_args()
    repo, revision = DATASETS[a.which]
    endpoint = a.endpoint.rstrip("/")
    value = manifest(endpoint, repo, revision)
    files = sorted((s for s in value["siblings"] if not a.include or any(s["rfilename"].startswith(i) for i in a.include)),
                   key=lambda s: s["rfilename"])
    if a.limit:
        files = files[: a.limit]
    a.dest.mkdir(parents=True, exist_ok=True)
    (a.dest / "hub-manifest.json").write_text(json.dumps(value, indent=1))
    print(json.dumps({"repo": repo, "revision": revision, "files": len(files), "dest": str(a.dest)}), flush=True)
    counts, started = {}, time.time()
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for i, status in enumerate(pool.map(lambda e: fetch(endpoint, repo, revision, a.dest, e), files), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 100 == 0 or i == len(files):
                print(json.dumps({"verified": i, "of": len(files), **counts, "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
