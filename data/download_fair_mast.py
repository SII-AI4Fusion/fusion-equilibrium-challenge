"""Download FAIR-MAST level-2 (v3) Zarr shots, only the groups the MAST pipeline reads.

FAIR-MAST is the public MAST archive (https://mastapp.site, CC BY-SA 4.0) served from the
anonymous S3 endpoint below. For each shot this copies the root ``zarr.json`` (consolidated
metadata) and every object under the requested groups, keeping the store layout, so
``<dest>/<shot>.zarr`` is a valid local Zarr 3 store. The ``equilibrium`` group (EFIT++) is never
downloaded. Direct connection only; files already present with the listed size are skipped.

    python data/download_fair_mast.py --shots-file mast/configs/data/fair_mast_shots.txt --dest data/mast_data_v3
    python data/download_fair_mast.py --shots 26497 --dest /tmp/fm --groups tf summary   # smoke test
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)

ENDPOINT = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast"
PREFIX = "level2/shots"
# Groups behind the seven cached modalities (summary, pf_active, tf, thomson_core/profiles,
# magnetics_slow, pf_passive).
GROUPS = ("summary", "pf_active", "tf", "thomson_scattering", "magnetics", "pf_passive")
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url, attempts=6, timeout=120):
    for attempt in range(attempts):
        try:
            with OPENER.open(url, timeout=timeout) as response:
                return response.read()
        except OSError as error:
            if attempt == attempts - 1:
                raise
            print(json.dumps({"retry": url, "attempt": attempt + 1, "error": str(error)[:160]}), flush=True)
            time.sleep(5 * (attempt + 1))


def list_objects(prefix):
    """(key, size) of every object under prefix (S3 ListObjectsV2, paginated)."""
    token, out = None, []
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        root = ET.fromstring(get(f"{ENDPOINT}/{BUCKET}?{urllib.parse.urlencode(query)}"))
        out += [(c.find(f"{NS}Key").text, int(c.find(f"{NS}Size").text)) for c in root.iter(f"{NS}Contents")]
        if root.findtext(f"{NS}IsTruncated") != "true":
            return out
        token = root.findtext(f"{NS}NextContinuationToken")


def fetch(key, size, dest: Path):
    target = dest / key.removeprefix(PREFIX + "/")
    if target.is_file() and target.stat().st_size == size:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    data = get(f"{ENDPOINT}/{BUCKET}/{urllib.parse.quote(key)}")
    if len(data) != size:
        raise RuntimeError(f"{key}: got {len(data)} bytes, listed {size}")
    tmp = target.with_name(target.name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(target)
    return size


def download_shot(shot, dest: Path, groups, workers):
    base = f"{PREFIX}/{shot}.zarr"
    objects = [(f"{base}/zarr.json", None)]
    for group in groups:
        objects += list_objects(f"{base}/{group}/")
    if objects[0][1] is None:  # root metadata: size from a listing of the root object itself
        objects[0] = next(o for o in list_objects(f"{base}/zarr.json") if o[0] == f"{base}/zarr.json")
    with cf.ThreadPoolExecutor(workers) as pool:
        nbytes = sum(pool.map(lambda o: fetch(o[0], o[1], dest), objects))
    return len(objects), nbytes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--shots", type=int, nargs="*", default=[])
    p.add_argument("--shots-file", type=Path, help="one shot number per line")
    p.add_argument("--groups", nargs="+", default=list(GROUPS))
    p.add_argument("--workers", type=int, default=16, help="parallel objects per shot")
    a = p.parse_args()
    if "equilibrium" in a.groups:
        p.error("the equilibrium group is not used by this entry and is never downloaded")
    shots = list(a.shots) + ([int(x) for x in a.shots_file.read_text().split()] if a.shots_file else [])
    if not shots:
        p.error("give --shots or --shots-file")
    started = time.time()
    for i, shot in enumerate(shots, 1):
        n, nbytes = download_shot(shot, a.dest, a.groups, a.workers)
        print(json.dumps({"shot": shot, "objects": n, "new_bytes": nbytes, "done": i, "of": len(shots),
                          "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
