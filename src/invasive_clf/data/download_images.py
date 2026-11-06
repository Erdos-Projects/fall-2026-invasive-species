"""Download the photos listed in the split tables (the dataset "manifest").

Usage (from the repository root)::

    python -m invasive_clf.data.download_images                         # data/splits/*.csv -> data/inat_dataset_final/images/
    python -m invasive_clf.data.download_images --limit 20              # quick check on 20 photos
    python -m invasive_clf.data.download_images --splits-dir data/inat_dataset_final   # tables from a fresh build_dataset run

Each row of ``train.csv``, ``val.csv``, ``test.csv`` and ``test_unseen.csv`` has a ``photo_url`` (the "medium", 500 px
version of the photo on iNaturalist's servers) and an ``image_path`` relative to the dataset folder
(``images/<label>/<observation_id>_<photo_id>.jpg``). This script downloads every photo that is not on disk yet and
checks that the file is a readable image. Files already present are skipped, so it can be re-run after an
interruption.

iNaturalist changes over time: an observer can delete a photo or change its license. Photos that can no longer be
downloaded are listed in ``download_failures.csv`` next to the images, and the summary says how many there are, so
anyone re-running the project knows whether their copy of the test set is complete.

This is the download cell (section 10) of the team notebook ``inat_final_query.ipynb``, without the step that copied
images from an earlier dataset folder.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

SPLITS = ("train", "val", "test", "test_unseen")
USER_AGENT = "invasive-species-detector/0.3 (Erdos Institute deep-learning project, fall 2026)"


def ok_image(path):
    """True if PIL can open and verify the file (catches truncated downloads and HTML error pages)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def download(session, url, path):
    """Fetch one photo to ``path``. Returns "ok", "404", "not an image", or the name of the last network error."""
    err = "unknown error"
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return "404"
            if r.status_code == 429 or r.status_code >= 500:          # rate limited / server error: back off and retry
                err = f"HTTP {r.status_code}"
                time.sleep(10 * (attempt + 1)); continue
            r.raise_for_status()
            tmp = path.with_suffix(".part")                            # write to a temp name so a crash never leaves a bad .jpg
            tmp.write_bytes(r.content)
            if not ok_image(tmp):
                tmp.unlink(missing_ok=True); return "not an image"
            tmp.rename(path)
            return "ok"
        except requests.RequestException as e:
            err = e.__class__.__name__
            time.sleep(5 * (attempt + 1))
    return err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits-dir", default="data/splits", help="folder with train/val/test/test_unseen.csv")
    ap.add_argument("--out-dir", default="data/inat_dataset_final", help="dataset folder; image_path is relative to it")
    ap.add_argument("--pause", type=float, default=0.25, help="seconds between downloads (photos are served from S3; be polite)")
    ap.add_argument("--limit", type=int, help="download at most this many rows (smoke test)")
    args = ap.parse_args(argv)

    splits_dir, out_dir = Path(args.splits_dir), Path(args.out_dir)
    todo = pd.concat([pd.read_csv(splits_dir / f"{s}.csv", usecols=["image_path", "observation_id", "photo_id", "photo_url"]).assign(split=s)
                      for s in SPLITS], ignore_index=True)
    if args.limit:
        todo = todo.head(args.limit)
    for d in todo["image_path"].map(lambda p: (out_dir / p).parent).unique():
        d.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    failures, had, fetched = [], 0, 0
    t0 = time.time()
    for i, r in enumerate(todo.itertuples(index=False), 1):
        path = out_dir / r.image_path
        if path.exists() and path.stat().st_size > 0:
            had += 1; continue
        status = download(session, r.photo_url, path)
        if status == "ok":
            fetched += 1
        else:
            failures.append(dict(split=r.split, observation_id=r.observation_id, photo_id=r.photo_id, url=r.photo_url, status=status))
        time.sleep(args.pause)
        if i % 500 == 0:
            print(f"  {i}/{len(todo)}  downloaded {fetched}, already had {had}, failed {len(failures)}  ({(time.time() - t0) / 60:.1f} min)")

    pd.DataFrame(failures, columns=["split", "observation_id", "photo_id", "url", "status"]).to_csv(out_dir / "download_failures.csv", index=False)
    print(f"done: {fetched} downloaded, {had} already present, {len(failures)} failed, of {len(todo)}")
    if failures:
        print("missing photos per split:", pd.DataFrame(failures)["split"].value_counts().to_dict(),
              f"(listed in {out_dir / 'download_failures.csv'})")


if __name__ == "__main__":
    main()
