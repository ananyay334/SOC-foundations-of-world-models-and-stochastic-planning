"""Download and extract CIFAR-10 in ImageFolder layout.

Uses the fast.ai mirror (reliable, unlike the flaky Toronto host). Produces:
    data/cifar10/train/<class>/*.png
    data/cifar10/test/<class>/*.png
Idempotent: skips work if the dataset is already present.
"""

import os
import tarfile
import urllib.request

URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
DATA_DIR = "data"
TGZ_PATH = os.path.join(DATA_DIR, "cifar10.tgz")
OUT_DIR = os.path.join(DATA_DIR, "cifar10")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    print(f"\r  downloading... {pct:3d}%  ({downloaded // (1 << 20)} MB)", end="")


def main():
    if os.path.isdir(os.path.join(OUT_DIR, "train")):
        print(f"Dataset already present at {OUT_DIR}/ — nothing to do.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TGZ_PATH):
        print(f"Fetching {URL}")
        urllib.request.urlretrieve(URL, TGZ_PATH, _progress)
        print()

    print("Extracting...")
    with tarfile.open(TGZ_PATH) as tar:
        tar.extractall(DATA_DIR)

    n_train = sum(len(files) for _, _, files in os.walk(os.path.join(OUT_DIR, "train")))
    n_test = sum(len(files) for _, _, files in os.walk(os.path.join(OUT_DIR, "test")))
    print(f"Done. {n_train} train images, {n_test} test images at {OUT_DIR}/")


if __name__ == "__main__":
    main()
