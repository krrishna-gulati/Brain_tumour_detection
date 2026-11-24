"""Build Dataset_prepared/ from the original Dataset/ folder.

Steps:
  1. Merge Dataset/Training and Dataset/Testing.
  2. Remove exact duplicates (identical pixels) and near-duplicates (same scan
     saved at a different size/quality), so no scan can end up in both the
     training and the test split. Groups whose copies carry different class
     labels are dropped entirely, since their label is unreliable.
  3. Crop each scan to the brain and even out brightness (brain_preprocessing.py).
  4. Split into train / val / test (70 / 15 / 15), stratified by class, with a
     fixed random seed so the split is reproducible.

Output: Dataset_prepared/{train,val,test}/<class>/*.jpg plus split_manifest.csv
The original Dataset/ folder is not modified.

Usage:  python prepare_data.py
"""
import argparse
import csv
import hashlib
import io
import os
import shutil
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from brain_preprocessing import prepare_image

CLASS_NAMES = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
SOURCE_SPLITS = ["Training", "Testing"]
MARKER = ".created_by_prepare_data"


def dhash(img):
    """64-bit difference hash: identical for the same scan at different sizes."""
    small = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten().tobytes()


def collect(src):
    """Load every image once: exact hash, near-duplicate hash, prepared JPEG bytes."""
    records = []
    for split in SOURCE_SPLITS:
        for cls in CLASS_NAMES:
            folder = os.path.join(src, split, cls)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Expected folder: {folder}")
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                path = os.path.join(folder, fname)
                try:
                    img = Image.open(path)
                    img.load()
                except Exception as e:
                    print(f"warning: could not read {path}: {e}")
                    continue
                gray = np.asarray(img.convert("L"))
                exact = hashlib.md5(gray.tobytes() + str(gray.shape).encode()).hexdigest()
                prepared = prepare_image(img)
                buf = io.BytesIO()
                prepared.save(buf, format="JPEG", quality=95)
                records.append(
                    {
                        "path": path,
                        "origin": split,
                        "class": cls,
                        "exact": exact,
                        "near": dhash(prepared),
                        "jpeg": buf.getvalue(),
                    }
                )
        print(f"Read and cropped {split}: {sum(r['origin'] == split for r in records)} images")
    return records


def deduplicate(records, key):
    """Keep one image per group; drop whole groups with conflicting labels."""
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    kept, removed_copies, removed_conflicts = [], 0, 0
    for group in groups.values():
        if len({r["class"] for r in group}) > 1:
            removed_conflicts += len(group)
            continue
        kept.append(group[0])
        removed_copies += len(group) - 1
    return kept, removed_copies, removed_conflicts


def main():
    p = argparse.ArgumentParser(description="Crop, de-duplicate and re-split the MRI dataset")
    p.add_argument("--src", default="Dataset", help="Original dataset (with Training/ and Testing/)")
    p.add_argument("--out", default="Dataset_prepared", help="Output folder")
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if os.path.exists(args.out):
        if not os.path.exists(os.path.join(args.out, MARKER)):
            raise SystemExit(
                f"'{args.out}' exists but was not created by this script; refusing to overwrite it."
            )
        print(f"Removing previous {args.out}/ ...")
        shutil.rmtree(args.out)

    records = collect(args.src)
    total = len(records)
    records, exact_copies, exact_conflicts = deduplicate(records, "exact")
    records, near_copies, near_conflicts = deduplicate(records, "near")
    print(f"\nImages read: {total}")
    print(f"Removed exact duplicates: {exact_copies}")
    print(f"Removed near-duplicates: {near_copies}")
    print(f"Removed (copies with conflicting labels): {exact_conflicts + near_conflicts}")
    print(f"Unique images kept: {len(records)}")

    labels = [r["class"] for r in records]
    train_val, test = train_test_split(
        records, test_size=args.test_size, stratify=labels, random_state=args.seed
    )
    val_fraction = args.val_size / (1 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=[r["class"] for r in train_val],
        random_state=args.seed,
    )

    manifest = []
    for split_name, items in (("train", train), ("val", val), ("test", test)):
        for r in items:
            folder = os.path.join(args.out, split_name, r["class"])
            os.makedirs(folder, exist_ok=True)
            stem = os.path.splitext(os.path.basename(r["path"]))[0]
            out_name = f"{r['origin'].lower()}_{stem}.jpg"
            with open(os.path.join(folder, out_name), "wb") as f:
                f.write(r["jpeg"])
            manifest.append([split_name, r["class"], out_name, os.path.relpath(r["path"])])

    with open(os.path.join(args.out, "split_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "file", "original_path"])
        w.writerows(manifest)
    open(os.path.join(args.out, MARKER), "w").close()

    print(f"\n{'class':18s} {'train':>6s} {'val':>6s} {'test':>6s}")
    counts = {name: Counter(r["class"] for r in items) for name, items in
              (("train", train), ("val", val), ("test", test))}
    for cls in CLASS_NAMES:
        print(f"{cls:18s} {counts['train'][cls]:6d} {counts['val'][cls]:6d} {counts['test'][cls]:6d}")
    print(f"{'total':18s} {len(train):6d} {len(val):6d} {len(test):6d}")
    print(f"\nWrote {args.out}/ (see split_manifest.csv for where each image came from)")


if __name__ == "__main__":
    main()
