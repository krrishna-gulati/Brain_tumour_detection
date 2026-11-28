"""Build Dataset_prepared/ from the original Dataset/ folder.

Steps:
  1. Merge Dataset/Training and Dataset/Testing.
  2. Attach patient IDs. Most tumour images in Dataset/ are copies of slices from the Figshare
     brain tumour dataset, which records each slice's patient; figshare_patient_ids.csv (made by
     evaluate_external.py) lists them. Where Figshare's label differs from ours (7 meningioma
     slices labelled pituitary here), Figshare's label is used, since it is the original source.
  3. Remove exact duplicates (identical pixels) and near-duplicates (same scan
     saved at a different size/quality), so no scan can end up in both the
     training and the test split. Groups whose copies carry different class
     labels are dropped entirely, since their label is unreliable.
  4. Crop each scan to the brain and even out brightness (brain_preprocessing.py).
  5. Split into train / val / test (about 70 / 15 / 15), stratified by class, keeping all slices
     of a patient in the same split, so the test split only has patients the models never
     trained on. Images without a patient ID (healthy scans and scans from other sources) count
     as one patient each. The split uses a fixed random seed, so it is reproducible.
     --split_by random gives the earlier split, which ignores patients (then other slices of
     a test patient can be in the training split).

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
import stat
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from brain_preprocessing import prepare_image

CLASS_NAMES = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
SOURCE_SPLITS = ["Training", "Testing"]
MARKER = ".created_by_prepare_data"


def dhash(img):
    """64-bit difference hash: identical for the same scan at different sizes."""
    small = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten().tobytes()


def remove_tree(path):
    """shutil.rmtree that also removes read-only folders (OneDrive marks synced folders read-only)."""
    def clear_readonly_and_retry(func, target, _):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    handler = {"onexc" if sys.version_info >= (3, 12) else "onerror": clear_readonly_and_retry}
    shutil.rmtree(path, **handler)


def path_key(path):
    return os.path.normcase(os.path.abspath(path))


def load_patient_ids(csv_path):
    """{image path: (patient ID, Figshare class)} from figshare_patient_ids.csv."""
    if not os.path.exists(csv_path):
        return {}
    base = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, encoding="utf-8") as f:
        return {path_key(os.path.join(base, r["path"])): (r["pid"], r["figshare_class"]) for r in csv.DictReader(f)}


def collect(src, patient_ids):
    """Load every image once: exact hash, near-duplicate hash, prepared JPEG bytes, patient ID."""
    records, relabelled = [], Counter()
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
                pid, label = patient_ids.get(path_key(path), ("", cls))
                if label != cls:
                    relabelled[cls, label] += 1
                gray = np.asarray(img.convert("L"))
                exact = hashlib.md5(gray.tobytes() + str(gray.shape).encode()).hexdigest()
                prepared = prepare_image(img)
                buf = io.BytesIO()
                prepared.save(buf, format="JPEG", quality=95)
                records.append(
                    {
                        "path": path,
                        "origin": split,
                        "class": label,
                        "pid": pid,
                        "exact": exact,
                        "near": dhash(prepared),
                        "jpeg": buf.getvalue(),
                    }
                )
        print(f"Read and cropped {split}: {sum(r['origin'] == split for r in records)} images")
    for (old, new), n in relabelled.items():
        print(f"Relabelled {n} images from {old} to {new} (the label in the original Figshare dataset)")
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
        first = dict(group[0])
        first["pid"] = next((r["pid"] for r in group if r["pid"]), "")  # any copy may carry the ID
        kept.append(first)
        removed_copies += len(group) - 1
    return kept, removed_copies, removed_conflicts


def split_off(records, size, seed, by_patient):
    """Split off about `size` of the records, stratified by class (and grouped by patient)."""
    labels = [r["class"] for r in records]
    if not by_patient:
        return train_test_split(records, test_size=size, stratify=labels, random_state=seed)
    groups = [r["pid"] or r["path"] for r in records]  # no ID: the image is its own group
    folds = StratifiedGroupKFold(n_splits=round(1 / size), shuffle=True, random_state=seed)
    keep, held = next(folds.split(np.zeros(len(records)), labels, groups))
    return [records[i] for i in keep], [records[i] for i in held]


def main():
    p = argparse.ArgumentParser(description="Crop, de-duplicate and re-split the MRI dataset")
    p.add_argument("--src", default="Dataset", help="Original dataset (with Training/ and Testing/)")
    p.add_argument("--out", default="Dataset_prepared", help="Output folder")
    p.add_argument("--patient_ids", default="figshare_patient_ids.csv",
                   help="Patient IDs from evaluate_external.py")
    p.add_argument("--split_by", choices=["patient", "random"], default="patient")
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    patient_ids = load_patient_ids(args.patient_ids)
    if args.split_by == "patient" and not patient_ids:
        raise SystemExit(f"'{args.patient_ids}' not found: run evaluate_external.py first, or use --split_by random")

    if os.path.exists(args.out):
        if not os.path.exists(os.path.join(args.out, MARKER)):
            raise SystemExit(
                f"'{args.out}' exists but was not created by this script; refusing to overwrite it."
            )
        print(f"Removing previous {args.out}/ ...")
        remove_tree(args.out)

    records = collect(args.src, patient_ids)
    total = len(records)
    records, exact_copies, exact_conflicts = deduplicate(records, "exact")
    records, near_copies, near_conflicts = deduplicate(records, "near")
    print(f"\nImages read: {total}")
    print(f"Removed exact duplicates: {exact_copies}")
    print(f"Removed near-duplicates: {near_copies}")
    print(f"Removed (copies with conflicting labels): {exact_conflicts + near_conflicts}")
    print(f"Unique images kept: {len(records)} "
          f"({sum(bool(r['pid']) for r in records)} with a patient ID, "
          f"from {len({r['pid'] for r in records if r['pid']})} patients)")

    by_patient = args.split_by == "patient"
    train_val, test = split_off(records, args.test_size, args.seed, by_patient)
    train, val = split_off(train_val, args.val_size / (1 - args.test_size), args.seed, by_patient)

    splits = (("train", train), ("val", val), ("test", test))
    patients = {name: {r["pid"] for r in items if r["pid"]} for name, items in splits}
    if by_patient:
        assert not (patients["train"] & patients["test"] or patients["val"] & patients["test"]
                    or patients["train"] & patients["val"]), "a patient ended up in two splits"

    manifest = []
    for split_name, items in splits:
        for r in items:
            folder = os.path.join(args.out, split_name, r["class"])
            os.makedirs(folder, exist_ok=True)
            stem = os.path.splitext(os.path.basename(r["path"]))[0]
            out_name = f"{r['origin'].lower()}_{stem}.jpg"
            with open(os.path.join(folder, out_name), "wb") as f:
                f.write(r["jpeg"])
            manifest.append([split_name, r["class"], out_name, os.path.relpath(r["path"]), r["pid"]])

    with open(os.path.join(args.out, "split_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "file", "original_path", "pid"])
        w.writerows(manifest)
    open(os.path.join(args.out, MARKER), "w").close()

    print(f"\nSplit by {args.split_by}:")
    print(f"{'class':18s} {'train':>6s} {'val':>6s} {'test':>6s}")
    counts = {name: Counter(r["class"] for r in items) for name, items in splits}
    for cls in CLASS_NAMES:
        print(f"{cls:18s} {counts['train'][cls]:6d} {counts['val'][cls]:6d} {counts['test'][cls]:6d}")
    print(f"{'total':18s} {len(train):6d} {len(val):6d} {len(test):6d}")
    print(f"{'patients (known)':18s} {len(patients['train']):6d} {len(patients['val']):6d} {len(patients['test']):6d}")
    shared = len(patients["test"] & (patients["train"] | patients["val"]))
    print(f"Test patients that also have slices in train/val: {shared}")
    print(f"\nWrote {args.out}/ (see split_manifest.csv for where each image came from)")


if __name__ == "__main__":
    main()
