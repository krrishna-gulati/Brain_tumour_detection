"""Test the models on the Figshare brain tumour dataset, and use its patient IDs to check our split.

Figshare brain tumour dataset (Cheng et al., 2015): 3,064 T1-weighted contrast-enhanced MRI slices
from 233 patients, scanned at two hospitals in China, in three classes (meningioma, glioma,
pituitary tumour); there are no healthy scans. Download the four brainTumorDataPublic_*.zip files
from https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 into one folder.

It turned out not to be an outside dataset for us: most of the tumour images in our Dataset/
are identical copies of its slices. So the script splits its slices by what the models have seen:
  - copies of scans in our train/val split (the models trained on these);
  - new slices of patients who have other slices in our train/val split;
  - slices of patients with no slice in our train/val split (the fair outside test).
The copies also give most of our own scans a patient ID. The script saves those IDs in
figshare_patient_ids.csv, which prepare_data.py uses to split by patient. It also checks our test
split for patient leakage: test scans whose patient has other slices in train/val, test scans
whose patient does not, and test scans from other sources.

Steps (each step saves its output in --data_dir, so a rerun skips the finished ones):
  1. convert:  read the .mat files straight from the zips and save each slice as an 8-bit PNG,
               using the min-max conversion from the dataset's own README.
  2. match:    find each slice's most similar image in our Dataset/ (correlation of 32x32
               thumbnails after our crop). Identical copies score 0.995 or more, while no two
               different slices score 0.99 (not even neighbouring slices of one patient), so
               0.99 is the cut-off.
  3. evaluate: run every model, and the combined model, exactly as the website does (raw file
               in, crop and brightness normalisation, then each model's own preprocessing).

Usage:
    python evaluate_external.py --data_dir <folder with the 4 zips>
"""
import argparse
import csv
import io
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))  # so the script also runs with python -I (isolated mode)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from brain_preprocessing import prepare_image  # noqa: E402

FIGSHARE_CLASSES = {1: "meningioma_tumor", 2: "glioma_tumor", 3: "pituitary_tumor"}
TUMOUR_CLASSES = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
SHORT = {"no_tumor": "no", "glioma_tumor": "glioma", "meningioma_tumor": "mening", "pituitary_tumor": "pituit"}
SAME_SCAN_CORRELATION = 0.99
PATIENT_IDS_FILE = PROJECT_DIR / "figshare_patient_ids.csv"  # read by prepare_data.py

OUTSIDE_GROUPS = [
    "copy of a scan in our train/val",
    "new slice, patient in our train/val",
    "patient not in our train/val",
]
INTERNAL_GROUPS = [
    "patient also in train/val",
    "Figshare patient not in train/val",
    "other source (no patient ID)",
]


def path_key(p):
    return os.path.normcase(os.path.abspath(p))


# --- 1. convert ----------------------------------------------------------------------------

def read_mat(raw):
    """Return (image, label, patient id) from one cjdata .mat file (MATLAB v7.3 or older)."""
    try:
        import h5py

        with h5py.File(io.BytesIO(raw), "r") as f:
            cj = f["cjdata"]
            image = np.array(cj["image"]).T  # MATLAB stores column-major
            label = int(np.array(cj["label"]).item())
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).flatten())
    except OSError:  # not an HDF5 file: an older MATLAB format
        from scipy.io import loadmat

        cj = loadmat(io.BytesIO(raw))["cjdata"][0, 0]
        image, label, pid = cj["image"], int(cj["label"].item()), str(cj["PID"].item())
    return image, label, pid.strip()


def to_uint8(image):
    """The dataset README's conversion: uint8(255 / (max - min) * (image - min))."""
    im = image.astype(np.float64)
    lo, hi = im.min(), im.max()
    return np.clip(np.round(255.0 / (hi - lo) * (im - lo)), 0, 255).astype(np.uint8)


def convert(data_dir):
    manifest_path = data_dir / "manifest.csv"
    if manifest_path.exists():
        return list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    zips = sorted(data_dir.glob("brainTumorDataPublic_*.zip"))
    if len(zips) != 4:
        raise SystemExit(f"Expected the 4 brainTumorDataPublic_*.zip files in {data_dir}, found {len(zips)}")
    rows = []
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                name = member.replace("\\", "/").split("/")[-1]
                if not re.fullmatch(r"\d+\.mat", name):  # only the numbered slice files
                    continue
                image, label, pid = read_mat(zf.read(member))
                cls = FIGSHARE_CLASSES[label]
                out = data_dir / "images" / cls / name.replace(".mat", ".png")
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(to_uint8(image)).save(out)
                rows.append({"file": out.name, "class": cls, "pid": pid,
                             "size": f"{image.shape[1]}x{image.shape[0]}"})
        print(f"converted {zpath.name}: {len(rows)} slices so far")
    rows.sort(key=lambda r: int(r["file"].split(".")[0]))
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# --- 2. match ------------------------------------------------------------------------------

def thumbnails(paths, label):
    """Normalised 32x32 thumbnail of each image after our crop and brightness normalisation."""
    out = []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        prepared = prepare_image(Image.open(p))
        t = np.asarray(prepared.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
        t -= t.mean()
        out.append(t / (np.linalg.norm(t) + 1e-8))
        if i % 500 == 0:
            print(f"  {label}: {i}/{len(paths)} ({time.time() - t0:.0f}s)")
    return np.stack(out)


def our_images():
    """Every image of the original Dataset/ (Training and Testing): (path, folder, class)."""
    found = []
    for folder in ("Training", "Testing"):
        for cls_dir in sorted((PROJECT_DIR / "Dataset" / folder).iterdir()):
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    found.append((p, folder, cls_dir.name))
    return found


def prepared_manifest():
    with open(PROJECT_DIR / "Dataset_prepared" / "split_manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def match(data_dir, rows):
    """Link Figshare slices and our images that are the same scan, and group both sides."""
    ours = our_images()
    cache = data_dir / "thumbnails.npz"
    if cache.exists():
        z = np.load(cache)
        fig_thumb, our_thumb = z["fig"], z["ours"]
    else:
        print("Making thumbnails of the Figshare slices and of our Dataset/ (about 5 minutes)...")
        fig_thumb = thumbnails([data_dir / "images" / r["class"] / r["file"] for r in rows], "Figshare")
        our_thumb = thumbnails([p for p, _, _ in ours], "ours")
        np.savez(cache, fig=fig_thumb, ours=our_thumb)

    sims = fig_thumb @ our_thumb.T
    split_of = {path_key(PROJECT_DIR / r["original_path"]): r["split"] for r in prepared_manifest()}

    # Our side: each of our images that is a copy of a Figshare slice gets that slice's patient ID.
    # They are saved in the project, so prepare_data.py can split by patient without this download.
    our_pid, copies_of, id_rows = {}, defaultdict(list), []
    fig_idx = sims.argmax(axis=0)
    for j, ((p, folder, cls), i) in enumerate(zip(ours, fig_idx)):
        if sims[i, j] >= SAME_SCAN_CORRELATION:
            our_pid[path_key(p)] = rows[i]["pid"]
            copies_of[i].append((folder, cls, split_of.get(path_key(p), "removed by de-duplication")))
            id_rows.append([p.relative_to(PROJECT_DIR).as_posix(), cls, rows[i]["file"], rows[i]["class"],
                            rows[i]["pid"], f"{sims[i, j]:.4f}"])
    with open(PATIENT_IDS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "our_class", "figshare_file", "figshare_class", "pid", "similarity"])
        w.writerows(id_rows)
    trained_pids = {pid for k, pid in our_pid.items() if split_of.get(k) in ("train", "val")}

    # Figshare side: where its copies ended up in our data, and which group it belongs to.
    for i, r in enumerate(rows):
        copies = copies_of.get(i, [])
        r["our_copies"] = ";".join(f"{folder}/{cls}->{split}" for folder, cls, split in copies)
        r["our_split"] = next((s for s in ("train", "val", "test") if any(c[2] == s for c in copies)), "")
        if r["our_split"] in ("train", "val"):
            r["group"] = OUTSIDE_GROUPS[0]
        elif r["pid"] in trained_pids:
            r["group"] = OUTSIDE_GROUPS[1]
        else:
            r["group"] = OUTSIDE_GROUPS[2]

    # Summary of the overlap
    pids = {r["pid"] for r in rows}
    copied = [i for i in range(len(rows)) if copies_of.get(i)]
    print(f"\nFigshare slices with an identical copy in our Dataset/: {len(copied)} of {len(rows)}, "
          f"from {len({rows[i]['pid'] for i in copied})} of {len(pids)} patients")
    print("Our Dataset/ images that are Figshare copies:")
    total = Counter((folder, cls) for _, folder, cls in ours)
    hits = Counter((folder, cls) for (p, folder, cls) in ours if path_key(p) in our_pid)
    for folder in ("Training", "Testing"):
        print(f"  {folder:<9}" + ", ".join(f"{SHORT[c]} {hits[folder, c]}/{total[folder, c]}"
                                          for c in ["no_tumor"] + TUMOUR_CLASSES))
    wrong = defaultdict(lambda: [set(), 0])  # (Figshare class, our class) -> [Figshare slices, our images]
    for i in copied:
        for _, cls, _ in copies_of[i]:
            if cls != rows[i]["class"]:
                wrong[rows[i]["class"], cls][0].add(i)
                wrong[rows[i]["class"], cls][1] += 1
    for (fig_cls, our_cls), (slices, n_ours) in wrong.items():
        print(f"  label differs: {len(slices)} Figshare {SHORT[fig_cls]} slices are labelled {SHORT[our_cls]} "
              f"in our Dataset/ ({n_ours} images)")
    print("\nFigshare slices by group:")
    for g in OUTSIDE_GROUPS:
        members = [r for r in rows if r["group"] == g]
        print(f"  {g:<38} {len(members):>5} slices, {len({r['pid'] for r in members}):>3} patients")
    return rows, our_pid, trained_pids


def internal_test_rows(our_pid, trained_pids):
    """Tumour scans of our test split, grouped by whether their patient is in train/val."""
    rows = []
    for r in prepared_manifest():
        if r["split"] != "test" or r["class"] == "no_tumor":
            continue
        pid = our_pid.get(path_key(PROJECT_DIR / r["original_path"]))
        group = INTERNAL_GROUPS[2] if pid is None else INTERNAL_GROUPS[0 if pid in trained_pids else 1]
        rows.append({**r, "pid": pid or "", "group": group})
    print("\nOur test split, tumour scans by group:")
    for g in INTERNAL_GROUPS:
        members = [r for r in rows if r["group"] == g]
        print(f"  {g:<38} {len(members):>5} scans, {len({r['pid'] for r in members if r['pid']}):>3} known patients")
    return rows


# --- 3. evaluate ---------------------------------------------------------------------------

def predict_files(clf, key, paths, chunk=64):
    """Class probabilities for each file, with the same preprocessing as the website."""
    out = []
    for i in range(0, len(paths), chunk):
        X = np.concatenate([clf._input(key, p) for p in paths[i:i + chunk]])
        out.append(clf.models[key].predict(X, batch_size=32, verbose=0))
    return np.concatenate(out)


def score_table(title, y, preds, names, labels, n_patients=None):
    """One line per model: accuracy, recall per tumour type, balanced accuracy, tumours called healthy."""
    n = len(y)
    if n == 0:
        return
    who = f", {n_patients} patients" if n_patients else ""
    print(f"\n{title}: {n} slices{who} "
          f"({', '.join(f'{SHORT[c]} {int(np.sum(y == names.index(c)))}' for c in TUMOUR_CLASSES)})")
    print(f"{'model':<22}{'accuracy':>16}{'glioma':>9}{'mening':>9}{'pituit':>9}{'balanced':>10}{'called healthy':>16}")
    for key, pred in preds.items():
        recalls = [np.mean(pred[y == names.index(c)] == names.index(c)) * 100 if np.any(y == names.index(c))
                   else np.nan for c in TUMOUR_CLASSES]
        correct = int(np.sum(pred == y))
        healthy = int(np.sum(pred == names.index("no_tumor")))
        print(f"{labels[key]:<22}{correct / n * 100:>7.1f}% ({correct:>4}){recalls[0]:>8.1f}%{recalls[1]:>8.1f}%"
              f"{recalls[2]:>8.1f}%{np.nanmean(recalls):>9.1f}%{healthy:>9} ({healthy / n * 100:.1f}%)")


def confusion(y, pred, names, title):
    print(f"\n{title} (rows = true class, columns = predicted class):")
    print(" " * 10 + "".join(f"{SHORT[c]:>9}" for c in names))
    for c in TUMOUR_CLASSES:
        row = [int(np.sum((y == names.index(c)) & (pred == j))) for j in range(len(names))]
        print(f"{SHORT[c]:>10}" + "".join(f"{v:>9}" for v in row))


def evaluate(data_dir, rows, internal):
    from brain_classifier import MODEL_SPECS, Classifier, combine

    clf = Classifier()
    clf.load()
    names = clf.class_names
    labels = {s["key"]: s["name"] for s in MODEL_SPECS}
    labels["combined"] = f"Combined ({len(clf.models)} models)"

    sets = {
        "Figshare": (rows, [data_dir / "images" / r["class"] / r["file"] for r in rows]),
        "our test": (internal, [PROJECT_DIR / r["original_path"] for r in internal]),
    }
    probs = {name: {} for name in sets}
    for key in clf.models:
        for name, (_, paths) in sets.items():
            t0 = time.time()
            probs[name][key] = predict_files(clf, key, paths)
            print(f"{labels[key]:<16} {name:<9} {len(paths):>5} scans in {time.time() - t0:.0f}s")

    preds = {}
    for name, (items, _) in sets.items():
        probs[name]["combined"] = np.array([combine([probs[name][k][i] for k in clf.models])
                                            for i in range(len(items))])
        preds[name] = {k: np.argmax(v, axis=1) for k, v in probs[name].items()}

    # Per-slice results, for the report and for looking at individual mistakes
    keys = list(probs["Figshare"])
    with open(data_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0]) + [f"{k}_pred" for k in keys] + [f"{k}_{SHORT[c]}" for k in keys for c in names])
        for i, r in enumerate(rows):
            w.writerow(list(r.values()) + [names[preds["Figshare"][k][i]] for k in keys]
                       + [f"{p:.4f}" for k in keys for p in probs["Figshare"][k][i]])

    for name, groups in (("our test", INTERNAL_GROUPS), ("Figshare", OUTSIDE_GROUPS)):
        items = sets[name][0]
        y = np.array([names.index(r["class"]) for r in items])
        group = np.array([r["group"] for r in items])
        print(f"\n===== {name.upper()} =====")
        score_table(f"{name}: all", y, preds[name], names, labels)
        for g in groups:
            mask = group == g
            n_pat = len({r["pid"] for r, m in zip(items, mask) if m and r["pid"]})
            score_table(f"{name}: {g}", y[mask], {k: v[mask] for k, v in preds[name].items()}, names, labels, n_pat)
        last = group == groups[-1] if name == "Figshare" else np.ones(len(y), bool)
        confusion(y[last], preds[name]["combined"][last], names,
                  f"Combined model, {name}: {groups[-1] if name == 'Figshare' else 'all tumour scans'}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description="Test the models on the Figshare brain tumour dataset")
    p.add_argument("--data_dir", required=True, help="Folder with the 4 brainTumorDataPublic_*.zip files")
    p.add_argument("--skip_eval", action="store_true", help="Only convert and match, no model runs")
    args = p.parse_args()
    data_dir = Path(args.data_dir).resolve()

    rows = convert(data_dir)
    counts = ", ".join(f"{SHORT[c]} {sum(r['class'] == c for r in rows)}" for c in TUMOUR_CLASSES)
    print(f"Figshare dataset: {len(rows)} slices from {len({r['pid'] for r in rows})} patients ({counts})")

    rows, our_pid, trained_pids = match(data_dir, rows)
    internal = internal_test_rows(our_pid, trained_pids)
    if not args.skip_eval:
        evaluate(data_dir, rows, internal)


if __name__ == "__main__":
    main()
