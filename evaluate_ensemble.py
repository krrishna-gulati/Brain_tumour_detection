"""Evaluate every model and the combined model (average of all models' probabilities).

Each model sees exactly the input its own evaluate mode uses (Dataset_prepared images with
that model's preprocessing), so the individual scores match the training scripts' results.

Usage:
    python evaluate_ensemble.py               # test split
    python evaluate_ensemble.py --split val   # validation split
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import train_and_infer_cnn as cnn
import train_and_infer_transfer as transfer
import train_and_infer_vgg19 as vgg
from brain_classifier import MODEL_SPECS, combine

LOADERS = {
    "vgg19": lambda d: vgg.load_images_and_labels(d, (224, 224)),
    "cnn": lambda d: cnn.load_images_and_labels(d, (240, 240)),
    "resnet50": lambda d: transfer.load_images_and_labels(d, transfer.ARCHS["resnet50"]["preprocess"], (224, 224)),
    "efficientnetb0": lambda d: transfer.load_images_and_labels(
        d, transfer.ARCHS["efficientnetb0"]["preprocess"], (224, 224)),
}


def main():
    p = argparse.ArgumentParser(description="Evaluate all models and their combination")
    p.add_argument("--split", choices=["test", "val"], default="test")
    args = p.parse_args()
    data_dir = os.path.join("Dataset_prepared", args.split)
    names = vgg.CLASS_NAMES

    probs, y = {}, None
    for spec in MODEL_SPECS:
        if not spec["path"].exists():
            print(f"skipping {spec['name']}: {spec['path']} not found")
            continue
        X, labels = LOADERS[spec["key"]](data_dir)
        if y is None:
            y = labels
        assert np.array_equal(y, labels), "loaders returned images in a different order"
        model = tf.keras.models.load_model(spec["path"], compile=False)
        probs[spec["key"]] = model.predict(X, batch_size=16, verbose=0)
        del X, model

    print(f"\n{args.split} split: {len(y)} scans\n")
    preds = {k: np.argmax(v, axis=1) for k, v in probs.items()}
    combined = np.array([combine([probs[k][i] for k in probs]) for i in range(len(y))])
    preds["combined"] = np.argmax(combined, axis=1)

    label = {spec["key"]: spec["name"] for spec in MODEL_SPECS}
    label["combined"] = f"Combined ({len(probs)} models)"
    for key, pred in preds.items():
        correct = int(np.sum(pred == y))
        print(f"{label[key]:<22} {correct / len(y) * 100:6.2f}%  ({correct}/{len(y)})")

    model_preds = np.stack([preds[k] for k in probs], axis=1)
    all_agree = np.all(model_preds == model_preds[:, :1], axis=1)
    print(f"\nAll {len(probs)} models agree on {all_agree.sum()} of {len(y)} scans; "
          f"the combined answer is right on {int(np.sum(preds['combined'][all_agree] == y[all_agree]))} of those.")
    print(f"On the {int((~all_agree).sum())} scans where they disagree, the combined answer is right on "
          f"{int(np.sum(preds['combined'][~all_agree] == y[~all_agree]))}.\n")

    print(f"Combined model ({args.split}):")
    print(classification_report(y, preds["combined"], target_names=names, digits=3))
    cm = confusion_matrix(y, preds["combined"], labels=list(range(len(names))))
    w = max(len(n) for n in names)
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print(" " * w + "  " + "  ".join(f"{n:>{w}}" for n in names))
    for name, row in zip(names, cm):
        print(f"{name:>{w}}  " + "  ".join(f"{v:>{w}}" for v in row))


if __name__ == "__main__":
    main()
