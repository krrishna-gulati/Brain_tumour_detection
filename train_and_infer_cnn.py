import os
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

from brain_preprocessing import prepare_image

# IMPORTANT: these names must match the class folder names inside
# Dataset_prepared/train, val and test (created by prepare_data.py)
CLASS_NAMES = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]


def load_images_and_labels(data_dir, target_size=(240, 240)):
    images = []
    labels = []

    for idx, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(data_dir, class_name)
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"Expected folder: {folder}. Run prepare_data.py first to create Dataset_prepared/."
            )

        print(f"Loading images from {folder} (label = {idx}, class = {class_name})")
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(root, fname)
                    try:
                        img = Image.open(path).convert("RGB")
                        img = img.resize(target_size)
                        arr = np.array(img).astype("float32") / 255.0
                        images.append(arr)
                        labels.append(idx)  # numeric label (0,1,2,3)
                    except Exception as e:
                        print(f"warning: could not read {path}: {e}")

    X = np.array(images)
    y = np.array(labels)
    print(f"Total samples loaded: {len(X)}")
    return X, y


def prepare_dataset(train_dir, val_dir, target_size=(240, 240)):
    # The train/val/test split is fixed on disk by prepare_data.py, so every
    # run (and both models) use exactly the same images.
    X_train, y_train = load_images_and_labels(train_dir, target_size)
    X_val, y_val = load_images_and_labels(val_dir, target_size)
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    return X_train, X_val, y_train, y_val


def build_multiclass_model(input_shape=(240, 240, 3), num_classes=4):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            # Data augmentation: random flips/rotations/zooms of each training
            # image so the model generalises to scans that look a bit different.
            # These layers are only active during training, not predict/evaluate.
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),

            # 4 outputs (one per class) → softmax for multiclass probs
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # y are 0,1,2,3
        metrics=["accuracy"],
    )
    return model


def train(args):
    X_train, X_val, y_train, y_val = prepare_dataset(
        args.train_dir, args.val_dir, target_size=(args.width, args.height)
    )

    model = build_multiclass_model(
        input_shape=(args.width, args.height, 3),
        num_classes=len(CLASS_NAMES),
    )
    model.summary()

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "multiclass_model.h5")

    callbacks = [
        # Save the model only when val_accuracy improves, so the file on disk
        # is always the best epoch so far (not just the last one).
        tf.keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        # Stop early if val_accuracy hasn't improved for `patience` epochs.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience, verbose=1
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    val_acc = history.history["val_accuracy"]
    best_epoch = int(np.argmax(val_acc)) + 1
    print(
        f"\nBest epoch: {best_epoch} of {len(val_acc)} "
        f"(val_accuracy {max(val_acc) * 100:.2f}%). "
        f"Model from that epoch saved to {save_path}"
    )


def preprocess_single_image(img_path, target_size=(240, 240)):
    # Same crop + brightness normalisation as the training images
    img = prepare_image(Image.open(img_path))
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(args):
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"Model file not found at: {model_path}")
        sys.exit(1)

    if not Path(args.image_path).exists():
        print(f"Image file not found at: {args.image_path}")
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    print(f"Preprocessing image: {args.image_path}")
    x = preprocess_single_image(
        args.image_path, target_size=(args.width, args.height)
    )

    print("Running prediction...")
    probs = model.predict(x)[0]  # vector length = num_classes

    for i, p in enumerate(probs):
        print(f"{CLASS_NAMES[i]}: {p * 100:.2f}%")

    best_idx = int(np.argmax(probs))
    best_class = CLASS_NAMES[best_idx]
    best_prob = probs[best_idx]

    print(f"\nFinal prediction: {best_class} ({best_prob * 100:.2f}% confidence)")


def evaluate(args):
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"Model file not found at: {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    # Same preprocessing as training, so results are comparable
    X_test, y_test = load_images_and_labels(
        args.test_dir, target_size=(args.width, args.height)
    )

    print("Running predictions on the test set...")
    probs = model.predict(X_test, batch_size=args.batch_size)
    y_pred = np.argmax(probs, axis=1)

    correct = int(np.sum(y_pred == y_test))
    print(f"\nTest accuracy: {correct / len(y_test) * 100:.2f}% ({correct}/{len(y_test)} correct)\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    w = max(len(n) for n in CLASS_NAMES)
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print(" " * w + "  " + "  ".join(f"{n:>{w}}" for n in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, cm):
        print(f"{name:>{w}}  " + "  ".join(f"{v:>{w}}" for v in row))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train or predict brain tumor type (no_tumor, glioma, meningioma, pituitary)"
    )
    p.add_argument(
        "--mode", choices=["train", "predict", "evaluate"], required=True
    )
    p.add_argument(
        "--train_dir",
        type=str,
        default="Dataset_prepared/train",
        help="Training images (one subfolder per class)",
    )
    p.add_argument(
        "--val_dir",
        type=str,
        default="Dataset_prepared/val",
        help="Validation images (one subfolder per class)",
    )
    p.add_argument(
        "--model_path", type=str, default="models/multiclass_model.h5"
    )
    p.add_argument("--image_path", type=str, help="Image file for prediction")
    p.add_argument(
        "--test_dir",
        type=str,
        default="Dataset_prepared/test",
        help="Directory with test images (one subfolder per class) for evaluate mode",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Stop early if val_accuracy doesn't improve for this many epochs",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--width", type=int, default=240)
    p.add_argument("--height", type=int, default=240)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        if not args.image_path:
            print("Error: --image_path is required in predict mode")
            sys.exit(1)
        predict(args)


if __name__ == "__main__":
    main()

