"""Transfer learning with other pre-trained networks (ResNet50, EfficientNetB0).

Same recipe as train_and_infer_vgg19.py, so results are comparable:
ImageNet weights, the last block fine-tuned at a low learning rate, the same augmentation,
the same head (global average pooling -> Dense 256 -> Dropout 0.5 -> softmax), and the same
data split, best-epoch saving and early stopping.

Usage:
    python train_and_infer_transfer.py --arch resnet50 --mode train
    python train_and_infer_transfer.py --arch efficientnetb0 --mode evaluate
    python train_and_infer_transfer.py --arch resnet50 --mode predict --image_path <image file>
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

from brain_preprocessing import prepare_image

# Must match the class folder names inside Dataset_prepared/train, val and test
CLASS_NAMES = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]

# For each network: the Keras builder, its own input preprocessing, and which layers to
# fine-tune (the last block, like VGG19's block5). BatchNorm layers always stay frozen.
ARCHS = {
    "resnet50": {
        "builder": tf.keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "finetune": lambda name: name.startswith("conv5_"),
        "label": "ResNet50",
    },
    "efficientnetb0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "finetune": lambda name: name.startswith(("block7", "top_")),
        "label": "EfficientNetB0",
    },
}


def model_path(arch):
    # Native .keras format: EfficientNet does not survive the legacy .h5 format
    # (a reloaded model gave predictions up to 0.15 different), while .keras reloads exactly.
    return os.path.join("models", f"multiclass_{arch}_model.keras")


def load_images_and_labels(data_dir, preprocess, target_size=(224, 224)):
    images, labels = [], []
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
                        img = Image.open(path).convert("RGB").resize(target_size)
                        images.append(preprocess(np.array(img).astype("float32")))
                        labels.append(idx)
                    except Exception as e:
                        print(f"warning: could not read {path}: {e}")
    X, y = np.array(images), np.array(labels)
    print(f"Total samples loaded: {len(X)}")
    return X, y


def build_model(arch, input_shape=(224, 224, 3), num_classes=4, learning_rate=1e-4):
    spec = ARCHS[arch]
    base = spec["builder"](weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = spec["finetune"](layer.name) and not isinstance(layer, layers.BatchNormalization)

    inputs = layers.Input(shape=input_shape)
    # Data augmentation (only active during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    # training=False keeps BatchNorm statistics fixed while the last block is fine-tuned
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(args):
    spec = ARCHS[args.arch]
    size = (args.width, args.height)
    X_train, y_train = load_images_and_labels(args.train_dir, spec["preprocess"], size)
    X_val, y_val = load_images_and_labels(args.val_dir, spec["preprocess"], size)
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    model = build_model(args.arch, (args.width, args.height, 3), len(CLASS_NAMES), args.learning_rate)
    model.summary()

    os.makedirs("models", exist_ok=True)
    save_path = model_path(args.arch)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )
    with open(os.path.join("models", f"history_{args.arch}.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    val_acc = history.history["val_accuracy"]
    best_epoch = int(np.argmax(val_acc)) + 1
    print(
        f"\nBest epoch: {best_epoch} of {len(val_acc)} (val_accuracy {max(val_acc) * 100:.2f}%). "
        f"{spec['label']} model from that epoch saved to {save_path}"
    )


def load_trained(args):
    path = args.model_path or model_path(args.arch)
    if not Path(path).exists():
        print(f"Model file not found at: {path}")
        sys.exit(1)
    print(f"Loading {ARCHS[args.arch]['label']} model from {path} ...")
    return tf.keras.models.load_model(path)


def evaluate(args):
    spec = ARCHS[args.arch]
    model = load_trained(args)
    X_test, y_test = load_images_and_labels(args.test_dir, spec["preprocess"], (args.width, args.height))
    print(f"Running predictions on the test set ({spec['label']})...")
    y_pred = np.argmax(model.predict(X_test, batch_size=args.batch_size), axis=1)

    correct = int(np.sum(y_pred == y_test))
    print(f"\nTest accuracy ({spec['label']}): {correct / len(y_test) * 100:.2f}% ({correct}/{len(y_test)} correct)\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    w = max(len(n) for n in CLASS_NAMES)
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print(" " * w + "  " + "  ".join(f"{n:>{w}}" for n in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, cm):
        print(f"{name:>{w}}  " + "  ".join(f"{v:>{w}}" for v in row))


def preprocess_single_image(img_path, preprocess, target_size=(224, 224)):
    # Same crop + brightness normalisation as the training images
    img = prepare_image(Image.open(img_path)).resize(target_size)
    return np.expand_dims(preprocess(np.array(img).astype("float32")), axis=0)


def predict(args):
    spec = ARCHS[args.arch]
    if not Path(args.image_path).exists():
        print(f"Image file not found at: {args.image_path}")
        sys.exit(1)
    model = load_trained(args)
    probs = model.predict(preprocess_single_image(args.image_path, spec["preprocess"], (args.width, args.height)))[0]
    for name, p in zip(CLASS_NAMES, probs):
        print(f"{name}: {p * 100:.2f}%")
    best = int(np.argmax(probs))
    print(f"\nFinal prediction ({spec['label']}): {CLASS_NAMES[best]} ({probs[best] * 100:.2f}% confidence)")


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate/predict with ResNet50 or EfficientNetB0 transfer learning")
    p.add_argument("--arch", choices=sorted(ARCHS), required=True)
    p.add_argument("--mode", choices=["train", "predict", "evaluate"], required=True)
    p.add_argument("--train_dir", default="Dataset_prepared/train")
    p.add_argument("--val_dir", default="Dataset_prepared/val")
    p.add_argument("--test_dir", default="Dataset_prepared/test")
    p.add_argument("--model_path", default=None, help="Defaults to models/multiclass_<arch>_model.keras")
    p.add_argument("--image_path", help="Image file for predict mode")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4, help="Stop early after this many epochs without improvement")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--height", type=int, default=224)
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
