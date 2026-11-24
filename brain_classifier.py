"""Shared prediction code for the desktop app (predict_app.py) and the website (web_app.py).

Loads the trained models once and classifies an image file with each of them, using the
same crop + brightness normalisation as training (via the training scripts' preprocessors).
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's info messages

BASE_DIR = Path(__file__).resolve().parent

# Main model first. Paths are the files written by the training scripts.
MODEL_SPECS = [
    {"key": "vgg19", "title": "VGG19 (main model)", "path": BASE_DIR / "models" / "multiclass_vgg19_model.h5"},
    {"key": "cnn", "title": "CNN (for comparison)", "path": BASE_DIR / "models" / "multiclass_model.h5"},
]

PRETTY_NAMES = {
    "no_tumor": "No tumour",
    "glioma_tumor": "Glioma tumour",
    "meningioma_tumor": "Meningioma tumour",
    "pituitary_tumor": "Pituitary tumour",
}

LOW_CONFIDENCE = 0.60  # below this, the apps warn that the model is unsure


class Classifier:
    """Loads the trained models once and classifies image files with each of them."""

    def __init__(self):
        self.models = {}
        self.class_names = []
        self._preprocess = {}

    def load(self):
        # Imported here so callers can show a window or message while TensorFlow loads.
        import tensorflow as tf
        import train_and_infer_cnn as cnn
        import train_and_infer_vgg19 as vgg

        self.class_names = vgg.CLASS_NAMES
        self._preprocess = {
            "vgg19": vgg.preprocess_single_image_vgg,
            "cnn": cnn.preprocess_single_image,
        }
        for spec in MODEL_SPECS:
            if spec["path"].exists():
                self.models[spec["key"]] = tf.keras.models.load_model(spec["path"], compile=False)
        if not self.models:
            raise FileNotFoundError("No trained model found in models/. Train VGG19 or the CNN first.")

    def classify(self, image_path):
        """Return {model key: list of class probabilities} for one image file."""
        results = {}
        for key, model in self.models.items():
            height, width = model.input_shape[1:3]
            x = self._preprocess[key](image_path, target_size=(width, height))
            results[key] = [float(p) for p in model(x, training=False).numpy()[0]]
        return results
