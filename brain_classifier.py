"""Shared prediction code for the desktop app (predict_app.py), the website (web_app.py)
and evaluate_ensemble.py.

Loads the trained models once and classifies an image file with each of them, using the
same crop + brightness normalisation as training (via the training scripts' preprocessors).
The combined model averages the class probabilities of all loaded models, each counting equally.
"""
import os
from functools import partial
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's info messages

BASE_DIR = Path(__file__).resolve().parent

# Display order on the website. "gradcam_layer": the layer Grad-CAM reads (None = no reliable
# heatmap, see gradcam.py). "test_accuracy": measured on the 423 test scans with
# evaluate_ensemble.py; update it if a model is retrained.
MODEL_SPECS = [
    {"key": "vgg19", "name": "VGG19", "title": "VGG19 (main model)",
     "path": BASE_DIR / "models" / "multiclass_vgg19_model.h5",
     "gradcam_layer": "block5_conv4", "test_accuracy": 0.908},
    {"key": "resnet50", "name": "ResNet50", "title": "ResNet50",
     "path": BASE_DIR / "models" / "multiclass_resnet50_model.keras",
     "gradcam_layer": "conv5_block3_out", "test_accuracy": 0.901},
    {"key": "efficientnetb0", "name": "EfficientNetB0", "title": "EfficientNetB0",
     "path": BASE_DIR / "models" / "multiclass_efficientnetb0_model.keras",
     "gradcam_layer": "top_activation", "test_accuracy": 0.882},
    {"key": "cnn", "name": "CNN", "title": "CNN (for comparison)",
     "path": BASE_DIR / "models" / "multiclass_model.h5",
     "gradcam_layer": None, "test_accuracy": 0.820},
]
SPECS_BY_KEY = {spec["key"]: spec for spec in MODEL_SPECS}

PRETTY_NAMES = {
    "no_tumor": "No tumour",
    "glioma_tumor": "Glioma tumour",
    "meningioma_tumor": "Meningioma tumour",
    "pituitary_tumor": "Pituitary tumour",
}

LOW_CONFIDENCE = 0.60  # below this, the apps warn that the model is unsure


# Combined model with all four models above, measured with evaluate_ensemble.py (396 of 423)
COMBINED_TEST_ACCURACY = 0.936


def combine(prob_lists):
    """Combined model: the average of the models' class probabilities (equal weights)."""
    return np.mean(np.asarray(prob_lists, dtype=float), axis=0).tolist()


class Classifier:
    """Loads the trained models once and classifies image files with each of them."""

    def __init__(self, keys=None):
        self.keys = keys  # which models to load (default: every model in MODEL_SPECS)
        self.models = {}
        self.class_names = []
        self._preprocess = {}
        self._predict_fns = {}    # model key -> compiled prediction function
        self._gradcam_fns = {}    # model key -> compiled Grad-CAM function

    def load(self):
        # Imported here so callers can show a window or message while TensorFlow loads.
        import tensorflow as tf
        import train_and_infer_cnn as cnn
        import train_and_infer_transfer as transfer
        import train_and_infer_vgg19 as vgg

        self.class_names = vgg.CLASS_NAMES
        self._preprocess = {
            "vgg19": vgg.preprocess_single_image_vgg,
            "cnn": cnn.preprocess_single_image,
            **{
                arch: partial(transfer.preprocess_single_image, preprocess=transfer.ARCHS[arch]["preprocess"])
                for arch in transfer.ARCHS
            },
        }
        for spec in MODEL_SPECS:
            if (self.keys is None or spec["key"] in self.keys) and spec["path"].exists():
                model = tf.keras.models.load_model(spec["path"], compile=False)
                self.models[spec["key"]] = model
                # Compiled once; running the big models layer by layer in Python is slow
                self._predict_fns[spec["key"]] = tf.function(
                    lambda x, m=model: m(x, training=False), reduce_retracing=True
                )
        if not self.models:
            raise FileNotFoundError("No trained model found in models/. Train a model first.")

    def warm_up(self, heatmaps=False):
        """Run each compiled function once on a blank image, so the first real scan is fast."""
        for key, model in self.models.items():
            x = np.zeros((1, *model.input_shape[1:]), dtype="float32")
            self._predict(key, x)
            if heatmaps and SPECS_BY_KEY[key]["gradcam_layer"]:
                self._explain(key, x)

    def _input(self, key, image_path):
        height, width = self.models[key].input_shape[1:3]
        return self._preprocess[key](image_path, target_size=(width, height))

    def _predict(self, key, x):
        return [float(p) for p in self._predict_fns[key](np.asarray(x, dtype="float32")).numpy()[0]]

    def _explain(self, key, x):
        import gradcam

        if key not in self._gradcam_fns:
            parts = gradcam.gradcam_parts(self.models[key], SPECS_BY_KEY[key]["gradcam_layer"])
            self._gradcam_fns[key] = gradcam.make_gradcam_function(parts)
        return gradcam.predict_with_gradcam(self._gradcam_fns[key], x)

    def classify(self, image_path):
        """Return {model key: list of class probabilities} for one image file."""
        return {key: self._predict(key, self._input(key, image_path)) for key in self.models}

    def classify_and_explain(self, image_path):
        """Return {model key: {"probs": [...], "cam": Grad-CAM map, or None}} for one image file."""
        results = {}
        for key in self.models:
            x = self._input(key, image_path)
            if SPECS_BY_KEY[key]["gradcam_layer"]:
                probs, cam = self._explain(key, x)
            else:
                probs, cam = self._predict(key, x), None
            results[key] = {"probs": probs, "cam": cam}
        return results
