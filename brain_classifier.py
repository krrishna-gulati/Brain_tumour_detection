"""Shared prediction code for the desktop app (predict_app.py) and the website (web_app.py).

Loads the trained models once and classifies an image file with each of them, using the
same crop + brightness normalisation as training (via the training scripts' preprocessors).
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's info messages

BASE_DIR = Path(__file__).resolve().parent

# Main model first. Paths are the files written by the training scripts.
# "heatmap": whether Grad-CAM gives a trustworthy map for this model (see gradcam.py).
MODEL_SPECS = [
    {"key": "vgg19", "name": "VGG19", "title": "VGG19 (main model)", "heatmap": True,
     "path": BASE_DIR / "models" / "multiclass_vgg19_model.h5"},
    {"key": "cnn", "name": "CNN", "title": "CNN (for comparison)", "heatmap": False,
     "path": BASE_DIR / "models" / "multiclass_model.h5"},
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
        self._gradcam_layers = {}  # model key -> (layers in order, index of last conv layer)

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

    def _input(self, key, image_path):
        height, width = self.models[key].input_shape[1:3]
        return self._preprocess[key](image_path, target_size=(width, height))

    def classify(self, image_path):
        """Return {model key: list of class probabilities} for one image file."""
        return {
            key: [float(p) for p in model(self._input(key, image_path), training=False).numpy()[0]]
            for key, model in self.models.items()
        }

    def classify_and_explain(self, image_path):
        """Return {model key: {"probs": [...], "cam": Grad-CAM map, or None}} for one image file.

        A map is made only for models whose spec says "heatmap": True.
        """
        import gradcam

        heatmap_keys = {spec["key"] for spec in MODEL_SPECS if spec["heatmap"]}
        results = {}
        for key, model in self.models.items():
            x = self._input(key, image_path)
            if key in heatmap_keys:
                if key not in self._gradcam_layers:
                    layers = gradcam.leaf_layers(model)
                    self._gradcam_layers[key] = (layers, gradcam.last_conv_index(layers))
                layers, conv_index = self._gradcam_layers[key]
                probs, cam = gradcam.predict_with_gradcam(layers, conv_index, x)
            else:
                probs, cam = [float(p) for p in model(x, training=False).numpy()[0]], None
            results[key] = {"probs": probs, "cam": cam}
        return results
