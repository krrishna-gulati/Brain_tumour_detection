"""Grad-CAM heatmaps: which regions of a scan pushed a model towards its answer.

Grad-CAM (Selvaraju et al., 2017) takes the feature maps of a late convolution layer, weights
each map by how much it raises the predicted class's score (the average gradient), and adds
them up. The result is a coarse map (14x14 for VGG19, 7x7 for ResNet50 and EfficientNetB0)
that is stretched over the image.

Works for the transfer-learning models, which all have the same shape:
augmentation layers -> pre-trained base (a nested model) -> pooling -> Dense -> Dropout -> Dense(softmax).
A sub-model of the base returns both the chosen layer's activations and the base output,
so this also works for bases with skip connections (ResNet50, EfficientNetB0).

The CNN gets no heatmap: it ends in Flatten + Dense, so every position has its own weights.
Its Grad-CAM maps lit up the black background and the HiResCAM variant gave only a few
isolated pixels; neither is a trustworthy picture of where the CNN looked.
"""
import numpy as np
import tensorflow as tf
from PIL import Image


def gradcam_parts(model, layer_name):
    """Split a model into (layers before the base, base sub-model, head layers after the base)."""
    base_index = next(i for i, layer in enumerate(model.layers) if isinstance(layer, tf.keras.Model))
    base = model.layers[base_index]
    before = [l for l in model.layers[:base_index] if not isinstance(l, tf.keras.layers.InputLayer)]
    features = tf.keras.Model(base.inputs, [base.get_layer(layer_name).output, base.output])
    head = model.layers[base_index + 1:]
    last = head[-1]
    if not (isinstance(last, tf.keras.layers.Dense) and getattr(last.activation, "__name__", "") == "softmax"):
        raise ValueError("Grad-CAM here expects the model to end in a Dense layer with softmax.")
    return before, features, head


def make_gradcam_function(parts):
    """Compile the Grad-CAM computation into a TensorFlow graph once per model.

    Run step by step in Python, ResNet50 and EfficientNetB0 (hundreds of layers) took
    seconds per scan; as a compiled graph it is a fraction of a second.
    """
    before, features, head = parts
    *middle, last = head

    @tf.function(reduce_retracing=True)
    def run(x):
        a = x
        for layer in before:
            a = layer(a, training=False)  # augmentation layers: off when not training
        with tf.GradientTape() as tape:
            conv, h = features(a, training=False)
            for layer in middle:
                h = layer(h, training=False)
            # Use the class score before softmax, as in the Grad-CAM paper: a probability
            # near 100% barely moves, so its gradients can vanish for very confident predictions.
            logits = tf.matmul(h, last.kernel) + last.bias
            score = tf.gather(logits[0], tf.argmax(logits[0]))
        grads = tape.gradient(score, conv)[0]              # (H, W, channels)
        weights = tf.reduce_mean(grads, axis=(0, 1))       # importance of each feature map
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1))
        return tf.nn.softmax(logits)[0], cam

    return run


def predict_with_gradcam(gradcam_function, x):
    """Return (class probabilities, Grad-CAM map in 0..1) for one preprocessed image batch x."""
    probs, cam = gradcam_function(tf.convert_to_tensor(x, dtype=tf.float32))
    cam = cam.numpy()
    peak = cam.max()
    cam = cam / peak if peak > 0 else cam
    return probs.numpy().astype(float).tolist(), cam


def _jet(v):
    """Blue -> cyan -> yellow -> red colour map for values in 0..1."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(img, cam, strength=0.6):
    """Colour the image by the Grad-CAM map; low-influence areas stay as the plain scan."""
    img = img.convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BICUBIC)
    v = np.asarray(cam_img, dtype=np.float32) / 255.0
    alpha = (strength * v)[..., None]
    base = np.asarray(img, dtype=np.float32) / 255.0
    out = base * (1 - alpha) + _jet(v) * alpha
    return Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255))
