"""Grad-CAM heatmaps: which regions of a scan pushed a model towards its answer.

Grad-CAM (Selvaraju et al., 2017) takes the last convolution layer's feature maps, weights
each map by how much it raises the predicted class's score (the average gradient), and adds
them up. The result is a coarse map, e.g. 14x14 for VGG19, that is stretched over the image.

Both models here are plain chains of layers (VGG19's base is a nested model), so the forward
pass is replayed layer by layer: up to the last conv layer, then the rest under a GradientTape.

Heatmaps are only shown for VGG19. Our CNN ends in Flatten + Dense, so every position has its
own weights; its Grad-CAM maps light up the black background, and the HiResCAM variant gives
only a few isolated pixels. Neither is a trustworthy picture of where the CNN looked.
"""
import numpy as np
import tensorflow as tf
from PIL import Image


def leaf_layers(model):
    """All layers in order, with nested models (VGG19's base) expanded and input layers dropped."""
    out = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if isinstance(layer, tf.keras.Model):
            out.extend(leaf_layers(layer))
        else:
            out.append(layer)
    return out


def last_conv_index(layers):
    last = layers[-1]
    if not (isinstance(last, tf.keras.layers.Dense) and getattr(last.activation, "__name__", "") == "softmax"):
        raise ValueError("Grad-CAM here expects the model to end in a Dense layer with softmax.")
    return max(i for i, layer in enumerate(layers) if isinstance(layer, tf.keras.layers.Conv2D))


def predict_with_gradcam(layers, conv_index, x, class_index=None):
    """Return (class probabilities, Grad-CAM map in 0..1) for one preprocessed image batch x."""
    a = tf.convert_to_tensor(x)
    for layer in layers[: conv_index + 1]:
        a = layer(a, training=False)  # training=False also keeps the augmentation layers off
    *middle, last = layers[conv_index + 1:]
    with tf.GradientTape() as tape:
        tape.watch(a)
        h = a
        for layer in middle:
            h = layer(h, training=False)
        # Use the class score before softmax, as in the Grad-CAM paper: a softmax probability
        # near 100% barely moves, so its gradients can vanish for very confident predictions.
        logits = tf.matmul(h, last.kernel) + last.bias
        probs = tf.nn.softmax(logits)[0]
        if class_index is None:
            class_index = int(tf.argmax(probs))
        score = logits[0, class_index]
    grads = tape.gradient(score, a)[0]                 # (H, W, channels)
    weights = tf.reduce_mean(grads, axis=(0, 1))       # importance of each feature map
    cam = tf.nn.relu(tf.reduce_sum(a[0] * weights, axis=-1)).numpy()
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
