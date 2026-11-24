"""Shared image preprocessing: crop an MRI scan to the brain and even out brightness.

Used by prepare_data.py (to build Dataset_prepared/) and by the predict mode of
both training scripts, so a single new scan gets exactly the same treatment as
the training images.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage


def crop_to_brain(img, margin=0.02, min_area_fraction=0.1):
    """Crop away the black background around the brain.

    Finds the largest bright region (the head/brain) and crops to its bounding
    box plus a small margin. If nothing sensible is found, returns the image
    unchanged.
    """
    gray = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(2)), dtype=np.uint8)

    # Threshold relative to the image's own brightness, so dark and bright
    # scans are handled the same way.
    threshold = max(10, 0.15 * np.percentile(gray, 99))
    mask = ndimage.binary_opening(gray > threshold, iterations=2)  # drop specks/text

    labels, n = ndimage.label(mask)
    if n == 0:
        return img
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)

    ys = np.where(largest.any(axis=1))[0]
    xs = np.where(largest.any(axis=0))[0]
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1

    h, w = gray.shape
    if (y1 - y0) * (x1 - x0) < min_area_fraction * h * w:
        return img  # region too small, probably not the brain

    my, mx = int(margin * h), int(margin * w)
    return img.crop((max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)))


def prepare_image(img):
    """Crop to the brain, then stretch contrast so every scan uses the full
    brightness range (removes the dark-vs-bright difference between sources)."""
    img = crop_to_brain(img.convert("RGB"))
    return ImageOps.autocontrast(img, cutoff=1)
