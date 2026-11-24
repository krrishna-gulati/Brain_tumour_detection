"""Local website: upload a brain MRI image in the browser and see the predicted tumour type.

Run it from VS Code (Run and Debug -> "Website: upload an MRI image") or with:
    python web_app.py
It loads the models (about 10 seconds), then opens http://127.0.0.1:5000 in your browser.
Press Ctrl+C in the terminal to stop it.

The site only listens on this computer (127.0.0.1), so nobody else on the network can reach it.
"""
import argparse
import base64
import csv
import io
import os
import random
import tempfile
import threading
import webbrowser
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from brain_classifier import BASE_DIR, LOW_CONFIDENCE, MODEL_SPECS, PRETTY_NAMES, Classifier
from brain_preprocessing import prepare_image

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
PREVIEW_SIZE = 360

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # reject uploads over 15 MB

classifier = Classifier()
classify_lock = threading.Lock()  # one prediction at a time; the models are shared


def preview_data_url(img):
    """Small JPEG of an image, as a data: URL the page can show directly."""
    img = img.convert("RGB")
    img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def describe(image_path, display_name):
    """Classify one image file and package everything the page shows."""
    img = Image.open(image_path)
    img.load()
    with classify_lock:
        results = classifier.classify(image_path)

    models = []
    for spec in MODEL_SPECS:
        probs = results.get(spec["key"])
        if probs is None:
            continue
        best = max(range(len(probs)), key=probs.__getitem__)
        models.append(
            {
                "key": spec["key"],
                "title": spec["title"],
                "prediction": PRETTY_NAMES.get(classifier.class_names[best]),
                "confidence": probs[best],
                "low_confidence": probs[best] < LOW_CONFIDENCE,
                "probabilities": [
                    {"name": PRETTY_NAMES.get(name, name), "p": p}
                    for name, p in zip(classifier.class_names, probs)
                ],
            }
        )
    return {
        "filename": display_name,
        "original": preview_data_url(img),
        "prepared": preview_data_url(prepare_image(img)),
        "models": models,
    }


@lru_cache(maxsize=1)
def test_samples():
    """Original files of the held-out test split (never used for training)."""
    manifest = BASE_DIR / "Dataset_prepared" / "split_manifest.csv"
    if not manifest.exists():
        return []
    with open(manifest, encoding="utf-8") as f:
        return [
            row for row in csv.DictReader(f)
            if row["split"] == "test" and (BASE_DIR / row["original_path"]).exists()
        ]


@app.get("/")
def index():
    return render_template("index.html", has_samples=bool(test_samples()))


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="No image received."), 400
    data = upload.read()
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return jsonify(error="That file is not a readable image. Please use a JPG or PNG scan."), 400

    suffix = Path(upload.filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix if suffix in ALLOWED_SUFFIXES else ".img", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return jsonify(describe(tmp.name, upload.filename))
    finally:
        os.unlink(tmp.name)


@app.get("/sample")
def sample():
    rows = test_samples()
    if not rows:
        return jsonify(error="No test split found. Run prepare_data.py first."), 404
    row = random.choice(rows)
    path = BASE_DIR / row["original_path"]
    out = describe(path, path.name)
    out["true_label"] = PRETTY_NAMES.get(row["class"], row["class"])
    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That file is too large (limit 15 MB)."), 413


def main():
    p = argparse.ArgumentParser(description="Brain tumour detection website (runs locally)")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    args = p.parse_args()

    print("Loading models, about 10 seconds...")
    classifier.load()
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nReady. Open {url} in your browser. Press Ctrl+C here to stop.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
