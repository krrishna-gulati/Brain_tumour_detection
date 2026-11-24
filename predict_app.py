"""Desktop app: choose a brain MRI image and see which tumour type the models predict.

Run it from VS Code (Run and Debug -> "Predict app: choose an MRI image") or with:
    python predict_app.py

VGG19 is the main model; the CNN is shown next to it for comparison. Both use the
same crop + brightness normalisation as training (brain_preprocessing.py).
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from brain_classifier import BASE_DIR, LOW_CONFIDENCE, MODEL_SPECS, PRETTY_NAMES, Classifier
from brain_preprocessing import prepare_image

PREVIEW_SIZE = 300


class ResultPanel(ttk.LabelFrame):
    """One model's answer: the predicted class and a bar per class."""

    BAR_WIDTH = 230

    def __init__(self, parent, title, class_names):
        super().__init__(parent, text=title, padding=10)
        self.class_names = class_names
        self.prediction = ttk.Label(self, text="-", font=("Segoe UI", 15, "bold"))
        self.prediction.grid(row=0, column=0, columnspan=3, sticky="w")
        self.warning = ttk.Label(self, text="", foreground="#b45309")
        self.warning.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self.bars, self.values = [], []
        for i, name in enumerate(class_names):
            ttk.Label(self, text=PRETTY_NAMES.get(name, name), width=18).grid(row=i + 2, column=0, sticky="w")
            bar = tk.Canvas(self, width=self.BAR_WIDTH, height=16, highlightthickness=0, bg="#e5e7eb")
            bar.grid(row=i + 2, column=1, padx=6, pady=2)
            value = ttk.Label(self, text="", width=7, anchor="e")
            value.grid(row=i + 2, column=2, sticky="e")
            self.bars.append(bar)
            self.values.append(value)

    def show(self, probs):
        best = max(range(len(probs)), key=lambda i: probs[i])
        self.prediction.config(text=f"{PRETTY_NAMES.get(self.class_names[best])}  ({probs[best] * 100:.1f}%)")
        self.warning.config(
            text="Low confidence: the model is unsure about this image." if probs[best] < LOW_CONFIDENCE else ""
        )
        for i, (bar, value, p) in enumerate(zip(self.bars, self.values, probs)):
            bar.delete("all")
            colour = "#2563eb" if i == best else "#9ca3af"
            bar.create_rectangle(0, 0, max(1, int(self.BAR_WIDTH * p)), 16, fill=colour, width=0)
            value.config(text=f"{p * 100:.1f}%")


class App:
    def __init__(self, root):
        self.root = root
        self.classifier = Classifier()
        self.events = queue.Queue()
        self.panels = {}
        self._photos = []  # keep references, or Tk drops the images

        root.title("Brain Tumour Detection")
        root.minsize(900, 560)
        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")
        self.choose_button = ttk.Button(top, text="Choose MRI image...", command=self.choose_image, state="disabled")
        self.choose_button.pack(side="left")
        self.status = ttk.Label(top, text="Loading models, please wait...")
        self.status.pack(side="left", padx=12)

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True, pady=12)

        images = ttk.Frame(body)
        images.pack(side="left", anchor="n")
        self.original_view = self._image_slot(images, "Uploaded image", 0)
        self.prepared_view = self._image_slot(images, "What the model sees (cropped, contrast-normalised)", 2)

        self.results = ttk.Frame(body)
        self.results.pack(side="left", fill="both", expand=True, padx=(16, 0), anchor="n")

        ttk.Label(
            outer,
            text=(
                "For learning and demonstration only, not a medical diagnosis. The models only know brain MRI "
                "scans of these four classes; any other picture will still be forced into one of them."
            ),
            foreground="#6b7280",
            wraplength=860,
        ).pack(fill="x")

        threading.Thread(target=self._load_models, daemon=True).start()
        root.after(100, self._poll)

    def _image_slot(self, parent, caption, row):
        ttk.Label(parent, text=caption).grid(row=row, column=0, sticky="w")
        view = ttk.Label(parent, relief="solid", anchor="center")
        view.grid(row=row + 1, column=0, pady=(2, 10))
        return view

    # --- background work (TensorFlow runs off the UI thread so the window stays responsive) ---
    def _load_models(self):
        try:
            self.classifier.load()
            self.events.put(("loaded", None))
        except Exception as e:
            self.events.put(("error", f"Could not load models: {e}"))

    def _classify(self, path):
        try:
            self.events.put(("result", (path, self.classifier.classify(path))))
        except Exception as e:
            self.events.put(("error", f"Could not classify {os.path.basename(path)}: {e}"))

    def _poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "loaded":
                    self._on_loaded()
                elif kind == "result":
                    self._on_result(*data)
                else:
                    self.status.config(text=data, foreground="#b91c1c")
                    self.choose_button.config(state="normal" if self.classifier.models else "disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # --- UI updates ---
    def _on_loaded(self):
        for spec in MODEL_SPECS:
            if spec["key"] in self.classifier.models:
                panel = ResultPanel(self.results, spec["title"], self.classifier.class_names)
                panel.pack(fill="x", pady=(0, 12))
                self.panels[spec["key"]] = panel
        self.choose_button.config(state="normal")
        self.status.config(text="Ready. Choose a brain MRI image (JPG or PNG).", foreground="")

    def choose_image(self):
        start_dir = BASE_DIR / "Dataset_prepared" / "test"
        path = filedialog.askopenfilename(
            title="Choose a brain MRI image",
            initialdir=start_dir if start_dir.exists() else BASE_DIR,
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            self.status.config(text=f"Not a readable image: {e}", foreground="#b91c1c")
            return
        self._photos = [self._show(self.original_view, img.convert("RGB")), self._show(self.prepared_view, prepare_image(img))]
        self.choose_button.config(state="disabled")
        self.status.config(text=f"Classifying {os.path.basename(path)}...", foreground="")
        threading.Thread(target=self._classify, args=(path,), daemon=True).start()

    def _show(self, view, img):
        img = img.copy()
        img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        photo = ImageTk.PhotoImage(img)
        view.config(image=photo)
        return photo

    def _on_result(self, path, results):
        for key, probs in results.items():
            self.panels[key].show(probs)
        self.choose_button.config(state="normal")
        self.status.config(text=f"Result for {os.path.basename(path)}", foreground="")


def main():
    try:  # sharp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
