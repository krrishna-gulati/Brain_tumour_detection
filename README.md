# Brain Tumour Detection from MRI: CNN vs VGG19

Classifies a brain MRI scan as **no tumour, glioma, meningioma or pituitary tumour**. Two models are compared: a small CNN trained from scratch, and VGG19 pre-trained on ImageNet with its last block fine-tuned on MRI scans. A local website lets you upload a scan and see the prediction, with a Grad-CAM heatmap of where VGG19 looked.

> For learning and demonstration only. This is not a medical device and must not be used for diagnosis.

## Results

Scores on 423 held-out test scans that were never used for training:

| Model | Test accuracy | No tumour | Glioma | Meningioma | Pituitary |
|---|---|---|---|---|---|
| CNN (trained from scratch, 30 epochs) | 82.0% | 68.9% | 73.7% | 78.3% | 99.2% |
| **VGG19 (block 5 fine-tuned)** | **90.8%** | **91.1%** | **90.2%** | **83.3%** | **98.4%** |

The per-class figures are recall: the share of each class's scans the model labelled correctly.

The original Training/Testing split was not usable as it stood. Much of the Testing folder was copies of Training images, and the two folders came from different-looking sources. `prepare_data.py` therefore merges them, crops every scan to the brain, normalises brightness, removes exact and near-duplicates, and makes a fresh stratified 70/15/15 split.

## Project structure

| File or folder | Purpose |
|---|---|
| `prepare_data.py` | Builds `Dataset_prepared/` from `Dataset/`: merge, crop, de-duplicate, split |
| `brain_preprocessing.py` | Crop to the brain and normalise brightness (used for training and prediction) |
| `train_and_infer_cnn.py` | CNN: train, evaluate, predict |
| `train_and_infer_vgg19.py` | VGG19: train, evaluate, predict |
| `brain_classifier.py` | Loads the trained models and classifies an image (shared by both apps) |
| `gradcam.py` | Grad-CAM heatmaps for VGG19 (the CNN's Flatten + Dense design gives unreliable maps) |
| `web_app.py`, `templates/` | Local website: upload an image in the browser |
| `predict_app.py` | Desktop app (Tkinter): choose an image and see the prediction |
| `models/` | Trained models, stored with Git LFS |
| `Dataset/` | Original images: `Training/` and `Testing/`, one folder per class |

## Setup

You need Python 3.12 (64-bit) and [Git LFS](https://git-lfs.com/), which stores the trained models. A GPU is not required; everything was trained on a laptop CPU.

```
git lfs install
git clone https://github.com/krrishna-gulati/Brain_tumour_detection.git
cd Brain_tumour_detection
python -m venv .venv
.venv\Scripts\activate          (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

**Website.** Run `python web_app.py`. It loads the models (about 10 seconds) and opens http://127.0.0.1:5000. Drag in an MRI scan to see VGG19's prediction and a Grad-CAM heatmap of the regions that drove it, with the CNN's prediction for comparison. Press Ctrl+C to stop it.

**Desktop app.** Run `python predict_app.py`.

**Rebuild the data split.** Run `python prepare_data.py` once. It creates `Dataset_prepared/`, which is needed for training, evaluation and the website's "Try a random test scan" button. The split uses a fixed seed, so it matches the one the models were trained on.

**Train and evaluate:**

```
python train_and_infer_cnn.py --mode train --epochs 30 --patience 30   # about 45 min on a laptop CPU
python train_and_infer_vgg19.py --mode train      # about 1.5 h on a laptop CPU
python train_and_infer_vgg19.py --mode evaluate   # accuracy, per-class report, confusion matrix
python train_and_infer_vgg19.py --mode predict --image_path <image file>
```

Training saves the best epoch (by validation accuracy) and stops early after 4 epochs without improvement.

## Dataset

`Dataset/` holds 3,264 JPEG images in `Training/` and `Testing/`, one subfolder per class. The folder layout and image counts match the public Kaggle dataset "Brain Tumor Classification (MRI)". Check that dataset's licence before reusing the images.

## Limitations

- The dataset has no patient IDs, so slices from one patient may appear in both training and test data. The test score may be slightly optimistic.
- All images come from one dataset; scans from other hospitals or scanners may score lower.
- The models see single 2D slices and know only these four classes. Any other image is still forced into one of them.
