# Brain Tumour Detection from MRI: CNN, VGG19, ResNet50, EfficientNetB0 and their combination

Classifies a brain MRI scan as **no tumour, glioma, meningioma or pituitary tumour**. Four models are compared: a small CNN trained from scratch, and three ImageNet networks (VGG19, ResNet50, EfficientNetB0) whose last block is fine-tuned on MRI scans. Averaging the four models' predictions gives the best result. A local website lets you upload a scan and see the combined answer, each model's answer, and Grad-CAM heatmaps of where the models looked.

> For learning and demonstration only. This is not a medical device and must not be used for diagnosis.

## Results

Scores on 402 test scans from 30 patients whose scans were never used for training (the split keeps all slices of a patient together):

| Model | Test accuracy | No tumour | Glioma | Meningioma | Pituitary | Model size |
|---|---|---|---|---|---|---|
| CNN (trained from scratch, 30 epochs) | 77.4% | 83.7% | 78.6% | 64.7% | 86.3% | 148 MB |
| EfficientNetB0 (fine-tuned) | 87.1% | 79.1% | 77.8% | 90.5% | 96.6% | 29 MB |
| ResNet50 (fine-tuned) | 90.0% | **95.3%** | 76.2% | **94.8%** | 98.3% | 211 MB |
| VGG19 (fine-tuned) | 91.0% | 86.0% | **85.7%** | 92.2% | 97.4% | 150 MB |
| **Combined (average of all four)** | **91.5%** | 93.0% | 81.0% | **94.8%** | **99.1%** | |

The per-class figures are recall: the share of each class's scans the model labelled correctly. The combined model averages the four models' class probabilities with equal weights. The weights were fixed in advance and not tuned on the test set.

The original Training/Testing split was not usable as it stood. Much of the Testing folder was copies of Training images, and the two folders came from different-looking sources. `prepare_data.py` therefore merges them, crops every scan to the brain, normalises brightness, removes exact and near-duplicates, and makes a fresh stratified split (about 70/15/15).

Most tumour images turned out to be copies of the [Figshare brain tumour dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) (Cheng et al., 2015), which records each slice's patient. `evaluate_external.py` matched them and saved the patient IDs in `figshare_patient_ids.csv`, and the split now keeps each patient on one side. An earlier random split had put other slices of 164 test patients into training. That flattered the CNN (82.0% then, 77.4% now) and the combined model (93.6% then, 91.5% now), while the three pre-trained networks moved by at most 1.1 points. The same check found 7 slices labelled pituitary here but meningioma in Figshare; they now carry Figshare's label.

## Project structure

| File or folder | Purpose |
|---|---|
| `prepare_data.py` | Builds `Dataset_prepared/` from `Dataset/`: merge, crop, de-duplicate, fix labels, split by patient |
| `evaluate_external.py` | Tests the models on the Figshare dataset, matches its slices to our images and saves their patient IDs |
| `figshare_patient_ids.csv` | Patient ID of each image that is a Figshare copy (2,345 images), used by `prepare_data.py` |
| `brain_preprocessing.py` | Crop to the brain and normalise brightness (used for training and prediction) |
| `train_and_infer_cnn.py` | CNN: train, evaluate, predict |
| `train_and_infer_vgg19.py` | VGG19: train, evaluate, predict |
| `train_and_infer_transfer.py` | ResNet50 and EfficientNetB0 (`--arch`): train, evaluate, predict |
| `evaluate_ensemble.py` | Scores every model and the combined model on the test (or validation) split |
| `brain_classifier.py` | Loads the trained models, classifies an image, combines the answers |
| `gradcam.py` | Grad-CAM heatmaps for the three pre-trained networks (the CNN's Flatten + Dense design gives unreliable maps) |
| `web_app.py`, `templates/` | Local website: upload an image in the browser |
| `predict_app.py` | Desktop app (Tkinter) with VGG19 and the CNN |
| `models/` | Trained models, stored with Git LFS (`.h5` for VGG19 and the CNN, `.keras` for ResNet50 and EfficientNetB0) |
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

**Website.** Run `python web_app.py`. It loads and prepares the four models (about 40 seconds) and opens http://127.0.0.1:5000. Drag in an MRI scan to see the combined final answer, how many models agree, and each model's own answer with a Grad-CAM heatmap. Each scan takes about 2 seconds. Press Ctrl+C to stop it.

**Desktop app.** Run `python predict_app.py`.

**Rebuild the data split.** Run `python prepare_data.py` once. It creates `Dataset_prepared/`, which is needed for training, evaluation and the website's "Try a random test scan" button. The split uses `figshare_patient_ids.csv` and a fixed seed, so it matches the one the models were trained on. `--split_by random` gives a split that ignores patients, for comparison.

**Train and evaluate** (times on a laptop CPU):

```
python train_and_infer_cnn.py --mode train --epochs 30 --patience 30   # about 50 min
python train_and_infer_vgg19.py --mode train                           # 1.5 to 3 h
python train_and_infer_transfer.py --arch resnet50 --mode train        # about 1 h
python train_and_infer_transfer.py --arch efficientnetb0 --mode train  # about 20 min
python train_and_infer_vgg19.py --mode evaluate      # accuracy, per-class report, confusion matrix
python evaluate_ensemble.py                          # all models and the combined model
python train_and_infer_vgg19.py --mode predict --image_path <image file>
```

**Outside test and patient check.** Download the four `brainTumorDataPublic_*.zip` files from the [Figshare dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) into one folder, then run `python evaluate_external.py --data_dir <that folder>` (about 35 minutes). It rewrites `figshare_patient_ids.csv` and scores every model on the Figshare slices and on the test split, grouped by whether the patient was seen in training.

Training saves the best epoch (by validation accuracy) and stops early after 4 epochs without improvement.

## Dataset

`Dataset/` holds 3,264 JPEG images in `Training/` and `Testing/`, one subfolder per class. The folder layout and image counts match the public Kaggle dataset "Brain Tumor Classification (MRI)". About 2,300 of its tumour images are copies of the Figshare brain tumour dataset (Cheng et al., 2015). Check both datasets' licences before reusing the images.

## Limitations

- Only the images that are Figshare copies have patient IDs. The 530 healthy and other-source images count as one patient each, so their patients could still appear on both sides of the split.
- About four in five images come from two hospitals in China. On 30 test scans from other sources the combined model scored only 73%, so scans from other hospitals or scanners may score lower.
- The models see single 2D slices and know only these four classes. Any other image is still forced into one of them.
