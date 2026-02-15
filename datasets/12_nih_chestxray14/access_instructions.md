# NIH ChestX-ray14 — Access Instructions

## Overview

NIH ChestX-ray14 (also known as ChestX-ray8 in its original CVPR 2017 publication) is a large-scale chest radiograph dataset released by the NIH Clinical Center. It contains 112,120 frontal-view X-ray images from 30,805 unique patients, each annotated with text-mined labels for 14 common thoracic diseases. The labels were extracted from associated radiological reports using natural language processing (NLP) techniques, with an expected accuracy of over 90%. This dataset has been widely used as a benchmark for automated chest X-ray interpretation and deep learning research.

## Access Level

**Open** -- The dataset is publicly available with no restrictions on use. No account or registration is required for download. Attribution is required when publishing results.

## Prerequisites

- **A modern web browser** for accessing the download links.
- **Sufficient storage**: The full dataset is approximately 42 GB.
- **Tools for handling PNG images** and CSV files for the labels.
- **Optional**: A Kaggle account (free) for downloading via Kaggle, or a Google Cloud account for accessing via Google Cloud Healthcare API.

## Step-by-Step Access

### Step 1: Access the NIH Clinical Center Box Repository

Navigate to the primary download site:
[https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)

This Box folder contains the image files, label CSV, and README documentation provided by the NIH Clinical Center.

### Step 2: Download the Image Archives

The images are split into multiple zip archives for manageable download sizes. Download all image archive files (`images_001.tar.gz` through `images_012.tar.gz`). Each archive contains a batch of PNG images.

### Step 3: Download the Labels and Metadata

Download the following key files from the same Box folder:

- `Data_Entry_2017_v2020.csv` -- The main label file mapping each image to its 14 disease labels.
- `BBox_List_2017.csv` -- Bounding box annotations for a subset of images (disease localization).
- `README_ChestXray.pdf` -- Official documentation from NIH.

### Step 4: Extract and Organize

Extract the tar.gz archives:

```bash
tar -xzf images_001.tar.gz
tar -xzf images_002.tar.gz
# ... repeat for all archives
```

All images will be extracted as PNG files named by their image index (e.g., `00000001_000.png`).

### Step 5: Alternative Download via Kaggle (Optional)

The dataset is also available on Kaggle:
[https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)

1. Create a free Kaggle account if you do not have one.
2. Navigate to the dataset page.
3. Click **Download** or use the Kaggle CLI:

```bash
kaggle datasets download -d nih-chest-xrays/data
```

### Step 6: Alternative Download via Google Cloud (Optional)

The dataset is available through the Google Cloud Healthcare API:
[https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)

This also provides access to expert-labeled subsets with additional annotations.

### Step 7: Access Expert Labels (Optional)

For higher-quality expert annotations on a subset of images, access is available through Google Cloud Healthcare API. Two expert label sets exist:

1. **Set 1**: Four findings -- airspace opacity, pneumothorax, nodule/mass, and fracture.
2. **Set 2**: All 14 original findings plus a normal/abnormal label.

A form must be completed through Google Cloud to access these expert labels.

## Data Format

- **Images**: PNG format, frontal-view chest X-rays.
- **Labels**: CSV files with multi-label annotations per image.
- **Bounding boxes**: CSV file with localization annotations for a subset.
- **Archives**: `.tar.gz` compressed files for download.

## Key Tables / Files

| File | Description |
|------|-------------|
| `images_001.tar.gz` through `images_012.tar.gz` | Image archives containing 112,120 frontal-view chest X-ray PNG files |
| `Data_Entry_2017_v2020.csv` | Main label file: Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, Patient Gender, View Position, Original Image Size |
| `BBox_List_2017.csv` | Bounding box annotations for disease localization on a subset of images |
| `train_val_list.txt` / `test_list.txt` | Official train/validation and test split definitions |
| `README_ChestXray.pdf` | Official NIH documentation |

## Disease Labels (14 Classes)

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax. An image may also be labeled "No Finding."

## Important Restrictions

- **No restrictions on use** of the NIH chest X-ray images.
- **Attribution required**: When publishing, you must:
  1. Provide a link to the NIH download site.
  2. Cite the original paper: Wang et al., "ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," IEEE CVPR, 2017.
  3. Acknowledge the NIH Clinical Center as the data provider.
- Labels are NLP-extracted (not manually verified for all images) -- expected accuracy >90%. Exercise caution when using labels for clinical validation.

## Useful Links

- [NIH Clinical Center Box Download](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Kaggle Mirror](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [Hugging Face Mirror](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)
- [Google Cloud Healthcare API](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)
- [Original Paper (CVPR 2017)](https://www.researchgate.net/publication/323597745)
- [CheXNet (Stanford) - Model Trained on This Dataset](https://stanfordmlgroup.github.io/projects/chexnet/)
