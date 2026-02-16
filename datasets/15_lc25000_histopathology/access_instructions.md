# LC25000 — Lung and Colon Cancer Histopathological Image Dataset — Access Instructions

## Overview

LC25000 is a histopathological image dataset containing 25,000 color images across 5 tissue classes, created by Andrew A. Borkowski and colleagues at James A. Haley Veterans' Hospital in Tampa, Florida. The dataset was designed to address the need for machine-learning-ready image datasets in cancer pathology. All images are de-identified, HIPAA-compliant, and validated. The five classes cover both lung and colon tissue: lung adenocarcinoma, lung squamous cell carcinoma, benign lung tissue, colon adenocarcinoma, and benign colonic tissue, with 5,000 images per class.

## Access Level

**Open** -- The dataset is freely available for download with no registration required on most platforms. Kaggle requires a free account. The images are de-identified and HIPAA-compliant, and the authors have made them freely available for AI research.

## Prerequisites

- **A Kaggle account** (free) if downloading from Kaggle, or no account needed for GitHub/Academic Torrents.
- **Sufficient storage**: The dataset is approximately 1.85 GB (compressed zip file).
- **Image viewing/processing tools**: Any software capable of handling JPEG images (e.g., Python with PIL/Pillow, OpenCV, or standard image viewers).

## Step-by-Step Access

### Step 1: Choose a Download Source

The dataset is available from multiple platforms:

1. **Kaggle** (most popular): [https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
2. **GitHub**: [https://github.com/tampapath/lung_colon_image_set](https://github.com/tampapath/lung_colon_image_set)
3. **Academic Torrents**: [https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af)
4. **Hugging Face**: [https://huggingface.co/datasets/1aurent/LC25000](https://huggingface.co/datasets/1aurent/LC25000)

### Step 2: Download from Kaggle (Recommended)

#### Option A: Via Web Browser

1. Navigate to the Kaggle dataset page.
2. Sign in to your Kaggle account (or create a free one).
3. Click the **Download** button to download the full dataset as a zip file (~1.85 GB).

#### Option B: Via Kaggle CLI

```bash
# Install the Kaggle CLI if not already installed
pip install kaggle

# Configure Kaggle API credentials (~/.kaggle/kaggle.json)
# Download the dataset
kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images
```

### Step 3: Download from GitHub (Alternative)

```bash
git clone https://github.com/tampapath/lung_colon_image_set.git
```

Note: Due to the large number of files, this may take considerable time.

### Step 4: Extract the Dataset

If downloaded as a zip file:

```bash
unzip lung-and-colon-cancer-histopathological-images.zip
```

### Step 5: Verify the Directory Structure

After extraction, the dataset should be organized as follows:

```
lung_colon_image_set/
  colon_image_sets/
    colon_aca/      (5,000 images - colon adenocarcinoma)
    colon_n/        (5,000 images - benign colonic tissue)
  lung_image_sets/
    lung_aca/       (5,000 images - lung adenocarcinoma)
    lung_scc/       (5,000 images - lung squamous cell carcinoma)
    lung_n/         (5,000 images - benign lung tissue)
```

Verify that each folder contains exactly 5,000 images (25,000 total).

## Data Format

- **Image format**: JPEG color images.
- **Image resolution**: 768 x 768 pixels per image.
- **Color space**: RGB.
- **Total images**: 25,000 (5,000 per class).
- **Total size**: Approximately 1.85 GB compressed.

## Key Tables / Files

| Directory / File | Description | Count |
|-----------------|-------------|-------|
| `colon_aca/` | Colon adenocarcinoma histopathology images | 5,000 |
| `colon_n/` | Benign colonic tissue histopathology images | 5,000 |
| `lung_aca/` | Lung adenocarcinoma histopathology images | 5,000 |
| `lung_scc/` | Lung squamous cell carcinoma histopathology images | 5,000 |
| `lung_n/` | Benign lung tissue histopathology images | 5,000 |

## Image Classes (5 Classes)

| Class | Abbreviation | Tissue Type | Pathology |
|-------|-------------|-------------|-----------|
| Colon Adenocarcinoma | `colon_aca` | Colon | Malignant |
| Benign Colonic Tissue | `colon_n` | Colon | Benign |
| Lung Adenocarcinoma | `lung_aca` | Lung | Malignant |
| Lung Squamous Cell Carcinoma | `lung_scc` | Lung | Malignant |
| Benign Lung Tissue | `lung_n` | Lung | Benign |

## Important Restrictions

- All images are **de-identified and HIPAA-compliant**.
- The dataset is **freely available** for AI researchers, but check the specific terms on the platform you download from.
- When publishing results, cite the original paper: Borkowski, A.A., Bui, M.M., Thomas, L.B., Wilson, C.P., DeLand, L.A., and Mastorides, S.M. "Lung and Colon Cancer Histopathological Image Dataset (LC25000)." arXiv:1912.12142, 2019.
- The images were augmented from a smaller original set -- be aware of potential data leakage if creating custom train/test splits.

## Useful Links

- [Kaggle Dataset Page](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- [GitHub Repository](https://github.com/tampapath/lung_colon_image_set)
- [Hugging Face Dataset](https://huggingface.co/datasets/1aurent/LC25000)
- [Academic Torrents](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af)
- [Original Paper (arXiv:1912.12142)](https://arxiv.org/abs/1912.12142)
