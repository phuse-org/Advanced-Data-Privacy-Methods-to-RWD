# CheXpert — Access Instructions

## Overview

CheXpert (Chest eXpert) is a large chest radiograph dataset created by the Stanford Machine Learning Group and Stanford Center for Artificial Intelligence in Medicine and Imaging (AIMI). It contains 224,316 chest radiographs from 65,240 patients who underwent radiographic examinations at Stanford Health Care between October 2002 and July 2017, in both inpatient and outpatient settings. The dataset features automated labels for 14 observations extracted from radiology reports, with a novel approach to capturing diagnostic uncertainty through explicit uncertainty labels. An extended version, CheXpert Plus, is also available with DICOM images and free-text radiology reports.

## Access Level

**Controlled** -- Access requires free registration on the Stanford AIMI Shared Datasets platform and acceptance of the Stanford University Dataset Research Use Agreement. No fee is charged. The dataset is restricted to non-commercial research use only.

## Prerequisites

- **A Stanford AIMI account**: Free registration at the Stanford AIMI Shared Datasets portal.
- **Agreement to the Research Use Agreement**: Must be reviewed and accepted before data access is granted.
- **Sufficient storage**: The full dataset is approximately 439 GB (original CheXpert) or varies for the downsampled version (~11 GB).
- **Tools for handling JPEG/PNG images** and CSV label files.
- **Optional**: For CheXpert Plus, tools capable of reading DICOM files.

## Step-by-Step Access

### Step 1: Visit the Stanford AIMI Dataset Page

Navigate to the CheXpert dataset page on the Stanford AIMI portal:
[https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7890a](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7890a)

Alternatively, visit the AIMI datasets overview:
[https://aimi.stanford.edu/datasets/chexpert-chest-x-rays](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)

### Step 2: Register for a Stanford AIMI Account

If you do not already have an account:

1. Click the registration or sign-in link on the dataset page.
2. Provide your name, email, institutional affiliation, and intended use.
3. Verify your email address.

### Step 3: Review and Accept the Research Use Agreement

Read the Stanford University Dataset Research Use Agreement carefully. Key terms include:

- The dataset may only be used for **non-commercial research purposes**.
- You may **not distribute, publish, or reproduce** a copy of the dataset.
- You may not use the data to identify any individual patient.
- Proper citation is required in any publications.

Accept the agreement to gain access to the download links.

### Step 4: Download the Dataset

After acceptance, download links become available. Two versions are typically offered:

1. **CheXpert Original**: Full-resolution images (~439 GB).
2. **CheXpert Small**: Downsampled images (~11 GB), suitable for initial experiments and prototyping.

Download the version appropriate for your research needs.

### Step 5: Extract and Explore

Extract the downloaded archive. The dataset is organized into:

```
CheXpert-v1.0/
  train/
    patient00001/
      study1/
        view1_frontal.jpg
        view2_lateral.jpg
    ...
  valid/
    ...
  train.csv
  valid.csv
```

### Step 6: Access CheXpert Plus (Optional)

For the extended CheXpert Plus dataset (DICOM format with radiology reports):
[https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1)

CheXpert Plus includes 223,462 unique pairs of radiology reports and chest X-rays in DICOM format with 47 DICOM metadata elements.

## Data Format

- **Images**: JPEG format (original CheXpert); DICOM format (CheXpert Plus).
- **Labels**: CSV files (`train.csv`, `valid.csv`) with columns for each of the 14 observations.
- **Label encoding**: 1 (positive), 0 (negative), -1 (uncertain), blank (unmentioned).
- **Directory structure**: Organized by patient ID and study number.

## Key Tables / Files

| File / Directory | Description |
|-----------------|-------------|
| `train/` | Training set images organized by patient/study (~223,414 images) |
| `valid/` | Validation set images (~234 images, with expert consensus labels) |
| `train.csv` | Training labels: Path, Sex, Age, Frontal/Lateral, AP/PA, and 14 observation columns |
| `valid.csv` | Validation labels with expert-adjudicated ground truth |
| Individual study folders | JPEG images per patient per study (frontal and/or lateral views) |

## Observation Labels (14 Classes)

No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices.

Each label can be: **positive (1)**, **negative (0)**, **uncertain (-1)**, or **blank** (not mentioned in the report).

## Important Restrictions

- **Non-commercial research use only** -- The Stanford University Dataset Research Use Agreement strictly prohibits commercial use.
- **No redistribution** -- You may not distribute, publish, or reproduce copies of the dataset.
- **No re-identification** -- Do not attempt to identify individual patients.
- **Citation required** -- Cite the original paper: Irvin, J., Rajpurkar, P., et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI 2019. (arXiv: 1901.07031)
- **Uncertainty labels** -- Be aware that many labels are marked as "uncertain" -- this is a feature of the dataset design and requires careful handling in model development.

## Useful Links

- [CheXpert Dataset Page (Stanford AIMI)](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7890a)
- [CheXpert Overview (Stanford AIMI)](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)
- [CheXpert Plus](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1)
- [CheXpert Competition Page](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [Original Paper (arXiv:1901.07031)](https://arxiv.org/abs/1901.07031)
- [CheXpert Plus Paper (arXiv:2405.19538)](https://arxiv.org/html/2405.19538v2)
- [CheXpert Demo Data (Small Sample)](https://aimi.stanford.edu/datasets/chexpert-demo-data)
