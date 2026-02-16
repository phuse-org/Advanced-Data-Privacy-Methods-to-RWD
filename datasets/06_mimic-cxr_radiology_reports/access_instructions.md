# MIMIC-CXR (MIMIC Chest X-ray Database) — Access Instructions

## Overview

MIMIC-CXR is a large, publicly available dataset of de-identified chest radiographs with free-text radiology reports. It contains **377,110 chest X-ray images** corresponding to **227,835 radiographic studies** from **65,379 patients** who presented to the Beth Israel Deaconess Medical Center (BIDMC) Emergency Department between 2011 and 2016. The dataset provides images in their native DICOM format along with the associated free-text radiology reports. A companion dataset, MIMIC-CXR-JPG, provides the same images in compressed JPG format with structured labels derived from the radiology reports.

MIMIC-CXR is hosted on [PhysioNet](https://physionet.org/) and maintained by the MIT Laboratory for Computational Physiology.

The latest version is **MIMIC-CXR v2.1.0**.

## Access Level

**Credentialed Access** — MIMIC-CXR is a restricted-access dataset requiring PhysioNet credentialing and an individual Data Use Agreement (DUA). The same credentialing used for other MIMIC datasets applies here, but you must separately sign the DUA for this project. The data is free of charge for all approved researchers.

## Prerequisites

Before starting the access process, you will need:

1. A valid email address (institutional/academic email is strongly preferred and speeds up credentialing).
2. An affiliation with a research institution, hospital, or university (students and postdocs need a supervisor/reference).
3. Access to the [CITI Program](https://about.citiprogram.org/) training platform (free for this course).
4. A web browser to complete all registration and training steps.
5. **Significant storage capacity.** The full DICOM dataset is approximately **4.7 TB**. The JPG alternative (MIMIC-CXR-JPG) is approximately **558 GB**. Plan your storage accordingly.
6. If you already have PhysioNet credentialed access, you only need to sign the DUA for this project (skip to Step 4).

## Step-by-Step Access

### Step 1: Create a PhysioNet Account

1. Go to [https://physionet.org/register/](https://physionet.org/register/).
2. Fill in your name, email address, and create a password.
3. Confirm your email address by clicking the verification link sent to your inbox.
4. Log in to your new PhysioNet account.

### Step 2: Complete CITI "Data or Specimens Only Research" Training

1. Go to [https://about.citiprogram.org/](https://about.citiprogram.org/) and create a CITI Program account (if you do not already have one).
2. Navigate to **My Courses** and click **Add Affiliation**.
3. Search for and select **"Massachusetts Institute of Technology Affiliates"** as your affiliation. This designated affiliation allows non-MIT personnel to take the required course without additional fees.
4. Enroll in and complete the **"Data or Specimens Only Research"** course.
5. After completion, go to **Records** at the top of the CITI website. Under your completion record, click **View-Print-Share** and download your **Completion Report** (not the certificate — PhysioNet requires the full report).

### Step 3: Submit Credentialing Application on PhysioNet

1. Log in to PhysioNet and navigate to [https://physionet.org/settings/credentialing/](https://physionet.org/settings/credentialing/).
2. Fill in your personal details, institutional affiliation, and research purpose.
3. If you are a student or postdoc, provide your supervisor's name and contact information in the reference section.
4. Upload your CITI **Completion Report** (the full report PDF, not just the certificate).
5. Submit your credentialing application and wait for approval (typically a few business days).

### Step 4: Sign the Data Use Agreement (DUA)

1. Navigate to the MIMIC-CXR project page: [https://physionet.org/content/mimic-cxr/](https://physionet.org/content/mimic-cxr/).
2. Scroll to the bottom of the page and look for the **"Sign the data use agreement"** prompt.
3. Read the DUA carefully and click to sign. Key provisions include:
   - You will use the data solely for lawful scientific research.
   - You will not attempt to re-identify any individual or institution.
   - You will not share the data with others.
   - Any publication using the data must also make the relevant analysis code publicly available.
   - You will report any suspected identifiable information to PHI-report@physionet.org.
4. Access is typically granted immediately after signing.

### Step 5: Download the Data

Given the large size of this dataset, choose your download approach carefully:

**Option A: Google Cloud Platform (Recommended for DICOM)**
- The dataset maintainers recommend using the data within Google Cloud Platform (GCP) rather than downloading the full 4.7 TB DICOM dataset.
- See: [https://mimic.mit.edu/docs/gettingstarted/cloud/](https://mimic.mit.edu/docs/gettingstarted/cloud/)

**Option B: wget (For Selective or Full Download)**
```bash
# Full DICOM dataset (~4.7 TB)
wget -r -N -c -np --user <your-username> --ask-password https://physionet.org/files/mimic-cxr/2.1.0/

# Or download only the metadata and reports (much smaller)
wget --user <your-username> --ask-password https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
```

**Option C: MIMIC-CXR-JPG (Smaller Alternative, ~558 GB)**
- If you do not need DICOM metadata and prefer a more manageable download, consider using MIMIC-CXR-JPG instead.
- URL: [https://physionet.org/content/mimic-cxr-jpg/](https://physionet.org/content/mimic-cxr-jpg/)
- Requires a separate DUA signing.

**Option D: Web Browser**
- You can download individual files or folders from the Files section on the project page.

## Data Format

### MIMIC-CXR (Primary Dataset)

- **Images**: DICOM format (`.dcm`) — the native clinical imaging format containing full 16-bit pixel depth along with structured metadata (patient positioning, imaging parameters, etc.). Built according to DICOM Standard version 2017e.
- **Reports**: Free-text radiology reports stored alongside the images.
- **Metadata**: Gzip-compressed CSV files (`.csv.gz`) containing study-level and image-level metadata.

### MIMIC-CXR-JPG (Companion Dataset)

- **Images**: JPG format — 8-bit compressed images derived from the DICOMs using a standardized pipeline (normalization, intensity inversion, contrast enhancement, JPEG quality 95).
- **Labels**: Structured multi-label annotations extracted from radiology reports using CheXpert and NegBio NLP labelers.

### Directory Structure

The images are organized into 10 top-level folders (`p10` through `p19`), each containing approximately 6,500 patient subfolders. Within each patient folder, images and reports are organized by study.

## Key Tables / Files

| File | Description |
|------|-------------|
| `mimic-cxr-2.0.0-metadata.csv.gz` | Image-level metadata extracted from DICOM headers (dicom_id, subject_id, study_id, view position, procedure description, data split). |
| `mimic-cxr-2.0.0-chexpert.csv.gz` | Structured labels from the CheXpert NLP labeler (14 pathology labels per study). Available in MIMIC-CXR-JPG. |
| `mimic-cxr-2.0.0-negbio.csv.gz` | Structured labels from the NegBio NLP labeler. Available in MIMIC-CXR-JPG. |
| `mimic-cxr-2.0.0-split.csv.gz` | Official train/validate/test split assignments. |
| `cxr-record-list.csv.gz` | Master list linking subject_id, study_id, and dicom_id. |
| `cxr-study-list.csv.gz` | Study-level listing with paths to radiology report text files. |
| `cxr-provider-list.csv.gz` | De-identified provider identifiers associated with studies. |
| `files/` | Directory tree containing DICOM images organized by patient and study. |

### Pathology Labels (14 Categories in CheXpert)

Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices.

### View Distribution

- Frontal views: ~251,714 images (66.8%)
- Lateral views: ~122,538 images (32.5%)
- Other projections: ~858 images (0.2%)

## Important Restrictions

- **No data sharing.** Each researcher must independently complete credentialing and sign the DUA. You may not redistribute the images or reports.
- **No re-identification attempts.** You must not attempt to identify any patients, providers, or institutions. All dates have been shifted (first admission mapped to years 2100-2200).
- **PHI reporting obligation.** If you discover identifiable information (especially in DICOM metadata or report text), report it to PHI-report@physionet.org.
- **Code availability required.** Any publication using MIMIC-CXR data must also make the relevant analysis code publicly available.
- **Citation requirement.** You must cite the dataset in any publications.
- **Separate DUA for MIMIC-CXR-JPG.** If you also want the JPG version, you need to sign a separate DUA on its project page.

## Useful Links

- [MIMIC-CXR on PhysioNet (DICOM, latest version)](https://physionet.org/content/mimic-cxr/)
- [MIMIC-CXR v2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/)
- [MIMIC-CXR-JPG on PhysioNet (JPG with labels)](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [MIMIC-CXR Documentation](https://mimic.mit.edu/docs/iv/modules/cxr/)
- [MIMIC-CXR GitHub Repository](https://github.com/MIT-LCP/mimic-cxr)
- [MIMIC-CXR Scientific Data Paper](https://www.nature.com/articles/s41597-019-0322-0)
- [PhysioNet Credentialing Page](https://physionet.org/settings/credentialing/)
- [CITI Program Training](https://about.citiprogram.org/)
- [PhysioNet CITI Course Instructions](https://physionet.org/about/citi-course/)
