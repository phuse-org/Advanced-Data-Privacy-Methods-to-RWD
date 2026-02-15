# BIMCV COVID-19+ — Access Instructions

## Overview

BIMCV COVID-19+ is a large, annotated medical imaging dataset from the Valencian Region Medical Image Bank (BIMCV) in Spain, containing chest X-ray (CXR) and computed tomography (CT) images from COVID-19 positive patients. The dataset was created by FISABIO, Miguel Hernandez University, University of Alicante, Hospital San Juan de Alicante, and collaborators including MedBravo, GE, and CIPF. It provides de-identified imaging data along with radiological findings, anatomical labels, PCR test results, and immunoglobulin (IgG/IgM) diagnostic antibody test results. Data was collected from 11 hospitals in the Valencian Region between February 26 and April 18, 2020.

## Access Level

**Open** -- The dataset is freely available for research purposes and can also be used for commercial purposes under certain conditions. Before downloading, users must accept the End-User License Agreement (Research Use Agreement). The code repository is released under the MIT license.

## Prerequisites

- **A modern web browser** for accessing the download portal and reading documentation.
- **Sufficient storage**: The full dataset comprises approximately 571 compressed archives (.tgz files) and is very large (tens of GB to over 100 GB depending on iteration).
- **Archive extraction tools** capable of handling `.tgz` files (e.g., `tar` on Linux/macOS, 7-Zip on Windows).
- **Agreement to the Research Use Agreement** before downloading.
- **WebDAV client** recommended for large downloads due to high demand on HTTP servers.

## Step-by-Step Access

### Step 1: Visit the BIMCV COVID-19 Project Page

Navigate to [https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/) to review the dataset description, available iterations, and download instructions.

### Step 2: Review and Accept the Research Use Agreement

Before downloading, read and accept the End-User License Agreement at:
[http://bimcv.cipf.es/bimcv-projects/bimcv-covid19/bimcv-covid19-dataset-research-use-agreement-2/](http://bimcv.cipf.es/bimcv-projects/bimcv-covid19/bimcv-covid19-dataset-research-use-agreement-2/)

### Step 3: Choose a Download Repository

The dataset is hosted in two repositories:

1. **Center for Open Science (OSF)** -- Located in Germany, accessible at [https://osf.io/nh7g8/](https://osf.io/nh7g8/) (DOI: 10.17605/OSF.IO/NH7G8).
2. **EUDAT** -- The largest integrated data services infrastructure in Europe, hosted through TransBioNet / Task Force COVID-19 at Barcelona Supercomputing Center (BSC).

### Step 4: Download the Data

The dataset is distributed as approximately 571 compressed `.tgz` archives plus three supplementary archives containing metadata, derivative information, and session data. Each archive has a corresponding manifest file listing its contents.

- **Recommended**: Use the WebDAV protocol for downloading large volumes due to high demand.
- **Verify integrity**: SHA1 checksums are provided for each archive -- verify downloads against these checksums before extraction.

```bash
# Example: verify a downloaded archive
sha1sum <downloaded_file.tgz>
# Compare with the published checksum
```

### Step 5: Extract and Organize

Extract the `.tgz` archives:

```bash
tar -xzf <archive_name>.tgz
```

The data follows the Medical Imaging Data Structure (MIDS) format with images stored in high resolution alongside anatomical labels.

### Step 6: Consult the GitHub Repository for Documentation

Visit the GitHub repository for additional documentation, code, and updates:
[https://github.com/BIMCV-CSUSP/BIMCV-COVID-19](https://github.com/BIMCV-CSUSP/BIMCV-COVID-19)

## Data Format

- **Imaging format**: DICOM files for CXR (CR, DX modalities) and CT studies.
- **Data structure**: Medical Imaging Data Structure (MIDS) format.
- **Distribution format**: Compressed `.tgz` archives with manifest files.
- **Annotations**: UMLS-coded radiological findings with anatomical labels. A subset of 10 images includes semantic segmentation of radiological findings by radiologists.
- **Metadata**: Supplementary archives with session data, derivatives, and metadata.

## Key Tables / Files

| Resource | Description |
|----------|-------------|
| CXR images (CR, DX) | Chest X-ray images in DICOM format (21,342 CR + 34,829 DX studies across all iterations) |
| CT images | Computed tomography studies in DICOM format (7,918 CT studies across all iterations) |
| Radiological findings | UMLS-coded labels for thoracic entities with anatomical localization |
| PCR / antibody test results | Laboratory data including PCR, IgG, IgM, and IgA results |
| Radiological reports | Original reports in Spanish |
| Segmentation annotations | Semantic segmentation of findings by radiologists (subset of 10 images) |
| Manifest files | Content listings for each `.tgz` archive |

## Important Restrictions

- You must accept the **Research Use Agreement** before downloading.
- The dataset is available for research and, under certain conditions, commercial use -- review the license carefully.
- **Do not claim diagnostic performance** of a model without a proper clinical study. As noted by the creators: "This is not a Kaggle competition dataset."
- Cite the original paper when publishing results: de la Iglesia Vaya et al., "BIMCV COVID-19+: a large annotated dataset of RX and CT images from COVID-19 patients" (arXiv: 2006.01174).
- All data is de-identified; do not attempt re-identification.

## Useful Links

- [BIMCV COVID-19 Project Page](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/)
- [Research Use Agreement](http://bimcv.cipf.es/bimcv-projects/bimcv-covid19/bimcv-covid19-dataset-research-use-agreement-2/)
- [OSF Repository (DOI: 10.17605/OSF.IO/NH7G8)](https://osf.io/nh7g8/)
- [GitHub Repository](https://github.com/BIMCV-CSUSP/BIMCV-COVID-19)
- [ArXiv Paper](https://arxiv.org/abs/2006.01174)
- [BIMCV Main Site](https://bimcv.cipf.es/)
