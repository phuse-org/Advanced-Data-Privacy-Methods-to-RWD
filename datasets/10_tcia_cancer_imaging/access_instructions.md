# The Cancer Imaging Archive (TCIA) — Access Instructions

## Overview

The Cancer Imaging Archive (TCIA) is an open-access database of medical images for cancer research, funded by the National Cancer Institute's (NCI) Cancer Imaging Program and operated by the University of Arkansas for Medical Sciences. TCIA hosts a large, growing archive of de-identified medical images organized as "collections" -- typically grouped by disease type (e.g., lung cancer), imaging modality (MRI, CT, digital histopathology), or research focus. The archive includes hundreds of curated collections with supporting data such as patient outcomes, treatment details, genomics, and expert analyses when available.

## Access Level

**Open with terms** -- Most collections on TCIA can be accessed without creating an account. A small number of restricted collections require an account and approval. All data downloads are subject to TCIA's Data Usage Policy. DICOM is the primary file format for radiology imaging.

## Prerequisites

- **A modern web browser** for browsing and searching collections via the TCIA Radiology Portal.
- **NBIA Data Retriever** installed on your system (Windows, Mac, or Linux) for downloading DICOM images in bulk. This is a free, one-time installation.
- **Java Runtime Environment (JRE)** may be required for the NBIA Data Retriever (desktop version).
- **For restricted collections**: A TCIA user account (free registration).
- **For programmatic access** (optional): Python environment with the `tcia_utils` package, or familiarity with REST APIs.

## Step-by-Step Access

### Step 1: Browse Collections on the TCIA Website

Visit [https://www.cancerimagingarchive.net/](https://www.cancerimagingarchive.net/) and navigate to the collections listing. Each collection page describes the imaging modalities, number of subjects, disease type, and data usage terms.

### Step 2: Search for Images Using the TCIA Radiology Portal

Use the TCIA Radiology Portal (also called the NBIA Data Portal) to search, filter, and browse imaging data within and across collections. You can filter by collection, modality, body part, and other criteria.

### Step 3: Install the NBIA Data Retriever

Download and install the NBIA Data Retriever from the [TCIA wiki](https://wiki.cancerimagingarchive.net/display/NBIA/9.3+Downloading+the+NBIA+Data+Retriever). This is the recommended tool for downloading DICOM data. On Linux, a command-line interface is available that does not require a desktop environment. You only need to install it once.

### Step 4: Create a Manifest File

In the TCIA Radiology Portal:

1. Search for images of interest.
2. Add selected subjects and/or series to your cart.
3. Click the cart icon in the top-right corner.
4. Select **Download > Download Cart** to download a `.tcia` manifest file to your browser.

### Step 5: Download Images via NBIA Data Retriever

Open the downloaded `.tcia` manifest file with the NBIA Data Retriever. The tool will:

1. Present the list of series to download.
2. Allow you to choose a download directory.
3. Let you select a folder naming convention:
   - **Descriptive**: Collection Name > Patient ID > Study Date + Study ID + Study Description > Series Number + Series Description
   - **Classic**: Collection Name > Patient ID > Study Instance UID > Series Instance UID
4. Download all DICOM files along with a `LICENSE` file and a `metadata.csv` file containing key DICOM header information.

### Step 6: Programmatic Access (Optional)

For scripting and automation, TCIA offers REST APIs:

- **TCIA REST API**: Query and download data programmatically.
- **Python notebooks**: Available at [github.com/kirbyju/TCIA_Notebooks](https://github.com/kirbyju/TCIA_Notebooks) for guided examples.
- **CLI version of NBIA Data Retriever**: Supports automated pipelines with flags to accept license terms non-interactively.

```python
# Example using tcia_utils
from tcia_utils import nbia
# List available collections
collections = nbia.getCollections()
```

## Data Format

- **Primary format**: DICOM (Digital Imaging and Communications in Medicine) for all radiology data.
- **Supporting files**: CSV metadata, LICENSE files, and clinical data files (varies by collection).
- **Other formats**: Some collections include non-DICOM data such as NIfTI segmentations, pathology images, or spreadsheets, depending on the collection.

## Key Tables / Files

| File / Resource | Description |
|----------------|-------------|
| DICOM image files (`.dcm`) | The primary imaging data, organized by patient, study, and series |
| `metadata.csv` | Key DICOM header fields extracted for each downloaded series |
| `LICENSE` | Data Usage Policy for the downloaded collection |
| Clinical data files | Patient outcomes, treatment details, genomics (varies by collection) |
| `.tcia` manifest files | Cart export files used by the NBIA Data Retriever |

## Important Restrictions

- All public TCIA data is free for research use, but you must comply with the TCIA Data Usage Policy.
- Some collections have specific licenses (e.g., CC-BY, CC-BY-NC) -- check each collection page.
- Restricted collections require a TCIA account and may require additional data use agreements.
- When publishing, cite both TCIA and the specific collection's publication.
- Do not attempt to re-identify subjects from the de-identified imaging data.

## Useful Links

- [TCIA Homepage](https://www.cancerimagingarchive.net/)
- [Access the Data](https://www.cancerimagingarchive.net/access-data/)
- [NBIA Data Retriever Download](https://wiki.cancerimagingarchive.net/display/NBIA/9.3+Downloading+the+NBIA+Data+Retriever)
- [NBIA Data Retriever FAQ](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+FAQ)
- [NBIA Data Retriever User Guide](https://wiki.cancerimagingarchive.net/display/NBIA/Current+NBIA+Data+Retriever+Help)
- [TCIA Radiology Portal Guide](https://wiki.cancerimagingarchive.net/display/NBIA)
- [TCIA Python Notebooks (GitHub)](https://github.com/kirbyju/TCIA_Notebooks)
- [TCIA REST API Guides](https://wiki.cancerimagingarchive.net/display/Public/Frequently+Asked+Questions)
- [TCIA on Wikipedia](https://en.wikipedia.org/wiki/The_Cancer_Imaging_Archive)
