# NCI Imaging Data Commons (IDC) — Access Instructions

## Overview

The NCI Imaging Data Commons (IDC) is a cloud-based repository of publicly available cancer imaging data, operated as a node within the National Cancer Institute's Cancer Research Data Commons (CRDC). IDC provides access to over 85 TB of cancer imaging data, including radiology images (CT, MRI, PET), brightfield (H&E) and fluorescence slide microscopy images, image-derived data (annotations, segmentations, quantitative measurements), and accompanying clinical data. All images are harmonized into the DICOM standard. IDC ingests collections from sources such as TCGA, TCIA, CPTAC, CCDI, HTAN, LIDC, NCI QIN, VHP, and NLST.

## Access Level

**Open** -- All data in IDC is publicly available with no registration and no access requests required. Over 95% of the data is covered by the permissive CC-BY license, which allows commercial reuse. The remaining data carries other Creative Commons or similar open licenses.

## Prerequisites

- **For basic browsing and visualization**: A modern web browser is sufficient. No account required.
- **For programmatic access via `idc-index`**: Python 3.7+ installed on your system.
- **For BigQuery queries**: A Google account (free). You can use BigQuery Sandbox for free-tier querying without a credit card.
- **For Google Colab notebooks**: A Google account provides free access to Colab with GPU support.
- **For bulk cloud downloads**: Familiarity with S3-compatible tools (e.g., `s5cmd`) or the `idc-index` CLI.

## Step-by-Step Access

### Step 1: Explore via the IDC Portal

Visit the IDC Portal at [https://portal.imaging.datacommons.cancer.gov/](https://portal.imaging.datacommons.cancer.gov/). You can browse, filter, and search across all collections without logging in. The portal is integrated with open-source viewers including OHIF (for radiology), Slim (for digital pathology), and VolView (for volumetric visualization).

### Step 2: Install the `idc-index` Python Package

For programmatic access, install the lightweight Python package:

```bash
pip install idc-index
```

Then instantiate the client in Python:

```python
from idc_index import IDCClient
client = IDCClient.client()
```

### Step 3: Search and Build Cohorts

Use `idc-index` to search IDC metadata and build cohorts programmatically:

```python
# List all available collections
collections = client.get_collections()

# Download a specific collection (e.g., rider_pilot, ~10.5 GB)
client.download_from_selection(collection_id="rider_pilot", downloadDir="./data")
```

You can also copy identifiers from the IDC Portal (collection, case, study, or series IDs) and use them directly with `idc-index` download functions.

### Step 4: Use the Command-Line Interface

Once `idc-index` is installed, you can download from the terminal:

```bash
idc download <manifest_file>
idc download <collection_id>
```

### Step 5: Advanced Metadata Queries with BigQuery (Optional)

For comprehensive metadata exploration beyond what `idc-index` provides:

1. Go to the [Google Cloud BigQuery Console](https://console.cloud.google.com/bigquery).
2. Create a BigQuery Sandbox (free, no credit card needed).
3. Navigate to the IDC public dataset and run SQL queries against DICOM metadata tables.
4. Alternatively, use the Python BigQuery client in a Google Colab notebook.

BigQuery gives access to the full set of DICOM metadata for all IDC images, enabling precise cohort definitions using standard SQL.

### Step 6: Run Tutorials (Optional)

Clone the IDC Tutorials repository for guided notebooks:

```bash
git clone https://github.com/ImagingDataCommons/IDC-Tutorials.git
```

These notebooks cover basic `idc-index` usage, BigQuery querying, and integration with analysis tools.

## Data Format

- **Primary format**: DICOM (Digital Imaging and Communications in Medicine). All images and image-derived data in IDC are harmonized to DICOM.
- **Metadata**: Available via BigQuery tables (SQL-queryable) and via `idc-index` local index.
- **Cloud storage**: Data is stored in both Google Cloud Storage and AWS S3 public buckets. Files can be fetched using S3 API clients or `idc-index`.

## Key Tables / Files

| Resource | Description |
|----------|-------------|
| `dicom_all` (BigQuery) | Comprehensive DICOM metadata for all IDC files |
| `idc-index` local index | Lightweight local index of collection, patient, study, and series metadata |
| DICOM image files | Original imaging data (CT, MRI, PET, pathology slides, etc.) |
| Annotation/segmentation files | DICOM-SR, DICOM-SEG, and RTSTRUCT files for image-derived data |
| Clinical data | Accompanying patient-level clinical information where available |

## Important Restrictions

- Over 95% of IDC data is under the **CC-BY 4.0** license, which permits commercial use with attribution.
- Some collections may have other Creative Commons licenses; check the license field per collection.
- When publishing results based on IDC data, cite the IDC platform paper and the specific collection(s) used.
- Respect patient privacy: all data is de-identified, but do not attempt to re-identify subjects.

## Useful Links

- [IDC Portal](https://portal.imaging.datacommons.cancer.gov/)
- [IDC User Guide](https://learn.canceridc.dev/)
- [Getting Started with IDC](https://learn.canceridc.dev/getting-started-with-idc)
- [idc-index Python Package (GitHub)](https://github.com/ImagingDataCommons/idc-index)
- [IDC Tutorials (GitHub)](https://github.com/ImagingDataCommons/IDC-Tutorials)
- [IDC on CRDC](https://datacommons.cancer.gov/repository/imaging-data-commons)
- [IDC on AWS Open Data Registry](https://registry.opendata.aws/nci-imaging-data-commons/)
- [IDC on Google Cloud Marketplace](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/idc)
- [IDC User Forum](https://discourse.canceridc.dev/)
- [IDC Platform Paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8373794/)
