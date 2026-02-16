# OpenNeuro — Access Instructions

## Overview

OpenNeuro is a free, open platform for sharing human and non-human brain imaging data, hosted by the Stanford Center for Reproducible Neuroscience. It is a BRAIN Initiative designated data archive that hosts hundreds of neuroscience datasets across multiple modalities including functional MRI (fMRI), structural MRI, diffusion MRI, electroencephalography (EEG), magnetoencephalography (MEG), intracranial EEG (iEEG), positron emission tomography (PET), and behavioral data. All uploaded datasets must conform to the Brain Imaging Data Structure (BIDS) standard, ensuring consistent organization and metadata across datasets.

## Access Level

**Open** -- The majority of datasets on OpenNeuro are shared under the Creative Commons CC0 license (public domain dedication), which places minimal restrictions on data reuse. Some datasets may use other Creative Commons licenses (e.g., CC-BY). No registration is required to browse or download public datasets.

## Prerequisites

- **A modern web browser** for browsing and downloading datasets via the OpenNeuro web interface.
- **For command-line downloads (recommended for large datasets)**:
  - **DataLad** (recommended): A data management tool built on Git and git-annex. Install via `pip install datalad` or your package manager.
  - **AWS CLI** (alternative): For direct S3 bucket access via the AWS Open Data Registry.
  - **Node.js / npm** (alternative): For the OpenNeuro CLI tool.
- **Sufficient storage**: Dataset sizes range from a few MB to hundreds of GB depending on the modality and number of subjects.
- **BIDS-compatible analysis tools** (optional): Software that works with the BIDS standard (e.g., fMRIPrep, MRIQC, FreeSurfer, MNE-Python, Brainstorm, FastSurfer).

## Step-by-Step Access

### Step 1: Browse Datasets on OpenNeuro

Visit [https://openneuro.org/](https://openneuro.org/) and use the search functionality to find datasets by keyword, modality, task, or other criteria. Each dataset page shows:

- Dataset description and README
- Number of subjects and sessions
- Imaging modalities
- License information
- Download options
- File browser for examining the BIDS structure

### Step 2: Download via the Web Interface (Small Datasets)

For smaller datasets:

1. Navigate to the dataset page on OpenNeuro.
2. Click the **Download** button.
3. Select the desired version/snapshot.
4. The dataset will download as a compressed archive.

### Step 3: Download via DataLad (Recommended for Large Datasets)

DataLad provides efficient, versioned access to OpenNeuro datasets:

```bash
# Install DataLad
pip install datalad

# Install a specific dataset (e.g., ds000228)
datalad install https://github.com/OpenNeuroDatasets/ds000228.git

# Navigate into the dataset
cd ds000228

# Get specific files (data is fetched on demand)
datalad get sub-01/anat/sub-01_T1w.nii.gz

# Or get all data
datalad get .
```

All OpenNeuro datasets are available as DataLad datasets via GitHub:
`https://github.com/OpenNeuroDatasets/<dataset_accession_number>`

### Step 4: Download via AWS S3 (Alternative)

OpenNeuro data is available through the AWS Open Data Registry:

```bash
# List contents of a dataset
aws s3 ls --no-sign-request s3://openneuro.org/<dataset_accession_number>/

# Download a full dataset
aws s3 sync --no-sign-request s3://openneuro.org/<dataset_accession_number>/ ./<dataset_accession_number>/
```

No AWS account is required (use `--no-sign-request`).

### Step 5: Download via OpenNeuro CLI (Alternative)

```bash
# Install the OpenNeuro CLI
npm install -g @openneuro/cli

# Download a dataset
openneuro download <dataset_accession_number> <output_directory>
```

### Step 6: Validate the BIDS Structure (Optional)

After downloading, validate that the dataset conforms to the BIDS standard:

```bash
# Install the BIDS validator
pip install bids-validator

# Validate the dataset
bids-validator /path/to/dataset/
```

## Data Format

- **Standard**: Brain Imaging Data Structure (BIDS) -- a community-developed standard for organizing neuroimaging data.
- **MRI images**: NIfTI format (`.nii` or `.nii.gz`) with JSON sidecar files for metadata.
- **EEG data**: BrainVision (`.vhdr`, `.vmrk`, `.eeg`), European Data Format (`.edf`), or EEGLAB (`.set`) formats.
- **MEG data**: Various formats depending on the acquisition system (e.g., `.fif` for Elekta/MEGIN).
- **Metadata**: JSON sidecar files, TSV participant tables, and dataset description files.
- **Events**: TSV files describing experimental events and timing.

## Key Tables / Files

| File / Resource | Description |
|----------------|-------------|
| `dataset_description.json` | Dataset name, BIDS version, license, authors, references |
| `participants.tsv` | Participant demographics (age, sex, group, handedness, etc.) |
| `participants.json` | Data dictionary for the participants table |
| `sub-XX/` | Per-subject directories containing modality-specific subdirectories |
| `sub-XX/anat/` | Structural MRI data (T1w, T2w, FLAIR, etc.) |
| `sub-XX/func/` | Functional MRI data (BOLD) with event files |
| `sub-XX/dwi/` | Diffusion-weighted imaging data |
| `sub-XX/eeg/` | EEG recordings |
| `sub-XX/meg/` | MEG recordings |
| `README` | Human-readable dataset description |
| `CHANGES` | Version history of the dataset |

## Common Modalities Available

| Modality | Description | Typical Format |
|----------|-------------|----------------|
| T1w MRI | Structural brain anatomy | NIfTI (.nii.gz) |
| fMRI (BOLD) | Functional brain activity | NIfTI (.nii.gz) + events.tsv |
| Diffusion MRI | White matter microstructure | NIfTI (.nii.gz) + bvec/bval |
| EEG | Electrical brain activity | BrainVision / EDF / EEGLAB |
| MEG | Magnetic brain activity | FIF / CTF / other |
| PET | Metabolic/receptor imaging | NIfTI (.nii.gz) + JSON |
| iEEG | Intracranial recordings | BrainVision / EDF |

## Important Restrictions

- Most datasets use the **CC0 license** (public domain) -- no restrictions on reuse, including commercial use.
- Some datasets may use **CC-BY** or other licenses -- check the `dataset_description.json` file or the dataset page.
- Even with CC0 licensing, **academic norms** strongly encourage citing the original dataset and associated publications.
- Be aware that some datasets may contain sensitive demographic information even though imaging data is de-identified.
- BRAIN Initiative-funded datasets may have additional sharing requirements.
- Always respect the wishes of data contributors by providing proper attribution.

## Useful Links

- [OpenNeuro Homepage](https://openneuro.org/)
- [OpenNeuro FAQ](https://openneuro.org/faq)
- [OpenNeuro Datasets on GitHub (DataLad)](https://github.com/OpenNeuroDatasets)
- [OpenNeuro on AWS Open Data Registry](https://registry.opendata.aws/openneuro/)
- [BIDS Specification](https://bids.neuroimaging.io/)
- [BIDS Validator](https://bids-standard.github.io/bids-validator/)
- [OpenNeuro Platform Paper (eLife)](https://elifesciences.org/articles/71774)
- [OpenNeuro Platform Paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8550750/)
- [DataLad Handbook](https://handbook.datalad.org/)
- [OpenNeuro CLI (npm)](https://www.npmjs.com/package/@openneuro/cli)
