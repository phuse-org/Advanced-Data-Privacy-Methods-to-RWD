# COVIDx-US — Access Instructions

## Overview

COVIDx-US is an open-access benchmark dataset of COVID-19 related lung ultrasound imaging data, developed by the National Research Council Canada (NRC) and the University of Waterloo as part of the COVID-Net open-source initiative. The dataset compiles lung ultrasound content from nine public data sources: ButterflyNetwork, GrepMed, The POCUS Atlas, LITFL, Radiopaedia, CoreUltrasound, University of Florida, scientific publications, and Clarius. As of version 1.5 (May 2022), it contains 242 ultrasound videos and 29,651 processed images categorized into COVID-19 positive, pneumonia, normal, and other lung condition classes. Each video includes a standardized lung ultrasound score (LUSS) for clinical interpretation.

## Access Level

**Open -- Research terms** -- The repository code and materials are released under the GNU Affero General Public License 3.0 (AGPL-3.0). However, the underlying source data carries mixed licensing: some sources use Creative Commons licenses (CC BY-NC, CC BY-NC-SA), while others have no stated usage restrictions. Users are responsible for verifying permissible use with each unlicensed data source.

## Prerequisites

- **Python 3.6+** installed on your system.
- **Jupyter Notebook** or JupyterLab for running the dataset creation notebook.
- **Git** for cloning the repository.
- **Internet connection** for downloading source videos from the nine public data sources.
- **Sufficient storage**: The processed dataset is several GB depending on frame extraction settings.
- **Required Python packages**: Listed in the repository (e.g., OpenCV, pandas, NumPy, requests).

## Step-by-Step Access

### Step 1: Clone the GitHub Repository

```bash
git clone https://github.com/nrc-cnrc/COVID-US.git
cd COVID-US
```

The repository contains processing scripts, metadata, masks, and the Jupyter notebook needed to create the dataset. **Note**: The repository does not host the imaging data directly -- you must generate the dataset using the provided scripts.

### Step 2: Set Up the Python Environment

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have Jupyter Notebook installed:

```bash
pip install jupyter
```

### Step 3: Run the Dataset Creation Notebook

Open and execute the `create_COVIDxUS.ipynb` notebook:

```bash
jupyter notebook create_COVIDxUS.ipynb
```

This notebook will:

1. Download ultrasound videos from the nine public data sources.
2. Process and extract frames from the videos.
3. Integrate the frames into the standardized COVIDx-US dataset structure.
4. Apply the provided masks and metadata.

**Important**: Modify the file paths in the notebook code to match your local directory structure if needed.

### Step 4: Verify the Dataset

After running the notebook, verify that you have the expected output:

- **242 ultrasound videos** (as of v1.5)
- **29,651 processed images** (as of v1.5)
- Organized by category: COVID-19 positive, pneumonia, normal, other lung conditions

### Step 5: Review Licensing for Each Source

The dataset aggregates data from nine sources with varying licenses. Before using the data, review the licensing terms for each source as documented in the repository. Some sources use Creative Commons licenses while others have no explicit license.

## Data Format

- **Source data**: Ultrasound video files downloaded from public sources.
- **Processed images**: Extracted frames from ultrasound videos (PNG/JPEG format).
- **Metadata**: CSV files with video-level and frame-level annotations.
- **Masks**: Provided for preprocessing and region-of-interest extraction.
- **LUSS scores**: Standardized lung ultrasound scores per video (added in v1.5).

## Key Tables / Files

| File / Resource | Description |
|----------------|-------------|
| `create_COVIDxUS.ipynb` | Main Jupyter notebook to generate the dataset from source videos |
| `dataset/` | Output directory for processed images and metadata |
| `masks/` | Preprocessing masks for ultrasound frame extraction |
| `metadata/` | Source video metadata, labels, and LUSS scores |
| `README.md` | Repository documentation with version history and instructions |
| `LICENSE` | GNU AGPL 3.0 license file |

## Dataset Categories

| Category | Description |
|----------|-------------|
| COVID-19 | Lung ultrasound from confirmed COVID-19 patients |
| Pneumonia | Non-COVID pneumonia cases |
| Normal | Healthy lung ultrasound |
| Other | Other lung diseases and conditions |

## Important Restrictions

- The **repository code** is licensed under GNU AGPL 3.0.
- **Source data has mixed licensing** -- some sources use CC BY-NC or CC BY-NC-SA, which prohibit commercial use. Others have no stated license.
- **Users must verify** permissible usage with each unlicensed data source independently.
- The team takes **no responsibility** for data use by downstream users.
- When publishing results, cite: Ebadi et al., "COVIDx-US -- An open-access benchmark dataset of ultrasound imaging data for AI-driven COVID-19 analytics," Frontiers in Bioscience-Landmark, 2022.
- The processed dataset should not be redistributed without verifying the licensing terms of each source.

## Useful Links

- [COVIDx-US GitHub Repository](https://github.com/nrc-cnrc/COVID-US)
- [Dataset Creation Notebook](https://github.com/nrc-cnrc/COVID-US/blob/main/create_COVIDxUS.ipynb)
- [Original Paper (arXiv:2103.10003)](https://arxiv.org/abs/2103.10003)
- [Published Paper (Frontiers in Bioscience-Landmark)](https://www.imrpress.com/journal/FBL/27/7/10.31083/j.fbl2707198/htm)
- [PubMed Entry](https://pubmed.ncbi.nlm.nih.gov/35866396/)
- [COVID-Net Initiative](https://alexswong.github.io/COVID-Net/)
