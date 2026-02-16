# SatHealth — Access Instructions

## Overview

SatHealth is a novel multimodal public health dataset that combines satellite-based environmental data, satellite imagery, all-disease prevalence estimates from medical claims, and Social Determinants of Health (SDoH) indicators. Developed by researchers at the Artificial Intelligence in Medicine (AIMED) Lab, SatHealth is designed to enable the integration of living environment characteristics into clinical AI and population health models.

To the knowledge of the authors, SatHealth is the first dataset in the United States that combines regional environmental characteristics with a healthcare database. The dataset was published as part of the KDD 2025 Datasets and Benchmarks Track (ACM SIGKDD Conference on Knowledge Discovery and Data Mining).

The dataset currently covers Ohio regions with plans for U.S.-wide expansion.

## Access Level

**Open** — SatHealth data are publicly accessible through a web application and a GitHub repository. The environmental and SDoH components are derived from public data sources (Google Earth Engine, American Community Survey). The disease prevalence data are estimated from the IBM MarketScan medical claims database. No formal registration or data use agreement is required for downloading, though users should comply with the terms of the underlying data sources.

## Prerequisites

- A web browser to access the SatHealth Explorer web application
- For reconstructing the dataset from scratch (optional):
  - Google Earth Engine account (free for research)
  - Google Maps Static API key
  - Python 3 environment with PyTorch and standard data science libraries
- Familiarity with geospatial data and machine learning concepts is recommended

## Step-by-Step Access

### Step 1: Review the Paper

Read the SatHealth paper to understand the dataset structure, methodology, and intended use cases:

- **arXiv:** [arxiv.org/abs/2506.13842](https://arxiv.org/abs/2506.13842)
- **Full HTML:** [arxiv.org/html/2506.13842](https://arxiv.org/html/2506.13842)
- **ACM Digital Library:** [dl.acm.org/doi/10.1145/3711896.3737440](https://dl.acm.org/doi/10.1145/3711896.3737440)

The paper describes the data collection process, feature engineering, geographic coverage, and benchmark experiments.

### Step 2: Access the SatHealth Explorer Web Application

Navigate to the SatHealth Explorer:

[https://aimed-sathealth.net](https://aimed-sathealth.net)

This web-based application allows you to:
- Explore SatHealth data interactively by region
- Visualize environmental features, satellite imagery, disease prevalence, and SDoH indicators
- Download pre-processed data files

### Step 3: Download Pre-Processed Data

From the SatHealth Explorer or the GitHub repository, download the pre-processed data archives:

- **`sathealth_dataset.zip`** — The full SatHealth dataset with environmental features, disease prevalence, and SDoH data
- **`sathealth_embeddings.zip`** — Pre-computed regional embedding vectors that can be directly integrated into clinical AI models

### Step 4: Clone the GitHub Repository

For code, data processing pipelines, and model implementations:

```bash
git clone https://github.com/Wang-Yuanlong/SatHealth.git
cd SatHealth
```

The repository contains:
- **Data processing scripts (01--03):**
  - `01_*` — Environmental data extraction from Google Earth Engine
  - `02_*` — Satellite image feature generation
  - `03_*` — Multimodal feature integration
- **Analysis scripts (04a--c):**
  - `04a_*` — Regional health modeling
  - `04b_*` — Correlation analysis
  - `04c_*` — Additional analyses
- **Model implementations:** LSTM, baseline models, and sequence models
- **Dataset utilities:** Custom PyTorch datasets and dataloaders

### Step 5: Reconstruct the Dataset from Scratch (Optional)

If you want to recreate or extend the dataset rather than using pre-processed files:

1. **Set up Google Earth Engine:** Register at [earthengine.google.com](https://earthengine.google.com/) and authenticate your environment
2. **Obtain a Google Maps Static API key:** Required for satellite imagery retrieval
3. **Run the extraction pipeline:** Follow the numbered scripts (01, 02, 03) in the repository to extract environmental data, retrieve satellite images, and integrate features
4. **Process disease prevalence:** The disease prevalence estimates are derived from the MarketScan medical claims database. Access to MarketScan requires a separate institutional license from Merative (formerly IBM Watson Health)

### Step 6: Use Regional Embeddings in Your Models

The pre-computed regional embeddings can be linked to any dataset with geographic identifiers (county FIPS, ZCTA, census tract, or CBSA codes). These embeddings encode environmental and satellite information and can be used as features in clinical AI models.

## Data Format

- **Tabular data:** CSV files with environmental features, disease prevalence, and SDoH indicators
- **Satellite imagery:** Over 400,000 aerial view images from Google Maps, each covering approximately a 500m-wide square area
- **Embeddings:** Pre-computed numerical vectors (tensors) in standard ML-ready formats
- **Code:** Python scripts using PyTorch

## Key Tables / Files

**Environmental Features (from Google Earth Engine):**
- Land surface temperature
- Vegetation indices (NDVI, EVI)
- Air quality indicators
- Precipitation and climate variables
- Land use / land cover classification

**Satellite Imagery:**
- Aerial view images at ~500m resolution
- Encoded as image embeddings for ML use

**Disease Prevalence:**
- All-disease prevalence estimates at regional level
- Derived from MarketScan medical claims data
- Covers multiple disease categories

**Social Determinants of Health (SDoH):**
- Social Deprivation Index (SDI) from the American Community Survey (ACS)
- Socioeconomic factors: poverty, education, employment, housing
- Community context variables

**Geographic Levels:**
- County
- ZIP Code Tabulation Area (ZCTA)
- Census Tract
- Core Based Statistical Area (CBSA)

## Important Restrictions

- **Geographic coverage:** The current release covers Ohio regions. U.S.-wide expansion is planned but not yet available.
- **MarketScan data:** The disease prevalence component is derived from the MarketScan commercial claims database, which primarily represents commercially insured and employer-sponsored populations. This may not be representative of the entire U.S. population (e.g., uninsured, Medicaid, Medicare populations).
- **Google Maps imagery terms:** Satellite images were retrieved from Google Maps Static API. Redistribution of raw Google Maps imagery may be subject to Google's Terms of Service. The dataset distributes image embeddings (features extracted from images) rather than raw images.
- **Research use:** The dataset is intended for research purposes. It should not be used for clinical decision-making without appropriate validation.
- **Citation required:** If you use SatHealth in your research, cite the paper:
  > Wang, Y., et al. "SatHealth: A Multimodal Public Health Dataset with Satellite-based Environmental Factors." Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25), 2025.
- **No explicit license stated:** The GitHub repository does not specify a formal open-source license as of the current release. Contact the authors for clarification on reuse terms.

## Useful Links

- [SatHealth Paper (arXiv)](https://arxiv.org/abs/2506.13842)
- [SatHealth Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3711896.3737440)
- [SatHealth GitHub Repository](https://github.com/Wang-Yuanlong/SatHealth)
- [SatHealth Explorer Web Application](https://aimed-sathealth.net)
- [Google Earth Engine](https://earthengine.google.com/)
- [KDD 2025 Datasets and Benchmarks Track](https://kdd2025.kdd.org/datasets-and-benchmarks-track-papers-2/)
