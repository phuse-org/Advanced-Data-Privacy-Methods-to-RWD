# EHRShot Benchmark -- Access Instructions

## Overview

EHRShot is a benchmark for evaluating foundation models on electronic health records (EHR) in few-shot and transfer learning settings. Developed by the Shah Lab at Stanford University School of Medicine, it provides full longitudinal structured EHR data for 6,739 patients from Stanford Medicine, encompassing 41.6 million clinical events across 921,499 visits. Unlike prior EHR benchmarks that focused exclusively on ICU data, EHRShot covers the full spectrum of clinical settings. The benchmark includes 15 clinically meaningful prediction tasks and is accompanied by CLMBR-T-Base, a 141-million-parameter clinical foundation model pre-trained on 2.57 million patients.

## Access Level

**Open / Research Terms** -- The dataset and model are available for research use after signing a research usage agreement. The source code is released under the Apache License 2.0. The dataset is hosted on the Redivis data platform, and the pre-trained model is available on Hugging Face. Both require agreeing to a research usage agreement before download.

## Prerequisites

Before accessing EHRShot, ensure you have:

- **Redivis Account**: Create a free account at [redivis.com](https://redivis.com) to access the dataset.
- **Hugging Face Account**: Create a free account at [huggingface.co](https://huggingface.co) to access the pre-trained CLMBR-T-Base model.
- **System Requirements** (for running the benchmark):
  - Linux machine
  - CUDA 11.8 and cuDNN 8.7.0 (recommended)
  - Python 3.10
  - Sufficient GPU memory for model inference
- **Software Dependencies**:
  - FEMR (Framework for Electronic Medical Records): `femr-cuda==0.1.16`
  - JAX with CUDA support: `jax[cuda]==0.4.8`
  - Haiku: `dm-haiku==0.0.9`
  - Optax: `optax==0.1.4`
  - Conda or Mamba for environment management

## Step-by-Step Access

### Step 1: Access the Dataset on Redivis

Navigate to the EHRShot dataset on Redivis at [https://redivis.com/datasets/53gc-8rhx41kgt](https://redivis.com/datasets/53gc-8rhx41kgt). Log in or create a Redivis account if you do not already have one.

### Step 2: Sign the Research Usage Agreement

On the Redivis dataset page, review and sign the research usage agreement. This agreement governs the terms under which you may use the de-identified patient data for research purposes.

### Step 3: Download the Dataset

Three versions of the dataset are available:

- **Original (EHRSHOT_ASSETS.zip)**: The primary format compatible with the EHRShot repository code.
- **MEDS (EHRSHOT_MEDS.zip)**: Data in the Medical Event Data Standard (MEDS) format for interoperability.
- **OMOP**: Full OMOP CDM table dumps for use with OHDSI tools.

Download the version(s) appropriate for your use case.

### Step 4: Access the Pre-trained Model (Optional)

Navigate to the CLMBR-T-Base model on Hugging Face at [https://huggingface.co/StanfordShahLab/clmbr-t-base](https://huggingface.co/StanfordShahLab/clmbr-t-base). Sign the research usage agreement and download the model weights.

### Step 5: Set Up the Development Environment

```bash
# Create a conda environment
conda create -n ehrshot python=3.10 -y
conda activate ehrshot

# Clone the repository
git clone https://github.com/som-shahlab/ehrshot-benchmark.git
cd ehrshot-benchmark

# Install the package
pip install -e .

# Install FEMR and JAX CUDA components
pip install femr-cuda==0.1.16
pip install jax[cuda]==0.4.8
pip install dm-haiku==0.0.9
pip install optax==0.1.4
```

### Step 6: Organize and Run the Benchmark

Place the downloaded dataset into the `EHRSHOT_ASSETS/` directory within the cloned repository, then execute the full benchmark pipeline:

```bash
bash run_all.sh
```

## Data Format

The EHRShot dataset is available in three formats:

- **Original Format (ZIP)**: Custom format compatible with the FEMR framework, where patient timelines consist of ordered sequences of clinical events (diagnoses, procedures, prescriptions).
- **MEDS Format (ZIP)**: Medical Event Data Standard format for broader interoperability.
- **OMOP CDM Format**: Standard OHDSI OMOP Common Data Model table dumps.

Patient records contain structured EHR data including timestamped diagnoses (ICD codes), procedures (CPT codes), prescriptions, laboratory results, and visit metadata.

## Key Tables / Files

| File / Component | Description |
|-----------------|-------------|
| **EHRSHOT_ASSETS.zip** | Primary dataset with patient timelines, labels, and features |
| **EHRSHOT_MEDS.zip** | Dataset in MEDS interoperability format |
| **OMOP tables** | Standard OMOP CDM table exports (Person, Condition, Procedure, Drug, Measurement, etc.) |
| **CLMBR-T-Base model** | 141M-parameter pre-trained clinical foundation model weights |
| **Label files** | Ground truth labels for all 15 prediction tasks |

### Prediction Tasks (15 total)

| Category | Tasks |
|----------|-------|
| **Operational Outcomes (3)** | Long length of stay, 30-day readmission, ICU transfer |
| **Anticipating Lab Results (5)** | Thrombocytopenia, hyperkalemia, hypoglycemia, hyponatremia, anemia |
| **Assignment of New Diagnoses (6)** | Hypertension, hyperlipidemia, pancreatic cancer, celiac disease, lupus, acute MI |
| **Chest X-ray Findings (1)** | 14-way multilabel classification of radiographic findings |

## Important Restrictions

- **Research Use Only**: The dataset and model are licensed separately from the source code. The dataset is available under a research usage agreement; consult the Redivis and Hugging Face pages for specific terms.
- **Source Code License**: The benchmark code is released under the Apache License 2.0.
- **No Re-identification**: Attempting to re-identify patients in the dataset is prohibited.
- **Citation Required**: When using EHRShot in publications, cite the original paper:
  > Wornow M, Thapa R, Steinberg E, Fries J, Shah N. EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models. *arXiv preprint arXiv:2307.02028*. 2023.

## Useful Links

- [EHRShot Documentation Website](https://ehrshot.stanford.edu)
- [EHRShot GitHub Repository](https://github.com/som-shahlab/ehrshot-benchmark)
- [EHRShot Dataset on Redivis](https://redivis.com/datasets/53gc-8rhx41kgt)
- [CLMBR-T-Base Model on Hugging Face](https://huggingface.co/StanfordShahLab/clmbr-t-base)
- [FEMR Framework (GitHub)](https://github.com/som-shahlab/femr)
- [Research Paper (arXiv)](https://arxiv.org/abs/2307.02028)
