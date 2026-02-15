# PhysioNet Challenge Datasets (Computing in Cardiology Challenge Signals) — Access Instructions

## Overview

The George B. Moody PhysioNet Challenges (formerly the PhysioNet/Computing in Cardiology Challenges) are a series of annual competitions that have been running since 2000, co-hosted by [PhysioNet](https://physionet.org/) and Computing in Cardiology (CinC). Each challenge focuses on a clinically important and unsolved question, typically involving physiological signal analysis — most commonly ECG, but also including other modalities such as heart sounds, blood pressure waveforms, EEG, respiratory signals, and structured clinical data.

Each year's challenge produces curated datasets that are generally made available to the research community. These datasets span a wide range of signal types and clinical problems and are a valuable resource for developing and benchmarking algorithms in physiological signal processing and clinical informatics.

The challenges were led by George Moody at the MIT Laboratory for Computational Physiology from 2000 to 2014. Since 2015, they have been led by Gari Clifford at Emory University and the Georgia Institute of Technology.

## Access Level

**Varies by Challenge — Mostly Open Access.** Most PhysioNet Challenge datasets are openly available (no credentialing required), though some may require a free PhysioNet account. A few challenge datasets that include sensitive clinical data may require credentialed access. Always check the specific challenge page for the access requirements of that year's dataset.

## Prerequisites

For most challenge datasets:

1. A web browser to navigate to the specific challenge page.
2. A free PhysioNet account (for some datasets).

For credentialed challenge datasets (rare):

3. CITI "Data or Specimens Only Research" training completion.
4. PhysioNet credentialing (same process as for MIMIC-IV, described below).

## Step-by-Step Access

### Step 1: Identify the Challenge Dataset You Need

Visit the main challenge hub to find the specific challenge year and topic:

- **Current/recent challenges**: [https://moody-challenge.physionet.org/](https://moody-challenge.physionet.org/)
- **Challenge dataset archive**: [https://archive.physionet.org/pn3/challenge/](https://archive.physionet.org/pn3/challenge/)
- **PhysioNet challenge index**: [https://physionet.org/about/challenge/](https://physionet.org/about/challenge/)

### Step 2: Navigate to the Specific Challenge Page

Each challenge has its own page with dataset descriptions, rules, and download links. For example:

- 2024 Challenge (ECG Image Digitization): [https://moody-challenge.physionet.org/2024/](https://moody-challenge.physionet.org/2024/)
- 2025 Challenge (Chagas Disease Detection): [https://moody-challenge.physionet.org/2025/](https://moody-challenge.physionet.org/2025/)

### Step 3: Check Access Requirements

On the challenge page, look for the data download section. Most challenge datasets fall into one of these categories:

- **Open Access**: Download directly — no account needed.
- **PhysioNet Account Required**: Create a free account at [https://physionet.org/register/](https://physionet.org/register/) and log in before downloading.
- **Credentialed Access**: Complete the full PhysioNet credentialing process (see Step 4 below).

### Step 4: Complete Credentialing (If Required)

If the specific challenge dataset requires credentialed access:

1. Create a PhysioNet account at [https://physionet.org/register/](https://physionet.org/register/).
2. Go to [https://about.citiprogram.org/](https://about.citiprogram.org/) and create a CITI Program account.
3. Add the **"Massachusetts Institute of Technology Affiliates"** affiliation and complete the **"Data or Specimens Only Research"** course.
4. Download your CITI **Completion Report** (not the certificate) from the Records section.
5. Go to [https://physionet.org/settings/credentialing/](https://physionet.org/settings/credentialing/) and submit your credentialing application with the CITI report.
6. Once approved, navigate to the specific dataset page and sign the Data Use Agreement (DUA).

### Step 5: Download the Data

Challenge datasets can typically be downloaded via:

**Option A: Direct Download from Challenge Page**
- Most challenge pages provide direct download links to the training and test data.

**Option B: PhysioNet Files Interface**
- Navigate to the dataset on PhysioNet and use the Files section.

**Option C: wget**
```bash
# Example for a specific challenge dataset (replace URL with the actual dataset URL)
wget -r -N -c -np https://physionet.org/files/<challenge-dataset-slug>/<version>/
```

**Option D: PhysioNet Archive**
- Older challenge datasets may be available from: [https://archive.physionet.org/pn3/challenge/](https://archive.physionet.org/pn3/challenge/)

## Data Format

Data formats vary by challenge year and topic. Common formats include:

| Format | Typical Use | Description |
|--------|-------------|-------------|
| **WFDB** (`.hea`, `.dat`) | ECG, waveform signals | Standard PhysioNet waveform format with header and binary signal files. |
| **CSV** | Clinical data, labels | Comma-separated values for structured data. |
| **WAV** | Heart sound recordings | Audio waveform files (e.g., 2016 heart sound challenge). |
| **MAT** | Signal data | MATLAB-formatted data files. |
| **PNG / JPG** | ECG images | Image files (e.g., 2024 ECG digitization challenge). |
| **TSV** | Annotations, labels | Tab-separated values. |

## Key Tables / Files

| Year | Topic | Signal Type |
|------|-------|-------------|
| 2000 | Detecting Sleep Apnea from the ECG | ECG |
| 2001 | Predicting Paroxysmal Atrial Fibrillation | ECG (RR intervals) |
| 2002 | RR Interval Time Series Modeling | RR intervals |
| 2003 | Distinguishing Ischemic from Non-Ischemic ST Changes | ECG |
| 2004 | Spontaneous Termination of Atrial Fibrillation | ECG |
| 2006 | QT Interval Measurement | ECG |
| 2008 | Detecting and Quantifying T-Wave Alternans | ECG |
| 2009 | Predicting Acute Hypotensive Episodes | Multi-parameter (ECG, ABP, etc.) |
| 2010 | Mind the Gap (Filling Gaps in Physiologic Data) | Multi-parameter |
| 2011 | Improving ECG Quality from Mobile Phones | ECG |
| 2012 | Predicting Mortality of ICU Patients | Clinical data + signals |
| 2013 | Noninvasive Fetal ECG | ECG (abdominal) |
| 2014 | Robust Detection of Heart Beats in Multimodal Data | ECG, ABP, PPG |
| 2015 | Reducing False Arrhythmia Alarms in the ICU | Multi-parameter ICU |
| 2016 | Classification of Heart Sound Recordings | Phonocardiogram (PCG) |
| 2017 | AF Classification from Short Single-Lead ECG | ECG (single-lead, AliveCor) |
| 2018 | You Snooze, You Win (Sleep Staging) | PSG (EEG, EOG, EMG) |
| 2019 | Early Prediction of Sepsis from Clinical Data | Clinical time series |
| 2020 | Classification of 12-lead ECGs | 12-lead ECG |
| 2021 | Varying Dimensions in Electrocardiography | Reduced-lead ECG |
| 2022 | Heart Murmur Detection from Auscultation | Phonocardiogram (PCG) |
| 2023 | Predicting Neurological Recovery from Coma | EEG |
| 2024 | Digitization and Classification of ECG Images | ECG images + signals |
| 2025 | Detecting Chagas Disease from ECGs | 12-lead ECG |

## Important Restrictions

- **Varies by challenge.** Each challenge dataset may have its own specific terms and conditions. Always read the data description and any associated DUA on the challenge page.
- **General PhysioNet terms apply.** For datasets hosted on PhysioNet, standard PhysioNet data use terms apply: use for research purposes only, no re-identification attempts, report any discovered PHI.
- **Challenge participation rules.** If you are participating in an active challenge (not just using historical data), you must follow the challenge rules regarding team registration, code submission, and abstract submission to Computing in Cardiology.
- **Citation requirements.** Each challenge has specific citation requirements. Typically you should cite both the challenge description paper and the specific dataset used.
- **Open-source code.** Many challenge participants are required to make their code open-source, and these contributions are archived alongside the datasets.

## Useful Links

- [George B. Moody PhysioNet Challenge Hub](https://moody-challenge.physionet.org/)
- [PhysioNet Challenge Index](https://physionet.org/about/challenge/)
- [PhysioNet Challenge Dataset Archive](https://archive.physionet.org/pn3/challenge/)
- [Challenge Overview and History](https://physionet.org/about/challenge/moody-challenge-overview)
- [2024 Challenge — ECG Image Digitization](https://moody-challenge.physionet.org/2024/)
- [2025 Challenge — Chagas Disease Detection](https://moody-challenge.physionet.org/2025/)
- [PhysioNet Main Site](https://physionet.org/)
- [PhysioNet Account Registration](https://physionet.org/register/)
- [PhysioNet Credentialing Page](https://physionet.org/settings/credentialing/)
- [CITI Program Training](https://about.citiprogram.org/)
- [Computing in Cardiology](https://www.cinc.org/)
