# MIMIC-IV Waveform Database — Access Instructions

## Overview

The MIMIC-IV Waveform Database is a collection of physiological waveform recordings from patients admitted to the ICU at Beth Israel Deaconess Medical Center (BIDMC), linked to the MIMIC-IV clinical database. These waveforms are a rich source of high-frequency patient data including ECG (electrocardiogram), PPG (photoplethysmogram), ABP (arterial blood pressure), and other bedside monitor signals. The database also contains time-stamped numeric (vital sign) summaries.

The initial release (**v0.1.0**, released 2022) contains **200 records** from **198 patients**. Larger releases with approximately 10,000 records are planned for subsequent versions. The waveforms can be linked to MIMIC-IV clinical data using `subject_id` and `hadm_id`.

This dataset is hosted on [PhysioNet](https://physionet.org/) and was the subject of a workshop at IEEE EMBC 2022 led by Peter Charlton demonstrating waveform feature extraction with the WFDB-Python package.

## Access Level

**Credentialed Access** — The MIMIC-IV Waveform Database is a restricted-access dataset on PhysioNet. You must complete the standard PhysioNet credentialing process and sign the Data Use Agreement (DUA) for this specific project. The data is free of charge for all approved researchers.

## Prerequisites

Before starting the access process, you will need:

1. A valid email address (institutional/academic email is strongly preferred and speeds up credentialing).
2. An affiliation with a research institution, hospital, or university (students and postdocs need a supervisor/reference).
3. Access to the [CITI Program](https://about.citiprogram.org/) training platform (free for this course).
4. A web browser to complete all registration and training steps.
5. **WFDB-compatible software** to read the waveform files. Recommended: [WFDB-Python](https://github.com/MIT-LCP/wfdb-python) (must be a recent version with FLAC compression support).
6. If you already have PhysioNet credentialed access (e.g., for MIMIC-IV), you only need to sign the DUA for this project (skip to Step 4).

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

1. Navigate to the MIMIC-IV Waveform Database project page: [https://physionet.org/content/mimic4wdb/](https://physionet.org/content/mimic4wdb/).
2. Scroll to the bottom of the page and look for the **"Sign the data use agreement"** prompt.
3. Read the DUA carefully and click to sign. Key provisions include:
   - You will use the data solely for lawful scientific research.
   - You will not attempt to re-identify any individual or institution.
   - You will not share the data with others.
   - You will report any suspected identifiable information to PHI-report@physionet.org.
4. Access is typically granted immediately after signing.

### Step 5: Download the Data

Once you have signed the DUA, you can download the data:

**Option A: Web Browser**
- Go to the Files section on the project page and download individual patient record folders.

**Option B: wget (Recommended for Full Download)**
```bash
wget -r -N -c -np --user <your-username> --ask-password https://physionet.org/files/mimic4wdb/0.1.0/
```

**Option C: WFDB-Python (Programmatic Access)**
```python
import wfdb

# Read a specific record
record = wfdb.rdrecord('record_name', pn_dir='mimic4wdb/0.1.0/waves/p100/p10020306/83404654')
```

## Data Format

### Waveform Files (WFDB Format)

- Waveforms are stored in **WFDB format** — the standard format for physiological signal data used by PhysioNet.
- Each patient record consists of multiple WFDB files:
  - **Header files** (`.hea`) — Describe the signal characteristics (sampling rate, number of channels, signal names, units, gain, baseline).
  - **Signal files** (`.dat`) — Contain the actual waveform sample values.
- Signal files are **FLAC-compressed** (using FLAC v1.3.2-3 with options `-8 -r8 -e`) to reduce storage. You need a recent version of WFDB that supports FLAC compression to read these files.

### Numerics Files (CSV)

- Numeric (vital sign) summaries are stored as **compressed CSV files** using **dictzip** compression (compatible with gzip).
- These contain periodic summary measurements (heart rate, blood pressure, SpO2, etc.) alongside the high-frequency waveform data.

### Signal Types

Common signals found in the waveform records include:

| Signal | Description | Typical Sampling Rate |
|--------|-------------|----------------------|
| ECG (II, V, etc.) | Electrocardiogram leads | 125-500 Hz |
| PPG / SpO2 | Photoplethysmogram / Pulse oximetry waveform | 125 Hz |
| ABP | Arterial blood pressure waveform | 125 Hz |
| RESP | Respiratory waveform | 62.5 Hz |
| CVP | Central venous pressure | 125 Hz |

### Record Splitting

Records are automatically split whenever there is a gap of more than one hour with no waveform or numeric data, to avoid incorrectly combining data from different patients who may have shared the same monitor.

## Key Tables / Files

| File/Directory | Description |
|----------------|-------------|
| `waves/` | Root directory containing all waveform records, organized by patient folder (e.g., `waves/p100/p10020306/`). |
| `RECORDS` | Master list of all available records in the database. |
| `SHA256SUMS.txt` | Checksums for data integrity verification. |
| Per-patient folders | Each folder contains WFDB header (`.hea`) and signal (`.dat`) files for waveform segments, plus compressed CSV numerics files. |

### Patient Identifiers

- `subject_id` — Unique patient identifier (links to MIMIC-IV `patients` table).
- `hadm_id` — Hospital admission identifier (links to MIMIC-IV `admissions` table).
- These identifiers enable linkage of waveform data with the full clinical record in MIMIC-IV.

## Important Restrictions

- **No data sharing.** Each researcher must independently complete credentialing and sign the DUA.
- **No re-identification attempts.** You must not attempt to identify any patients. All protected health information (names, dates of birth, medical record numbers, recording dates) has been removed or replaced with de-identified surrogates using the same date-shift offset as MIMIC-IV.
- **PHI reporting obligation.** If you discover any potentially identifying information, report it to PHI-report@physionet.org.
- **Citation requirement.** You must cite the dataset in any publications. The reference is: Moody B, Hao S, Gow B, Pollard T, Zong W, Mark R. "MIMIC-IV Waveform Database" (version 0.1.0). PhysioNet (2022).
- **Software requirements.** Ensure your WFDB library supports FLAC decompression before attempting to read the signal files.

## Useful Links

- [MIMIC-IV Waveform Database on PhysioNet](https://physionet.org/content/mimic4wdb/)
- [MIMIC-IV Waveform Database v0.1.0](https://physionet.org/content/mimic4wdb/0.1.0/)
- [MIMIC-IV Waveform Announcement](https://physionet.org/news/post/401/)
- [WFDB-Python Package (GitHub)](https://github.com/MIT-LCP/wfdb-python)
- [MIMIC-IV Clinical Database (for linkage)](https://physionet.org/content/mimiciv/)
- [PhysioNet Credentialing Page](https://physionet.org/settings/credentialing/)
- [CITI Program Training](https://about.citiprogram.org/)
- [PhysioNet CITI Course Instructions](https://physionet.org/about/citi-course/)
