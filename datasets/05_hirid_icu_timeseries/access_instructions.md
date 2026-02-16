# HiRID (High Time-Resolution ICU Dataset) — Access Instructions

## Overview

HiRID is a freely accessible critical care dataset containing de-identified data from nearly **34,000 patient admissions** to the Department of Intensive Care Medicine of the Bern University Hospital, Switzerland — an interdisciplinary 60-bed ICU admitting more than 6,500 patients per year. The dataset covers admissions from January 2008 to June 2016 and includes de-identified demographic information along with **681 routinely collected physiological variables**, diagnostic test results, and treatment parameters. HiRID's distinguishing feature is its uniquely high time resolution, with most bedside monitoring data recorded **every 2 minutes** — higher than any other published ICU dataset.

HiRID was developed as a collaboration between Bern University Hospital and the Swiss Federal Institute of Technology (ETH Zurich). It is hosted on [PhysioNet](https://physionet.org/).

The latest version is **HiRID v1.1.1** (released 2021).

## Access Level

**Credentialed Access** — HiRID is a restricted-access dataset on PhysioNet. You must create a PhysioNet account, complete human subjects research training, become a credentialed user, and sign an individual Data Use Agreement (DUA). The data is free of charge for approved researchers.

## Prerequisites

Before starting the access process, you will need:

1. A valid email address (institutional/academic email is strongly preferred and speeds up credentialing).
2. An affiliation with a research institution, hospital, or university (students and postdocs need a supervisor/reference).
3. Access to the [CITI Program](https://about.citiprogram.org/) training platform (free for this course).
4. A web browser to complete all registration and training steps.
5. If you already have PhysioNet credentialed access (e.g., for MIMIC-IV), you only need to sign the DUA for this project (skip to Step 4).

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

1. Navigate to the HiRID project page: [https://physionet.org/content/hirid/](https://physionet.org/content/hirid/).
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
- Go to the Files section on the HiRID project page and download individual compressed archives.

**Option B: wget (Recommended for Full Download)**
```bash
wget -r -N -c -np --user <your-username> --ask-password https://physionet.org/files/hirid/1.1.1/
```

## Data Format

HiRID data is available in two formats:

- **Apache Parquet** (`.parquet`) — Strongly typed binary columnar format, recommended for performance. Supported by pandas, Spark, R, MATLAB, and many other tools.
- **Compressed CSV** (`.csv.gz`) — Standard comma-separated values for broad compatibility.

The data uses a **long table format** where each row represents a single measurement: the value of a specific variable at a specific time for a specific patient.

### Data States

The data is provided in two processing states:

1. **Raw data** — Minimally processed (only de-identification changes). Contains all 681 source variables.
2. **Pre-processed data** — Intermediary pipeline stages from the accompanying Nature Medicine publication. Source variables representing the same clinical concepts are merged into meta-variables. Contains the 18 most predictive meta-variables.

### File Partitioning

The data is split into partitions for parallel processing. A lookup table (`{data_set}_index.csv`) maps patient IDs to partition IDs. Partitions are aligned across tables so that a given patient's data is always in the same partition ID.

## Key Tables / Files

### Raw Data Tables

| File/Table | Description |
|------------|-------------|
| `observation_tables_parquet.tar.gz` | Observation data (vital signs, lab results, monitoring data) for all patients. 681 variables at 2-minute resolution. |
| `pharma_records_parquet.tar.gz` | Pharmaceutical records (drug administrations, infusions, treatments). |
| `reference_data.tar.gz` | General patient data (demographics, admission/discharge info) and reference tables. |

### Reference Tables

| File | Description |
|------|-------------|
| `hirid_variable_reference.csv` | Variable reference lookup table for raw data (maps variable IDs to names and units). |
| `ordinal_vars_ref.csv` | Reference table for categorical/ordinal variables (maps coded values to string labels). |
| `hirid_variable_reference_preprocessed.csv` | Variable reference table for the pre-processed data (merged meta-variables). |

### Pre-processed Data

| File/Table | Description |
|------------|-------------|
| Merged stage | Source variables merged into meta-variables by clinical concept. |
| Imputed stage | Merged data with imputation applied. |

## Important Restrictions

- **No data sharing.** Each researcher must independently complete credentialing and sign the DUA. You may not redistribute the data.
- **No re-identification attempts.** You must not attempt to identify any patients or institutions.
- **PHI reporting obligation.** If you discover any information that could identify an individual, report it to PHI-report@physionet.org.
- **Citation requirement.** You must cite the dataset in any resulting publications. The reference is: Faltys M, Zimmermann M, Lyu X, Huser M, Hyland S, Ratsch G, Merz T. "HiRID, a high time-resolution ICU dataset" (version 1.1.1). PhysioNet (2021).
- **Swiss data origin.** Note that HiRID originates from a Swiss hospital, which may have implications for certain regulatory frameworks (e.g., Swiss data protection law in addition to HIPAA-style de-identification).

## Useful Links

- [HiRID on PhysioNet (latest version)](https://physionet.org/content/hirid/)
- [HiRID v1.1.1](https://physionet.org/content/hirid/1.1.1/)
- [HiRID Official Documentation](https://hirid.intensivecare.ai/)
- [HiRID Data Structure Details](https://hirid.intensivecare.ai/structure-of-the-published-data)
- [HiRID Data Details](https://hirid.intensivecare.ai/data-details)
- [HiRID Getting Started Guide](https://hirid.intensivecare.ai/getting-started)
- [HiRID-ICU-Benchmark (GitHub)](https://github.com/ratschlab/HIRID-ICU-Benchmark)
- [PhysioNet Credentialing Page](https://physionet.org/settings/credentialing/)
- [CITI Program Training](https://about.citiprogram.org/)
- [PhysioNet CITI Course Instructions](https://physionet.org/about/citi-course/)
