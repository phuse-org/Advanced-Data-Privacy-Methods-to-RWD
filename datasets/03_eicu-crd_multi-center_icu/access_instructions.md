# eICU Collaborative Research Database (eICU-CRD) — Access Instructions

## Overview

The eICU Collaborative Research Database (eICU-CRD) is a large, multi-center critical care database comprising de-identified health data from patients admitted to ICUs across the United States. The database was created through the Philips eICU Research Institute program, which collects data from participating ICUs via the eICU telehealth system. eICU-CRD contains data for over **200,000 patient unit encounters** from more than **139,000 unique patients** admitted between 2014 and 2015, spanning **335 ICU units** at **208 hospitals** located throughout the US. It is hosted on [PhysioNet](https://physionet.org/) and documented at [eicu-crd.mit.edu](https://eicu-crd.mit.edu/).

The current version is **eICU-CRD v2.0** (released May 17, 2018).

## Access Level

**Credentialed Access** — eICU-CRD is a restricted-access dataset. You must create a PhysioNet account, complete human subjects research training, become a credentialed PhysioNet user, and sign an individual Data Use Agreement (DUA) before accessing the data. The data is free of charge for all approved researchers.

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

1. Navigate to the eICU-CRD project page: [https://physionet.org/content/eicu-crd/](https://physionet.org/content/eicu-crd/).
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
- Go to the Files section on the eICU-CRD project page and download individual CSV files.

**Option B: wget (Recommended for Full Download)**
```bash
wget -r -N -c -np --user <your-username> --ask-password https://physionet.org/files/eicu-crd/2.0/
```

**Option C: Google Cloud / BigQuery**
- eICU-CRD is also available on Google BigQuery for cloud-based querying.
- See: [https://eicu-crd.mit.edu/gettingstarted/cloud/](https://eicu-crd.mit.edu/gettingstarted/cloud/)

## Data Format

- All tables are provided as **gzip-compressed CSV files** (`.csv.gz`).
- The database uses a highly denormalized relational schema where most tables can be independently accessed and linked through the central `patient` table.
- Key linking identifiers include `patientunitstayid` (unique ICU stay), `patienthealthsystemstayid` (hospital admission), and `hospitalid` (hospital).
- Vital signs in the database were originally collected at 1-minute intervals, with 5-minute medians archived in eICU-CRD.

## Key Tables / Files

The eICU-CRD contains **31 tables**:

### Core Tables

| Table | Description |
|-------|-------------|
| `patient` | Core table defining patients, hospital admissions, and ICU stays |
| `hospital` | Hospital information (region, number of beds, teaching status) |

### Clinical Data Tables

| Table | Description |
|-------|-------------|
| `diagnosis` | Patient diagnoses from structured problem list |
| `treatment` | Treatment information from structured list |
| `medication` | Medication orders |
| `infusionDrug` | Infusion drug data |
| `lab` | Standard laboratory measurements (~160 harmonized lab types) |
| `customLab` | Custom/non-standard laboratory results |
| `microLab` | Microbiology lab results |

### Vital Signs Tables

| Table | Description |
|-------|-------------|
| `vitalPeriodic` | Periodic vital signs (heart rate, SpO2, respiratory rate, etc.) |
| `vitalAperiodic` | Aperiodic vital signs (non-invasive BP, etc.) |
| `nurseCharting` | Nurse charting data (additional vital signs and observations) |

### Assessment and Documentation Tables

| Table | Description |
|-------|-------------|
| `nurseAssessment` | Nurse assessment data |
| `nurseCare` | Nurse care data |
| `physicalExam` | Physical examination findings |
| `note` | Clinical notes |
| `pastHistory` | Patient past medical history |
| `admissionDx` | Admission diagnoses (APACHE) |
| `admissionDrug` | Drugs on admission |
| `allergy` | Patient allergy information |

### APACHE Scoring Tables

| Table | Description |
|-------|-------------|
| `apacheApsVar` | APACHE Acute Physiology Score variables |
| `apachePatientResult` | APACHE patient results/severity scores |
| `apachePredVar` | APACHE prediction variables |

### Care Plan Tables

| Table | Description |
|-------|-------------|
| `carePlanGeneral` | General care plan entries |
| `carePlanGoal` | Care plan goals |
| `carePlanCareProvider` | Care plan care provider information |
| `carePlanEOL` | Care plan end-of-life documentation |
| `carePlanInfectiousDisease` | Care plan infectious disease entries |

### Respiratory Tables

| Table | Description |
|-------|-------------|
| `respiratoryCare` | Respiratory care data |
| `respiratoryCharting` | Respiratory charting data |

### Other Tables

| Table | Description |
|-------|-------------|
| `intakeOutput` | Intake and output data |

## Important Restrictions

- **No data sharing.** Each researcher must independently complete credentialing and sign the DUA. You may not redistribute or share the downloaded data.
- **No re-identification attempts.** You must not attempt to identify any patients, providers, or institutions. The data has been de-identified in accordance with HIPAA Safe Harbor provisions.
- **PHI reporting obligation.** If you discover any information that could identify an individual, report it to PHI-report@physionet.org.
- **Code availability.** Publications using this data should make analysis code publicly available.
- **Citation requirement.** You must cite the dataset in any publications. The reference is: Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. "The eICU Collaborative Research Database, a freely available multi-center database for critical care research." Scientific Data (2018). DOI: 10.1038/sdata.2018.178.
- **Multi-center considerations.** Because eICU-CRD spans 208 hospitals, be aware that data completeness and documentation practices may vary across sites.

## Useful Links

- [eICU-CRD on PhysioNet (latest version)](https://physionet.org/content/eicu-crd/)
- [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/)
- [eICU-CRD Official Documentation](https://eicu-crd.mit.edu/)
- [eICU-CRD Schema Diagram](https://eicu-crd.mit.edu/about/schema/)
- [eICU-CRD Demo Dataset (Open Access)](https://physionet.org/content/eicu-crd-demo/)
- [PhysioNet Credentialing Page](https://physionet.org/settings/credentialing/)
- [CITI Program Training](https://about.citiprogram.org/)
- [PhysioNet CITI Course Instructions](https://physionet.org/about/citi-course/)
- [eICU-CRD Scientific Data Paper](https://www.nature.com/articles/sdata2018178)
