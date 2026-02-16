# MIMIC-IV (Medical Information Mart for Intensive Care IV) — Access Instructions

## Overview

MIMIC-IV is a large, freely available database of de-identified electronic health records (EHR) from patients admitted to the Beth Israel Deaconess Medical Center (BIDMC) in Boston, MA, USA. The database covers hospital admissions from 2008 through 2022 and includes a wide range of clinical data such as demographics, vital signs, laboratory measurements, medications, procedures, diagnoses, and ICU-specific charted events. MIMIC-IV is hosted on [PhysioNet](https://physionet.org/) and maintained by the MIT Laboratory for Computational Physiology (LCP).

The latest version is **MIMIC-IV v3.1** (released October 11, 2024), which contains records for **364,627 unique patients**, **546,028 hospitalizations**, and **94,458 unique ICU stays**.

## Access Level

**Credentialed Access** — MIMIC-IV is a restricted-access dataset. You must create a PhysioNet account, complete human subjects research training, become a credentialed PhysioNet user, and sign an individual Data Use Agreement (DUA) before you can access the data. The data is free of charge for all approved researchers.

## Prerequisites

Before starting the access process, you will need:

1. A valid email address (institutional/academic email is strongly preferred and speeds up credentialing).
2. An affiliation with a research institution, hospital, or university (students and postdocs need a supervisor/reference).
3. Access to the [CITI Program](https://about.citiprogram.org/) training platform (free for this course).
4. A web browser to complete all registration and training steps.

## Step-by-Step Access

### Step 1: Create a PhysioNet Account

1. Go to [https://physionet.org/register/](https://physionet.org/register/).
2. Fill in your name, email address, and create a password.
3. Confirm your email address by clicking the verification link sent to your inbox.
4. Log in to your new PhysioNet account.

### Step 2: Complete CITI "Data or Specimens Only Research" Training

1. Go to [https://about.citiprogram.org/](https://about.citiprogram.org/) and create a CITI Program account (if you do not already have one).
2. Once logged in, navigate to **My Courses** and click **Add Affiliation**.
3. Search for and select **"Massachusetts Institute of Technology Affiliates"** as your affiliation. This is the designated affiliation for non-MIT personnel who need this course to access PhysioNet data. Do not register as an independent learner, as this may incur fees.
4. Enroll in and complete the **"Data or Specimens Only Research"** course.
5. After completion, go to **Records** at the top of the CITI website. Under your completion record, click **View-Print-Share** and download your **Completion Report** (not the certificate — PhysioNet requires the full report).

### Step 3: Submit Credentialing Application on PhysioNet

1. Log in to PhysioNet and navigate to [https://physionet.org/settings/credentialing/](https://physionet.org/settings/credentialing/).
2. Fill in your personal details, institutional affiliation, and research purpose.
3. If you are a student or postdoc, provide your supervisor's name and contact information in the reference section.
4. Upload your CITI **Completion Report** (the full report PDF, not just the certificate).
5. Submit your credentialing application.
6. Wait for PhysioNet administrators to review and approve your application. This typically takes a few business days but may vary. Having an academic email and/or ORCID linked to your institution can expedite the process.

### Step 4: Sign the Data Use Agreement (DUA)

1. Once your credentialing is approved, navigate to the MIMIC-IV project page: [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/).
2. Scroll to the bottom of the page. You will see a section prompting you to **"Sign the data use agreement"**.
3. Read the DUA carefully and click to sign. Key provisions include:
   - You will use the data solely for lawful scientific research.
   - You will not attempt to re-identify any individual or institution.
   - You will not share the data with others (each user must obtain their own access).
   - You will report any suspected identifiable information to PHI-report@physionet.org.
4. Once signed, access is typically granted immediately.

### Step 5: Download the Data

Once you have signed the DUA, you can download the data using any of these methods:

**Option A: Web Browser**
- Go to the Files section on the MIMIC-IV project page and download individual files or folders.

**Option B: wget (Recommended for Full Download)**
```bash
wget -r -N -c -np --user <your-username> --ask-password https://physionet.org/files/mimiciv/3.1/
```

**Option C: Google Cloud / BigQuery**
- MIMIC-IV v3.1 is also available on Google BigQuery for cloud-based querying without downloading.
- See: [https://mimic.mit.edu/docs/gettingstarted/cloud/](https://mimic.mit.edu/docs/gettingstarted/cloud/)

## Data Format

- All tables are provided as **gzip-compressed CSV files** (`.csv.gz`).
- The database uses a relational structure with foreign keys linking tables (e.g., `subject_id`, `hadm_id`, `stay_id`).
- Data is organized into two modules: **hosp** (hospital-wide EHR) and **icu** (ICU clinical information system).

## Key Tables / Files

### hosp Module (Hospital-wide EHR Data)

| Table | Description |
|-------|-------------|
| `patients` | Patient demographics (age, gender, anchor year) |
| `admissions` | Hospital admission and discharge details |
| `transfers` | Intra-hospital unit transfers |
| `diagnoses_icd` | Billed ICD-9/ICD-10 diagnosis codes |
| `procedures_icd` | Billed ICD-9/ICD-10 procedure codes |
| `labevents` | Laboratory test results |
| `microbiologyevents` | Microbiology culture results |
| `prescriptions` | Medication prescriptions |
| `pharmacy` | Pharmacy dispensing information |
| `emar` | Electronic Medication Administration Record |
| `emar_detail` | Supplementary medication administration details |
| `poe` | Provider order entry |
| `poe_detail` | Provider order entry details |
| `services` | Hospital service assignments |
| `drgcodes` | Diagnosis-Related Group codes |
| `hcpcsevents` | HCPCS/CPT billed events |
| `omr` | Online Medical Record (height, weight, BMI, blood pressure) |
| `d_labitems` | Dictionary of laboratory item IDs |
| `d_icd_diagnoses` | Dictionary of ICD diagnosis codes |
| `d_icd_procedures` | Dictionary of ICD procedure codes |
| `d_hcpcs` | Dictionary of HCPCS codes |
| `provider` | De-identified provider identifiers |

### icu Module (ICU Clinical Information System — MetaVision)

| Table | Description |
|-------|-------------|
| `icustays` | ICU stay tracking (admission/discharge times, unit type) |
| `chartevents` | Charted observations (vitals, assessments, scores) |
| `inputevents` | IV fluid and medication inputs |
| `ingredientevents` | Ingredients of IV administrations |
| `outputevents` | Patient outputs (urine, drainage, etc.) |
| `procedureevents` | Procedures documented during ICU stay |
| `datetimeevents` | Date/time-type observations |
| `d_items` | Dictionary of item IDs used in ICU events tables |
| `caregiver` | De-identified caregiver identifiers |

## Important Restrictions

- **No data sharing.** Each researcher must independently complete credentialing and sign the DUA. You may not redistribute or share the downloaded data with anyone, including collaborators.
- **No re-identification attempts.** You must not attempt to identify any patients, providers, or institutions in the data.
- **PHI reporting obligation.** If you discover any information that could potentially identify an individual, you must report it to PHI-report@physionet.org.
- **Code availability.** Any publication using MIMIC-IV data should also make the relevant analysis code publicly available.
- **Citation requirement.** You must cite the dataset in any resulting publications.

## Useful Links

- [MIMIC-IV on PhysioNet (latest version)](https://physionet.org/content/mimiciv/)
- [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC-IV Official Documentation](https://mimic.mit.edu/docs/iv/)
- [MIMIC-IV on BigQuery (Cloud Access)](https://mimic.mit.edu/docs/gettingstarted/cloud/)
- [PhysioNet Credentialing Page](https://physionet.org/settings/credentialing/)
- [CITI Program Training](https://about.citiprogram.org/)
- [PhysioNet CITI Course Instructions](https://physionet.org/about/citi-course/)
- [MIMIC Code Repository (GitHub)](https://github.com/MIT-LCP/mimic-code)
- [MIMIC-IV Nature Scientific Data Paper](https://www.nature.com/articles/s41597-022-01899-x)
