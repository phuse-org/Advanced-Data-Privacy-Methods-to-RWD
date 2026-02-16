# MIMIC-IV-Note (Deidentified Free-Text Clinical Notes) — Access Instructions

## Overview

MIMIC-IV-Note is a companion dataset to MIMIC-IV that provides de-identified free-text clinical notes for patients in the MIMIC-IV clinical database. It contains **331,794 discharge summaries** from 145,915 patients and **2,321,355 radiology reports** from 237,427 patients admitted to the Beth Israel Deaconess Medical Center (BIDMC) in Boston, MA, USA. The notes were de-identified using a combined rule-based and neural network approach, with all protected health information (PHI) replaced by three underscores (`___`). MIMIC-IV-Note is hosted on [PhysioNet](https://physionet.org/) and maintained by the MIT Laboratory for Computational Physiology.

The latest version is **MIMIC-IV-Note v2.2** (released January 2023).

## Access Level

**Credentialed Access** — MIMIC-IV-Note is a restricted-access dataset requiring PhysioNet credentialing and an individual Data Use Agreement (DUA). The same credentialing used for MIMIC-IV applies here, but you must separately sign the DUA for this specific project. The data is free of charge for all approved researchers.

## Prerequisites

Before starting the access process, you will need:

1. A valid email address (institutional/academic email is strongly preferred and speeds up credentialing).
2. An affiliation with a research institution, hospital, or university (students and postdocs need a supervisor/reference).
3. Access to the [CITI Program](https://about.citiprogram.org/) training platform (free for this course).
4. A web browser to complete all registration and training steps.
5. If you already have credentialed access for MIMIC-IV on PhysioNet, you only need to sign the DUA for this project (skip to Step 4).

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

1. Navigate to the MIMIC-IV-Note project page: [https://physionet.org/content/mimic-iv-note/](https://physionet.org/content/mimic-iv-note/).
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
- Go to the Files section on the project page and download individual files.

**Option B: wget (Recommended)**
```bash
wget -r -N -c -np --user <your-username> --ask-password https://physionet.org/files/mimic-iv-note/2.2/
```

## Data Format

- All tables are provided as **gzip-compressed CSV files** (`.csv.gz`).
- Each row in the note tables corresponds to a unique clinical note, identified by a unique `note_id`.
- Notes are linkable to MIMIC-IV clinical data via `subject_id` and `hadm_id`.
- PHI has been replaced with exactly three underscores (`___`) wherever it appeared in the original text.

## Key Tables / Files

| Table | Description |
|-------|-------------|
| `discharge` | De-identified discharge summary notes (331,794 notes from 145,915 patients). These are comprehensive narrative summaries written at the end of a hospitalization covering the patient's hospital course. |
| `discharge_detail` | Supplementary metadata associated with discharge summaries. |
| `radiology` | De-identified radiology report notes (2,321,355 notes from 237,427 patients). These are free-text reports generated by radiologists interpreting imaging studies. |
| `radiology_detail` | Supplementary metadata associated with radiology reports, including Current Procedural Terminology (CPT) codes, exam names, and links between parent reports and addendums. |

### Key Columns

- `note_id` — Unique identifier for each note.
- `subject_id` — Patient identifier (links to MIMIC-IV `patients` table).
- `hadm_id` — Hospital admission identifier (links to MIMIC-IV `admissions` table).
- `chartdate` / `charttime` — Date/time associated with the note.
- `text` — The de-identified free-text content of the note.

## Important Restrictions

- **No data sharing.** Each researcher must independently complete credentialing and sign the DUA. You may not redistribute or share the data.
- **No re-identification attempts.** You must not attempt to identify any patients, providers, or institutions from the notes, even using contextual clues in the free text.
- **PHI reporting obligation.** Despite de-identification, if you discover any residual information that could identify an individual, report it immediately to PHI-report@physionet.org.
- **Code availability.** Publications using this data should make analysis code publicly available.
- **Citation requirement.** You must cite the dataset in any resulting publications.
- **Separate DUA from MIMIC-IV.** Even if you have MIMIC-IV access, you must sign the DUA specifically for MIMIC-IV-Note.

## Useful Links

- [MIMIC-IV-Note on PhysioNet (latest version)](https://physionet.org/content/mimic-iv-note/)
- [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/)
- [MIMIC-IV Official Documentation](https://mimic.mit.edu/docs/iv/)
- [PhysioNet Credentialing Page](https://physionet.org/settings/credentialing/)
- [CITI Program Training](https://about.citiprogram.org/)
- [PhysioNet CITI Course Instructions](https://physionet.org/about/citi-course/)
- [MIMIC Code Repository (GitHub)](https://github.com/MIT-LCP/mimic-code)
