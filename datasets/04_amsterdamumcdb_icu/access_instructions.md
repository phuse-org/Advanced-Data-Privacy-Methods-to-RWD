# AmsterdamUMCdb (Amsterdam University Medical Centers Database) -- Access Instructions

## Overview

AmsterdamUMCdb is the first freely accessible European intensive care database. It is hosted by Amsterdam Medical Data Science at Amsterdam University Medical Centers (Amsterdam UMC). The database contains de-identified health data from 23,106 intensive care unit (ICU) and high dependency unit (HDU) admissions of 20,109 adult patients admitted between 2003 and 2016, totaling approximately 1 billion clinical data points. The data includes demographics, vital signs, laboratory tests, medications, and other clinical observations recorded during ICU stays.

## Access Level

**Controlled Access** -- AmsterdamUMCdb is free of charge, but access requires completing a recognized ethics/research training course, signing a Data Use Agreement (End User License Agreement), and having a practicing intensivist serve as a reference. Applications are reviewed and approved by the database administrators. The database fully complies with the European General Data Protection Regulation (GDPR) and U.S. HIPAA standards; an independent privacy audit concluded that re-identification is not reasonably likely.

## Prerequisites

Before starting the access request process, ensure you have the following:

- **Ethics / Research Training Certificate**: A valid certificate from one of the following recognized courses:
  - CITI Program "Data or Specimens Only Research" (DSOR) course (available free of charge at [citiprogram.org](https://about.citiprogram.org/))
  - NFU Basic Course for Clinical Investigators (BROK)
  - An equivalent institutional ethics training course
- **Practicing Intensivist Reference**: You need a practicing intensivist who is willing to serve as your reference and co-sign your application. The intensivist must be easily identifiable through an online directory, institutional web page, or equivalent source.
- **Institutional Affiliation**: While not strictly required, having a research institution affiliation strengthens your application.

## Step-by-Step Access

### Step 1: Complete the Required Training Course

Enroll in and complete the CITI Program DSOR course (or equivalent). This same training is also required for accessing other ICU databases such as MIMIC and eICU. Upon completion, download or save your certificate of completion, as you will need to submit it with your application.

### Step 2: Download and Complete the Access Request Form

Download the Access Request Form and End User License Agreement from the Amsterdam Medical Data Science website or directly from the supplementary materials of the original publication. The form is also available as a PDF at: [Access Request Form (PDF)](https://cdn-links.lww.com/permalink/ccm/g/ccm_49_6_2021_01_18_thoral_ccmed-d-20-02227_sdc3.pdf).

Fill in all required fields, including your personal details, institutional affiliation, and a description of your intended research use.

### Step 3: Obtain Your Intensivist Reference Signature

Have a practicing intensivist review and co-sign the Access Request Form. Both you (the main applicant) and your reference must agree to the terms of the End User License Agreement and share the associated responsibilities.

### Step 4: Submit Your Application

Email the completed and signed Access Request Form along with your training course certificate to:

**access@amsterdammedicaldatascience.nl**

Ensure that all fields are filled in and all required signatures are present. Incomplete applications will not be processed.

### Step 5: Await Approval

Allow up to five (5) business days for review and approval of your application. You will receive confirmation via email once your access has been granted.

### Step 6: Download the Data

Upon approval, AmsterdamUMCdb will be made available for download through EASY at DANS (Data Archiving and Networked Services), the Netherlands institute for permanent access to digital research resources. Follow the download instructions provided in your approval email.

## Data Format

AmsterdamUMCdb version 1.5.0 (released May 2024) is provided in the **OMOP Common Data Model version 5.4** format. The data can be loaded into a relational database management system, with **PostgreSQL** being the recommended and tested platform. Earlier versions were also available as CSV files.

Key technical details:
- **Storage**: Approximately 300 GB after successful conversion (600 GB during conversion)
- **Recommended RDBMS**: PostgreSQL
- **Python package**: Available via PyPI (`pip install amsterdamumcdb`)
- **OMOP CDM conversion**: Via the AMSTEL R package and ETL pipeline

## Key Tables / Files

Since AmsterdamUMCdb focuses on ICU admissions, the following OMOP CDM tables are populated:

| Table | Description |
|-------|-------------|
| **Person** | Patient demographics and death records for ICU-admitted patients |
| **Visit Occurrence** | ICU admission and discharge events |
| **Condition Occurrence** | Primary/secondary admission diagnoses and event-based conditions |
| **Procedure Occurrence** | Procedures performed before (e.g., surgery) or during ICU admission |
| **Drug Exposure** | Medication administrations during ICU stay |
| **Measurement** | Vital signs, laboratory results, and clinical observations |
| **Observation** | Additional clinical observations |
| **Condition Era** | Spans of time with assumed conditions |
| **Drug Era** | Spans of time with assumed drug exposures |

A data dictionary is available via the `amsterdamumcdb.get_dictionary()` Python function or the `dictionary.csv` file included with the dataset.

## Important Restrictions

- **No Re-identification**: You must avoid any attempt to re-identify any individual or entity in AmsterdamUMCdb.
- **No Unauthorized Sharing**: You may not share access to the database or any portion of it with anyone whose own access request has not been approved by the administrators.
- **Breach Notification**: You must immediately notify the administrators if you suspect unauthorized access or any possibility of re-identification.
- **Respectful Use**: Although de-identified, the data contains detailed clinical care information and must be treated with appropriate care and respect.
- **Citation Requirement**: When using AmsterdamUMCdb in research, you must cite the original publication:
  > Thoral PJ, et al. Sharing ICU Patient Data Responsibly Under the Society of Critical Care Medicine/European Society of Intensive Care Medicine Joint Data Science Collaboration: The Amsterdam University Medical Centers Database (AmsterdamUMCdb) Example. *Crit Care Med.* 2021 Jun 1;49(6):e563-e577. doi: 10.1097/CCM.0000000000004916.

## Useful Links

- [Amsterdam Medical Data Science -- AmsterdamUMCdb](https://amsterdammedicaldatascience.nl/#amsterdamumcdb)
- [AmsterdamUMCdb GitHub Repository](https://github.com/AmsterdamUMC/AmsterdamUMCdb)
- [AmsterdamUMCdb GitHub Wiki](https://github.com/AmsterdamUMC/AmsterdamUMCdb/wiki)
- [AMSTEL -- OMOP CDM Conversion Repository](https://github.com/AmsterdamUMC/AMSTEL)
- [AMSTEL Documentation](https://amsterdamumc.github.io/AMSTEL/index.html)
- [AmsterdamUMCdb Python Package (PyPI)](https://pypi.org/project/amsterdamumcdb/)
- [Access Request Form (PDF)](https://cdn-links.lww.com/permalink/ccm/g/ccm_49_6_2021_01_18_thoral_ccmed-d-20-02227_sdc3.pdf)
- [Original Publication (PubMed Central)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8132908/)
- [CITI Program Training](https://about.citiprogram.org/)
