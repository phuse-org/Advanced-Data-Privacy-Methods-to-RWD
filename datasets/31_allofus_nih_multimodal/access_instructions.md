# NIH All of Us Research Program -- Access Instructions

## Overview

The All of Us Research Program is a landmark longitudinal cohort study led by the National Institutes of Health (NIH) that aims to advance precision medicine by partnering with over one million participants from across the United States. The program prioritizes enrolling individuals historically underrepresented in biomedical research -- currently over 45% of participants are from racial and ethnic minorities, and more than 80% are from groups underrepresented in biomedical research. The dataset includes electronic health records (EHRs), participant surveys, physical measurements, genomic data (whole genome sequencing and genotyping arrays), wearable device data (Fitbit), and linked biospecimen information. All data is accessed through the cloud-based Researcher Workbench, a secure analysis environment powered by Google Cloud Platform.

## Access Level

**Controlled Access -- Tiered Registration** -- All of Us uses a tiered access model with three data tiers:

- **Public Tier**: Aggregate, de-identified data available to anyone through the Data Browser and Data Snapshots. No registration required.
- **Registered Tier**: Individual-level data from EHRs, surveys, physical measurements, and wearable devices. Requires Researcher Workbench registration, identity verification, training, and signing the Data User Code of Conduct.
- **Controlled Tier**: Everything in the Registered Tier plus genomic data (WGS, genotyping arrays), unshifted dates, and previously suppressed demographic fields. Requires institutional Data Use and Registration Agreement (DURA) with All of Us, plus additional Controlled Tier training.

Access is free of charge. Computation costs on Google Cloud Platform are the researcher's responsibility, though $300 in initial credits are provided.

## Prerequisites

Before registering for the Researcher Workbench, ensure you have:

- **Institutional Affiliation**: A recognized research institution (academic, government, non-profit, or industry).
- **U.S. Identity Verification**: A Social Security Number (SSN) and valid U.S. government-issued ID for Login.gov verification. International researchers without a SSN can use ID.me for identity verification.
- **Institutional DURA** (for Controlled Tier): Your institution must have a Data Use and Registration Agreement in place with All of Us that includes Controlled Tier access. If your institution does not yet have a DURA, you can initiate the process through the All of Us website.
- **Responsible Conduct of Research Training**: Willingness to complete the All of Us Responsible Conduct of Research training module (provided within the Workbench).
- **Compute Budget** (optional): While $300 in free Google Cloud credits are provided, larger analyses may require additional funding for cloud compute and storage.

## Step-by-Step Access

### Step 1: Create a Researcher Workbench Profile

Navigate to [https://www.researchallofus.org/register/](https://www.researchallofus.org/register/) and create your Researcher Workbench account. Complete your researcher profile, including your name, institutional affiliation, research interests, and intended use of the data. You must sign the Terms of Service and agree to the Privacy Policy.

### Step 2: Verify Your Identity

Verify your identity using one of two methods:
- **Login.gov**: Available for researchers in the United States who have a Social Security Number and a valid U.S. government-issued ID.
- **ID.me**: Available for researchers who do not have a SSN or who are located outside the United States.

This step confirms that you are who you claim to be and is a one-time requirement.

### Step 3: Complete Responsible Conduct of Research Training

Complete the Responsible Conduct of Research Training module provided within the Researcher Workbench. This training covers ethical considerations for conducting research with data from All of Us participants, including privacy protections, responsible data use, and participant respect. This training must be renewed annually.

### Step 4: Sign the Data User Code of Conduct

Review and sign the Data User Code of Conduct, which outlines the program's expectations for researchers, including commitments to responsible use, participant privacy, and appropriate publication practices.

After completing Steps 1-4, you will have **Registered Tier** access.

### Step 5: Obtain Controlled Tier Access (for Genomic Data)

To access genomic data and other Controlled Tier resources:

1. **Institutional DURA**: Confirm that your institution has a DURA in place with All of Us that covers Controlled Tier access. If not, work with your institutional officials to initiate the DURA process through the All of Us website.
2. **Controlled Tier Training**: Complete the additional Controlled Tier Training module within the Researcher Workbench. This training must be renewed annually.
3. **Institutional Confirmation**: Your institution will confirm your access authorization on a monthly basis.

### Step 6: Create a Workspace and Begin Research

Once registered, create a workspace in the Researcher Workbench. Each workspace requires a description of your research purpose. You can then use Jupyter notebooks (Python, R, or SAS) to query, analyze, and visualize the data within the secure cloud environment.

## Data Format

All of Us data is organized according to the **OMOP Common Data Model (CDM) version 5.3**, maintained by the OHDSI community. Genomic and wearable data use specialized formats.

| Data Type | Format |
|-----------|--------|
| **EHR Data** | OMOP CDM v5.3 tables (queryable via SQL, Python, R within the Workbench) |
| **Survey Data** | OMOP CDM tables (PPI vocabulary mapped to LOINC) |
| **Physical Measurements** | OMOP Measurement table (mapped to LOINC) |
| **Whole Genome Sequencing (WGS)** | Hail VariantDataset (VDS) for full callset; VCF, Hail MatrixTable (MT), and PLINK 1.9 for subsets |
| **Genotyping Arrays** | VCF, Hail MatrixTable, PLINK 1.9, raw IDAT files |
| **Structural Variants** | Specialized callset format |
| **Long-Read WGS** | Available for a subset (~2,700 participants) |
| **Wearable / Fitbit Data** | Custom All of Us tables (not OMOP); 7 tables organized by domain and granularity |

## Key Tables / Files

### OMOP CDM Tables (EHR, Surveys, Physical Measurements)

| Table | Description |
|-------|-------------|
| **Person** | Demographics (age, sex, race, ethnicity) |
| **Visit Occurrence** | Healthcare encounters (inpatient, outpatient, ER) |
| **Condition Occurrence** | Diagnoses and medical conditions |
| **Drug Exposure** | Medication prescriptions and administrations |
| **Measurement** | Lab results, vital signs, physical measurements |
| **Procedure Occurrence** | Clinical and surgical procedures |
| **Observation** | Surveys, social history, and other observations |
| **Device Exposure** | Medical devices used during care |
| **Death** | Mortality information |
| **Location / Care Site / Provider** | Healthcare setting information |
| **Fact Relationship** | Links between records across tables |
| **Specimen** | Biospecimen information |

### Wearable Device Tables (Fitbit)

| Table | Description |
|-------|-------------|
| **activity_summary** | Daily activity summaries (steps, calories, distance) |
| **heart_rate_summary** | Daily heart rate zone summaries |
| **heart_rate_minute_level** | Minute-by-minute heart rate data |
| **steps_intraday** | Minute-level step counts |
| **sleep_daily_summary** | Daily sleep metrics |
| **sleep_level** | Sleep stage data (deep, light, REM, wake) |
| **device** | Device type and model information |

### Genomic Data (Controlled Tier Only)

- Short-read WGS for ~414,000 participants
- Genotyping arrays for ~447,000 participants
- Structural variants for ~97,000 participants
- Long-read WGS for ~2,700 participants

## Important Restrictions

- **No Re-identification**: Any attempt to identify participants is strictly prohibited.
- **Workspace Purpose**: Each workspace must have a stated research purpose that is reviewed by All of Us.
- **Cloud-Only Access**: All data analysis must be performed within the Researcher Workbench. Data cannot be downloaded or removed from the secure environment (with limited exceptions for summary statistics and aggregate results).
- **Responsible Use**: Researchers must follow the Data User Code of Conduct, including commitments to equitable research practices and participant respect.
- **Annual Training Renewal**: Responsible Conduct of Research training (and Controlled Tier training, if applicable) must be renewed annually.
- **Publication Requirements**: Acknowledge the All of Us Research Program in all publications and presentations. Do not publish or present results that could enable re-identification.
- **No Unauthorized Sharing**: Do not share data or workspace access with unregistered individuals.
- **IRB**: The NIH All of Us IRB serves as the single IRB for use of the data. Individual projects do not typically require separate IRB approval.

## Useful Links

- [All of Us -- Register for Researcher Workbench](https://www.researchallofus.org/register/)
- [Researcher Workbench Overview](https://www.researchallofus.org/data-tools/workbench/)
- [Data Types and Organization](https://support.researchallofus.org/hc/en-us/articles/4619151535508-Data-Types-and-Organization)
- [Data Methods](https://www.researchallofus.org/data-tools/methods/)
- [Understanding OMOP Basics](https://support.researchallofus.org/hc/en-us/articles/360039585391-Understanding-OMOP-Basics)
- [Genomic Data Organization](https://support.researchallofus.org/hc/en-us/articles/29475228181908-How-the-All-of-Us-Genomic-data-are-organized)
- [All of Us Genomic Data (Nature, 2023)](https://www.nature.com/articles/s41586-023-06957-x)
- [All of Us Frequently Asked Questions](https://researchallofus.org/frequently-asked-questions/)
- [All of Us Research Program (NIH)](https://allofus.nih.gov/)
- [UC Irvine -- How to Gain Access Guide](https://guides.lib.uci.edu/AllofUs/access)
