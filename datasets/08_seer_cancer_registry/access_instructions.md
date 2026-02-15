# SEER Cancer Registry -- Access Instructions

## Overview

The Surveillance, Epidemiology, and End Results (SEER) Program is a premier source for cancer statistics in the United States, operated by the National Cancer Institute (NCI). SEER collects and publishes cancer incidence, survival, and prevalence data from population-based cancer registries covering approximately 46% of the U.S. population. The program has been collecting data since 1975 and includes millions of cancer cases with information on patient demographics, primary tumor site, tumor morphology, stage at diagnosis, first course of treatment, and follow-up for vital status.

## Access Level

**Controlled Access** -- SEER data access is free of charge but requires registration, identity verification, and agreement to a Research Data Use Agreement (DUA). There are two tiers of research data:

- **SEER Research Data**: Standard incidence and population data by age, sex, race, year of diagnosis, and geographic area. Requires DUA acknowledgment.
- **SEER Research Plus Data**: Extended data with additional variables. Requires user authentication with an eRA Commons or HHS account, plus DUA signing.

Each individual who will access the data must submit a separate request and sign the DUA independently.

## Prerequisites

Before starting the access process, ensure you have:

- **Institutional Affiliation**: A research or academic institution affiliation (recommended but not strictly required for basic SEER Research data).
- **eRA Commons or HHS Account** (for SEER Research Plus data): If you do not already have one, you will need to register through your institution's signing official.
- **SEER*Stat Software**: Download and install the SEER*Stat statistical software, which is the primary tool for analyzing SEER data. Available for Windows; the latest version is SEER*Stat 9.0.42.2.

## Step-by-Step Access

### Step 1: Visit the SEER Data Access Page

Navigate to the SEER data access page at [https://seer.cancer.gov/data/access.html](https://seer.cancer.gov/data/access.html). Review the available data products and determine which level of access you need (SEER Research vs. SEER Research Plus).

### Step 2: Create a SEER*Stat Account

Click on the link to request access to SEER data. You will be prompted to create a SEER*Stat account by providing your name, institutional affiliation, email address, and research purpose.

### Step 3: Sign the Data Use Agreement

Read and acknowledge the SEER Research Data Use Agreement. The DUA includes the following key commitments:

- You will not link or attempt to link the data with any other database at the individual level, including linking two or more SEER databases.
- You will not attempt to learn the identity of any individual in the database.
- You will not release the data to any other person; each team member must sign their own DUA.
- You will suppress statistics based on counts of 1 to 4 in all publications.
- You will acknowledge the SEER database and specific version in all publications.

### Step 4: Authenticate (for Research Plus Data)

If requesting SEER Research Plus data, authenticate using your eRA Commons or HHS account credentials. This step verifies your identity and institutional affiliation.

### Step 5: Download and Install SEER*Stat

Download the SEER*Stat software from [https://seer.cancer.gov/data-software/](https://seer.cancer.gov/data-software/). Install and configure it on your Windows computer. SEER*Stat connects to the SEER data servers using your registered credentials.

### Step 6: Access the Data

Once approved, log in to SEER*Stat with your credentials to access the SEER databases. You can run queries for incidence, mortality, survival, and prevalence analyses directly within the software. Data can also be exported for use in other statistical packages.

## Data Format

SEER data is available in multiple formats:

- **SEER*Stat Binary Database Format**: The primary format for use with the SEER*Stat software. Created from ASCII text files using the SEER*Prep conversion tool.
- **ASCII Fixed-Width Text Files**: Raw data files with fixed record lengths. Can be converted to SEER*Stat format using SEER*Prep.
- **CSV Format**: Available for newer NAACCR (North American Association of Central Cancer Registries) format data, including incidence, mortality, and census tract files.
- **Exported Data**: SEER*Stat allows export of query results to CSV or tab-delimited text files for use in R, Python, SAS, or other tools.

## Key Tables / Files

The SEER databases contain the following major data categories:

| Database / File | Description |
|----------------|-------------|
| **SEER Incidence Database** | Cancer incidence data from 1975 to 2022, including tumor site, morphology, stage, grade, and demographics |
| **SEER Mortality Database** | Cause-of-death data linked to cancer registrations |
| **SEER Populations Database** | County-level population denominators for rate calculations |
| **U.S. Cancer Statistics (USCS) Database** | Combined CDC/NCI database (requires separate request form) |
| **SEER-Medicare Linked Data** | SEER data linked with Medicare claims (separate application required through NCI) |
| **Database Dictionary** | Variable definitions, valid values, and code descriptions |

Each November, a new submission of data is released (e.g., the November 2024 submission covers data through 2022).

## Important Restrictions

- **No Data Linkage**: You may not link SEER data with any other database at the individual level, nor link two or more SEER databases together.
- **No Re-identification**: Any attempt to identify individuals in the database is strictly prohibited.
- **No Redistribution**: You may not share the data with others. Each researcher must sign their own DUA.
- **Publication Requirements**: Statistics based on counts of 1 to 4 must be suppressed in all publications. Case-listing information for identifiable individuals must never be published.
- **Citation Required**: Acknowledge the specific SEER database version in all publications and presentations.
- **Annual Renewal**: A new DUA and (if applicable) USCS request form is required for each annual data submission release.
- **SEER-Medicare Restrictions**: CMS data access policies require transition to federally managed enclave environments. NCI is not approving new cloud-based storage requests for SEER-Medicare data.

## Useful Links

- [How to Request Access to SEER Data](https://seer.cancer.gov/data/access.html)
- [SEER Data and Software](https://seer.cancer.gov/data-software/)
- [SEER Incidence Data, 1975-2022](https://seer.cancer.gov/data/)
- [SEER*Stat Software Download](https://seer.cancer.gov/data-software/software.html)
- [SEER*Prep Software (Data Conversion)](https://seer.cancer.gov/seerprep/)
- [SEER Research DUA (November 2024 Submission)](https://seer.cancer.gov/data-software/documentation/seerstat/nov2024/seer-dua-nov2024.html)
- [SEER*Stat Database Documentation](https://seer.cancer.gov/data-software/documentation/seerstat/nov2024/)
- [SEER-Medicare Data Requests](https://healthcaredelivery.cancer.gov/seermedicare/obtain/requests.html)
- [SEER Program Overview](https://seer.cancer.gov/)
