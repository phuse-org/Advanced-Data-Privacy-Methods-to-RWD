# NHIS (National Health Interview Survey) — Access Instructions

## Overview

The National Health Interview Survey (NHIS) is the principal source of information on the health of the civilian noninstitutionalized population of the United States. Conducted continuously since 1957 by the National Center for Health Statistics (NCHS), part of the Centers for Disease Control and Prevention (CDC), the NHIS collects data through in-person household interviews on a broad range of health topics including health status, health care access and utilization, health insurance coverage, health behaviors, and chronic conditions.

The NHIS interviews approximately 35,000 households annually, covering roughly 87,500 persons. It uses a multistage area probability design to produce nationally representative estimates. The survey underwent a major redesign in 2019, modernizing content and streamlining the questionnaire structure.

## Access Level

**Open** — Public-use NHIS data files are freely available for download from the CDC/NCHS website. No registration, application, or data use agreement is required for public-use files.

Restricted-use data (with detailed geographic and other sensitive variables) are available only through the NCHS Research Data Center (RDC) via a formal proposal process.

## Prerequisites

None for public-use data. Files can be downloaded immediately.

For restricted-use data:
- An approved research proposal submitted to the NCHS Research Data Center
- Payment of applicable user fees
- Completion of required training and administrative forms
- All output is subject to disclosure review

## Step-by-Step Access

### Step 1: Navigate to the NHIS Data Page

Go to the NHIS Questionnaires, Datasets, and Documentation page:

[https://www.cdc.gov/nchs/nhis/data-questionnaires-documentation.htm](https://www.cdc.gov/nchs/nhis/data-questionnaires-documentation.htm)

Or the updated documentation index page:

[https://www.cdc.gov/nchs/nhis/documentation/index.html](https://www.cdc.gov/nchs/nhis/documentation/index.html)

### Step 2: Select a Survey Year

The page provides links to individual year pages. Currently available years include:
- **2019--2024** (redesigned questionnaire)
- **1997--2018** (prior questionnaire design)
- Earlier years (pre-1997) are available through the CDC archive

Each year page contains all data files, questionnaires, codebooks, and supporting documentation for that year.

### Step 3: Review the Survey Components

**For 2019 onward (redesigned NHIS):**
The redesigned survey collects data from one randomly selected sample adult per family. Key components include:
- **Sample Adult** — The primary data file with health status, conditions, functioning, health care, health behaviors, and demographics
- **Paradata** — Interview process data
- **Imputed Income** — Multiple imputation files for family income

**For 1997--2018 (prior design):**
- **Household** — Household-level variables
- **Family** — Family-level demographic and socioeconomic data
- **Person** — Basic demographic and health data for all family members
- **Sample Adult** — Detailed health information for one randomly selected adult (18+) per family
- **Sample Child** — Detailed health information for one randomly selected child (0--17) per family
- **Supplements** — Topical modules that vary by year (e.g., cancer, complementary medicine, immunization)

### Step 4: Download Data Files

Each survey year page provides downloadable ZIP archives containing:
- **CSV data files** — Comma-separated value files openable in Excel or any statistical software
- **ASCII (fixed-width) data files** — Plain text data files
- **SAS input statements** — Programs to read the ASCII files into SAS
- **SPSS input statements** — Programs to read the ASCII files into SPSS
- **Stata input statements** — Programs to read the ASCII files into Stata
- **Codebooks** — Variable-by-variable documentation with frequencies and descriptions
- **Survey description documents** — Technical documentation on survey design, weighting, and methodology

Click the download link for the format you need. Files are compressed in ZIP format.

### Step 5: Import the Data

**Using CSV files (simplest method):**
Open the CSV file directly in R, Python, Stata, or Excel:

```r
# R
nhis_data <- read.csv("adult24.csv")
```

```python
# Python
import pandas as pd
nhis_data = pd.read_csv("adult24.csv")
```

**Using ASCII files with provided input programs:**
Run the SAS, SPSS, or Stata input program provided alongside the ASCII data file. The input program reads the fixed-width file and assigns variable names and labels.

### Step 6: Merge Files as Needed

To build an analytic dataset spanning multiple components, use the household identifier (`HHX`), family identifier (`FMX`), and person identifier (`FPX`) to merge files. Sample SAS code for merging is provided in the Survey Description Document for each year.

## Data Format

- **Primary formats:** CSV files and ASCII (fixed-width) text files
- **Import programs:** SAS, SPSS, and Stata input statements provided
- **Documentation:** HTML/PDF codebooks, survey description documents, and questionnaire PDFs
- All files are distributed as compressed ZIP archives

## Key Tables / Files

For the redesigned NHIS (2019 onward), the primary file is:

- **adult** (e.g., `adult24.csv`) — Sample Adult interview data with hundreds of variables on:
  - Demographics (age, sex, race, ethnicity, education, income)
  - Health status and conditions (diabetes, heart disease, cancer, asthma, arthritis)
  - Health behaviors (smoking, alcohol, physical activity, sleep)
  - Health care access and utilization (insurance, usual source of care, ER visits)
  - Functioning and disability
  - Mental health (anxiety, depression screening)
  - Immunizations

- **child** (e.g., `child24.csv`) — Sample Child interview data (if collected that year)

Key linking variables:
- `HHX` — Household serial number
- `FMX` — Family serial number
- `FPX` — Person number within family

Key analytic variables:
- `WTFA_A` — Sample adult weight (final annual)
- `STRAT_P` — Variance estimation stratum
- `PSU_P` — Primary sampling unit for variance estimation

## Important Restrictions

- **Survey weights required:** NHIS uses a complex multistage probability sample design. All analyses must use the provided survey weights, strata, and PSU variables to produce valid national estimates.
- **Questionnaire redesign in 2019:** Data from 2019 onward are not directly comparable to data from 1997--2018 due to substantial changes in question wording, skip patterns, and survey structure. Use caution when analyzing trends across the redesign boundary.
- **No individual identifiers:** Public-use files do not include names, addresses, Social Security numbers, or detailed geographic identifiers.
- **Citation requirement:** Users should cite NHIS data using the recommended format from the survey documentation.
- **Early Release estimates:** Preliminary estimates for key health indicators are released biannually before the final data files. These are available through the NHIS Early Release Program but may differ slightly from final estimates.

## Useful Links

- [NHIS Home Page](https://www.cdc.gov/nchs/nhis/index.html)
- [NHIS Questionnaires, Datasets, and Documentation](https://www.cdc.gov/nchs/nhis/documentation/index.html)
- [2024 NHIS Data and Documentation](https://www.cdc.gov/nchs/nhis/documentation/2024-nhis.html)
- [NHIS Early Release Key Indicators](https://www.cdc.gov/nchs/nhis/early-release/key-indicators.html)
- [NHIS Survey Description Documents](https://www.cdc.gov/nchs/nhis/documentation/index.html)
- [NCHS Research Data Center (for restricted data)](https://www.cdc.gov/rdc/)
- [IPUMS NHIS (harmonized data across years)](https://nhis.ipums.org/nhis/)
- [NHIS on Healthy People 2030](https://odphp.health.gov/healthypeople/objectives-and-data/data-sources-and-methods/data-sources/national-health-interview-survey-nhis)
