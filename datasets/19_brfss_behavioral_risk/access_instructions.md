# BRFSS (Behavioral Risk Factor Surveillance System) — Access Instructions

## Overview

The Behavioral Risk Factor Surveillance System (BRFSS) is the largest continuously conducted health survey system in the world. Operated by the Centers for Disease Control and Prevention (CDC), BRFSS is a state-based system of telephone surveys that collects data from U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services.

Established in 1984 with 15 states, BRFSS now collects data in all 50 states, the District of Columbia, and three U.S. territories (Guam, Puerto Rico, and the U.S. Virgin Islands). The system completes more than 400,000 adult interviews each year, making it an invaluable resource for monitoring state-level and sub-state health trends.

## Access Level

**Open** — All BRFSS public-use data files and documentation are freely available for download from the CDC website. No registration, account, or data use agreement is required.

## Prerequisites

None. BRFSS data can be downloaded immediately from the CDC website.

## Step-by-Step Access

### Step 1: Navigate to the BRFSS Data Page

Go to the BRFSS Survey Data and Documentation page:

[https://www.cdc.gov/brfss/annual_data/annual_data.htm](https://www.cdc.gov/brfss/annual_data/annual_data.htm)

Or start from the main BRFSS page and navigate to the data section:

[https://www.cdc.gov/brfss/data_documentation/index.htm](https://www.cdc.gov/brfss/data_documentation/index.htm)

### Step 2: Select a Survey Year

The annual data page lists all available survey years from 1984 to the present. Click on the year of interest. Each year page provides:
- Data files
- Codebooks
- Questionnaires
- Technical documentation (survey methodology, weighting, response rates)
- Calculated variables documentation

### Step 3: Understand the Survey Modules

BRFSS data are collected through three types of modules:

1. **Core Component** — A fixed set of questions asked by all states every year, covering:
   - Health status
   - Healthy days (health-related quality of life)
   - Health care access
   - Exercise / physical activity
   - Diabetes
   - Inadequate sleep
   - Demographics

2. **Optional Modules** — Standardized sets of questions on specific topics that states can choose to include:
   - Examples: cancer screening, cardiovascular health, cognitive decline, social determinants of health, adverse childhood experiences (ACE), sexual orientation and gender identity
   - Availability varies by state and year

3. **State-Added Questions** — Questions added individually by states; these are not included in the national dataset

### Step 4: Download the Data Files

Each year page provides data files in multiple formats:

- **SAS Transport Format (.XPT)** — The primary distribution format
- **ASCII (fixed-width) data files** — Plain text files with corresponding SAS, SPSS, and Stata import programs
- **Codebook** — PDF or HTML document describing each variable, its coding, and frequencies

Download the file format you prefer. Files are typically distributed as ZIP archives.

### Step 5: Import the Data

**In SAS:**
```sas
LIBNAME brfss XPORT 'path/to/LLCP2023.XPT';
PROC COPY IN=brfss OUT=work;
RUN;
```

**In R:**
```r
library(haven)
brfss <- read_xpt("path/to/LLCP2023.XPT")
```

**In Python:**
```python
import pandas as pd
brfss = pd.read_sas("path/to/LLCP2023.XPT", format="xport")
```

**In Stata:**
```stata
import sasxport5 "path/to/LLCP2023.XPT", clear
```

Note: BRFSS files can be very large (400,000+ records with hundreds of variables). Ensure you have adequate memory and disk space.

### Step 6: Review Calculated Variables and Weighting

BRFSS provides pre-calculated variables for commonly used risk factors. Review the Calculated Variables documentation for each year. All analyses must use the raking-based survey weights (`_LLCPWT`) to produce valid population estimates.

## Data Format

- **Primary format:** SAS Transport files (`.XPT`)
- **Alternative format:** ASCII fixed-width text files with SAS/SPSS/Stata import programs
- **Documentation:** PDF/HTML codebooks, questionnaires, calculated variables documentation, data quality reports
- **Supplementary:** GIS/map data files available in ZIP format for geographic analysis

## Key Tables / Files

The main annual data file is typically named with a prefix like `LLCP` followed by the year (e.g., `LLCP2023.XPT`). Key variables include:

**Demographics:**
- `_STATE` — FIPS state code
- `_AGE_G` / `_AGE80` — Age categories
- `_RACE` / `_RACEGR3` — Race/ethnicity
- `SEX` / `SEXVAR` — Sex
- `INCOME3` — Income level
- `EDUCA` — Education level

**Health Behaviors:**
- `_SMOKER3` — Smoking status (calculated)
- `_RFBING5` — Binge drinking (calculated)
- `_TOTINDA` — Physical activity (calculated)
- `_BMI5CAT` — BMI category (calculated)
- `SLEPTIM1` — Sleep duration

**Health Conditions:**
- `DIABETE4` — Diabetes status
- `_MICHD` — Coronary heart disease/MI (calculated)
- `_ASTHMS1` — Asthma status (calculated)
- `CHCCOPD3` — COPD

**Health Access:**
- `HLTHPLN1` — Health care coverage
- `MEDCOST1` — Could not see doctor due to cost
- `CHECKUP1` — Last routine checkup

**Survey Design:**
- `_LLCPWT` — Final weight (raking-adjusted)
- `_STSTR` — Sample design stratification variable
- `_PSU` — Primary sampling unit

## Important Restrictions

- **Survey weights required:** BRFSS uses a complex sample design with disproportionate stratification and raking. All analyses must use the provided survey weights to produce valid estimates.
- **State-level data:** While the combined national file is available, BRFSS is fundamentally a state-based surveillance system. National estimates require appropriate weighting.
- **Telephone survey limitations:** BRFSS is conducted by telephone (landline and cell phone). Households without telephones are excluded. The survey has shifted toward a dual-frame (landline + cell) design over time.
- **No individual identifiers:** Public-use files do not contain personal identifying information.
- **Citation:** Cite as "Centers for Disease Control and Prevention. Behavioral Risk Factor Surveillance System Survey Data. Atlanta, Georgia: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, [YEAR]."
- **Optional module availability varies:** Not all states administer all optional modules in any given year. Check the module availability tables before conducting state comparisons.

## Useful Links

- [BRFSS Home Page](https://www.cdc.gov/brfss/index.html)
- [BRFSS Annual Survey Data](https://www.cdc.gov/brfss/annual_data/annual_data.htm)
- [BRFSS Survey Data and Documentation](https://www.cdc.gov/brfss/data_documentation/index.htm)
- [BRFSS on CDC WONDER](https://wonder.cdc.gov/brfss.html)
- [BRFSS Web Enabled Analysis Tool (WEAT)](https://www.cdc.gov/brfss/data_tools.htm)
- [BRFSS Data User Guide](https://www.cdc.gov/brfss/data_documentation/index.htm)
- [BRFSS on Data.gov](https://catalog.data.gov/dataset/cdc-behavioral-risk-factor-surveillance-system-brfss)
- [BRFSS on HealthData.gov](https://healthdata.gov/dataset/CDC-Behavioral-Risk-Factor-Surveillance-System-BRF/w6eg-52ny)
