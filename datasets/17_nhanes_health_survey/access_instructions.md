# NHANES (National Health and Nutrition Examination Survey) — Access Instructions

## Overview

The National Health and Nutrition Examination Survey (NHANES) is a program of studies conducted by the National Center for Health Statistics (NCHS), part of the Centers for Disease Control and Prevention (CDC). NHANES is designed to assess the health and nutritional status of adults and children in the United States. The survey combines in-home personal interviews with physical examinations and laboratory tests conducted in mobile examination centers (MECs). NHANES has been running since the early 1960s and transitioned to a continuous annual survey format in 1999.

NHANES examines a nationally representative sample of approximately 5,000 persons each year. The survey is unique in that it combines interviews and physical examinations, making it one of the most comprehensive health data sources in the world. NHANES data are referenced in approximately 5,000 publications annually.

## Access Level

**Open** — All public-use NHANES data files and documentation are freely available for download from the CDC/NCHS website without registration or application. No data use agreement is required for public-use files.

Note: Certain sensitive variables are only available as limited-access data through the NCHS Research Data Center (RDC), which requires a formal proposal and approval process.

## Prerequisites

None for public-use data. You can download files immediately from the NHANES website.

For limited-access data at the NCHS Research Data Center:
- An approved research proposal submitted to NCHS
- Payment of user fees
- Completion of training and administrative forms
- All output must pass a disclosure review

## Step-by-Step Access

### Step 1: Navigate to the NHANES Website

Go to the NHANES home page at:

[https://wwwn.cdc.gov/nchs/nhanes/Default.aspx](https://wwwn.cdc.gov/nchs/nhanes/Default.aspx)

This is the centralized location for all publicly available NHANES datasets and documentation, covering both historic and current survey cycles.

### Step 2: Select a Survey Cycle

NHANES data are organized by two-year survey cycles. Navigate to the survey cycle of interest. Available continuous NHANES cycles include:

- 1999--2000, 2001--2002, 2003--2004, 2005--2006, 2007--2008
- 2009--2010, 2011--2012, 2013--2014, 2015--2016, 2017--2018
- 2017--March 2020 (pre-pandemic), 2021--2023

Earlier surveys (NHANES I, II, III) are also available. NHANES III (1988--1994) data are accessible through a separate section of the website.

### Step 3: Browse Data Components

Within each survey cycle, data files are organized into five main components:

1. **Demographics** — Age, sex, race/ethnicity, income, education, household composition
2. **Dietary** — 24-hour dietary recall data collected via the USDA Automated Multiple-Pass Method (released as "What We Eat in America")
3. **Examination** — Body measurements, blood pressure, dental examinations, body composition (DXA), audiometry, vision, and other physical measurements
4. **Laboratory** — Biomarker measurements from blood, serum, and urine specimens (e.g., cholesterol, glucose, heavy metals, nutritional biomarkers)
5. **Questionnaire** — Self-reported data on health conditions, health behaviors, smoking, physical activity, medications, food security, healthcare access, and more

### Step 4: Download Data Files

Click on the specific data file you need. Each data component page lists individual data files with links to:
- The data file itself (SAS Transport format, `.XPT`)
- The accompanying documentation/codebook

**If SAS is installed:** Left-clicking the data file link may prompt you to save the `.XPT` file directly.

**If SAS is not installed:** Right-click the data file link and choose "Save As" to download the file to your local drive.

### Step 5: Import and Convert the Data

SAS Transport (`.XPT`) files must be imported/converted before analysis. They are not directly usable without this step.

**In SAS:**
```sas
LIBNAME xptfile XPORT 'path/to/datafile.xpt';
LIBNAME outlib 'path/to/output/folder';
PROC COPY IN=xptfile OUT=outlib;
RUN;
```

**In R:**
```r
library(haven)
data <- read_xpt("path/to/datafile.xpt")
```

**In Python:**
```python
import pandas as pd
data = pd.read_sas("path/to/datafile.xpt", format="xport")
```

**In Stata:**
```stata
import sasxport5 "path/to/datafile.xpt", clear
```

You can also use the free SAS Universal Viewer to view and export data files. It is available at: [SAS Universal Viewer](https://wwwn.cdc.gov/nchs/nhanes/SasViewer.aspx)

### Step 6: Merge Files as Needed

To build an analytic dataset, you will typically need to merge files across components (e.g., demographics + laboratory + questionnaire) using the respondent sequence number (`SEQN`) as the linking variable. To combine multiple survey cycles, append datasets after confirming that variable names and coding are consistent across cycles.

## Data Format

- **Primary format:** SAS Transport files (`.XPT`, Version 5)
- **Documentation:** HTML codebooks and frequency tables for each data file
- **Additional resources:** SAS, R, and Stata sample code for importing and analyzing data
- Files can be converted to CSV, Stata `.dta`, or other formats using standard statistical software

## Key Tables / Files

Each survey cycle typically includes dozens of individual data files. Key files include:

- **DEMO** — Demographic variables and survey design variables (survey weights, strata, PSU)
- **DR1TOT / DR2TOT** — Total nutrient intakes from Day 1 and Day 2 dietary recalls
- **BPX** — Blood pressure measurements
- **BMX** — Body measures (height, weight, BMI, waist circumference)
- **GLU / GHB** — Fasting glucose and glycohemoglobin (HbA1c)
- **TCHOL / HDL / TRIGLY** — Lipid panel data
- **SMQ** — Smoking and tobacco use
- **PAQ** — Physical activity
- **DIQ** — Diabetes
- **MCQ** — Medical conditions
- **RXQ_RX** — Prescription medication use

The variable `SEQN` (Respondent Sequence Number) is the unique person identifier used to merge files within a cycle.

## Important Restrictions

- **No single-year public releases:** Due to disclosure risk concerns, continuous NHANES data are released only in multi-year cycles (typically two-year periods). No single-year files are publicly released for 1999 onward.
- **Limited-access data:** Some variables (e.g., detailed geographic identifiers, certain genetic data) are restricted and available only through the NCHS Research Data Center.
- **Survey weights required:** NHANES uses a complex, multistage probability sampling design. Proper analysis requires the use of survey weights, strata, and PSU variables provided in the Demographics file. Failure to account for the survey design will produce biased estimates.
- **Citation:** Users should cite NHANES data appropriately in publications. Consult the NHANES documentation for recommended citation formats.
- **Variable changes across cycles:** Variable names, coding schemes, and laboratory methods may change between cycles. Always review the documentation for each cycle before combining data across years.

## Useful Links

- [NHANES Home Page](https://wwwn.cdc.gov/nchs/nhanes/Default.aspx)
- [NHANES Questionnaires, Datasets, and Related Documentation](https://wwwn.cdc.gov/nchs/nhanes/)
- [NHANES Tutorials](https://wwwn.cdc.gov/nchs/nhanes/tutorials/default.aspx)
- [NHANES Datasets and Documentation Tutorial Module](https://wwwn.cdc.gov/nchs/nhanes/tutorials/datasets.aspx)
- [NHANES Frequently Asked Questions](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/faq.aspx)
- [SAS Universal Viewer Download](https://wwwn.cdc.gov/nchs/nhanes/SasViewer.aspx)
- [NCHS Research Data Center (for restricted data)](https://www.cdc.gov/rdc/)
- [CDC NHANES Overview](https://www.cdc.gov/nchs/nhanes/index.html)
- [NHANES on Healthy People 2030](https://odphp.health.gov/healthypeople/objectives-and-data/data-sources-and-methods/data-sources/national-health-and-nutrition-examination-survey-nhanes)
