# County Health Rankings & Roadmaps — Access Instructions

## Overview

County Health Rankings & Roadmaps (CHR&R) is a program of the University of Wisconsin Population Health Institute, funded by the Robert Wood Johnson Foundation. CHR&R ranks the health of nearly every county in the United States, providing a revealing snapshot of how health is influenced by where people live, learn, work, and play.

The program measures and ranks counties within each state using a model of population health that includes health outcomes (length of life and quality of life) and health factors (health behaviors, clinical care, social and economic factors, and the physical environment). It draws on over 30 data sources and covers all U.S. counties (~3,000).

CHR&R data have been released annually since 2010 and are widely used in public health practice, community health assessment, health equity research, and policy development.

## Access Level

**Open** — National data files and state-level data are freely available for download. No registration is required for standard data files.

For custom data requests beyond what is publicly available, you may need to contact CHR&R directly, and a Data Use Agreement or IRB approval may be required depending on the nature of the request.

## Prerequisites

None for standard data downloads. Files can be downloaded immediately.

For working with the data, a spreadsheet application (Excel, Google Sheets) or statistical software (R, SAS, Stata, Python) is sufficient.

## Step-by-Step Access

### Step 1: Navigate to the Data and Documentation Page

Go to the CHR&R data and documentation page:

[https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation](https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation)

Or navigate from the main site:

[https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation](https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation)

### Step 2: Download National Data Files

The annual data release is available in multiple formats:

- **CSV format** — Comma-separated values for analytic use in R, Python, SAS, or other software
- **SAS format** — SAS datasets for direct use in SAS
- **Excel format** — Spreadsheet files for viewing and basic analysis

Select the desired year and format, then click the download link. National files contain data for all U.S. counties.

### Step 3: Download State-Specific Data (Alternative)

State-specific data files can be downloaded from the individual state pages. Navigate to:

https://www.countyhealthrankings.org/health-data/{state-name}/data-and-resources

(Replace `{state-name}` with the state of interest, e.g., `wisconsin`)

Each state page provides data files and visualizations specific to that state.

### Step 4: Download Trend Data

Trend datasets are also available for download in CSV and SAS format on the Data and Documentation page. These allow analysis of how county health measures have changed over time.

### Step 5: Explore Data Interactively (Optional)

The CHR&R website provides an interactive exploration tool:

[https://www.countyhealthrankings.org/health-data](https://www.countyhealthrankings.org/health-data)

Use this tool to compare counties, view rankings, explore individual measures, and create custom visualizations before downloading data for deeper analysis.

### Step 6: Review Technical Documentation

Download the Technical Documentation PDF for your year of interest. This document describes:
- The health model and ranking methodology
- All measures with their definitions, data sources, and years of data
- Changes to measures from year to year
- Guidance on comparing data across states and over time

Example: [2025 Technical Documentation (PDF)](https://www.countyhealthrankings.org/sites/default/files/media/document/CHRR%20Technical%20Documentation%202025_2.pdf)

## Data Format

- **CSV files** — Comma-separated values (one row per county)
- **SAS files** — SAS datasets
- **Excel files** — XLS/XLSX workbooks, often with multiple tabs for different measure categories
- **Documentation:** PDF technical documentation, measure definitions, and data source descriptions

## Key Tables / Files

The annual data release contains county-level values for dozens of health measures organized into the CHR&R health model:

**Health Outcomes:**
- *Length of Life:* Premature death (years of potential life lost before age 75), life expectancy
- *Quality of Life:* Poor or fair health, poor physical health days, poor mental health days, low birthweight

**Health Factors:**

*Health Behaviors:*
- Adult smoking, adult obesity, food environment index
- Physical inactivity, access to exercise opportunities
- Excessive drinking, alcohol-impaired driving deaths
- Sexually transmitted infections, teen births

*Clinical Care:*
- Uninsured adults and children
- Primary care physicians, dentists, mental health providers (per population)
- Preventable hospital stays, mammography screening, flu vaccinations

*Social and Economic Factors:*
- High school completion, some college
- Unemployment, children in poverty, income inequality
- Children in single-parent households, social associations
- Violent crime, injury deaths
- Residential segregation

*Physical Environment:*
- Air pollution (PM2.5), drinking water violations
- Severe housing problems, driving alone to work, long commute (driving alone)

**Key identifier variables:**
- `FIPS` — 5-digit county FIPS code
- `State` — State name
- `County` — County name
- `statecode` — 2-digit state FIPS code
- `countycode` — 3-digit county FIPS code

Each measure includes:
- Raw value
- Numerator and denominator (where applicable)
- Confidence intervals
- Rank within state
- Quartile classification

## Important Restrictions

- **Citation required:** Users must cite County Health Rankings & Roadmaps when publishing work that uses CHR&R data. Recommended citation: "University of Wisconsin Population Health Institute. County Health Rankings & Roadmaps [Year]. www.countyhealthrankings.org."
- **Data source restrictions:** CHR&R compiles data from many sources, each with its own data use terms. Some source data may have additional restrictions on redistribution.
- **Not all measures available for all counties:** Some measures may be suppressed for counties with small populations due to statistical reliability concerns.
- **Ranking methodology changes:** Measures, data sources, and weighting may change from year to year. Review the technical documentation before making temporal comparisons.
- **Cross-state comparisons:** Rankings are produced within states, not across states. Cross-state comparisons should use raw values rather than within-state ranks.
- **Custom data requests:** For data not available in the standard download files, contact CHR&R through their website. IRB approval and/or a Data Use Agreement may be required.

## Useful Links

- [County Health Rankings Home Page](https://www.countyhealthrankings.org/)
- [Data and Documentation](https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation)
- [Explore Health Rankings (Interactive)](https://www.countyhealthrankings.org/health-data)
- [Methodology and Sources](https://www.countyhealthrankings.org/health-data/methodology-and-sources)
- [2025 Technical Documentation (PDF)](https://www.countyhealthrankings.org/sites/default/files/media/document/CHRR%20Technical%20Documentation%202025_2.pdf)
- [UW Population Health Institute](https://uwphi.pophealth.wisc.edu/chrr/)
- [Robert Wood Johnson Foundation](https://www.rwjf.org/)
