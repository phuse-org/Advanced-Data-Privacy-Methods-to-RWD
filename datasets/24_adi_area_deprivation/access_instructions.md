# ADI (Area Deprivation Index) — Access Instructions

## Overview

The Area Deprivation Index (ADI) is a neighborhood-level measure of socioeconomic disadvantage developed and maintained by the Health Innovation Program at the University of Wisconsin School of Medicine and Public Health. Originally created by the Health Resources and Services Administration (HRSA) and subsequently refined by Amy Kind, MD, PhD, and her research team, the ADI ranks neighborhoods (census block groups) by their level of socioeconomic deprivation.

The ADI is calculated from 17 measures spanning four domains: income and employment, education, housing quality, and household characteristics. It is derived from American Community Survey (ACS) 5-year estimates and is available at the census block group level -- the most geographically precise unit available from the ACS (approximately 600--3,000 people per block group). The ADI is widely used in health disparities research, clinical care targeting, and health equity policy.

The ADI is distributed through the **Neighborhood Atlas**, a web platform hosted by the University of Wisconsin-Madison.

## Access Level

**Open** — The ADI data and mapping tools are available at no cost. However, downloading the datasets requires creating a free account on the Neighborhood Atlas website. No institutional affiliation or data use agreement is required to register.

## Prerequisites

- A valid email address to create a free Neighborhood Atlas account
- No institutional approval, IRB, or data use agreement is required for general use
- For GIS analysis, familiarity with census block group geography is recommended

## Step-by-Step Access

### Step 1: Navigate to the Neighborhood Atlas

Go to the Neighborhood Atlas website:

[https://www.neighborhoodatlas.medicine.wisc.edu/](https://www.neighborhoodatlas.medicine.wisc.edu/)

### Step 2: Create a Free Account

Click the "Register" or "Sign Up" link on the website. Provide:
- Your name
- Email address
- A password
- Basic information about your intended use

Registration is free and generally approved immediately. You will need to confirm your email address.

### Step 3: Explore the Interactive Map (Optional)

Before downloading, you can explore ADI rankings visually:
1. Use the mapping tool to search for a specific address, city, or state
2. View block group-level ADI rankings overlaid on the map
3. The map displays the 2023 ADI (most recent available) and allows toggling between national rankings and state deciles

### Step 4: Download ADI Datasets

After logging in, navigate to the data download section:
1. Select the geographic scope: **National** or **Individual State**
2. Select the ADI version/year you need
3. Click the download link

Available versions include ADI data based on recent ACS 5-year estimates (the 2023 ADI is the most current as of this writing). Earlier versions are also available.

### Step 5: Review the Scoring System

The ADI provides two types of scores:

**National Ranking (1--100):**
- Ranks all U.S. block groups against each other nationally
- 1 = least disadvantaged, 100 = most disadvantaged
- Block groups with insufficient data are coded as missing

**State Decile (1--10):**
- Ranks block groups within each state separately
- 1 = least disadvantaged, 10 = most disadvantaged
- Constructed by ranking the ADI from low to high within each state without consideration of the national ADI

### Step 6: Link to Your Data

The ADI is designed to be linked to individual-level data using the census block group FIPS code (12-digit code: 2-digit state + 3-digit county + 6-digit tract + 1-digit block group). You can geocode addresses to block groups using the U.S. Census Bureau geocoder or commercial geocoding services.

**Important:** The ADI team recommends only linking at the block group level. Linking to larger geographic units (ZIP codes, ZCTAs, census tracts) is not a validated approach and will introduce error.

## Data Format

- **Primary format:** CSV files (comma-separated values)
- **Columns include:** Block group FIPS code, state FIPS, county FIPS, national ranking (1--100), state decile (1--10)
- **Mapping:** Interactive web-based maps on the Neighborhood Atlas portal
- **Documentation:** Methodology descriptions and FAQs on the Neighborhood Atlas website

## Key Tables / Files

The ADI download file contains the following key fields:

- `FIPS` (or `GISJOIN`) — 12-digit census block group FIPS code (state + county + tract + block group)
- `ADI_NATRANK` — National ADI ranking (1--100; 1 = least disadvantaged, 100 = most disadvantaged)
- `ADI_STATEFIPS` — State FIPS code
- `ADI_STATERANK` (or `ADI_STATEDECILE`) — State-level decile ranking (1--10; 1 = least disadvantaged, 10 = most disadvantaged)

The 17 component measures used to compute the ADI span four domains:

**Income and Employment:**
- Median family income
- Income disparity (percentage of households with income below a threshold vs. above)
- Percentage of families below poverty level
- Percentage of population below 150% of poverty threshold
- Unemployment rate (age 16+)

**Education:**
- Percentage of population aged 25+ with less than 9 years of education
- Percentage of population aged 25+ with at least a high school diploma

**Housing Quality:**
- Median home value
- Median gross rent
- Median monthly mortgage
- Percentage of owner-occupied housing units
- Percentage of housing units without complete plumbing
- Percentage of housing units without a telephone
- Percentage of occupied housing units with more than one person per room (crowding)

**Household Characteristics:**
- Percentage of single-parent households with dependents
- Percentage of households with persons under 65 with a disability

## Important Restrictions

- **Block group level only:** The ADI is validated only at the census block group level. Aggregating or linking to larger geographic units (ZIP codes, census tracts, counties) introduces unvalidated error and is discouraged by the ADI team.
- **Cross-version comparisons:** ADI rankings from different years/versions should not be directly compared because the underlying ACS data, census geography, and in some cases the methodology may differ.
- **Known methodological critique:** A published critique in Health Affairs Scholar notes that because the ADI component variables are not standardized before computing scores, indicators measured in dollars (particularly median home value) receive disproportionately greater weight. Users should be aware of this when interpreting ADI scores.
- **Citation required:** Users should cite the ADI and Neighborhood Atlas in publications. Recommended citations:
  - Kind AJH, Buckingham WR. "Making Neighborhood-Disadvantage Metrics Accessible -- The Neighborhood Atlas." New England Journal of Medicine, 2018; 378(26): 2456-2458.
  - University of Wisconsin School of Medicine and Public Health. Area Deprivation Index. Downloaded from [Neighborhood Atlas](https://www.neighborhoodatlas.medicine.wisc.edu/)
- **Registration required for downloads:** While free, you must create an account before downloading data files.

## Alternative Data Source

A standardized version of the ADI is also available through the **National Neighborhood Data Archive (NaNDA)** at ICPSR (University of Michigan):

[https://www.openicpsr.org/openicpsr/project/210581](https://www.openicpsr.org/openicpsr/project/210581)

The NaNDA version provides ADI scores for 2015, 2020, and 2022 with standardized construction methodology.

## Useful Links

- [Neighborhood Atlas Home Page](https://www.neighborhoodatlas.medicine.wisc.edu/)
- [ADI Datasets on HIPxChange](https://hipxchange.org/toolkit/adi/)
- [Kind et al. 2018, NEJM (original paper)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6051533/)
- [NaNDA Standardized ADI (ICPSR)](https://www.openicpsr.org/openicpsr/project/210581)
- [ADI on SparkMap](https://sparkmap.org/data-info/area-deprivation-index/)
- [U.S. Census Bureau Geocoder (for address-to-block-group matching)](https://geocoding.geo.census.gov/geocoder/)
