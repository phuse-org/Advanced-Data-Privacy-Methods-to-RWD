# EPA AQS (Air Quality System) — Access Instructions

## Overview

The Air Quality System (AQS) is the EPA's repository for ambient air quality monitoring data collected across the United States. AQS contains data from over 10,000 monitors operated by EPA, state, local, and tribal air pollution control agencies. The database includes measurements of criteria pollutants (ozone, particulate matter, carbon monoxide, sulfur dioxide, nitrogen dioxide, lead), hazardous air pollutants (air toxics), and meteorological data.

AQS also stores descriptive information about each monitoring station (geographic location, operating organization, measurement methods) and comprehensive data quality assurance/quality control information. Data in AQS date back to the 1950s for some pollutants.

## Access Level

**Open** — AQS data are freely available through multiple access channels. Pre-generated data files require no registration. The AQS Data Mart API requires a free email-based registration to obtain an API key but has no other restrictions.

## Prerequisites

- **For pre-generated downloads:** None. Files can be downloaded immediately.
- **For AQS Data Mart API:** A valid email address to register for a free API key.
- **For programming packages:** R or Python 3 environment with the respective EPA packages installed.

## Step-by-Step Access

### Step 1: Choose Your Access Method

EPA provides multiple ways to access AQS data. Choose the method that best fits your needs:

| Method | Best For | Registration |
|--------|----------|-------------|
| Pre-generated files | Bulk national data, historical analysis | None |
| AQS Data Mart API | Custom queries, specific monitors/regions | Free API key |
| AirData interactive tool | Exploration, visualization, quick lookups | None |
| R package (RAQSAPI) | Programmatic access in R | Free API key |
| Python package (pyaqsapi) | Programmatic access in Python | Free API key |

### Step 2a: Download Pre-Generated Data Files

Navigate to the pre-generated data files page:

[https://aqs.epa.gov/aqsweb/airdata/download_files.html](https://aqs.epa.gov/aqsweb/airdata/download_files.html)

This page provides national-level data files organized by:

- **Annual Summary Data** — One record per monitor per year, with summary statistics
- **Daily Summary Data** — One record per monitor per day
- **Hourly Data** — Individual hourly measurements (very large files)
- **8-Hour Average Data** — Rolling 8-hour averages (primarily for ozone)
- **AQI by County / CBSA** — Daily Air Quality Index values aggregated by geography
- **Monitor Descriptions** — Metadata about each monitoring site
- **Blanks Data** — Quality assurance blank sample data

Files are available by pollutant group:
- Ozone (44201)
- SO2 (42401)
- CO (42101)
- NO2 (42602)
- PM2.5 (FRM/FEM, 88101)
- PM2.5 (non-FRM/FEM, 88502)
- PM10 (81102)
- Lead (various parameter codes)
- HAPs (Hazardous Air Pollutants)
- VOCs (Volatile Organic Compounds)
- Nonoxides of Nitrogen (NONOxNOy)

Select the pollutant, time aggregation, and year, then download the ZIP file.

**Update schedule:** Files are updated twice per year — in June (complete data for the prior year) and in December (summer/ozone season data update).

### Step 2b: Use the AQS Data Mart API

**API documentation:** [AQS Data Mart API](https://aqs.epa.gov/aqsweb/documents/data_api.html)

**Step 1 — Register for an API key:**

Send a request to:
```
https://aqs.epa.gov/data/api/signup?email=your.email@example.com
```

An API key will be emailed to you. This key is required for all subsequent API calls.

**Step 2 — Query data:**

All API calls follow this template URI pattern (replace braced values with your actual parameters):
```
https://aqs.epa.gov/data/api/{service}/{filter}?email={email}&key={key}&param={parameters}
```

Available services include:
- `monitors` — Monitor metadata and operational information
- `sampleData` — Raw sample-level data (finest grain)
- `dailyData` — Daily summary data
- `annualData` — Annual summary data
- `qaOnePointQcRawData` — QA data

Data can be filtered by:
- State FIPS code
- County FIPS code
- Site number
- Parameter code (pollutant)
- Date range (begin and end dates, format: YYYYMMDD)
- Core-Based Statistical Area (CBSA)
- Monitoring Agency
- Bounding box (latitude/longitude)

**Step 3 — Parse the response:**

All output is in JSON format with two top-level elements:
- `Header` — Request metadata (status, row count, processing time)
- `Data` — Array of data records

### Step 2c: Use the RAQSAPI Package (R)

```r
install.packages("RAQSAPI")
library(RAQSAPI)

# Register (one-time)
aqs_sign_up(email = "your.email@example.com")

# Set credentials
aqs_credentials(username = "your.email@example.com", key = "your_api_key")

# Example: Get annual PM2.5 data for a county
data <- aqs_annualsummary_by_county(
  parameter = "88101",
  bdate = as.Date("2023-01-01"),
  edate = as.Date("2023-12-31"),
  stateFIPS = "06",
  countycode = "037"
)
```

### Step 2d: Use the pyaqsapi Package (Python)

```python
import pyaqsapi as aqs

# Register (one-time)
aqs.aqs_sign_up("your.email@example.com")

# Set credentials
aqs.aqs_credentials("your.email@example.com", "your_api_key")

# Example: Get daily ozone data for a state
data = aqs.bystate.dailydata(
    parameter="44201",
    bdate="20230101",
    edate="20231231",
    stateFIPS="06"
)
```

### Step 3: Understand the File Formats

Review the file format documentation at:

[https://aqs.epa.gov/aqsweb/airdata/FileFormats.html](https://aqs.epa.gov/aqsweb/airdata/FileFormats.html)

This page describes all columns in the pre-generated files, including parameter codes, units, measurement methods, and data quality flags.

## Data Format

- **Pre-generated files:** CSV files compressed in ZIP archives
- **API output:** JSON
- **R/Python packages:** Return data as data frames
- **File sizes:** Hourly files can be very large (hundreds of MB per pollutant per year); annual and daily summary files are more manageable

## Key Tables / Files

**Common variables across pre-generated files:**

- `State Code` / `County Code` / `Site Num` — Geographic identifiers (FIPS codes)
- `Parameter Code` — Numeric code identifying the pollutant (e.g., 44201 = Ozone, 88101 = PM2.5)
- `Parameter Name` — Human-readable pollutant name
- `POC` — Parameter Occurrence Code (distinguishes multiple monitors at one site)
- `Latitude` / `Longitude` — Monitor coordinates
- `Datum` — Geographic datum (typically WGS84 or NAD83)
- `Date Local` / `Date GMT` — Measurement date in local time and GMT
- `Sample Duration` — Averaging period (e.g., 1 Hour, 24 Hour)
- `Arithmetic Mean` — Average concentration for the period
- `1st Max Value` / `1st Max Hour` — Maximum concentration and when it occurred
- `AQI` — Air Quality Index value (where applicable)
- `Units of Measure` — Measurement units (e.g., Parts per million, Micrograms/cubic meter)
- `Method Name` — Analytical method used
- `Observation Count` — Number of valid observations in the summary period
- `Observation Percent` — Data completeness percentage

**Key pollutant parameter codes:**
| Code | Pollutant |
|------|-----------|
| 44201 | Ozone |
| 88101 | PM2.5 (FRM/FEM) |
| 88502 | PM2.5 (non-FRM/FEM) |
| 81102 | PM10 |
| 42401 | SO2 |
| 42101 | CO |
| 42602 | NO2 |
| 14129 | Lead (PM10) |

## Important Restrictions

- **Not real-time data:** AQS data undergo quality assurance review and are not suitable for real-time air quality alerts. For real-time data, use the [AirNow API](https://www.airnow.gov/).
- **Data completeness varies:** Not all monitors report data for all hours/days. Check the observation count and completeness percentage before analysis.
- **Method changes over time:** Measurement methods and federal reference methods have evolved. Be cautious when analyzing long-term trends and consult the method codes.
- **Pre-generated files are national:** There is no state-level filtering for pre-generated files; you must download the national file and filter locally.
- **API rate limits:** The Data Mart API may have usage limits. Avoid excessive concurrent requests.
- **Citation:** Cite as "United States Environmental Protection Agency. Air Quality System Data Mart [internet database]. Available from: [https://aqs.epa.gov/aqsweb/airdata/download_files.html](https://aqs.epa.gov/aqsweb/airdata/download_files.html). Accessed [Date]."

## Useful Links

- [AQS Home Page](https://www.epa.gov/aqs)
- [Obtaining AQS Data](https://www.epa.gov/aqs/obtaining-aqs-data)
- [Pre-Generated Data Files Download](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
- [AQS Data Mart API Documentation](https://aqs.epa.gov/aqsweb/documents/data_api.html)
- [AirData File Format Documentation](https://aqs.epa.gov/aqsweb/airdata/FileFormats.html)
- [AQS Data Mart Welcome Page](https://aqs.epa.gov/aqsweb/documents/data_mart_welcome.html)
- [RAQSAPI R Package (GitHub)](https://github.com/USEPA/RAQSAPI)
- [pyaqsapi Python Package (GitHub)](https://github.com/USEPA/pyaqsapi)
- [Best Way to Access Outdoor Air Monitoring Data (EPA Guide)](https://www.epa.gov/outdoor-air-quality-data/what-best-way-access-outdoor-air-monitoring-data)
- [AirNow (real-time data)](https://www.airnow.gov/)
