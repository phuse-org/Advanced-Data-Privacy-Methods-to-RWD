# WHO GHO (Global Health Observatory) — Access Instructions

## Overview

The Global Health Observatory (GHO) is the World Health Organization's gateway to health-related statistics for its 194 member states. The GHO provides access to over 2,300 health indicators spanning decades of data on topics including mortality, disease burden, health systems, environmental health, nutrition, risk factors, and the Sustainable Development Goals (SDGs).

The GHO serves as WHO's primary open data platform, offering interactive data exploration, visualization tools, a RESTful OData API, and bulk download capabilities. All data are freely available without registration.

## Access Level

**Open** — All GHO data are freely available without registration, API keys, or data use agreements. The OData API requires no authentication.

## Prerequisites

None. Data can be accessed immediately through the web portal, API, or bulk downloads.

For programmatic access, familiarity with REST APIs and JSON/XML formats is helpful. R and Python wrapper packages are available.

## Step-by-Step Access

### Step 1: Explore the GHO Web Portal

Navigate to the GHO home page:

[https://www.who.int/data/gho](https://www.who.int/data/gho)

The portal provides browsable themes and topics including:
- Air pollution
- Child health
- Immunization
- HIV/AIDS
- Malaria
- Tuberculosis
- Noncommunicable diseases
- Mortality and global health estimates
- Health workforce
- Universal health coverage
- Sustainable Development Goals (SDGs)

Click on any theme or indicator to view interactive visualizations and data tables.

### Step 2: Access Data via the OData API (Recommended for Programmatic Use)

The GHO OData API is the primary programmatic access method.

**Base URL:** `https://ghoapi.azureedge.net/api/`

No API key or authentication is required.

**List all available indicators:**
```
https://ghoapi.azureedge.net/api/Indicator
```

**Search for indicators by keyword:**
```
https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,'life expectancy')
```

**Download data for a specific indicator (by code):**
```
https://ghoapi.azureedge.net/api/WHOSIS_000001
```
(This example retrieves "Life expectancy at birth" for all countries and years.)

**List dimension values (e.g., countries):**
```
https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues
```

**Filter data by dimension:**
You can apply OData filters to restrict results by country, year, sex, and other dimensions using standard OData `$filter` syntax.

### Step 3: Use the API from R

```r
library(httr)
library(jsonlite)

# Fetch life expectancy data
url <- "https://ghoapi.azureedge.net/api/WHOSIS_000001"
response <- GET(url)
data <- fromJSON(content(response, "text", encoding = "UTF-8"))
df <- data$value
```

Alternatively, community-maintained R packages are available for working with GHO data.

### Step 4: Use the API from Python

```python
import requests
import pandas as pd

# Fetch life expectancy data
url = "https://ghoapi.azureedge.net/api/WHOSIS_000001"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data["value"])
```

A Python wrapper package (`WHO_GHO_API_client`) is also available on GitHub for simplified data retrieval.

### Step 5: Use the API from Power BI

The GHO OData API can be directly connected to Power BI using the OData connector. Enter the base URL (`https://ghoapi.azureedge.net/api/`) as the data source.

### Step 6: Download Data via the Web Interface

For manual downloads of individual indicators:
1. Navigate to a specific indicator page on the GHO portal
2. Click the data table view
3. Use the download option to export data as CSV or Excel

For bulk access, use the GHO Data Repository:

[https://apps.who.int/gho/data/node.main](https://apps.who.int/gho/data/node.main)

### Step 7: Explore Additional WHO Data Platforms

WHO is transitioning GHO content to a newer platform:

[https://data.who.int](https://data.who.int)

This platform features SDG indicator pages, dashboards, and additional content. Data are available for download in CSV format.

## Data Format

- **API output:** JSON (default) or XML
- **Web downloads:** CSV, Excel (XLS/XLSX)
- **Data structure:** Each API response contains a header (metadata about the request) and a body (the data records), with fields for indicator code, country code, year, sex, value, and other dimensions

## Key Tables / Files

GHO data are organized by indicators rather than files. Key indicators include:

**Mortality and Life Expectancy:**
- `WHOSIS_000001` — Life expectancy at birth (years)
- `WHOSIS_000002` — Healthy life expectancy (HALE) at birth
- `MDG_0000000001` — Infant mortality rate (per 1,000 live births)
- `NCDMORT3070` — Probability of dying from NCDs between ages 30 and 70

**Disease Burden:**
- `MALARIA001` — Malaria incidence (per 1,000 population at risk)
- `TB_e_inc_num` — Tuberculosis incidence
- `HIV_0000000001` — HIV prevalence

**Health Systems:**
- `HWF_0006` — Physicians per 10,000 population
- `UHC_INDEX_REPORTED` — Universal health coverage index

**Risk Factors:**
- `TOBACCO_0000000253` — Prevalence of tobacco smoking
- `NCD_BMI_30A` — Prevalence of obesity (BMI >= 30)

**Environmental Health:**
- `AIR_41` — Ambient air pollution (PM2.5, annual mean)
- `WSH_SANITATION_SAFELY_MANAGED` — Safely managed sanitation services

Key data dimensions:
- `COUNTRY` — ISO 3-letter country code
- `YEAR` — Data year
- `SEX` — Sex disaggregation
- `AGEGROUP` — Age group disaggregation
- `REGION` — WHO region

## Important Restrictions

- **Terms of use:** WHO data are provided under the WHO open data policy. Users must cite the World Health Organization as the data source.
- **Data quality varies by country:** Data availability, completeness, and quality vary significantly across countries and indicators. WHO applies modeling and estimation methods for many indicators; check the indicator metadata for methodology details.
- **API deprecation notice:** The current GHO OData API (at `ghoapi.azureedge.net`) is planned for deprecation as WHO transitions to a new OData implementation. Users should monitor WHO announcements for migration details. The newer platform at `data.who.int` will host future data releases.
- **Not real-time data:** GHO data represent compiled statistics, typically with a lag of one to several years.
- **Citation:** Cite as "World Health Organization. Global Health Observatory data repository. [Indicator name]. Geneva: World Health Organization; [Year accessed]."

## Useful Links

- [GHO Home Page](https://www.who.int/data/gho)
- [GHO OData API Documentation](https://www.who.int/data/gho/info/gho-odata-api)
- [GHO Data Repository (legacy)](https://apps.who.int/gho/data/node.main)
- [WHO Data Portal (new platform)](https://data.who.int)
- [GHO Indicator List via API](https://ghoapi.azureedge.net/api/Indicator)
- [WHO GHO API Python Client (GitHub)](https://github.com/tobykylaw/WHO_GHO_API_client)
- [GHO on World Bank Data360](https://data360.worldbank.org/en/dataset/WHO_GHO)
