# OpenFDA — Access Instructions

## Overview

OpenFDA is the U.S. Food and Drug Administration's (FDA) open data initiative, providing public access to FDA datasets through a RESTful API built on Elasticsearch. OpenFDA serves publicly available data about drugs, medical devices, foods, tobacco products, and other FDA-regulated products, including adverse event reports, product recalls, product labeling, and registration data.

OpenFDA was created to make it easier for developers, researchers, and the public to access and analyze FDA data. It is not intended to be a comprehensive source of all FDA information but rather a convenient programmatic interface to key public datasets.

## Access Level

**Open** — OpenFDA data are freely available with no registration required for basic use. An optional API key is available for higher rate limits. No data use agreement is required.

- **Without API key:** 40 requests per minute, 1,000 requests per day
- **With API key (free):** 240 requests per minute, 120,000 requests per day

## Prerequisites

- **For basic API use:** None. You can query the API directly from a web browser or command line.
- **For higher rate limits:** Register for a free API key at [open.fda.gov/apis/authentication](https://open.fda.gov/apis/authentication/)
- **For bulk downloads:** None. Download files are publicly accessible.
- **For programmatic use:** Familiarity with REST APIs and JSON format. R and Python packages are available.

## Step-by-Step Access

### Step 1: Navigate to the OpenFDA Website

Go to the OpenFDA home page:

[https://open.fda.gov/](https://open.fda.gov/)

Explore the available API endpoints and documentation.

### Step 2: Understand the Available Endpoints

OpenFDA provides endpoints organized by regulated product category:

**Drug Endpoints:**
| Endpoint | Description | URL Path |
|----------|-------------|----------|
| Adverse Events | FAERS adverse event and medication error reports | `/drug/event.json` |
| Product Labeling | Structured Product Labeling (SPL) drug labels | `/drug/label.json` |
| NDC Directory | National Drug Code directory | `/drug/ndc.json` |
| Recall Enforcement | Drug recall and enforcement reports | `/drug/enforcement.json` |
| Drugs@FDA | Drug application information | `/drug/drugsfda.json` |

**Device Endpoints:**
| Endpoint | Description | URL Path |
|----------|-------------|----------|
| Adverse Events | Medical device adverse event reports (MAUDE) | `/device/event.json` |
| Recalls | Device recall reports | `/device/recall.json` |
| Classification | Device classification data | `/device/classification.json` |
| 510(k) | Premarket notifications | `/device/510k.json` |
| PMA | Premarket approval data | `/device/pma.json` |
| Registration & Listing | Device establishment registrations | `/device/registrationlisting.json` |
| UDI | Unique Device Identification data | `/device/udi.json` |

**Food Endpoints:**
| Endpoint | Description | URL Path |
|----------|-------------|----------|
| Recall Enforcement | Food recall and enforcement reports | `/food/enforcement.json` |
| Adverse Events | CFSAN Adverse Event Reporting System (CAERS) | `/food/event.json` |

**Other Endpoints:**
| Endpoint | Description | URL Path |
|----------|-------------|----------|
| Animal & Veterinary Events | Animal drug adverse events | `/animalandveterinary/event.json` |
| Tobacco Problem Reports | Tobacco product problem reports | `/tobacco/problem.json` |

### Step 3: Query the API

**Base URL:** `https://api.fda.gov`

**Basic query structure** (template — replace braced values with actual parameters):
```
https://api.fda.gov/{endpoint}?search={field}:{term}&limit={number}
```

**Example — Search drug adverse events for aspirin:**
```
https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:"aspirin"&limit=5
```

**Example — Count adverse events by reaction:**
```
https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:"aspirin"&count=patient.reaction.reactionmeddrapt.exact
```

**Example — Search drug labels:**
```
https://api.fda.gov/drug/label.json?search=openfda.brand_name:"advil"&limit=1
```

**Example — Food recalls in a date range** (note: `[…+TO+…]` is Elasticsearch range syntax, not a placeholder):
```
https://api.fda.gov/food/enforcement.json?search=report_date:[20230101+TO+20231231]&limit=10
```

You can test queries directly in your browser or using the interactive tool:

[https://open.fda.gov/apis/try-the-api/](https://open.fda.gov/apis/try-the-api/)

### Step 4: Register for an API Key (Optional but Recommended)

For higher rate limits, register for a free API key:

1. Go to [https://open.fda.gov/apis/authentication/](https://open.fda.gov/apis/authentication/)
2. Enter your email address
3. An API key will be emailed to you
4. Include the key in your requests: `&api_key=YOUR_KEY_HERE`

### Step 5: Download Bulk Data Files

For large-scale analysis, download complete endpoint datasets as bulk files rather than using the API:

[https://open.fda.gov/data/downloads/](https://open.fda.gov/data/downloads/)

The downloads page provides ZIP archives of the complete dataset for each endpoint. Files are in JSON format (one JSON record per line, organized into partitioned files).

Available downloads include:
- Drug Adverse Events
- Drug Labels
- Drug NDC Directory
- Drug Enforcement Reports
- Device Adverse Events (MAUDE)
- Device Recalls
- Device 510(k)
- Device PMA
- Device Classification
- Device Registration & Listing
- Device UDI
- Food Enforcement
- Food Adverse Events
- Animal & Veterinary Events
- Tobacco Problem Reports

### Step 6: Use Programming Libraries

**R Package (openFDA):**
```r
install.packages("openFDA")
library(openFDA)

# Query drug adverse events
results <- openFDA::fda_query("/drug/event.json") |>
  fda_search(patient.drug.openfda.generic_name = "metformin") |>
  fda_limit(10) |>
  fda_exec()
```

**Python (using requests):**
```python
import requests
import json

url = "https://api.fda.gov/drug/event.json"
params = {
    "search": 'patient.drug.openfda.generic_name:"metformin"',
    "limit": 10
}
response = requests.get(url, params=params)
data = response.json()
results = data["results"]
```

**JavaScript / Node.js:**
```javascript
const response = await fetch(
  'https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:"metformin"&limit=10'
);
const data = await response.json();
```

## Data Format

- **API responses:** JSON format with two top-level objects:
  - `meta` — Metadata including disclaimer, terms, license, pagination info, and total result count
  - `results` — Array of data records
- **Bulk downloads:** JSON files packaged in ZIP archives, partitioned into multiple files per endpoint
- **Query parameters:** Support for search, count, limit, and skip operations using OpenFDA's query syntax

## Key Tables / Files

**Drug Adverse Events (`/drug/event.json`) — Key Fields:**
- `safetyreportid` — Unique report identifier
- `receivedate` / `receiptdate` — When FDA received the report
- `serious` — Whether the event was serious (1 = yes, 2 = no)
- `seriousnessdeath` / `seriousnesshospitalization` — Specific seriousness criteria
- `patient.drug[]` — Array of drugs the patient was taking
  - `medicinalproduct` — Drug name as reported
  - `drugindication` — Reason the drug was taken
  - `openfda.brand_name` / `openfda.generic_name` — Standardized drug names
- `patient.reaction[]` — Array of adverse reactions
  - `reactionmeddrapt` — Reaction term (MedDRA preferred term)
  - `reactionoutcome` — Outcome (1=recovered, 2=recovering, 3=not recovered, 4=resolved with sequelae, 5=fatal, 6=unknown)
- `patient.patientonsetage` — Patient age at onset
- `patient.patientsex` — Patient sex (0=unknown, 1=male, 2=female)

**Drug Labels (`/drug/label.json`) — Key Fields:**
- `openfda.brand_name` / `openfda.generic_name` — Drug identifiers
- `indications_and_usage` — Approved indications
- `warnings` / `boxed_warning` — Safety warnings
- `adverse_reactions` — Adverse reactions section
- `dosage_and_administration` — Dosing information

**Recall Enforcement (`/drug/enforcement.json`) — Key Fields:**
- `recall_number` — Unique recall identifier
- `product_description` — Description of recalled product
- `reason_for_recall` — Why the product was recalled
- `classification` — Class I (most serious), Class II, or Class III
- `status` — Ongoing, completed, or terminated
- `distribution_pattern` — Geographic distribution of the product

## Important Restrictions

- **Not for clinical decision-making:** OpenFDA data should not be used to make decisions regarding medical care. Always consult a health provider about the risks and benefits of FDA-regulated products.
- **Not for public alerts:** The data should not be used as a method to issue alerts to the public or to track the lifecycle of a recall in real time.
- **Adverse event limitations:** FAERS adverse event reports are voluntarily submitted and do not establish causation. Reporting rates vary by drug, condition, and time period. The data cannot be used to calculate incidence rates or compare drug safety profiles without proper pharmacoepidemiologic methods.
- **Rate limits:** Without an API key, you are limited to 40 requests per minute and 1,000 per day. With a key, 240 per minute and 120,000 per day. FDA may restrict access for violations of the Terms of Service.
- **Data quality:** Not all data in OpenFDA have been validated for clinical or production use. Some fields may contain errors, duplicates, or missing values.
- **Terms of Service:** Use is subject to the [OpenFDA Terms of Service](https://open.fda.gov/terms/)
- **No PII:** OpenFDA does not contain Personally Identifiable Information about patients or other sensitive information.

## Useful Links

- [OpenFDA Home Page](https://open.fda.gov/)
- [OpenFDA API Documentation](https://open.fda.gov/apis/)
- [Try the API (Interactive)](https://open.fda.gov/apis/try-the-api/)
- [Bulk Data Downloads](https://open.fda.gov/data/downloads/)
- [API Authentication (Get API Key)](https://open.fda.gov/apis/authentication/)
- [Drug Adverse Events Endpoint](https://open.fda.gov/apis/drug/event/)
- [Drug Label Endpoint](https://open.fda.gov/apis/drug/label/)
- [Drug Enforcement/Recalls Endpoint](https://open.fda.gov/apis/drug/enforcement/)
- [OpenFDA Terms of Service](https://open.fda.gov/terms/)
- [openFDA R Package (CRAN)](https://cran.r-project.org/web/packages/openFDA/)
- [OpenFDA GitHub Organization](https://github.com/FDA)
