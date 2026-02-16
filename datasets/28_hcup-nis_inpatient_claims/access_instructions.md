# HCUP National Inpatient Sample (NIS) -- Access Instructions

## Overview

The Healthcare Cost and Utilization Project (HCUP) National Inpatient Sample (NIS) is the largest publicly available all-payer inpatient care database in the United States. It is developed and maintained through a Federal-State-Industry partnership sponsored by the Agency for Healthcare Research and Quality (AHRQ). The NIS contains data from approximately 7 million unweighted hospital stays each year (estimating more than 35 million weighted hospitalizations nationally) and is designed to produce U.S. regional and national estimates of inpatient utilization, access, cost, quality, and outcomes. Data is available annually beginning in 1988. Starting with the 2012 data year, the NIS is a sample of discharges from all HCUP-participating hospitals (covering more than 97% of the U.S. population), whereas prior years used a sample of hospitals.

## Access Level

**Restricted Access -- Fee-Based** -- The NIS is available for purchase through the HCUP Central Distributor. Access requires completion of a mandatory online Data Use Agreement (DUA) training course, signing a Nationwide Data Use Agreement, and paying a per-year database fee. The data is considered restricted because, although de-identified, it contains granular discharge-level clinical and demographic information. Institutional IRB review documentation may also be required.

## Prerequisites

Before ordering NIS data, ensure you have:

- **HCUP DUA Training**: Complete the mandatory online HCUP Data Use Agreement Training Course at [https://www.hcup-us.ahrq.gov/tech_assist/dua.jsp](https://www.hcup-us.ahrq.gov/tech_assist/dua.jsp). Save your certificate of completion.
- **Signed Data Use Agreement**: Download, review, and sign the Nationwide Data Use Agreement, available at [https://www.hcup-us.ahrq.gov/team/NationwideDUA.pdf](https://www.hcup-us.ahrq.gov/team/NationwideDUA.pdf).
- **IRB Documentation** (if required by your institution): Obtain documentation of IRB review status from your institution.
- **Statistical Software**: SAS, SPSS, or Stata to load and analyze the data. Load programs for all three are provided by HCUP.
- **Budget**: The NIS is purchased on a per-data-year basis. Student pricing is available at a discount. Historically, costs have ranged from approximately $350 to $500 per year, though current pricing should be confirmed through the Central Distributor.

## Step-by-Step Access

### Step 1: Complete the DUA Training Course

Navigate to [https://www.hcup-us.ahrq.gov/tech_assist/dua.jsp](https://www.hcup-us.ahrq.gov/tech_assist/dua.jsp) and complete the online HCUP Data Use Agreement Training Course. This course covers the terms and conditions of data use, reporting restrictions, and your obligations as a data recipient. Save or print your certificate of completion.

### Step 2: Sign the Nationwide Data Use Agreement

Download the Nationwide DUA form from [https://www.hcup-us.ahrq.gov/team/NationwideDUA.pdf](https://www.hcup-us.ahrq.gov/team/NationwideDUA.pdf). Read the agreement carefully, complete all fields, and sign it. The DUA must be signed by anyone seeking permission from AHRQ to access HCUP Nationwide databases.

### Step 3: Register on the HCUP Central Distributor

Create an account on the HCUP Central Distributor Online Reporting System (CDORS) at [https://cdors.ahrq.gov/](https://cdors.ahrq.gov/). This portal manages the ordering, purchasing, and distribution of HCUP databases.

### Step 4: Submit Your Data Application

Through CDORS, submit a data application that includes:
- Your completed DUA training certificate
- Your signed Nationwide DUA
- IRB documentation (if applicable)
- Identification of the specific NIS data year(s) you wish to purchase

### Step 5: Pay the Database Fee

After your application is reviewed and approved, you will receive instructions for payment. Payment processing is performed by NORC at the University of Chicago. Each data year is purchased separately. Student researchers may qualify for discounted pricing.

### Step 6: Receive and Load the Data

Upon payment confirmation, you will receive access to download the NIS data files. The data is delivered as compressed ASCII files (SecureZIP format). Download the corresponding SAS, SPSS, or Stata load programs from the HCUP-US website to convert the ASCII data into your preferred statistical software format.

## Data Format

The NIS is distributed as **fixed-width ASCII text files** compressed with SecureZIP from PKWARE. Load programs are provided for:

- **SAS**: Available from 1998 onward
- **SPSS**: Available from 1998 onward
- **Stata**: Available from 2004 onward

The load programs convert the ASCII files into analysis-ready datasets in the respective software formats.

## Key Tables / Files

Each NIS data year includes the following files:

| File | Unit of Observation | Description |
|------|---------------------|-------------|
| **Inpatient Core File** | Inpatient stay (discharge) | Primary file containing clinical and demographic data: ICD-9/10 diagnosis and procedure codes, patient demographics, admission/discharge dates, total charges, payer information, disposition |
| **Hospital Weights File** | Hospital | One record per hospital; contains sampling weights for producing national estimates and variance estimation data elements |
| **Disease Severity Measures File** | Inpatient stay | Severity and comorbidity measures (e.g., All Patient Refined DRGs, Elixhauser comorbidity measures); links to Core file via KEY |
| **Diagnosis and Procedure Groups File** | Inpatient stay | Categorization of diagnoses and procedures into clinically meaningful groups (e.g., Clinical Classifications Software) |

Key data elements in the Core file include:
- **KEY**: Unique record identifier linking across files
- **DX1-DXn**: ICD diagnosis codes (up to 40 per record)
- **PR1-PRn**: ICD procedure codes (up to 25 per record)
- **AGE, FEMALE, RACE**: Patient demographics
- **LOS**: Length of stay
- **TOTCHG**: Total charges
- **PAY1**: Expected primary payer
- **DISPUNIFORM**: Disposition of patient

## Important Restrictions

- **Cell Size Suppression**: The HCUP DUA prohibits reporting any counts (unweighted or weighted) less than 11. All statistics based on small cell sizes must be suppressed in publications and presentations.
- **No Re-identification**: Any attempt to identify patients, providers, or hospitals is prohibited.
- **No Redistribution**: Data may not be shared with individuals who have not completed their own DUA and training.
- **Research Use Only**: Data must be used solely for research, statistical reporting, and related purposes.
- **Data Security**: You must maintain appropriate physical and electronic security for all data files.
- **Annual DUA Renewal**: The DUA applies to specific data years; additional purchases require confirmation that terms remain in effect.
- **Citation Required**: Acknowledge the HCUP NIS and AHRQ in all publications. Use the citation format specified in the HCUP documentation.

## Useful Links

- [Purchase HCUP Data (Central Distributor)](https://hcup-us.ahrq.gov/tech_assist/centdist.jsp)
- [HCUP Central Distributor Online Ordering (CDORS)](https://cdors.ahrq.gov/)
- [NIS Overview](https://hcup-us.ahrq.gov/nisoverview.jsp)
- [NIS Database Documentation](https://hcup-us.ahrq.gov/db/nation/nis/nisdbdocumentation.jsp)
- [NIS Data Element Descriptions](https://hcup-us.ahrq.gov/db/nation/nis/nisdde.jsp)
- [DUA Training Course](https://www.hcup-us.ahrq.gov/tech_assist/dua.jsp)
- [Nationwide DUA Form (PDF)](https://www.hcup-us.ahrq.gov/team/NationwideDUA.pdf)
- [SAS Load Programs](https://hcup-us.ahrq.gov/db/nation/sasloadprog.jsp)
- [Stata Load Programs](https://hcup-us.ahrq.gov/db/nation/stataloadprog.jsp)
- [HCUP Home Page](https://hcup-us.ahrq.gov/)
- [HCUP FAQs](https://hcup-us.ahrq.gov/tech_assist/faq.jsp)
- [HCUP Central Distributor Contact: HCUPDistributor@AHRQ.gov / 866-556-4287](mailto:HCUPDistributor@AHRQ.gov)
