# UK Biobank -- Access Instructions

## Overview

UK Biobank is a large-scale biomedical database and research resource containing genetic, lifestyle, and health information from approximately 500,000 volunteer participants recruited across the United Kingdom between 2006 and 2010. It is operated as an independent charity and is one of the most significant health research resources in the world, supporting over 22,000 researchers across more than 60 countries and contributing to more than 18,000 peer-reviewed publications. UK Biobank data spans electronic health records, imaging (brain, cardiac, and body MRI; DEXA scans), genomics (whole genome sequencing, genotyping arrays, exome sequencing), proteomics, physical activity monitoring, and extensive questionnaire data covering demographics, lifestyle, and medical history.

## Access Level

**Restricted Access -- Institutional Application + Fee** -- UK Biobank data is available to all bona fide researchers from academic, charity, government, and commercial organizations worldwide for health-related research that is in the public interest. Access requires registration, a formal application describing the proposed research, review by the UK Biobank Access Committee, execution of a Material Transfer Agreement (MTA), and payment of an access fee. Since 2024, all research data is accessed exclusively through the cloud-based UK Biobank Research Analysis Platform (UKB-RAP), powered by DNAnexus.

## Prerequisites

Before applying for UK Biobank access, ensure you have:

- **Bona Fide Researcher Status**: You must be affiliated with a recognized academic, government, charity, or commercial research organization.
- **Research Proposal**: A well-defined research question that falls within health-related research in the public interest.
- **Budget for Fees**:
  - Standard access fee (check current rates at [ukbiobank.ac.uk/use-our-data/fees](https://www.ukbiobank.ac.uk/use-our-data/fees/))
  - Student researchers: Reduced fee of 500 GBP (+VAT) for data-only access
  - Lower-income country researchers: 500 GBP for 3 years (175 GBP/year extension); may qualify for full fee coverage through the Global Researcher Access Fund
  - UKB-RAP compute and storage costs (charged via AWS/Google Cloud; initial credits provided)
- **Institutional Support**: Your institution's authorized signatory must execute the Material Transfer Agreement.

## Step-by-Step Access

### Step 1: Register on the Access Management System (AMS)

Navigate to [https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) and register on the UK Biobank Access Management System (AMS). You will need to provide your identity, institutional affiliation, and contact information. Registration confirms your bona fide researcher status.

### Step 2: Submit a Research Application

Through the AMS, submit a formal application describing:
- Your proposed research question and objectives
- The specific data fields, samples, or data types you require
- How the research is in the public interest
- Whether you require depletable biological samples (which undergo additional review)
- Your team members (each must register individually on the AMS)

The UK Biobank Access Management Team will perform an initial review of your application within approximately 10 working days.

### Step 3: Access Committee Review

Most applications are approved by the epidemiologist review team. Applications involving depletable samples, participant recontact, or potentially contentious research are escalated to the Access Committee (AC), a sub-committee of the UK Biobank Board that meets quarterly. The AC makes final decisions on these complex applications.

**Note**: UK Biobank holds Research Tissue Bank (RTB) approval from its governing ethics committee, which covers the majority of proposed research uses. Researchers typically do not need to obtain separate ethics approval.

### Step 4: Execute the Material Transfer Agreement (MTA)

Once your application is approved, you (or your institution's authorized signatory) must execute the Material Transfer Agreement. The MTA defines the legal terms under which data and/or samples are provided.

### Step 5: Pay the Access Fee

Pay the applicable access fee before data release. Fees are tiered based on researcher status (standard, student, lower-income country). Financial support may be available through the Global Researcher Access Fund (supported by AstraZeneca, Bristol Myers Squibb, GSK, Johnson & Johnson, and Regeneron).

### Step 6: Access Data on the Research Analysis Platform (UKB-RAP)

Once payment is confirmed, data is dispensed to your project on the UK Biobank Research Analysis Platform (UKB-RAP), hosted on DNAnexus. You will receive:
- 40 GBP in initial DNAnexus credits
- 1,000 GBP in project credits to support research (available until Summer 2026)
- Access to Jupyter notebooks (Python/R) and command-line tools for analysis

All analysis is performed within the secure cloud environment. Data egress is restricted and incurs additional charges.

## Data Format

UK Biobank data is delivered through the UKB-RAP in the following formats:

| Data Type | Format |
|-----------|--------|
| **Phenotypic / Tabular Data** | Parquet tables on UKB-RAP (dispensed as `participant_0001`, `participant_0002`, etc.); exportable to CSV, XLSX, TSV |
| **Genotyping Arrays** | PLINK (bed/bim/fam), BGEN (imputed data) |
| **Whole Genome Sequencing** | CRAM files (alignment), pVCF (variant calls) |
| **Exome Sequencing** | CRAM and pVCF files |
| **Imaging (MRI)** | Bulk files organized by field category (DICOM-derived, NIFTI for processed data) |
| **DEXA Scans** | Bulk imaging files |
| **Accelerometer / Physical Activity** | Bulk data files (CWA format, processed summary statistics) |
| **Proteomics** | Tabular data integrated into participant tables |

Bulk data files follow a naming convention: `ukb<FIELD-ID>_c<CHROM>_b<BLOCK>_v<VERSION>.<SUFFIX>` for genomic files; participant-organized subfolders for imaging data.

## Key Tables / Files

| Category | Description |
|----------|-------------|
| **Participant Tables** | Horizontally split Parquet tables (`participant_0001` through `participant_NNNN`) containing one row per participant and columns for each approved data field |
| **Genotyping Data** | ~800K SNP array data for all 500K participants; imputed variants (~90M SNPs) in BGEN format |
| **Whole Genome Sequencing** | Short-read WGS for ~500K participants in CRAM and pVCF format |
| **Brain MRI** | Structural, functional, and diffusion MRI for ~100K participants |
| **Cardiac MRI** | Heart structure and function imaging |
| **Body MRI (Abdominal)** | Organ volume and fat distribution imaging |
| **DEXA** | Full-body dual-energy X-ray absorptiometry scans |
| **Accelerometer Data** | 7-day wrist-worn accelerometer data for ~100K participants |
| **Primary Care (GP) Data** | Linked general practitioner records (diagnoses, prescriptions, referrals) |
| **Hospital Episode Statistics** | Linked inpatient, outpatient, and A&E records from NHS Digital |
| **Death Registry** | Linked mortality data from national death registries |
| **Cancer Registry** | Linked cancer registration data |
| **COVID-19 Data** | Test results, primary care, and hospital data related to COVID-19 |

## Important Restrictions

- **Public Interest Requirement**: All research using UK Biobank data must be health-related and in the public interest.
- **No Re-identification**: Any attempt to identify participants is strictly prohibited. Participant IDs are pseudonymized (randomized per application).
- **Secure Environment**: All analysis must be performed within the UKB-RAP cloud environment. Data egress is restricted and monitored.
- **No Unauthorized Sharing**: Data must not be shared with individuals not listed on the approved application. Each team member must register individually.
- **MTA Compliance**: All terms of the Material Transfer Agreement must be followed. Violations may result in access revocation.
- **Return of Results**: Research findings should be returned to UK Biobank to enrich the resource for the broader community.
- **Publication Requirements**: UK Biobank must be acknowledged in all publications. Research outputs should be reported back to UK Biobank.
- **Time-Limited Access**: Access is granted for a defined period (typically 3 years) and must be renewed thereafter.

## Useful Links

- [Apply for Access](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access/)
- [UK Biobank -- Enable Your Research](https://www.ukbiobank.ac.uk/enable-your-research)
- [Access Fees](https://www.ukbiobank.ac.uk/use-our-data/fees/)
- [Financial Support / Global Researcher Access Fund](https://www.ukbiobank.ac.uk/use-our-data/fees/financial-support/)
- [Access Procedures Document (PDF)](https://www.ukbiobank.ac.uk/wp-content/uploads/2025/01/Access-procedures.pdf)
- [AMS Registration Guide](https://community.ukbiobank.ac.uk/hc/en-gb/articles/28023058927261-AMS-User-Guide-Registering)
- [Applying to UK Biobank FAQ](https://community.ukbiobank.ac.uk/hc/en-gb/articles/28002689502237-Applying-to-UK-Biobank-and-accessing-data-FAQ)
- [UKB-RAP Documentation (DNAnexus GitBook)](https://dnanexus.gitbook.io/uk-biobank-rap)
- [UKB-RAP Notebooks -- Access (GitHub)](https://github.com/UK-Biobank/UKB-RAP-Notebooks-Access)
- [UKB-RAP Notebooks -- Genomics (GitHub)](https://github.com/UK-Biobank/UKB-RAP-Notebooks-Genomics)
- [Genetic Data Description (Resource 531)](https://biobank.ctsu.ox.ac.uk/crystal/refer.cgi?id=531)
- [UK Biobank Home](https://www.ukbiobank.ac.uk/)
