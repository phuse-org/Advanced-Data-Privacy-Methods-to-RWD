# i2b2 / n2c2 Clinical NLP Shared Task Data -- Access Instructions

## Overview

The i2b2/n2c2 datasets are a collection of annotated clinical text corpora originally created through the i2b2 (Informatics for Integrating Biology and the Bedside) project, a former NIH-funded National Center for Biomedical Computing. Beginning in 2018, these shared tasks are officially known as n2c2 (National NLP Clinical Challenges), a name that pays tribute to their i2b2 origins. The datasets are now hosted and administered by the Department of Biomedical Informatics (DBMI) at Harvard Medical School. They consist of fully de-identified clinical notes -- primarily discharge summaries and progress notes from Partners Healthcare (now Mass General Brigham), Beth Israel Deaconess Medical Center, and the University of Pittsburgh Medical Center -- with expert annotations for various NLP tasks spanning from 2006 through 2022.

## Access Level

**Controlled Access** -- The datasets are freely available to the research community at no cost, but access requires registration on the DBMI Data Portal, signing a Data Use Agreement (DUA), and agreeing to the Rules of Conduct. Each individual user must independently complete the registration and DUA process. Access approval may take several business days.

## Prerequisites

Before applying for access, ensure you have:

- **Research Purpose**: A legitimate research use case for clinical NLP data.
- **Individual Registration**: Each person who will use the data must register separately. Under no circumstances may data files be shared with or copied for additional individuals.
- **Faculty Sponsor** (for challenge participation): If participating in an active n2c2 challenge, team leaders must be faculty members (post-doc, research scientist, or professor). Teams consisting of solo students will not be approved for challenge participation (though data access for general research is available to all researchers).

## Step-by-Step Access

### Step 1: Visit the DBMI Data Portal

Navigate to the n2c2 NLP Research Data Sets page on the DBMI Data Portal at [https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

### Step 2: Register or Log In

Click "Login / Register" on the portal. If you are a first-time user, create a new account by providing your name, institutional affiliation, email address, and other required information. If you are a returning user, log in with your existing credentials.

### Step 3: Sign the Rules of Conduct

After logging in, you will be prompted to review and sign the Rules of Conduct, which govern responsible use of the data portal and its resources.

### Step 4: Sign the Data Use and Confidentiality Agreement

Review and sign the Data Use Agreement (DUA). The DUA outlines the terms under which the clinical data may be used, including prohibitions on redistribution, re-identification attempts, and posting data to external websites (including GitHub).

### Step 5: Request Access to Specific Datasets

Browse the available n2c2 datasets and request access to the specific challenge data you need. Each challenge year and track may have its own dataset with separate download links.

### Step 6: Download the Data

Once approved, download the datasets from the DBMI Data Portal. The data is organized by challenge year and track.

**Note**: As of the most recent check, some n2c2 datasets may be temporarily unavailable due to portal maintenance. The 2019 Challenge tracks 1 (Clinical Semantic Textual Similarity) and 2 (Family History Extraction) are available directly through Mayo Clinic.

## Data Format

The datasets consist of:

- **Clinical Notes**: Plain text files containing fully de-identified discharge summaries, progress notes, and other clinical documents.
- **Annotations**: XML format files containing expert annotations (ground truth labels) corresponding to each clinical note. The XML annotations encode entities, relations, attributes, and other task-specific labels.
- **Evaluation Scripts**: Python scripts for computing precision, recall, and F-measure against the ground truth annotations.

Each challenge typically provides separate training and test sets, and some include unannotated corpora for unsupervised or semi-supervised approaches.

## Key Tables / Files

The datasets are organized by challenge year. Major shared tasks include:

| Year | Challenge | Description |
|------|-----------|-------------|
| **2006** | De-identification & Smoking | Automatic de-identification of clinical text; smoking status classification from discharge records |
| **2008** | Obesity | Recognition of obesity and 15 co-morbidities from sparse clinical data |
| **2010** | Concepts, Assertions, Relations | Extraction of medical concepts (problems, tests, treatments), assertion classification, and relation extraction |
| **2011** | Coreference Resolution | Identifying when different mentions refer to the same entity |
| **2012** | Temporal Relations | Identifying clinical events and temporal expressions and their relationships |
| **2014** | De-identification & Heart Disease | Longitudinal clinical narratives; de-identification and risk factor identification for heart disease |
| **2018 Track 1** | Cohort Selection | Clinical trial cohort selection from longitudinal patient records |
| **2018 Track 2** | Adverse Drug Events | Medication extraction and adverse drug event detection in EHR notes |
| **2019 Track 1** | Clinical Semantic Similarity | Measuring semantic similarity between clinical sentence pairs (via Mayo Clinic) |
| **2019 Track 2** | Family History Extraction | Extracting family history information from clinical notes (via Mayo Clinic) |
| **2022 Track 1** | Medication Events | Contextualized medication event extraction with change classification |
| **2022 Track 2** | Social Determinants of Health | Extracting SDOH (substance use, employment, living status) from clinical notes |
| **2022 Track 3** | Progress Note Understanding | Assessment and plan reasoning in daily progress notes |

## Important Restrictions

- **No Redistribution**: Under no circumstances may copies of any data files be provided to additional individuals or posted to other websites, including GitHub, public repositories, or shared drives.
- **Individual Access Only**: Each researcher must independently register and sign the DUA through the DBMI Data Portal. Group or shared accounts are not permitted.
- **No Re-identification**: Any attempt to identify the patients, providers, or institutions in the clinical notes is strictly prohibited.
- **Research Use Only**: The data is provided exclusively for research purposes.
- **Citation Required**: When publishing results using i2b2/n2c2 data, cite the relevant challenge publications and acknowledge the data source.

## Useful Links

- [n2c2 NLP Research Data Sets (DBMI Data Portal)](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
- [DBMI Data Portal Home](https://portal.dbmi.hms.harvard.edu/)
- [n2c2 Homepage (Harvard DBMI)](https://n2c2.dbmi.hms.harvard.edu/)
- [n2c2 Data Use Agreement](https://n2c2.dbmi.hms.harvard.edu/data-use-agreement)
- [n2c2 Data Sets Overview](https://n2c2.dbmi.hms.harvard.edu/data-sets)
- [Legacy i2b2 NLP Data Sets Page](https://www.i2b2.org/NLP/DataSets/)
- [i2b2 Project Home](https://www.i2b2.org/)
