# Open mHealth / mPower / Beiwe -- Digital Health Wearables Data -- Access Instructions

## Overview

This dataset folder covers digital health and wearable sensor data from three interconnected platforms and studies:

- **mPower Study**: A landmark clinical observational study of Parkinson's disease (PD) conducted through an iPhone app built on Apple's ResearchKit framework, managed by Sage Bionetworks. The study collected data from 8,320 participants through smartphone surveys and sensor-based tasks (tapping, voice, walking, memory) to monitor key PD indicators. It is one of the largest open-access Parkinson's disease mobile health studies.

- **Open mHealth (OmH)**: An open interoperability standard and data schema for patient-generated health data from wearable devices and mobile health apps. Open mHealth provides standardized JSON schemas for representing health measurements (e.g., heart rate, blood glucose, physical activity) and tools for converting vendor-specific data into standardized formats compatible with HL7 FHIR.

- **Beiwe Research Platform**: A high-throughput data collection platform for smartphone-based digital phenotyping, developed by the Onnela Lab at Harvard T.H. Chan School of Public Health. Beiwe collects passive smartphone sensor data (GPS, accelerometer, gyroscope, screen activity, communication logs) and active survey data for research in behavioral and mental health, particularly schizophrenia and other psychiatric conditions.

## Access Level

**Controlled Access** -- The mPower study data is available through the Synapse platform (operated by Sage Bionetworks) after completing registration and governance requirements. Open mHealth schemas and tools are fully open source. Beiwe platform software is open source, but data collected through Beiwe in specific studies is governed by each study's own IRB and data sharing agreements.

## Prerequisites

### For mPower Data (Primary Dataset)
- **Synapse Account**: Create a free account at [synapse.org](https://www.synapse.org/).
- **Qualified Researcher Status**: You may need to validate your researcher profile on Synapse, including your institutional affiliation.
- **Governance Compliance**: Complete the data-specific governance requirements on the Synapse platform (data use agreement, intended use description).

### For Open mHealth Tools
- **No Prerequisites**: The Open mHealth schemas and conversion tools are open source and freely available.

### For Beiwe Platform
- **AWS Account** (for deploying Beiwe backend): The Beiwe server runs on Amazon Web Services.
- **IRB Approval**: Required for collecting new data from human subjects.
- **No Prerequisites for Software**: The Beiwe source code is open source on GitHub.

## Step-by-Step Access

### mPower Study Data (via Synapse)

#### Step 1: Create a Synapse Account

Navigate to [https://www.synapse.org/](https://www.synapse.org/) and register for a free account. Provide your name, email, institutional affiliation, and set up your profile.

#### Step 2: Navigate to the mPower Project

Go to the mPower Public Researcher Portal at [https://www.synapse.org/mPower](https://www.synapse.org/mPower). This Synapse project serves as the central documentation and data access point for the mPower study.

#### Step 3: Complete Governance Requirements

Review and fulfill the data governance requirements, which include:
- Agreeing to the data use conditions established to respect participants' privacy
- Describing your intended use of the data
- Acknowledging the terms of data sharing

The governance structures reflect the balance between participants' desire to share their data and the need to protect their privacy.

#### Step 4: Access and Download the Data

Once governance requirements are fulfilled, you can access the coded study data through:
- **Synapse Web Interface**: Browse and download data files directly
- **Synapse Python Client**: `pip install synapseclient` for programmatic access
- **Synapse R Client**: `install.packages("synapser")` for R-based access
- **Synapse Command Line Client**: For batch downloads

You can also explore the data through the [dHealth Digital Health Data Portal](https://dhealth.synapse.org/Explore/Collections/DetailsPage?study=mPower+Mobile+Parkinson+Disease+Study).

### Open mHealth Schemas

#### Step 1: Visit the Open mHealth Website

Navigate to [https://www.openmhealth.org/](https://www.openmhealth.org/) to review the available schemas and tools.

#### Step 2: Access the Schema Library

Review the Open mHealth JSON schema specifications at [https://www.openmhealth.org/mhealth-schema/](https://www.openmhealth.org/mhealth-schema/). Schemas cover domains including physical activity, vital signs, sleep, and clinical measurements.

#### Step 3: Use Conversion Tools

Use the Open mHealth tools (Shimmer, OMH-to-FHIR converters) to transform vendor-specific wearable data into standardized formats.

### Beiwe Platform

#### Step 1: Access the Source Code

Clone the Beiwe repositories from GitHub:
- Frontend (Android): Java/Kotlin-based mobile app
- Frontend (iOS): Native iOS app
- Backend: Python 3-based server running on AWS

#### Step 2: Deploy for Your Study

Deploy the Beiwe backend on AWS using the provided setup instructions. Configure sensor data collection parameters (GPS frequency, accelerometer sampling rate, survey schedules) for your specific study needs.

## Data Format

### mPower Study Data
The mPower data consists of:
- **Survey Responses**: Structured data from participant questionnaires (demographics, MDS-UPDRS, PDQ-8)
- **Sensor Measurements**: Time-series data from iPhone sensors captured during active tasks
  - **Tapping Task**: Touch coordinates and timestamps
  - **Voice Task**: Audio recordings (m4a format) with acoustic features
  - **Walking/Balance Task**: Accelerometer and gyroscope readings (JSON)
  - **Memory Task**: Response accuracy and timing data
- **Format**: JSON files and tabular data (TSV/CSV) hosted on Synapse

### Open mHealth
- **JSON Schemas**: IEEE-standard Open mHealth schemas define data points as JSON objects with a header (creation timestamp, schema reference) and a body (measurement values with units and timestamps).
- **FHIR Mapping**: Open mHealth provides mappings to HL7 FHIR Observation resources for interoperability with clinical systems.

### Beiwe Platform Data
- **Passive Sensor Data**: CSV files organized by sensor type (GPS, accelerometer, gyroscope, magnetometer, Wi-Fi, Bluetooth, screen state, call/text logs)
- **Active Survey Data**: JSON-formatted survey responses
- **Storage**: Raw data is stored in Amazon S3; PostgreSQL for metadata
- **Typical Volume**: High-throughput; a single participant can generate gigabytes of sensor data per week

## Key Tables / Files

### mPower Study Modules

| Module | Description |
|--------|-------------|
| **Demographics** | Participant age, sex, diagnosis status, medication use |
| **MDS-UPDRS** | Movement Disorder Society Unified Parkinson's Disease Rating Scale responses |
| **PDQ-8** | Parkinson's Disease Questionnaire (8-item quality of life measure) |
| **Tapping** | Finger tapping speed and regularity data from touchscreen |
| **Voice** | Sustained phonation recordings for vocal tremor analysis |
| **Walking** | Accelerometer and gyroscope data during gait tasks |
| **Memory** | Spatial memory game performance data |

### Beiwe Data Streams

| Stream | Sampling | Description |
|--------|----------|-------------|
| **GPS** | Configurable (e.g., 1 min on / 10 min off) | Location coordinates, altitude, accuracy |
| **Accelerometer** | Configurable (e.g., 10 Hz) | 3-axis acceleration data |
| **Gyroscope** | Configurable | 3-axis angular velocity |
| **Screen State** | Event-driven | Screen on/off/unlock events |
| **Call / Text Logs** | Event-driven | Hashed contact IDs, duration, type |
| **Wi-Fi** | Periodic scans | Nearby Wi-Fi networks (hashed BSSIDs) |
| **Bluetooth** | Periodic scans | Nearby Bluetooth devices |
| **Surveys** | Scheduled | Researcher-designed active data collection |

## Important Restrictions

### mPower Data
- **Qualified Researcher Use**: Data is intended for qualified researchers with legitimate research purposes.
- **Participant Privacy**: Governance structures are in place to protect participant privacy. The overwhelming majority of participants consented to broad data sharing, but researchers must still respect privacy protections.
- **No Re-identification**: Any attempt to identify individual participants is prohibited.
- **Citation Required**: When publishing results, cite the original mPower publication:
  > Bot BM, et al. The mPower study, Parkinson disease mobile data collected using ResearchKit. *Scientific Data*. 2016;3:160011. doi: 10.1038/sdata.2016.11.

### Open mHealth
- **Open Source**: Schemas and tools are available under open source licenses. No restrictions on use.

### Beiwe Platform
- **IRB Required**: Any new data collection using Beiwe requires IRB approval from your institution.
- **Data Security**: Beiwe implements end-to-end encryption and secure AWS storage for participant data.
- **Participant Consent**: Active informed consent is required from all participants in Beiwe-based studies.
- **HIPAA Compliance**: Beiwe is designed to support HIPAA-compliant data collection when properly configured.

## Useful Links

### mPower Study
- [mPower Public Researcher Portal (Synapse)](https://www.synapse.org/mPower)
- [dHealth Digital Health Data Portal -- mPower](https://dhealth.synapse.org/Explore/Collections/DetailsPage?study=mPower+Mobile+Parkinson+Disease+Study)
- [mPower Data Preparation Scripts (GitHub)](https://github.com/Sage-Bionetworks/mPower-sdata)
- [mPower Publication (Scientific Data, 2016)](https://www.nature.com/articles/sdata201611)

### Open mHealth
- [Open mHealth Home](https://www.openmhealth.org/)
- [Open mHealth Schema Documentation](https://www.openmhealth.org/mhealth-schema/)
- [Wearable Data Mapping to OmH and FHIR (PubMed)](https://pubmed.ncbi.nlm.nih.gov/37387034/)

### Beiwe Platform
- [Beiwe -- Onnela Lab, Harvard T.H. Chan School of Public Health](https://hsph.harvard.edu/research/onnela-lab/digital-phenotyping-and-beiwe-research-platform/)
- [Beiwe Publication (ResearchGate)](https://www.researchgate.net/publication/357079735_Beiwe_A_data_collection_platform_for_high-throughput_digital_phenotyping)

### General
- [Sage Bionetworks / Synapse Home](https://www.synapse.org/)
- [Synapse Python Client Documentation](https://python-docs.synapse.org/)
