# Dataset Registry

This folder contains **metadata only** (dataset cards, documentation). No raw patient-level data is stored in GitHub.

Each numbered subfolder corresponds to one dataset. Add dataset-specific documentation, cards, and notes inside the relevant folder.

---

## Quick Reference — All Datasets

| # | Dataset | Domain | Modalities | Access | LLM Suitability | Folder |
|---|---------|--------|------------|--------|------------------|--------|
| 1 | [MIMIC-IV](#1-mimic-iv) | Clinical EHR / ICU | EHR (structured), Notes (text) | Controlled | High | `01_mimic-iv_icu_ehr/` |
| 2 | [MIMIC-IV-Note](#2-mimic-iv-note) | Clinical EHR Notes | Notes (clinical text) | Controlled | High | `02_mimic-iv-note_clinical_notes/` |
| 3 | [eICU-CRD](#3-eicu-crd) | Clinical ICU (multi-center) | EHR (structured), some notes | Controlled | High | `03_eicu-crd_multi-center_icu/` |
| 4 | [AmsterdamUMCdb](#4-amsterdamumcdb) | Clinical ICU (EU) | EHR (structured) | Controlled | Medium | `04_amsterdamumcdb_icu/` |
| 5 | [HiRID](#5-hirid) | Clinical ICU (Switzerland) | EHR (structured, time series) | Controlled | Medium | `05_hirid_icu_timeseries/` |
| 6 | [MIMIC-CXR](#6-mimic-cxr) | Imaging + Reports | Imaging (CXR), Radiology Reports (text) | Controlled | High | `06_mimic-cxr_radiology_reports/` |
| 7 | [PhysioNet ICU Waveforms](#7-physionet-icu-waveforms) | Physiologic Signals | ECG, ABP, PPG, etc. | Controlled | Medium | `07_physionet_icu_waveforms/` |
| 8 | [SEER](#8-seer) | Cancer Registry (U.S.) | Registry (coded clinical) | Controlled | Low | `08_seer_cancer_registry/` |
| 9 | [NCI IDC](#9-nci-idc) | Oncology Imaging | Imaging (CT/MRI/PET), metadata | Open | Medium | `09_nci-idc_oncology_imaging/` |
| 10 | [TCIA](#10-tcia) | Oncology Imaging | Imaging (CT/MRI/PET), some reports | Open | High | `10_tcia_cancer_imaging/` |
| 11 | [BIMCV COVID-19+](#11-bimcv-covid-19) | Radiology (COVID) | Imaging (CXR/CT), reports (some) | Open | High | `11_bimcv_covid19_imaging/` |
| 12 | [NIH ChestX-ray14](#12-nih-chestx-ray14) | Radiology | Imaging (CXR) + labels | Open | Medium | `12_nih_chestxray14/` |
| 13 | [CheXpert](#13-chexpert) | Radiology | Imaging (CXR) + labels | Controlled | Medium | `13_chexpert_chest_xray/` |
| 14 | [COVIDx-US](#14-covidx-us) | Ultrasound | Imaging (LUS) | Open | Medium | `14_covidx-us_ultrasound/` |
| 15 | [LC25000](#15-lc25000) | Histopathology | Imaging (histopathology) | Open | Medium | `15_lc25000_histopathology/` |
| 16 | [OpenNeuro](#16-openneuro) | Neuroscience | Imaging (MRI, fMRI, MEG, EEG) | Open | Medium | `16_openneuro_brain_imaging/` |
| 17 | [NHANES](#17-nhanes) | Public Health Survey (U.S.) | Survey, lab tests, examination | Open | Low | `17_nhanes_health_survey/` |
| 18 | [NHIS](#18-nhis) | Public Health Survey (U.S.) | Survey | Open | Low | `18_nhis_health_interview/` |
| 19 | [BRFSS](#19-brfss) | Behavioral Risk (U.S.) | Survey | Open | Low | `19_brfss_behavioral_risk/` |
| 20 | [WHO GHO](#20-who-gho) | Global Public Health | Aggregated indicators | Open | Low | `20_who-gho_global_health/` |
| 21 | [CDC/ATSDR SVI](#21-cdcatsdr-svi) | Social Determinants | Area-level indices | Open | Low | `21_cdc-svi_social_vulnerability/` |
| 22 | [EPA AQS](#22-epa-aqs) | Environment / Exposure | Sensor (ambient monitors) | Open | Medium | `22_epa-aqs_air_quality/` |
| 23 | [County Health Rankings](#23-county-health-rankings) | Public Health / SDoH | Aggregated indicators | Open | Low | `23_countyhealthrankings_sdoh/` |
| 24 | [ADI](#24-adi) | SDoH / Deprivation | Index (area-level) | Open | Low | `24_adi_area_deprivation/` |
| 25 | [i2b2/n2c2](#25-i2b2n2c2) | Clinical Text (de-identified) | Notes (text) | Controlled | High | `25_i2b2_n2c2_clinical_nlp/` |
| 26 | [EHRShot](#26-ehrshot) | Clinical EHR | EHR (structured) | Open | Medium | `26_ehrshot_benchmark/` |
| 27 | [SatHealth](#27-sathealth) | Public Health + Environment | Aggregated + satellite features | Open | Low | `27_sathealth_environment_sdoh/` |
| 28 | [HCUP NIS](#28-hcup-nis) | Claims / Utilization | Claims (inpatient encounters) | Restricted | Low | `28_hcup-nis_inpatient_claims/` |
| 29 | [OpenFDA](#29-openfda) | Regulatory / Safety | Product labels, adverse events | Open | Low | `29_openfda_regulatory/` |
| 30 | [UK Biobank](#30-uk-biobank) | Genomic + Clinical + Imaging | EHR, Imaging, Genomics | Restricted | Medium | `30_uk_biobank_multimodal/` |
| 31 | [All of Us](#31-all-of-us) | Clinical / Genomic / Lifestyle | EHR, surveys, genomics, wearables | Controlled | Medium | `31_allofus_nih_multimodal/` |
| 32 | [PhysioNet Challenge](#32-physionet-challenge) | Clinical Signals / Sensors | ECG, PPG, accelerometer | Open | Low | `32_physionet_challenge_signals/` |
| 33 | [Open mHealth](#33-open-mhealth) | Digital Health / Wearables | Smartphone sensors, wearables | Controlled | Medium | `33_openmhealth_wearables/` |

---

## Access Level Legend

| Level | Meaning |
|-------|---------|
| **Open** | Public download, no approval needed |
| **Controlled** | Free but requires registration, training, and/or DUA |
| **Restricted** | Fee-based or institutional application required |
| **Other** | Dataset-specific terms (see individual entries) |

---

## Dataset Details

### 1. MIMIC-IV

| Field | Details |
|-------|---------|
| **Full name** | Medical Information Mart for Intensive Care IV |
| **Domain** | Clinical EHR / ICU / ED |
| **Modalities** | EHR (structured), Notes (text) |
| **Population** | BIDMC ICU/ED patients, ~2008-2022; ~300k+ patients |
| **Approx variables** | Hundreds of columns across >20 tables (demographics, labs, meds, vitals, procedures, notes) |
| **Access level** | Controlled (Registration + CITI training + DUA, free) |
| **LLM suitability** | High |
| **Notes** | Excellent for ML/LLM; rich structured+text; single-center; de-identified |
| **Data dictionary** | [mimic.mit.edu/docs/iv](https://mimic.mit.edu/docs/iv/) |
| **Access instructions** | [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/) |

---

### 2. MIMIC-IV-Note

| Field | Details |
|-------|---------|
| **Full name** | MIMIC-IV Clinical Notes |
| **Domain** | Clinical EHR Notes |
| **Modalities** | Notes (clinical text) |
| **Population** | Subset of MIMIC-IV; millions of note documents |
| **Approx variables** | Note text + metadata fields |
| **Access level** | Controlled (Registration + CITI training + DUA, free) |
| **LLM suitability** | High |
| **Notes** | High value for clinical NLP/LLM fine-tuning (discharge summaries, radiology, etc.) |
| **Data dictionary** | [mimic.mit.edu/docs/iv/modules/notes](https://mimic.mit.edu/docs/iv/modules/notes/) |
| **Access instructions** | [physionet.org/content/mimic-iv-note](https://physionet.org/content/mimic-iv-note/) |

---

### 3. eICU-CRD

| Field | Details |
|-------|---------|
| **Full name** | eICU Collaborative Research Database |
| **Domain** | Clinical ICU EHR (multi-center) |
| **Modalities** | EHR (structured), some notes |
| **Population** | ~200k ICU admissions across >200 U.S. hospitals |
| **Approx variables** | Hundreds of variables across relational tables |
| **Access level** | Controlled (Registration + CITI training + DUA, free) |
| **LLM suitability** | High |
| **Notes** | Multi-center ICU data; strong for generalization and benchmarking |
| **Data dictionary** | [eicu-crd.mit.edu](https://eicu-crd.mit.edu/) |
| **Access instructions** | [physionet.org/content/eicu-crd](https://physionet.org/content/eicu-crd/) |

---

### 4. AmsterdamUMCdb

| Field | Details |
|-------|---------|
| **Full name** | Amsterdam University Medical Centers Database |
| **Domain** | Clinical ICU EHR (EU) |
| **Modalities** | EHR (structured) |
| **Population** | Amsterdam UMC ICU; adult critical care stays |
| **Approx variables** | Dozens to hundreds of variables; relational DB + parquet |
| **Access level** | Controlled (Registration + DUA, free) |
| **LLM suitability** | Medium |
| **Notes** | Non-U.S. ICU cohort; complements MIMIC/eICU for external validation |
| **Data dictionary** | [amsterdammedicaldatascience.nl](https://amsterdammedicaldatascience.nl/) |
| **Access instructions** | [amsterdammedicaldatascience.nl/#amsterdamumcdb](https://amsterdammedicaldatascience.nl/#amsterdamumcdb) |

---

### 5. HiRID

| Field | Details |
|-------|---------|
| **Full name** | High Resolution ICU Dataset |
| **Domain** | Clinical ICU EHR (Switzerland) |
| **Modalities** | EHR (structured, high-resolution time series) |
| **Population** | Bern University Hospital ICU |
| **Approx variables** | Hundreds of time series signals + metadata |
| **Access level** | Controlled (Registration + DUA, free) |
| **LLM suitability** | Medium |
| **Notes** | Rich high-resolution ICU data for sequence models; good for DP/synthetic pipelines |
| **Data dictionary** | [hirid.intensivecare.ai](https://hirid.intensivecare.ai/) |
| **Access instructions** | [physionet.org/content/hirid](https://physionet.org/content/hirid/) |

---

### 6. MIMIC-CXR

| Field | Details |
|-------|---------|
| **Full name** | MIMIC Chest X-Ray |
| **Domain** | Clinical Imaging + Reports |
| **Modalities** | Imaging (CXR), Radiology Reports (text) |
| **Population** | Chest radiographs from BIDMC with associated reports |
| **Approx variables** | Images + DICOM headers + text reports |
| **Access level** | Controlled (Registration + DUA, free) |
| **LLM suitability** | High |
| **Notes** | Multimodal image+text; ideal for vision-language modelling |
| **Data dictionary** | [mimic.mit.edu/docs/iv/modules/cxr](https://mimic.mit.edu/docs/iv/modules/cxr/) |
| **Access instructions** | [physionet.org/content/mimic-cxr](https://physionet.org/content/mimic-cxr/) |

---

### 7. PhysioNet ICU Waveforms

| Field | Details |
|-------|---------|
| **Full name** | MIMIC-IV Waveform / PhysioNet ICU Waveforms |
| **Domain** | Clinical Physiologic Signals |
| **Modalities** | Sensor (ECG, ABP, PPG, etc.) |
| **Population** | ICU physiologic waveform recordings (various cohorts) |
| **Approx variables** | Dozens of channels/signals per patient; long time series |
| **Access level** | Controlled (varies by dataset; many require registration/DUA) |
| **LLM suitability** | Medium |
| **Notes** | Great for time series DL; can be linked to clinical tables in some cohorts |
| **Data dictionary** | [physionet.org/about/database](https://physionet.org/about/database/) |
| **Access instructions** | [physionet.org](https://physionet.org/) |

---

### 8. SEER

| Field | Details |
|-------|---------|
| **Full name** | Surveillance, Epidemiology, and End Results Program |
| **Domain** | Cancer Registry (U.S.) |
| **Modalities** | Registry (coded clinical, incidence, survival) |
| **Population** | U.S. population-based cancer registry; millions of cases since 1975 |
| **Approx variables** | Hundreds (site, histology, stage, treatment, survival) |
| **Access level** | Controlled (Registration + DUA, free) |
| **LLM suitability** | Low |
| **Notes** | Excellent for population cancer analytics; limited free-text; useful labels/outcomes |
| **Data dictionary** | [seer.cancer.gov/data](https://seer.cancer.gov/data/) |
| **Access instructions** | [seer.cancer.gov/data/access.html](https://seer.cancer.gov/data/access.html) |

---

### 9. NCI IDC

| Field | Details |
|-------|---------|
| **Full name** | NCI Imaging Data Commons |
| **Domain** | Oncology Imaging |
| **Modalities** | Imaging (CT/MRI/PET etc.), metadata |
| **Population** | Tens of thousands of studies; >80 TB imaging (public) |
| **Approx variables** | Image metadata dictionaries + collection-level clinical variables |
| **Access level** | Open (most collections CC-BY) |
| **LLM suitability** | Medium |
| **Notes** | Large-scale imaging for DL/vision-language; cloud-native access |
| **Data dictionary** | [learn.canceridc.dev](https://learn.canceridc.dev/) |
| **Access instructions** | [portal.imaging.datacommons.cancer.gov](https://portal.imaging.datacommons.cancer.gov/) |

---

### 10. TCIA

| Field | Details |
|-------|---------|
| **Full name** | The Cancer Imaging Archive |
| **Domain** | Oncology Imaging |
| **Modalities** | Imaging (CT/MRI/PET), some collections with reports |
| **Population** | Hundreds of curated collections; thousands of subjects |
| **Approx variables** | Collection-specific; DICOM metadata + annotations where available |
| **Access level** | Open (with terms) |
| **LLM suitability** | High |
| **Notes** | Gold-standard for open cancer imaging; many benchmarks |
| **Data dictionary** | [wiki.cancerimagingarchive.net](https://wiki.cancerimagingarchive.net/) |
| **Access instructions** | [cancerimagingarchive.net/access-data](https://www.cancerimagingarchive.net/access-data/) |

---

### 11. BIMCV COVID-19+

| Field | Details |
|-------|---------|
| **Full name** | BIMCV COVID-19 Positive |
| **Domain** | Imaging (Radiology, COVID) |
| **Modalities** | Imaging (CXR/CT), reports (some) |
| **Population** | COVID-19 positive patients in Valencia region (Spain) |
| **Approx variables** | Images + clinical/radiology metadata |
| **Access level** | Open (research terms) |
| **LLM suitability** | High |
| **Notes** | Useful for imaging ML and report-linked multimodal tasks |
| **Data dictionary** | [bimcv.org/datasets/bimcv-covid19](http://bimcv.org/datasets/bimcv-covid19/) |
| **Access instructions** | [bimcv.org/datasets](http://bimcv.org/datasets/) |

---

### 12. NIH ChestX-ray14

| Field | Details |
|-------|---------|
| **Full name** | NIH Clinical Center ChestX-ray14 |
| **Domain** | Imaging (Radiology) |
| **Modalities** | Imaging (CXR) + 14 disease labels |
| **Population** | 100k+ frontal chest X-rays |
| **Approx variables** | Image pixels + 14 binary labels + metadata |
| **Access level** | Open (registration form) |
| **LLM suitability** | Medium |
| **Notes** | Widely used benchmark for CXR classification |
| **Data dictionary** | [nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| **Access instructions** | [nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC) |

---

### 13. CheXpert

| Field | Details |
|-------|---------|
| **Full name** | CheXpert: Large Chest Radiograph Dataset |
| **Domain** | Imaging (Radiology) |
| **Modalities** | Imaging (CXR) + 14 labels + uncertainty indicators |
| **Population** | 224,000 chest radiographs from Stanford |
| **Approx variables** | Images + 14 labels + uncertainty indicators |
| **Access level** | Controlled (Registration + DUA, free) |
| **LLM suitability** | Medium |
| **Data dictionary** | [stanfordmlgroup.github.io/competitions/chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| **Access instructions** | [stanfordmlgroup.github.io/competitions/chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/) |

---

### 14. COVIDx-US

| Field | Details |
|-------|---------|
| **Full name** | COVIDx-US (Lung Ultrasound) |
| **Domain** | Imaging (Ultrasound) |
| **Modalities** | Imaging (LUS) |
| **Population** | 12k+ images from ~1.3k patients (COVID vs non-COVID) |
| **Approx variables** | Images + metadata |
| **Access level** | Open (research terms) |
| **LLM suitability** | Medium |
| **Notes** | Useful for ultrasound DL; COVID benchmarks |
| **Data dictionary** | [github.com/nrc-cnrc/COVID-US](https://github.com/nrc-cnrc/COVID-US) |
| **Access instructions** | [github.com/nrc-cnrc/COVID-US](https://github.com/nrc-cnrc/COVID-US) |

---

### 15. LC25000

| Field | Details |
|-------|---------|
| **Full name** | Lung and Colon Cancer Histopathological Images (25,000) |
| **Domain** | Histopathology |
| **Modalities** | Imaging (histopathology) |
| **Population** | 25,000 images (5 classes) |
| **Approx variables** | Images + class labels |
| **Access level** | Open (Kaggle terms, free) |
| **LLM suitability** | Medium |
| **Notes** | Balanced multi-class dataset for pathology DL |
| **Data dictionary** | [kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) |
| **Access instructions** | [kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) |

---

### 16. OpenNeuro

| Field | Details |
|-------|---------|
| **Full name** | OpenNeuro |
| **Domain** | Neuroscience (MRI/MEG/EEG) |
| **Modalities** | Imaging (MRI, fMRI, MEG, EEG), behavioral/metadata |
| **Population** | Hundreds of studies; thousands of subjects |
| **Approx variables** | BIDS metadata + imaging files; variables vary per study |
| **Access level** | Open (CC0 preferred) |
| **LLM suitability** | Medium |
| **Notes** | BIDS-standardized datasets; strong for neuro ML and multimodal |
| **Data dictionary** | [openneuro.org](https://openneuro.org/) |
| **Access instructions** | [openneuro.org](https://openneuro.org/) |

---

### 17. NHANES

| Field | Details |
|-------|---------|
| **Full name** | National Health and Nutrition Examination Survey |
| **Domain** | Public Health Survey (U.S.) |
| **Modalities** | Survey, lab tests, examination data |
| **Population** | Nationally representative U.S. sample, ongoing since 1960s |
| **Approx variables** | Thousands across many modules |
| **Access level** | Open |
| **LLM suitability** | Low |
| **Notes** | Rich health + labs; great for population modelling |
| **Data dictionary** | [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/Default.aspx) |
| **Access instructions** | [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/Default.aspx) |

---

### 18. NHIS

| Field | Details |
|-------|---------|
| **Full name** | National Health Interview Survey |
| **Domain** | Public Health Survey (U.S.) |
| **Modalities** | Survey |
| **Population** | Annual cross-sectional U.S. health interview survey |
| **Approx variables** | Thousands depending on year |
| **Access level** | Open |
| **LLM suitability** | Low |
| **Notes** | Health status, access, utilization; good for trend modelling |
| **Data dictionary** | [cdc.gov/nchs/nhis/data-questionnaires-documentation.htm](https://www.cdc.gov/nchs/nhis/data-questionnaires-documentation.htm) |
| **Access instructions** | [cdc.gov/nchs/nhis/data-questionnaires-documentation.htm](https://www.cdc.gov/nchs/nhis/data-questionnaires-documentation.htm) |

---

### 19. BRFSS

| Field | Details |
|-------|---------|
| **Full name** | Behavioral Risk Factor Surveillance System |
| **Domain** | Behavioral Risk (U.S.) |
| **Modalities** | Survey |
| **Population** | Largest continuously conducted health survey in the world (U.S. adults) |
| **Approx variables** | Hundreds per year plus modules |
| **Access level** | Open |
| **LLM suitability** | Low |
| **Notes** | Behavioral risk factors; state-level estimates |
| **Data dictionary** | [cdc.gov/brfss/annual_data/annual_data.htm](https://www.cdc.gov/brfss/annual_data/annual_data.htm) |
| **Access instructions** | [cdc.gov/brfss/annual_data/annual_data.htm](https://www.cdc.gov/brfss/annual_data/annual_data.htm) |

---

### 20. WHO GHO

| Field | Details |
|-------|---------|
| **Full name** | WHO Global Health Observatory |
| **Domain** | Global Public Health |
| **Modalities** | Aggregated indicators |
| **Population** | 194 countries; multiple health topics |
| **Approx variables** | Thousands of indicators |
| **Access level** | Open |
| **LLM suitability** | Low |
| **Notes** | Macro-level modelling; not patient-level |
| **Data dictionary** | [who.int/data/gho](https://www.who.int/data/gho) |
| **Access instructions** | [who.int/data/gho](https://www.who.int/data/gho) |

---

### <a id="21-cdcatsdr-svi"></a>21. CDC/ATSDR SVI

| Field | Details |
|-------|---------|
| **Full name** | CDC/ATSDR Social Vulnerability Index |
| **Domain** | Social Determinants / SDoH |
| **Modalities** | Area-level indices |
| **Population** | U.S. census tract/county level |
| **Approx variables** | ~15 themes/variables + component indicators |
| **Access level** | Open |
| **LLM suitability** | Low |
| **Notes** | Useful as contextual features linked to clinical datasets |
| **Data dictionary** | [atsdr.cdc.gov/placeandhealth/svi](https://www.atsdr.cdc.gov/placeandhealth/svi/index.html) |
| **Access instructions** | [atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html](https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html) |

---

### 22. EPA AQS

| Field | Details |
|-------|---------|
| **Full name** | EPA Air Quality System |
| **Domain** | Environment / Exposure |
| **Modalities** | Sensor (ambient monitors) |
| **Population** | U.S. nationwide monitoring network; decades of data |
| **Approx variables** | Hundreds of pollutants/metrics across stations |
| **Access level** | Open |
| **LLM suitability** | Medium |
| **Notes** | Exposure features for ML; can be linked by geocode/time |
| **Data dictionary** | [epa.gov/aqs](https://www.epa.gov/aqs) |
| **Access instructions** | [epa.gov/aqs/aqs-data-mart-api](https://www.epa.gov/aqs/aqs-data-mart-api) |

---

### 23. County Health Rankings

| Field | Details |
|-------|---------|
| **Full name** | County Health Rankings & Roadmaps |
| **Domain** | Public Health / SDoH |
| **Modalities** | Aggregated indicators |
| **Population** | U.S. counties; annual indicators |
| **Approx variables** | Hundreds of county-level measures |
| **Access level** | Open |
| **LLM suitability** | Low |
| **Notes** | Good for contextual features, fairness/inequity analyses |
| **Data dictionary** | [countyhealthrankings.org/explore-health-rankings/measures-data-sources](https://www.countyhealthrankings.org/explore-health-rankings/measures-data-sources) |
| **Access instructions** | [countyhealthrankings.org/explore-health-rankings/rankings-data-documentation](https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation) |

---

### 24. ADI

| Field | Details |
|-------|---------|
| **Full name** | Area Deprivation Index |
| **Domain** | SDoH / Deprivation |
| **Modalities** | Index (area-level) |
| **Population** | U.S. neighborhoods (block group) |
| **Approx variables** | 17 census-based indicators |
| **Access level** | Open (registration for downloads) |
| **LLM suitability** | Low |
| **Notes** | Common equity-related feature for RWE modelling |
| **Data dictionary** | [neighborhoodatlas.medicine.wisc.edu](https://www.neighborhoodatlas.medicine.wisc.edu/) |
| **Access instructions** | [neighborhoodatlas.medicine.wisc.edu/download](https://www.neighborhoodatlas.medicine.wisc.edu/download) |

---

### <a id="25-i2b2n2c2"></a>25. i2b2/n2c2

| Field | Details |
|-------|---------|
| **Full name** | i2b2/n2c2 Clinical NLP Challenges |
| **Domain** | Clinical Text (de-identified) |
| **Modalities** | Notes (text) + annotations |
| **Population** | Assorted tasks (de-id, relations, concepts) on real clinical notes |
| **Approx variables** | Text + annotations for tasks |
| **Access level** | Controlled (DUA + approval, free) |
| **LLM suitability** | High |
| **Notes** | High-value for LLM/NLP on clinical text |
| **Data dictionary** | [n2c2.dbmi.hms.harvard.edu](https://n2c2.dbmi.hms.harvard.edu/) |
| **Access instructions** | [portal.dbmi.hms.harvard.edu/projects/n2c2-nlp](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) |

---

### 26. EHRShot

| Field | Details |
|-------|---------|
| **Full name** | EHRShot |
| **Domain** | Clinical EHR |
| **Modalities** | EHR (structured) |
| **Population** | ~6,700 patients; few-shot learning benchmark |
| **Approx variables** | Diagnoses, meds, labs, demographics (dozens+) |
| **Access level** | Open (research terms) |
| **LLM suitability** | Medium |
| **Notes** | Purpose-built for foundation/transfer learning tasks |
| **Data dictionary** | [github.com/som-shahlab/ehrshot-benchmark](https://github.com/som-shahlab/ehrshot-benchmark) |
| **Access instructions** | [github.com/som-shahlab/ehrshot-benchmark](https://github.com/som-shahlab/ehrshot-benchmark) |

---

### 27. SatHealth

| Field | Details |
|-------|---------|
| **Full name** | SatHealth |
| **Domain** | Public Health + Environment + SDoH |
| **Modalities** | Aggregated health indicators, satellite/environmental features |
| **Population** | Regional U.S. (starting with Ohio) |
| **Approx variables** | Hundreds of engineered features (ENV + SDoH + prevalence) |
| **Access level** | Open (research terms) |
| **LLM suitability** | Low |
| **Notes** | Multimodal context features; useful for fairness and geospatial ML |
| **Data dictionary** | [arxiv.org/abs/2506.13842](https://arxiv.org/abs/2506.13842) |
| **Access instructions** | [arxiv.org/abs/2506.13842](https://arxiv.org/abs/2506.13842) |

---

### 28. HCUP NIS

| Field | Details |
|-------|---------|
| **Full name** | HCUP National Inpatient Sample |
| **Domain** | Claims / Utilization |
| **Modalities** | Claims (inpatient encounters) |
| **Population** | U.S. nationwide sample of hospital discharges (~7 million per year) |
| **Approx variables** | 100+ variables per record (diagnosis, procedure, demographics, costs) |
| **Access level** | Restricted (fee-based; DUA required) |
| **LLM suitability** | Low |
| **Notes** | Excellent for cost/utilization modelling and comorbidity analysis |
| **Data dictionary** | [hcup-us.ahrq.gov/db/nation/nis/nisdbdocumentation.jsp](https://www.hcup-us.ahrq.gov/db/nation/nis/nisdbdocumentation.jsp) |
| **Access instructions** | [hcup-us.ahrq.gov/tech_assist/centdist.jsp](https://www.hcup-us.ahrq.gov/tech_assist/centdist.jsp) |

---

### 29. OpenFDA

| Field | Details |
|-------|---------|
| **Full name** | OpenFDA |
| **Domain** | Regulatory / Safety |
| **Modalities** | Product labels, adverse events, device and drug recalls |
| **Population** | Millions of drug/device records from FDA databases |
| **Approx variables** | Hundreds across APIs (adverse events, recalls, NDC, etc.) |
| **Access level** | Open (API) |
| **LLM suitability** | Low |
| **Notes** | Structured text for NLP/LLM regulatory modelling |
| **Data dictionary** | [open.fda.gov/apis](https://open.fda.gov/apis/) |
| **Access instructions** | [open.fda.gov/data](https://open.fda.gov/data/) |

---

### 30. UK Biobank

| Field | Details |
|-------|---------|
| **Full name** | UK Biobank |
| **Domain** | Genomic + Clinical + Imaging |
| **Modalities** | EHR, Imaging, Genomics, Questionnaires |
| **Population** | ~500,000 UK participants aged 40-69 |
| **Approx variables** | >7,000 fields + genetic and imaging data |
| **Access level** | Restricted (fee + application) |
| **LLM suitability** | Medium |
| **Notes** | Extremely rich multimodal dataset for ML/LLM; strong governance |
| **Data dictionary** | [biobank.ndph.ox.ac.uk/showcase](https://biobank.ndph.ox.ac.uk/showcase/) |
| **Access instructions** | [ukbiobank.ac.uk/enable-your-research/apply-for-access](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) |

---

### 31. All of Us

| Field | Details |
|-------|---------|
| **Full name** | All of Us Research Program (NIH) |
| **Domain** | Clinical / Genomic / Lifestyle |
| **Modalities** | EHR, surveys, genomics, wearables |
| **Population** | >500,000 participants (U.S.) |
| **Approx variables** | Thousands of data fields (EHR + surveys + Fitbit + DNA) |
| **Access level** | Controlled (Registration + DUA, tiered access) |
| **LLM suitability** | Medium |
| **Notes** | Premier U.S. multimodal cohort for ML/LLM and fairness analyses |
| **Data dictionary** | [researchallofus.org/data-tools/data-browser](https://www.researchallofus.org/data-tools/data-browser/) |
| **Access instructions** | [researchallofus.org/register](https://www.researchallofus.org/register/) |

---

### 32. PhysioNet Challenge

| Field | Details |
|-------|---------|
| **Full name** | PhysioNet Challenge Datasets (e.g., 2023 AF Classification) |
| **Domain** | Clinical Signals / Sensors |
| **Modalities** | ECG, PPG, accelerometer |
| **Population** | Varies per challenge (hundreds-thousands of subjects) |
| **Approx variables** | High-frequency waveform + annotations |
| **Access level** | Open (varies) |
| **LLM suitability** | Low |
| **Notes** | Benchmark for sensor ML and federated/DP tasks |
| **Data dictionary** | [physionet.org/challenge](https://physionet.org/challenge/) |
| **Access instructions** | [physionet.org/about/database](https://physionet.org/about/database/) |

---

### 33. Open mHealth

| Field | Details |
|-------|---------|
| **Full name** | Open mHealth Datasets (e.g., Beiwe, mPower) |
| **Domain** | Digital Health / Wearables |
| **Modalities** | Smartphone sensors, wearables (accelerometer, GPS, surveys) |
| **Population** | Thousands of participants in Parkinson's, depression, etc. |
| **Approx variables** | Sensor streams + survey metadata |
| **Access level** | Controlled (registration) |
| **LLM suitability** | Medium |
| **Notes** | Digital phenotyping benchmark; ideal for ML sequence and behavioral analytics |
| **Data dictionary** | [openmhealth.org](https://www.openmhealth.org/) |
| **Access instructions** | [synapse.org/mPower](https://www.synapse.org/mPower) |
