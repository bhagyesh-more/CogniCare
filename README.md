# CogniCare - Clinical Cognitive and Arousal Monitoring Platform

> **Version 1.1 - Clinical Hospital Patient Intel Module**

A production-grade, multi-tenant clinical intelligence platform for hospitals and clinics.
Built on a full Responsible AI backbone - explainable by design, transparent by default, privacy-first.

---

## Table of Contents

- [Overview](#overview)
- [Core Architecture](#core-architecture)
- [Application Flow](#application-flow)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Database Schema](#database-schema)
- [Dashboard Pages](#dashboard-pages)
- [Installation and Setup](#installation-and-setup)
- [Demo Credentials](#demo-credentials)
- [Project Status](#project-status)
- [Ethical Commitments](#ethical-commitments)

---

## Overview

**CogniCare** is a clinical-grade cognitive monitoring platform designed exclusively for **hospitals, clinics, and medical professionals**.
Patients do not log in - doctors create and manage patient profiles and use CogniCare to monitor cognitive load and emotional arousal trends.

The system processes 9 physiological biomarkers from EDA (skin conductance), ECG (heart rate variability), and respiration signals
to classify stress and cognitive states using interpretable Random Forest classifiers with full SHAP explainability.

**This is NOT a consumer-facing or patient-facing application.**

---

## Core Architecture

Three-layer pipeline feeding into a Streamlit clinical platform backed by SQLite:

**Part 1 - Data Pipeline (main.py)**
DatasetProcessor loads WESAD .pkl files, FeatureExtractor computes 60-second EDA/HRV/Resp windows,
DataCleaner normalises with z-score capping.
Output: output/feature_dataset.csv

**Part 2 - ML Training (train.py)**
ModelTrainer trains Random Forest x2 (Arousal + Cognitive Load),
LabelEncoder maps WESAD labels to binary targets.
Output: models/arousal/ and models/cognitive_load/

**Part 3 - Responsible AI Layer**
- PredictionEngine: RF inference (load artifacts, scale, predict)
- ExplainabilityEngine: SHAP local and global explanations
- ConfidenceEngine: Tier classification (High/Medium/Low)
- TransparencyEngine: Narrative assembly from all engine outputs
- PrivacyEngine: Anonymous UUID4 sessions + PII scrubbing

**Clinical Platform (app.py)** - Streamlit multi-page app with SQLite persistence.
Database at data/cognicare.db (4 tables: hospitals, doctors, patients, assessments).

---

## Application Flow

`
Landing Page
     |
     v
Hospital Registration / Doctor Login
     |
     v
Hospital Dashboard  (stats, doctor list, system health)
     |
     v
Patient Directory   (search, register, view profiles)
     |
     v
Patient Profile     (demographics, risk indicators, SHAP, trends)
     |
     v
Cognitive Analysis  (manual input or CSV batch, bound to active patient)
     |
     v
Physiological Intel (biosignal charts, clinic patient roster)
     |
     v
Responsible AI Insights (SHAP global, model explainability)
     |
     v
Reports and History (PDF export, assessment log per patient)
`

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit 1.35+ |
| Charting | Plotly 5.18+ |
| Machine Learning | scikit-learn 1.4+ (Random Forest) |
| Explainability | SHAP 0.44+ (TreeExplainer) |
| Data Processing | NumPy 1.26+, Pandas 2.2+, SciPy 1.13+ |
| Database | SQLite 3 (context-managed connections) |
| PDF Reports | FPDF2 2.7+, ReportLab 4.0+ |
| Image Processing | Pillow 9.0+ |
| Persistence | Joblib 1.4+ |
| Design Theme | Neon Cyberpunk / Glassmorphism (custom CSS) |
| Fonts | Orbitron, Space Grotesk (Google Fonts) |

---

## Project Structure

`
CogniArousal/
|
+-- app.py                      # Streamlit entrypoint; page routing
+-- main.py                     # Part 1: physiological data pipeline
+-- train.py                    # Part 2: ML training pipeline
+-- explain.py                  # Standalone SHAP exploration script
+-- test_inference.py           # Smoke test for PredictionEngine
|
+-- src/
|   +-- database.py             # SQLite service layer (multi-tenant)
|   +-- responsible_ai.py       # Unified RAI facade
|   +-- prediction_engine.py    # Model inference (load + predict)
|   +-- explainability_engine.py# SHAP local + global explanations
|   +-- confidence_engine.py    # Confidence tier classification
|   +-- transparency_engine.py  # Narrative + prediction record
|   +-- privacy_engine.py       # Anonymous sessions + PII scrubbing
|   +-- model_trainer.py        # Random Forest training + evaluation
|   +-- feature_extractor.py    # Window-level EDA/HRV/Resp features
|   +-- dataset_processor.py    # WESAD .pkl loader + signal aligner
|   +-- data_cleaner.py         # Z-score normalisation + outlier filter
|   +-- label_encoder.py        # WESAD label to arousal/cognitive_load
|   +-- monitoring.py           # Runtime inference metrics tracker
|   +-- report_generator.py     # PDF + CSV clinical report generator
|   +-- demo_profiles.py        # Demo mode mock patient profiles
|   +-- validator.py            # Feature input validation
|
+-- dashboard/
|   +-- styles.py               # Injected CSS: cyberpunk design system
|   +-- components.py           # Shared Plotly + HTML UI components
|   +-- pages/
|       +-- landing.py          # Marketing landing page
|       +-- auth.py             # Clinic registration + doctor login
|       +-- mission_control.py  # Original single-user dashboard
|       +-- patients.py         # Patient directory + profile dashboards
|       +-- analysis.py         # Cognitive analysis (manual + CSV batch)
|       +-- physio.py           # Physiological Intel (EDA/HRV/Resp)
|       +-- dataset.py          # WESAD dataset exploration
|       +-- demo.py             # Demo mode (mock patient profiles)
|       +-- responsible_ai_page.py  # Global SHAP + RAI transparency
|       +-- timeline.py         # Assessment session timeline
|
+-- models/
|   +-- arousal/                # RF model + scaler + metadata
|   +-- cognitive_load/         # RF model + scaler + metadata
|
+-- output/
|   +-- feature_dataset.csv     # Part 1 output (ML-ready features)
|   +-- physiological_data.csv  # Raw aligned signals (all subjects)
|
+-- data/
|   +-- cognicare.db            # SQLite clinical database
|   +-- S2/, S3/, ...           # WESAD raw subject .pkl files
|
+-- reports/                    # Generated PDF/CSV reports
+-- assets/                     # Logos and static assets
+-- requirements.txt
+-- run.bat                     # Windows start script
`

---

## Module Reference

### Part 1: Data Pipeline

**main.py** - Pipeline orchestrator. Discovers WESAD .pkl files, runs extraction chain, saves CSVs.

**src/dataset_processor.py - DatasetProcessor**
- Loads WESAD chest sensor .pkl files
- Extracts 3 raw signal channels: ECG, EDA, Resp
- Aligns sample rates, maps affect labels (baseline, amusement, stress)
- Outputs sample-level labeled DataFrame

**src/feature_extractor.py - FeatureExtractor**
- Segments signals into 60-second non-overlapping windows at 700 Hz
- Extracts 9 domain-specific features per window
- scipy.signal.find_peaks for EDA peak and breath cycle detection
- Adaptive R-peak detection for ECG (physiological clipping 30-220 bpm)

**src/data_cleaner.py - DataCleaner**
- Z-score normalisation across all feature columns
- Outlier capping (z > 3 clipped to +-3)
- Drops rows with excessive NaN values

**src/label_encoder.py - add_target_labels**

Maps WESAD 3-class labels to two binary classification targets:

| WESAD Label | Emotional Arousal | Cognitive Load |
|---|---|---|
| baseline | Low | Low |
| amusement | High | Low |
| stress | High | High |

Key: amusement = high arousal but LOW cognitive load - creates genuinely distinct decision boundaries.

---

### Part 2: ML Training Pipeline

**train.py** - Full training pipeline with evaluation reports and inference smoke test.

**src/model_trainer.py - ModelTrainer**
- Trains Random Forest classifier (300 trees) per target
- class_weight=balanced to handle WESAD class imbalance
- 5-fold stratified cross-validation: accuracy, F1 macro, precision, recall
- Saves artifacts: model.pkl, scaler.pkl, feature_metadata.json, feature_importance.csv, confusion_matrix.csv

RF hyperparameters: n_estimators=300, max_depth=None, min_samples_split=4, min_samples_leaf=2,
class_weight=balanced, random_state=42, n_jobs=-1

---

### Part 3: Responsible AI Layer

**src/responsible_ai.py - ResponsibleAI**
Unified facade wiring all 5 engines. Exposes explain_prediction(session_id, target, data)
and global_importance(target, dataset). Returns ResponsibleAIPrediction with .narrative and .to_dict().

**src/prediction_engine.py - PredictionEngine**
- Loads model.pkl + scaler.pkl artifacts from disk
- Accepts single sample (dict) or batch (DataFrame)
- Returns: predicted_class, confidence, prob_Low, prob_High

**src/explainability_engine.py - ExplainabilityEngine**
- shap.TreeExplainer for exact SHAP values on Random Forests
- Local: top-N features ranked by |SHAP| with direction and magnitude
- Global: mean |SHAP| importance across dataset

**src/confidence_engine.py - ConfidenceEngine**
- Converts class probabilities to structured ConfidenceResult
- Tiers: High (>85%), Medium (60-85%), Low (<60%)
- Low tier sets flag_review=True; computes Shannon entropy

**src/transparency_engine.py - TransparencyEngine**
- Assembles ResponsibleAIPrediction from all engine outputs
- Auto-generates human-readable clinical narratives

**src/privacy_engine.py - PrivacyEngine**
- UUID4-based anonymous session IDs (non-reversible)
- PII key scrubbing before inference (name, email, dob, subject_id, etc.)
- Context manager guarantees ephemeral data cleanup

**src/monitoring.py - Monitor**
- Per-inference wall-clock timing (ms)
- Rolling inference count, avg/min/max latency
- Stored in Streamlit session state

**src/report_generator.py - ReportGenerator**
- 5-page branded PDF: header/metadata, features table, predictions, SHAP, methodology
- Also generates flat CSV reports (one row per session)

**src/validator.py - Validator**
- Input feature range and type validation
- Checks all 9 required feature columns are present

---

### Clinical Database Layer

**src/database.py - DatabaseService**
SQLite at data/cognicare.db. Context-managed connections prevent file descriptor leaks on Windows.
PRAGMA foreign_keys = ON enforced. Passwords hashed with salted SHA-256.

---

## Database Schema

**hospitals** - Multi-tenant clinic registry
- id, name, registration_code (UNIQUE), created_at

**doctors** - Clinician credentials (hospital-scoped)
- id, hospital_id (FK), name, email (UNIQUE), password_hash, created_at

**patients** - Clinical patient profiles
- id TEXT PRIMARY KEY (format: PAT-YYYYMMDD-XXX)
- doctor_id (FK), hospital_id (FK)
- name, age, gender, dob, contact_number, address
- occupation, industry, work_schedule, education_level
- clinical_notes, registration_date

**assessments** - Complete inference history per patient
- id, patient_id (FK), doctor_id (FK), timestamp, session_id
- All 9 feature values (eda_mean through resp_variability)
- arousal_pred, arousal_conf, arousal_tier, arousal_narrative
- load_pred, load_conf, load_tier, load_narrative
- shap_values TEXT (JSON array), flag_review INTEGER

**Key Methods:**

| Method | Description |
|---|---|
| register_hospital(name, code) | Create clinic with unique registration code |
| get_hospital_by_code(code) | Lookup clinic by registration code |
| get_hospital_stats(hospital_id) | Return doctor/patient/assessment counts |
| register_doctor(hospital_id, name, email, password) | Register clinician (hashed password) |
| authenticate_doctor(email, password) | Verify credentials (salted SHA-256) |
| create_patient(...) | Register patient, auto-generate PAT-ID |
| get_patient(patient_id) | Fetch patient with doctor name joined |
| get_patients_by_hospital(hospital_id) | List all hospital patients |
| search_patients(hospital_id, query) | Search by name, ID, or doctor |
| log_assessment(...) | Persist complete prediction record |
| get_patient_assessments(patient_id) | Full chronological assessment history |

**Seeded Development Data (auto-inserted on first run):**
- Clinic: NEO-GEN Hospital
- Doctor: Dr. Evelyn Vance (evelyn@cognicare.com / password123)
- Patients: Arthur Pendelton, Sarah Jenkins, Devon Cole, Lana Croft
- Mock Assessments: 12 sessions for Arthur, 8 for Sarah

---

## Feature Engineering

All 9 features extracted from 60-second non-overlapping windows at 700 Hz:

| Feature | Signal | Description |
|---|---|---|
| eda_mean | EDA | Mean skin conductance level |
| eda_std | EDA | Skin conductance variability |
| eda_peak_count | EDA | Number of skin conductance responses |
| eda_peak_amplitude | EDA | Mean peak prominence |
| heart_rate_bpm | ECG | Heart rate via R-peak detection (clipped 30-220 bpm) |
| rmssd | ECG | Root mean square of successive RR differences |
| sdnn | ECG | Standard deviation of RR intervals (HRV) |
| resp_rate_bpm | Respiration | Breaths per minute (clipped 3-60) |
| resp_variability | Respiration | Std of inter-breath intervals (seconds) |

All features Z-score normalised. Outliers beyond 3 standard deviations clipped.

---

## Machine Learning Models

Two independent Random Forest classifiers:

| Model | Target | Classes |
|---|---|---|
| Arousal Classifier | Emotional arousal state | Low, High |
| Cognitive Load Classifier | Mental workload level | Low, High |

Typical cross-validated performance on WESAD: Accuracy 88-93%, F1 macro 0.85-0.91.
Training subjects S2-S17 (S1 excluded per WESAD dataset notes).

---

## Dashboard Pages

### 1. Landing Page (landing.py)
Marketing overview before authentication. Platform intro, live stats, 4-step workflow diagram,
ethical AI principles cards, CTAs for Register Clinic and Doctor Login.

### 2. Auth Page (auth.py)
Doctor Login tab (SHA-256 verification). Register Clinic tab (unique code + first doctor account).

### 3. Hospital Dashboard (app.py)
Clinic KPIs (patients, assessments, active staff), system health gauges,
active doctors table, model and dataset status indicators.

### 4. Patient Directory (patients.py)
Searchable patient explorer with full registration panel.
Profile view: intake glassmorphism cards, latest prediction results,
clinical risk indicator (NOMINAL / ELEVATED FATIGUE / CRITICAL COGNITIVE SPUR),
SHAP attribution bars, AI narrative, trend charts, assessment log with PDF download,
and shortcut button to Physiological Intel.

### 5. Cognitive Analysis (analysis.py)
Dual-tab interface bound to the active patient context.

Manual Input tab: 9 feature inputs, dual inference (arousal + load),
prediction cards, probability charts, confidence gauges, SHAP explanation,
auto-save to patient history in SQLite, PDF/CSV download.

CSV Upload tab: Batch upload (all 9 columns required), each row saved as separate assessment,
batch summary KPIs, results dataframe preview, export as CSV.

### 6. Physiological Intel (physio.py)
Dual-mode monitoring page.

Clinic Roster mode: Left column searchable patient list with real-time text filter.
Right column active patient profile card + chronological biosignal line charts
(EDA/HRV/Resp KPIs + charts with assessment timestamps on x-axis), scatter plots, distribution histograms.

WESAD Benchmark mode: Subject selector dropdown, same charts populated with WESAD population data.

### 7. Dataset Intelligence (dataset.py)
WESAD population statistics, feature correlation heatmaps, label distributions.

### 8. Demo Mode (demo.py)
5 fictional patient archetypes, full assessment without database persistence, for presentations.

### 9. Responsible AI Insights (responsible_ai_page.py)
Global SHAP feature importance charts, model architecture documentation, ethical commitments.

---

## Dashboard UI System

**dashboard/styles.py** - Injected CSS design system:
- Neon cyberpunk palette: cyan #00E5FF, purple #9B6DFF, pink #FF4D7A, green #00FFB2
- Glassmorphism cards with accent borders and neon glow
- Fonts: Orbitron (headings), Space Grotesk (body)
- Animated pulse dots, hover transforms, neon box-shadows

**dashboard/components.py** - Shared component library:

| Function | Description |
|---|---|
| glass_card(content, accent) | Glassmorphism card wrapper HTML |
| kpi_card(value, label, icon, color) | Neon metric card with accent |
| prediction_card(...) | Full prediction result card with confidence bar |
| gauge_chart(value, title, color) | Plotly radial gauge |
| signal_line_chart(x, y, title, color) | Plotly filled biosignal line chart |
| scatter_chart(df, x, y, color_col, title) | Plotly scatter coloured by class label |
| histogram_chart(series, col, color) | Feature distribution histogram |
| probability_bar_chart(class_probs, title) | Class probability bar chart |
| shap_feature_row(label, val, max_abs, dir) | SHAP attribution HTML row |
| section_header(title, subtitle, icon) | Page section header HTML |
| narrative_box(text) | Styled AI narrative blockquote |
| status_badge(label, state) | Animated status indicator |
| neon_divider() | Horizontal neon separator |
| timeline_row(...) | Assessment history row HTML |

---

## Installation and Setup

**Prerequisites:** Python 3.11+. WESAD dataset files (download separately; not included).

**Install dependencies:**
`
pip install -r requirements.txt
`

**Part 1 - Process Raw Data:**
`
python main.py --data_dir data/ --output_dir output/
`

**Part 2 - Train Models:**
`
python train.py
`

**Start the Platform:**
`
streamlit run app.py
`

On Windows: run.bat

The SQLite database is auto-created and seeded with demo data on first run.
Navigate to http://localhost:8501.

---

## Testing and Verification

**Inference smoke test:**
`
python test_inference.py
`

**Database test:**
`
python scratch/test_db.py
`

---

## Demo Credentials

| Field | Value |
|---|---|
| Clinic Code | NEO-GEN |
| Doctor Email | evelyn@cognicare.com |
| Password | password123 |

**Sample Patients:**

| Patient ID | Name | Occupation | Pre-loaded Sessions |
|---|---|---|---|
| PAT-2026-0001 | Arthur Pendelton | Lead DevOps Engineer | 12 |
| PAT-2026-0002 | Sarah Jenkins | Financial Risk Analyst | 8 |
| PAT-2026-0003 | Devon Cole | Air Traffic Controller | 0 |
| PAT-2026-0004 | Lana Croft | Field Archaeologist | 0 |

---

## Project Status

| Feature | Status |
|---|---|
| WESAD data processing pipeline | Complete |
| Feature extraction (EDA / HRV / Resp) | Complete |
| Random Forest training (x2 targets) | Complete |
| SHAP explainability (local + global) | Complete |
| Confidence tier classification | Complete |
| AI transparency narratives | Complete |
| Privacy engine (anonymous sessions) | Complete |
| PDF + CSV report generation | Complete |
| Multi-tenant SQLite database | Complete |
| Marketing landing page | Complete |
| Clinic registration + doctor login | Complete |
| Hospital dashboard | Complete |
| Patient directory + registration | Complete |
| Patient profile dashboards | Complete |
| Cognitive analysis (manual + batch CSV) | Complete |
| Physiological Intel (roster + charts) | Complete |
| WESAD dataset exploration page | Complete |
| Demo mode (standalone, no auth) | Complete |
| Responsible AI insights page | Complete |
| Assessment PDF export per patient | Complete |
| Chronological biosignal trend charts | Complete |
| Searchable patient roster in Physio Intel | Complete |

---

## Ethical Commitments

CogniCare is designed with Responsible AI at its core:

- **Explainability** - Every prediction includes SHAP feature attribution so clinicians understand why, not just what
- **Confidence Transparency** - Low-confidence predictions are automatically flagged for clinician review
- **Privacy by Design** - Inference sessions use anonymous UUID4 IDs; no PII retained in prediction pipelines
- **Human in the Loop** - Clinical decision support tool, not a replacement for professional medical judgement
- **Data Minimisation** - Only 9 pre-computed physiological features processed; no raw biometric streams stored