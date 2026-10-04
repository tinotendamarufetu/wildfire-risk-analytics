# 🌋 Enterprise Wildfire Risk Analytics & Underwriting Platform

An end-to-end spatial data engineering and machine learning platform that evaluates property-level wildfire risk in Southern California. The system combines multi-source spatial data ingestion, DuckDB spatial query processing, XGBoost risk modeling, TreeSHAP model interpretability, and an interactive actuarial underwriting engine for property hardening simulations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Engine-2C8EBB)
![DuckDB](https://img.shields.io/badge/DuckDB-OLAP%20Engine-FFF000?logo=duckdb&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Executive Summary

Traditional property insurance underwriting relies heavily on static, regional risk maps that fail to account for dynamic micro-terrain features, proximity to historical fire perimeters, and specific property hardening measures (e.g., defensible space, ember-resistant vents). 

This platform bridges spatial data engineering and actuarial risk modeling by:
1. **Ingesting & Aggregating Multi-Source Data:** Extracting fire perimeters from CalFire GIS, historical weather indices from Open-Meteo, and elevation profiles from USGS.
2. **Predictive Risk Scoring:** Utilizing an **XGBoost Classifier** to evaluate high-risk wildfire probabilities ($0.0 - 1.0$).
3. **Explainable AI (TreeSHAP):** Decomposing global and local feature attributions to provide transparent underwriting rationale.
4. **Actuarial Simulation & Hardening Discounts:** Modeling "what-if" property mitigation scenarios (defensible space clearing, Class A fire-resistant roofing) to calculate dynamic risk index reductions and annual policy premium savings.

---

## 🏗️ Architecture & Pipeline Flow

┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA INGESTION                                  │
│   CalFire Perimeters (GIS)  │  Open-Meteo API  │  USGS 3DEP Terrain Data   │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  SPATIAL FEATURE ENGINEERING & DUCKDB                      │
│   - Buffer joins & spatial proximity calculation                            │
│   - Slope gradient & climate index extraction (VPD, Precip, Wind)           │
│   - Parquet storage & DuckDB analytical querying                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING & SHAP ENGINE                           │
│   - XGBoost & Random Forest Classifiers (MLflow tracked)                    │
│   - TreeSHAP feature contribution calculation                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   UNDERWRITING & SIMULATION DASHBOARD                       │
│   - Geocoding & Southern California bounding box validation                 │
│   - Dynamic PyDeck 3D map visualizer                                        │
│   - Property hardening & premium savings calculator                         │
└─────────────────────────────────────────────────────────────────────────────┘

---

## 🛠️ Key Features

* **Dynamic Geocoding & Regional Guardrails:** Converts street addresses or custom latitude/longitude inputs into coordinates with automatic bounding box checks (`32.5°N - 35.5°N, -120.5°W - -115.0°W`) to restrict inference to active Southern California coverage zones.
* **Spatial Feature Store (DuckDB):** Stores engineered spatial datasets in Parquet format and executes low-latency analytical queries over geographical boundaries.
* **Explainable Risk Scoring (SHAP):** Breaks down prediction outputs feature-by-feature (e.g., slope gradient, distance to past fires, annual max wind speed) to provide full transparency for regulatory compliance and policyholder review.
* **Interactive Property Hardening Engine:** Real-time simulation showing how structural mitigations (30–100 ft defensible space, Class A roofing, ember mesh vents) reduce risk scores and produce financial premium savings ($1,000+ / year).
* **Climate Stress Testing:** Supports scenario stress-testing (e.g., *Extreme Drought*) to evaluate property resilience under changing climate conditions.

---

## 📂 Project Structure

wildfire-risk-analytics/
│
├── data/
│   ├── raw/                      # Ingested shapefiles and weather archives
│   ├── processed/                # Engineered Parquet datasets
│   └── outputs/                  # SHAP summary plots & visual artifacts
│
├── models/
│   └── xgboost_wildfire_model.joblib # Serialized model artifacts
│
├── src/
│   ├── data_ingestion.py         # CalFire & Open-Meteo pipeline tasks
│   ├── feature_engineering.py    # Terrain, proximity, and climate features
│   ├── database.py               # DuckDB spatial database driver
│   ├── train.py                  # XGBoost/Random Forest training & MLflow logging
│   ├── evaluate.py               # Model evaluation & SHAP visual generation
│   └── underwriting_engine.py    # Actuarial pricing & mitigation logic
│
├── .gitignore
├── app.py                        # Streamlit web platform entrypoint
├── requirements.txt              # Cloud-ready environment dependencies
└── README.md

🚀 Installation & Local Setup
Prerequisites
Python 3.10+

Git

1. Clone the Repository
git clone [https://github.com/tinotendamarufetu/wildfire-risk-analytics.git](https://github.com/tinotendamarufetu/wildfire-risk-analytics.git)
cd wildfire-risk-analytics

2. Set Up Virtual EnvironmentBash# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install DependenciesBashpip
install -r requirements.txt

4. Execute the Pipeline End-to-EndBashpython -m src.feature_engineering
python -m src.database
python -m src.train
python -m src.evaluate
python -m src.underwriting_engine

6. Launch the Streamlit DashboardBash
7. streamlit run app.py


🌐 Cloud Deployment
The platform is deployed live on Streamlit Community Cloud.

Live Demo: Wildfire Risk Analytics Platform

Repository: tinotendamarufetu/wildfire-risk-analytics
