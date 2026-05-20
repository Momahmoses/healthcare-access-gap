# Healthcare Facility Access Gap Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A geospatial analytics platform identifying underserved Nigerian communities using travel-time modelling, facility density analysis, and K-Means gap-zone clustering — guiding evidence-based placement of new clinics and mobile health units.

---

## Problem Statement

Over 70 million Nigerians live more than 5 km from the nearest functional health facility. Without spatial analysis, resource allocation remains inequitable. This platform quantifies the access gap at LGA level and recommends priority intervention zones.

---

## Features

| Feature | Description |
|---------|-------------|
| Access Tier Scoring | Composite coverage score across 8 dimensions |
| K-Means Gap Clustering | Priority zones ranked by affected population |
| State-Level Heatmaps | Comparative access analysis across all 36 states |
| Interactive Dashboard | 4-page Streamlit app with Plotly Mapbox maps |
| Facility Placement Recommender | Top sites for new clinics based on need score |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Geospatial | GeoPandas, Folium, Plotly Mapbox |
| Machine Learning | scikit-learn (K-Means) |
| Dashboard | Streamlit |
| Data | pandas, NumPy |

---

## Project Structure

```
healthcare-access-gap/
├── src/
│   ├── data_loader.py       # Community and facility data ingestion
│   ├── analysis.py          # Travel-time modelling, access scoring, gap clustering
│   └── visualize.py         # Interactive maps and coverage charts
├── streamlit_app.py         # Multi-page Streamlit dashboard entry point
├── .streamlit/config.toml   # Theme and server config
├── requirements.txt
└── runtime.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/healthcare-access-gap.git
cd healthcare-access-gap
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Data Sources

- HMIS facility location registry (PHCs, General Hospitals, Specialist Hospitals)
- WorldPop gridded population data (100m resolution)
- OpenStreetMap Nigeria road network
- GRID3 Nigeria healthcare facility dataset

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
