# Decision Support Demo: Rule-based vs ML-based Anomaly Detection

This repository contains a small decision-support prototype that demonstrates how
rule-based methods and machine learning can be combined and compared in an
analysis and BI context.

The focus is on practical usability, interpretability, and maintainability rather
than on complex models or production infrastructure.

---

## Overview

The application visualizes time-series data for environmental indicators and supports
decision-making by automatically flagging potential anomalies.

Two complementary approaches are implemented:

- Rule-based anomaly detection using rolling statistics (baseline, fully interpretable)
- ML-based anomaly detection using Isolation Forest (data-driven, more adaptive)

A comparison view highlights overlaps and differences between the two methods,
supporting informed human decision-making.

---

## Key Features

- Interactive dashboard for time-series analysis
- User-selectable region and indicator
- Rule-based anomaly detection (rolling z-score)
- ML-based anomaly detection (Isolation Forest)
- Side-by-side comparison of methods
- Adjustable parameters for exploration and testing
- Synthetic data generation (no sensitive data)

---

## Technical Scope

This prototype is intentionally scoped as a testbed / proof of concept and focuses on:

- Data modeling for analytical use cases
- Feature engineering for anomaly detection
- Practical application of AI/ML in decision support
- Clear separation between baseline logic and ML-based methods
- Reproducibility and ease of experimentation

It does not aim to demonstrate:

- Enterprise BI platforms
- Production data pipelines
- Cloud infrastructure or deployment at scale

---

## Architecture (Simplified)

    Synthetic Data
         ↓
    Feature Engineering (Python / pandas)
         ↓
    Rule-based Logic   |   ML-based Logic (Isolation Forest)
         ↓
    Visualization & Interaction (Streamlit)
         ↓
    Decision Support (Human-in-the-loop)

---

## Running the Application Locally

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

    pip install -r requirements.txt

### Run the app

    streamlit run app.py

The application will be available at:

    http://localhost:8501

---

## Parameters and Exploration

The dashboard allows interactive tuning of:

- Rolling window size
- Rule-based z-score threshold
- Expected anomaly proportion for ML-based detection

This supports exploratory analysis and comparison of sensitivity versus
interpretability.

---

## Notes on AI Usage

Machine learning is used as a decision-support aid, not as an automated decision
mechanism.

- Rule-based detection provides transparency and a clear baseline
- ML-based detection provides adaptive pattern recognition
- Final interpretation and decisions remain with the user

This reflects a human-in-the-loop approach suitable for public-sector and
decision-support environments.

---

## License

This project is provided for demonstration and evaluation purposes.

