# Algerian Forest Fires - Model Training & FWI Prediction App

## Overview

This project is a step-by-step machine learning lifecycle implementation for the Algerian Forest Fires dataset.

It includes:

- **Data understanding and cleaning** in the EDA notebook.
- **Feature engineering and encoding** for modeling.
- **Model training and comparison** across multiple linear models.
- **Model serialization** with `pickle`.
- **Flask web app deployment** for predicting Fire Weather Index (FWI).

The final app takes weather/fire-related inputs and returns a predicted FWI score using a trained **Ridge Regression** model and a fitted **StandardScaler**.

---

## Project Summary

### 1) Dataset

- Raw source file: `notebook/Algerian_forest_fires_dataset_UPDATE.csv`
- Cleaned file: `notebook/Algerian_forest_fires_cleaned_dataset.csv`
- The data represents two Algerian regions and includes meteorological + FWI component features.

Core features used by the app model:

- `Temperature`, `RH`, `Ws`, `Rain`, `FFMC`, `DMC`, `ISI`, `Classes`, `Region`

Target:

- `FWI` (Fire Weather Index)

### 2) EDA & Feature Engineering

Notebook: `notebook/2.0-EDA And FE Algerian Forest Fires.ipynb`

Main work done in this notebook:

- Reads and inspects the raw dataset.
- Cleans irregular entries (including mixed header/region rows in raw data).
- Creates/normalizes the `Region` field.
- Handles types and missing/inconsistent values.
- Encodes `Classes` for analysis.
- Saves cleaned dataset to:
  - `notebook/Algerian_forest_fires_cleaned_dataset.csv`

### 3) Model Training

Notebook: `notebook/3.0-Model Training.ipynb`

Modeling pipeline used:

- Train/test split (`test_size=0.25`, `random_state=42`)
- Correlation-based feature reduction (threshold around `0.85`)
- Standardization using `StandardScaler`
- Model experiments with:
  - `LinearRegression`
  - `Lasso` / `LassoCV`
  - `Ridge` / `RidgeCV`
  - `ElasticNet` / `ElasticNetCV`
- Evaluation with `mean_absolute_error` and `r2_score`

Saved artifacts:

- `models/ridge.pkl`
- `models/scaler.pkl`

### 4) Flask Inference App

Backend:

- `application.py`

Templates:

- `templates/index.html`
- `templates/home.html`

Flow:

1. User opens `/`
2. App navigates to prediction form
3. Form values are scaled with saved scaler
4. Ridge model predicts FWI
5. Predicted value is rendered on page

Run command:

```bash
python application.py
```

Default server config in code:

- Host: `0.0.0.0`
- Port: `5001`

---

## Repository Structure

```text
application.py
requirements.txt
models/
	ridge.pkl
	scaler.pkl
notebook/
	2.0-EDA And FE Algerian Forest Fires.ipynb
	3.0-Model Training.ipynb
	Algerian_forest_fires_dataset_UPDATE.csv
	Algerian_forest_fires_cleaned_dataset.csv
templates/
	index.html
	home.html
```

---

## Installation & Setup

1. Create and activate virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies listed:

- Flask
- numpy
- pandas
- scikit-learn

---

## Notes

- The deployed inference uses the pre-trained Ridge model from `models/ridge.pkl`.
- The form field names in `templates/home.html` should match backend keys in `application.py` (for example `RH` and `Ws` naming) to avoid runtime input issues.

---

## Quick Summary

This repository demonstrates a complete mini ML lifecycle:

**Raw data -> EDA/Cleaning -> Feature engineering -> Model training/comparison -> Model persistence -> Flask inference app**.

It is a practical end-to-end project for learning how notebook-based model development is connected to a simple production-style prediction interface.
