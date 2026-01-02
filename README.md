# 🏦 Credit Risk Modeling System

A production-oriented credit risk prediction system that demonstrates how to build, track, and deploy
interpretable machine learning models using proxy labels, modern MLOps tooling, and a simple FastAPI service.

## 📊 Business Context

Financial institutions must actively measure and manage exposure to credit risk while satisfying regulatory
expectations around transparency and explainability. This project walks through an end‑to‑end workflow for
building a credit risk model when no explicit default label is available, using RFM‑style behavioral signals
as a proxy for high‑risk customers.

The main goals are:

- Create a proxy target that approximates credit risk.
- Engineer robust, interpretable features for downstream modeling.
- Train baseline and gradient‑boosted models with experiment tracking.
- Expose a prediction API suitable for integration into upstream systems.
- Provide enough documentation and structure to be auditable and maintainable.

## 🧱 Technical Overview

**Core components**

- Python, scikit‑learn, XGBoost, and LightGBM for modeling.
- MLflow for experiment tracking and model registry.
- FastAPI to serve predictions via a REST API.
- Docker and optional Docker Compose for containerized deployment.
- Structured project layout with notebooks, source code, configs, and tests.

## 📂 Project Structure

```bash
credit-risk-analysis/
├── artifacts/               # Serialized pipelines and model artifacts
├── configs/
│   └── model_config.yaml    # Example config for model hyperparameters and data settings
├── data/                    # Placeholder for raw / processed data (not tracked)
├── mlruns/                  # Local MLflow experiment tracking
├── notebooks/
│   ├── 1.0-eda.ipynb
│   ├── 2.0-feature_engineering.ipynb
│   ├── 3.0-model_training.ipynb
│   ├── 3.1-lightgbm-experiments.ipynb
│   └── 4.0-prediction_checking.ipynb
├── src/
│   ├── data_processing.py   # Feature engineering and preprocessing logic
│   ├── train.py             # Training script with MLflow logging
│   ├── inference.py         # Inference utilities used by the API
│   ├── utils.py             # Shared helpers and utilities
│   └── api/
│       ├── main.py          # FastAPI app exposing prediction endpoint(s)
│       └── pydantic_models.py
├── tests/                   # Basic unit tests for core functionality
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Install Dependencies

It is recommended to use a virtual environment (e.g. `venv` or `conda`):

```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests

```bash
pytest
```

### 3. Train a Model and Log to MLflow

```bash
python src/train.py
```

This will:

- Train the configured model.
- Log parameters and metrics to the `mlruns/` directory via MLflow.
- Persist the fitted pipeline to the `artifacts/` directory.

### 4. Start the Prediction API

Using FastAPI directly:

```bash
uvicorn src.api.main:app --reload
```

Or via Docker:

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

Or with Docker Compose:

```bash
docker-compose up
```

Once running, open `http://localhost:8000/docs` to explore the interactive Swagger UI.

## 🧪 Modeling and Experiments

The notebooks in the `notebooks/` folder cover:

- Exploratory data analysis and RFM‑style proxy target creation.
- Feature engineering for tabular modeling.
- Training baseline Logistic Regression, XGBoost, and LightGBM models.
- Comparing model performance and recording runs in MLflow.
- Basic prediction checking and sanity validation.

The additional `3.1-lightgbm-experiments.ipynb` notebook provides a template for running more advanced
LightGBM experiments and logging results for comparison with other models.

## 🧠 Extending the Project

Ideas for extension:

- Integrate SHAP or other explainability methods into the pipeline.
- Add monitoring for data drift or score distribution shifts.
- Connect the FastAPI service to a real message queue or event stream.
- Deploy the model behind an API gateway or serverless function.

This repository is intended as a realistic, end‑to‑end example of credit risk modeling with an emphasis on
structure, reproducibility, and deployment rather than as a production‑ready banking solution.