# Masters-Thesis: Sepsis Mortality Prediction

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-scaffolded-yellow)

<!--
## Motivation
Explain why this project matters and its real-world impact.

## Problem Statement
Clearly define the problem or task being solved.
-->

## 📌 Overview
This repository contains my Master's Thesis project in Biostatistics at the University of Pittsburgh.  
The goal is to forecast sepsis mortality using **multi-modal EHR data** (structured clinical variables + clinical notes).  
The pipeline integrates:
- Data extraction from MIMIC-IV (SQL)
- Preprocessing and feature engineering
- NLP embeddings (Word2Vec)
- Model training with and without SMOTE to handle class imbalance
- Model evaluation and statistical tests
- Deployment via API (FastAPI/Flask demo)

---

<!--
- **Goal:** [Short description of the prediction or classification task]
- **Dataset:** [Name + link if public; instructions if restricted]
- **Tech stack:** Python, scikit-learn, XGBoost, PyTorch, SHAP, Streamlit
- **Key results:** 
  - Best model: [Model + metric]
  - Interpretability: [e.g., SHAP feature importances]
  - Deployment: [Optional: Streamlit demo / API]



---

-->

## 🛠️ Tech Stack

- **Languages**: Python (3.11), R, SQL
- **Libraries**: PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM, pandas, numpy,  matplotlib, seaborn, plotly
- **NLP**: Word2Vec, BERT (HuggingFace Transformers)
- **API**: Flask
- **Experiment Tracking**: MLflow
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Testing**: PyTest

---

## Methodology
- Data preprocessing
- Feature engineering
- Model selection and training
- Model evaluation (AUROC, calibration, SHAP feature importance)
- Results visualization and interpretation

---
<!--
## 🏗️ Project Architecture

![Architecture](docs/architecture.png)

---

## 📊 Results

Internal AUROC: [to be added]

External validation AUROC: [to be added]

Feature importance and SHAP visualizations: see results/plots/

---

## Results Snapshot
| Model            | AUC  | Accuracy | Notes |
|------------------|------|----------|-------|
| Logistic Reg.    | 0.72 | 0.68     | Baseline |
| XGBoost          | 0.85 | 0.79     | Best performer |

---

## 🔍 Interpretability Example
![SHAP summary plot](reports/figures/shap_summary.png)

---
-->

## 🎓 Academic Context

This repository supports my Master’s Thesis defense by ensuring reproducibility and transparency:
- Committee members can follow the modular notebooks.
- Recruiters can inspect the scaffold and MLOps integrations.

## 📂 Repo Structure
```bash
Masters-Thesis/
├── data/               # raw, processed, external datasets (ignored in Git)
├── notebooks/          # modular workflow notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_external_validation.ipynb
│   ├── 07_api_demo.ipynb
│   └── archive/        # legacy notebooks + Rmd
├── src/                # reusable pipeline code
├── scripts/            # CLI and automation scripts
├── sql/                # SQL queries for MIMIC-IV extraction
├── results/            # trained models and plots (ignored in Git)
├── mlflow_tracking/    # MLflow experiment logs (ignored in Git)
├── configs/            # experiment configs
├── tests/              # unit tests
├── Makefile            # workflow automation
├── Dockerfile          # containerization
├── requirements.txt    # Python dependencies
├── environment.yml     # Conda environment
└── README.md           # this file
```
---

## ⚙️ Setup Instructions

### Setup
```bash
git clone https://github.com/tylerkelly7/Masters-Thesis.git
cd Masters-Thesis
conda env create -f environment.yml
conda activate Masters-Thesis
```

---
<!--
## 🔄 ML Pipeline

### Run the entire ML workflow:

```bash
make preprocess
make features
make train
make eval
```
-->

### Pipeline steps:

1. Extract raw data from MIMIC-IV (via SQL)

2. Clean & preprocess structured and text data

3. Engineer features (structured + embeddings)

4. Train models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, etc.)

5. Evaluate models (AUROC, calibration, SHAP/feature importance)

6. Save model artifacts with MLflow

---
<!--
## 🔬 API for Predictions

```bash
make run-api
```

---

## 🧪 End-to-End Test

### Run pipeline + API test with one command:

```bash
make pipeline_test
```

---
-->

## 📓 Notebooks
Modular Jupyter notebooks (01–07) implement the full pipeline, with legacy work preserved in `archive/`.

---

## ✅ Unit Tests
Unit tests (PyTest) cover data preprocessing, feature engineering, and model training modules.

- Currently in development
---
<!--
## 🐳 Containerization

### Build Docker image:

```bash
docker build -t masters_thesis .
```

### Run with Docker:

```bash
docker run -p 5000:5000 masters_thesis
```

### Or with Docker Compose:

```bash
docker-compose up --build
```

---
-->

<!--
## 📚 Documentation

Full project documentation is in [Full Documentation](docs/index.md)  
[GitHub Pages](https://tylerkelly7.github.io/Masters-Thesis/)

-->
## 📈 Future Work

- “Resampling has been modularized in src/resampling.py.
Currently only SMOTE is supported and the design allows extension to undersampling or hybrid methods (e.g., SMOTEENN, ADASYN). Future work could refactor training code (repeated_cv_with_mixed_search) to accept additional resampling strategies as parameters.”

- BERT Extensions
- Incorporating reinforcement and deep learning

---

## 📄 License

This project is licensed under the MIT License.

---

🙋 About Me

Built by Tyler Kelly
📧 Email: tylerjkelly77@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/tylerkelly7)  

🔗 [Portfolio](https://github.com/tylerkelly7)