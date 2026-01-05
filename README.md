# Sepsis Mortality Prediction with NLP — Master’s Thesis

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-scaffolded-yellow)

📘 **[Read the full thesis (PDF)](docs/Tyler%20Kelly%20Thesis.pdf)**

---

## 📌 Overview

This project was developed as a Master’s Thesis in Biostatistics at the University of Pittsburgh and focuses on ICU sepsis mortality prediction using EHR data and natural language processing of clinical text.


### Motivation
Sepsis remains a leading cause of mortality in intensive care units, motivating the need for reproducible and interpretable machine learning approaches to mortality prediction using routinely collected EHR data.

### Objective
The objective is to develop, evaluate, and statistically compare supervised machine learning models for predicting in-hospital mortality among ICU patients with sepsis using multi-modal EHR data from MIMIC-IV.

### Project Summary

This work frames sepsis mortality prediction as a binary classification task with class imbalance.
Models are tuned on the original training data and subsequently retrained on SMOTE-balanced data for final evaluation on both resampling schemes.

The pipeline integrates:
- Data ingestion and preprocessing using the cleaned MIMIC-IV–derived dataset released by [Gao et al.](https://github.com/yuyinglu2000/Sepsis-Mortality)
- Unstructured clinical feature engineering
- Clinical text representation using NLP embeddings (Word2Vec; experimental LLM extensions explored separately)
- Supervised model training with and without SMOTE to address class imbalance
- Rigorous evaluation using AUROC, calibration, and statistical testing
- Reproducible experiment tracking and artifact logging (MLflow)
- A lightweight API demonstration for inference (in progress)

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
- **Libraries**: scikit-learn, XGBoost, LightGBM, pandas, numpy,  matplotlib, seaborn, plotly
- **NLP**: Word2Vec (gensim; main thesis NLP technique); experimental LLM-based feature extraction extensions using gpt-4.1-mini and gpt-4o-mini
- **Experiment Tracking**: MLflow
- **API**: Flask (FastAPI planned)
- **Containerization**: Docker, Docker Compose (planned)
- **CI/CD**: GitHub Actions
- **Testing**: PyTest

---

## Methodology

1. Outcome labeling and cohort definition using the cleaned MIMIC-IV–derived dataset released by Gao et al.
2. Structured data preprocessing and normalization
3. Clinical text embedding using Word2Vec as the primary thesis representation
4. Model selection and hyperparameter tuning using repeated stratified cross-validation on non-resampled training data
5. Retraining of selected models on SMOTE-balanced training data
6. Performance evaluation using AUROC, calibration curves, and SHAP
7. Artifact persistence and experiment tracking with MLflow


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

### 📄 Full Thesis Document

The complete, submitted Master’s thesis is available here:

📘 **[Tyler Kelly — Master’s Thesis (PDF)](docs/Tyler%20Kelly%20Thesis.pdf)**

This repository supports my Master’s Thesis defense by ensuring reproducibility and transparency:
- Committee members can follow the modular notebooks.
- Recruiters can inspect the scaffold and MLOps integrations.

This repository is also intended for academic and educational purposes.
Results should not be interpreted as clinical decision support without external validation.

### 📊 Dataset Source

The primary dataset used in this project is the cleaned MIMIC-IV–derived cohort released by Gao et al.:

🔗 **Gao et al. — Sepsis Mortality Prediction Repository**  
https://github.com/yuyinglu2000/Sepsis-Mortality

Specifically, this work uses the cleaned dataset provided in `Data_after_Cleaning.csv`, with all downstream preprocessing, feature engineering, modeling, and evaluation performed in this repository.

---

## 📂 Repo Structure
```bash
Masters-Thesis/
├── data/               # raw, processed, external datasets (gitignored)
├── notebooks/          # modular workflow notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_baseline_model_evaluation.ipynb
│   ├── 06_visualization.ipynb
│   ├── 07_w2v_hyperparam_search.ipynb
│   ├── 08_w2v_optimized_training.ipynb
│   ├── 09_final_model_evaluation.ipynb
│   ├── 10_statistical_testing.ipynb
│   ├── 11_external_validation.ipynb  # currently unused
│   ├── 12_reporting.ipynb
│   ├── 13_Cosine_Similarity.ipynb
│   ├── 14_LLM_Extension.ipynb         # exploratory work, not part of core thesis results
│   ├── 15_api_demo.ipynb             # in development
│   └── archive/        # legacy notebooks + Rmd
├── src/                # reusable, testable pipeline modules (data, features, models, evaluation)
├── scripts/            # CLI and automation scripts
├── sql/                # SQL utilities for extracting unstructured notes from MIMIC-IV
├── results/            # trained models and plots (gitignored)
├── mlflow_tracking/    # MLflow experiment logs (gitignored)
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

> Note: Raw MIMIC-IV data are not included due to licensing restrictions. SQL scripts are provided for reproducible extraction.

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

1. Extract unstructured clinical notes from MIMIC-IV using SQL utilities and merge them with the cleaned MIMIC-IV–derived dataset released by Gao et al.

2. Clean and preprocess structured clinical variables and clinical text

3. Generate NLP embeddings from clinical text

4. Train and tune machine learning models using cross-validation

5. Retrain models on SMOTE-balanced training data

6. Generate model performance metrics, feature attributions, and evaluation visualizations

7. Log trained models, metrics, and artifacts using MLflow

8. Perform formal statistical testing (e.g., DeLong tests) to compare AUROC performance across models

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
Modular Jupyter notebooks (01–15) implement the full pipeline, with legacy work preserved in `archive/`.

---

## ✅ Unit Tests
Core pipeline logic has been refactored into `src/` to support testability and reuse.
PyTest-based unit tests validate data preprocessing, feature engineering, and model training logic.
Coverage is being expanded as part of ongoing pipeline hardening.

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

- Extend resampling support beyond SMOTE (e.g., SMOTEENN, ADASYN) via the modular resampling interface
- Expand NLP representations using contextual embeddings (BERT variants)
- Investigate deep learning approaches for longitudinal risk modeling using time-aware representations


---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋 About Me

Built by Tyler Kelly

📧 Email: tylerjkelly77@gmail.com

🔗 LinkedIn: [LinkedIn](https://www.linkedin.com/in/tylerkelly7)  

🔗 Portfolio: [Portfolio](https://github.com/tylerkelly7)