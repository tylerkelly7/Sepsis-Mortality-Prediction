# Masters-Thesis: Sepsis Mortality Prediction

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-scaffolded-yellow)


## Motivation
A reproducible machine learning pipeline for predicting ICU sepsis mortality using structured EHR data and clinical text from MIMIC-IV, developed as a Master’s thesis in Biostatistics.


## Objective
To develop, evaluate, and statistically compare machine learning models for predicting in-hospital mortality among ICU patients with sepsis using multi-modal electronic health record (EHR) data.


---

## 📌 Overview
This repository contains my Master’s Thesis project in Biostatistics at the University of Pittsburgh.

This work treats sepsis mortality prediction as a clinical risk stratification problem under class imbalance.
Model selection and hyperparameter tuning are performed on non-resampled training data, followed by retraining on SMOTE-balanced data for final evaluation.
This approach aligns with best practices for imbalanced clinical prediction tasks and is documented for reproducibility.

The pipeline integrates:
- Cohort construction and feature extraction from MIMIC-IV (SQL)
- Structured clinical feature engineering
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

1. Cohort definition and outcome labeling using MIMIC-IV
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
├── sql/                # SQL queries for MIMIC-IV extraction
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

🙋 About Me

Built by Tyler Kelly
📧 Email: tylerjkelly77@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/tylerkelly7)  

🔗 [Portfolio](https://github.com/tylerkelly7)