# MLOps Capstone 2 — Casting Defect Detection

Skeleton for the **Casting Defect Detection** Deep-Learning MLOps pipeline (ResNet18 transfer
learning). Infrastructure + the test harness are provided; **you implement the modelling + MLOps
logic** marked `# TODO`. Each notebook section's marks (red) match `MLOps2_Grading_Rubric.xlsx`.

## Provided (don't rewrite)
`config.py` · `src/__init__.py` · `src/dataset.py` · `requirements.txt` · `Dockerfile` ·
`.github/workflows/ci.yml` · `tests/`

## You build (the `# TODO`s)
| File | Stage | You implement |
|---|---|---|
| `Data_Preparation.ipynb` | 1, 2.1 | guided notebook: understanding, quality, versioning, EDA, preprocessing |
| `Model_Development_and_Tracking.ipynb` | 2.2–2.4, 3 | guided notebook: transfer learning, MLflow, eval, registry, FastAPI/Docker/CI |
| `Operations_Monitoring_and_Evidence.ipynb` | 4 | guided notebook: logging, statistical + embedding drift, retraining, rollback |
| `src/data_prep.py` | 1, 2.1 | discovery, quality validation, versioned splits, transforms, image features |
| `src/model.py` | 2.2, 4.3 | ResNet18 transfer-learning model + embedding extractor |
| `src/train.py` | 2.3–2.4, 3.2 | training loop, MLflow tracking + registry + @production, reference baseline |
| `src/evaluate.py` | 3.1 | recall-on-defect / F1 / ROC / confusion + failure cases |
| `app.py` | 3.3 | FastAPI `/health` + `/predict` (image upload) + prediction logging |
| `src/monitoring.py` | 4.1–4.3 | statistical (Evidently+PSI) + embedding drift + confidence |
| `src/retrain.py` | 4.4–4.5 | drift-triggered retrain, version compare, promote/rollback |

The three notebooks carry **descriptive per-task instructions** (objective · stage/task · inputs→outputs
· TODO expectations · dependencies · where to document). In addition, each stage now opens with a
**Markdown sub-task checklist** — every sub-task shows its ID and marks (e.g. `2.2.1 — ImageNet-pretrained
ResNet18, backbone frozen [4]`) so you can see exactly what each mark rewards. Stage 4.3 in the operations notebook also includes a **conceptual walkthrough of embedding drift** (penultimate-layer 512-dim embedding → distance-to-centroid → PSI) so you are not left
to infer it.

## Setup
```bash
pip install -r requirements.txt
# Download the casting dataset (Kaggle) and extract into data/ so you have:
#   data/.../train/{ok_front,def_front}/   and   data/.../test/{ok_front,def_front}/
#   https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
```

## Run (after implementing the TODOs)
```bash
python -m src.train          # train → MLflow → register @production
python -m src.monitoring     # statistical + embedding drift → drift_summary.json
python -m src.retrain        # drift-triggered retrain + rollback decision
pytest -q                    # API + pipeline tests
uvicorn app:app --port 8000  # serve; POST an image to /predict
```

## What you submit 
Upload these individual files; all 100 marks are graded from them:
1. `Casting_Defect_MLOps_Report` as **PDF** (Stages 1, 3, 4 + Stage-3/4 evidence).
2. `Data_Preparation.ipynb` (executed, with outputs).
3. `Model_Development_and_Tracking.ipynb` (executed, with outputs).
4. `Operations_Monitoring_and_Evidence.ipynb` (executed, with outputs).
5. `app.py`.

**Note:** Repository-generated evidence (MLflow UI, Docker build, CI green run, drift report) must be captured
**inside the notebooks/report** as screenshots, code output and summaries — we don't accept
repositories, ZIPs, HTML or JSON files.

## Acceptance targets
* `pytest` green; `/health` returns `model_loaded: true` after training.
* Test **recall on defects** is the headline metric; report F1 + ROC-AUC + confusion.
* `monitoring` flags **statistical and embedding** drift; `retrain` makes a promote/rollback decision.
* MLflow registry shows `casting_defect_classifier` with a `@production` alias.


# Casting Defect Detection using End-to-End MLOps Pipeline

## Project Overview

This project implements an end-to-end MLOps pipeline for automated visual inspection of submersible pump impeller castings using Deep Learning and modern MLOps practices.

The objective is to classify casting images into:

-  OK Casting (`ok_front`)
-  Defective Casting (`def_front`)

The project demonstrates the complete machine learning lifecycle, including data preparation, transfer learning, experiment tracking, model evaluation, monitoring, retraining governance, Docker deployment, and CI/CD automation.

This capstone was implemented as part of the **MLOps Specialization** using production-oriented engineering practices.

---

# Business Problem

Manual inspection of industrial castings is time-consuming, expensive, and susceptible to human error.

Manufacturing industries require automated inspection systems capable of:

- Detecting defective castings accurately
- Reducing inspection time
- Improving production quality
- Supporting continuous monitoring in production environments

This project addresses these challenges using computer vision, transfer learning, and an end-to-end MLOps workflow.

---

# Dataset

**Dataset**

Casting Product Image Data for Quality Inspection

Images represent top-view grayscale photographs of submersible pump impellers.

### Dataset Statistics

| Split | OK | Defective | Total |
|--------|----:|----------:|------:|
| Train | 2875 | 3758 | 6633 |
| Test | 262 | 453 | 715 |
| **Total** | **3137** | **4211** | **7348** |

Image Size

- 300 × 300 pixels
- Grayscale
- Augmented dataset supplied by Kaggle

---

# Project Architecture

```
                Kaggle Dataset
                      │
                      ▼
            Data Validation
                      │
                      ▼
          Dataset Versioning
                      │
                      ▼
      Image Preprocessing & Augmentation
                      │
                      ▼
      Transfer Learning (ResNet18)
                      │
                      ▼
          Model Training
                      │
                      ▼
         MLflow Experiment Tracking
                      │
                      ▼
          Model Evaluation
                      │
                      ▼
        Model Registry (MLflow)
                      │
                      ▼
       Monitoring & Drift Detection
                      │
                      ▼
     Automatic Retraining Decision
                      │
                      ▼
         FastAPI Inference API
                      │
                      ▼
        Docker + GitHub Actions
```

---

# Project Structure

```
.
├── artifacts/
├── data/
├── src/
│   ├── data_prep.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── monitoring.py
│   ├── retrain.py
│   └── train.py
│
├── Data_Preparation.ipynb
├── Model_Development_and_Tracking.ipynb
├── Operations_Monitoring_and_Evidence.ipynb
├── app.py
├── config.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Category | Technology |
|------------|----------------|
| Language | Python 3.11 |
| Deep Learning | PyTorch |
| Computer Vision | TorchVision |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Deployment | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently + PSI |
| Data Processing | NumPy, Pandas |
| Visualisation | Matplotlib |
| Testing | PyTest |

---

# MLOps Workflow

The project follows an end-to-end MLOps workflow consisting of:

- Business Understanding
- Data Quality Validation
- Dataset Versioning
- Exploratory Data Analysis
- Image Preprocessing
- Transfer Learning
- Model Training
- Experiment Tracking
- Model Registry
- Model Evaluation
- Model Deployment
- Monitoring
- Drift Detection
- Automated Retraining
- Governance
- Continuous Integration

---

# Model Development

## Backbone

- ResNet18
- ImageNet Pre-trained

Transfer Learning Strategy

- Frozen Backbone
- Trainable Classification Head

Training Configuration

| Parameter | Value |
|------------|---------|
| Epochs | 8 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Weight Decay | 1e-4 |
| Early Stopping | Patience = 3 |

---

# Experiment Tracking

MLflow was used for:

- Experiment Tracking
- Hyperparameter Logging
- Metric Logging
- Model Versioning
- Model Registry

Registered Model

```
casting_defect_classifier
```

Latest Registered Version

```
Version 6
```

---

# Model Performance

Evaluation was performed on the independent test dataset.

| Metric | Score |
|----------|---------|
| Accuracy | **93.57%** |
| Precision | **98.34%** |
| Recall | **91.39%** |
| F1 Score | **94.74%** |
| Macro F1 | **93.23%** |
| ROC-AUC | **0.9860** |

Confusion Matrix

| | Predicted OK | Predicted Defect |
|---|---:|---:|
| Actual OK | 255 | 7 |
| Actual Defect | 39 | 414 |

These results demonstrate strong defect detection capability with a high ROC-AUC and excellent precision.

---

# Monitoring & Drift Detection

The monitoring pipeline evaluates production data using:

- Statistical Drift (PSI)
- Embedding Drift
- Confidence Monitoring

Example Monitoring Results

| Metric | Value |
|----------|---------|
| Mean PSI | 13.069 |
| Embedding PSI | 12.374 |
| Confidence Drop | 0.0907 |
| Drift Detected | Yes |
| Retraining Triggered | Yes |

Monitoring outputs include:

- Drift Summary
- Drift Report
- Confidence Monitoring
- Embedding Drift Detection
- Retraining Decision

---

# Retraining & Governance

When significant drift is detected:

1. Candidate model is retrained.
2. Candidate model is evaluated.
3. Candidate model is registered in MLflow.
4. Governance policy determines promotion or rollback.

This project demonstrates an automated retraining workflow with model governance.

---

# Docker Deployment

The inference service is containerised using Docker.

```
docker build -t casting-defect-api .
```

Run

```
docker run -p 8000:8000 casting-defect-api
```

---

# Continuous Integration

GitHub Actions automatically performs:

- Dependency Installation
- Unit Testing
- Docker Build Validation

All implemented tests passed successfully.

```
8 tests passed
```

---

# Key Features

✔ Modular Project Structure

✔ Transfer Learning using ResNet18

✔ MLflow Experiment Tracking

✔ Model Registry

✔ Automated Evaluation

✔ PSI-based Drift Detection

✔ Embedding Drift Detection

✔ Confidence Monitoring

✔ Automated Retraining

✔ Docker Deployment

✔ GitHub Actions CI

✔ Production-ready Project Organisation

---

# Future Improvements

Potential future enhancements include:

- Fine-tuning the complete backbone
- GPU training support
- Kubernetes deployment
- Real-time monitoring dashboard
- Model signature logging in MLflow
- Explainable AI using Grad-CAM
- Multi-class defect classification

---

# Results Summary

- Successfully trained on the complete Kaggle dataset (7,348 images)
- Achieved **94.74% F1-score** on the test dataset
- Achieved **98.60% ROC-AUC**
- Implemented complete MLOps lifecycle
- Automated monitoring and retraining pipeline
- Containerised inference service
- Integrated CI/CD using GitHub Actions

---

# Author

**Abhinav Mathur**

Senior Quality Engineering Manager | AI & MLOps Enthusiast

MLOps Capstone Project
