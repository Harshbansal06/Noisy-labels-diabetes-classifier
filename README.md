# Noisy-labels-diabetes-classifier

A robust machine learning pipeline built to handle malicious data corruption, developed for the **[Summer of ML'26 - Classification Hackathon](https://www.kaggle.com/competitions/summer-of-ml-26-classification-hacakthon/overview)** hosted by The Programming Club - IIITDMJ.

---
## 📊 Dataset Access
To comply with the competition's data protection rules, the raw datasets are not included in this repository. 

To run this project locally:
1. Register and download the data from the official **[Summer of ML'26 Leaderboard / Data Tab](https://www.kaggle.com/competitions/summer-of-ml-26-classification-hacakthon/leaderboard)**.
2. Place `train.csv` and `test.csv` directly into your local root directory before executing the notebooks.


## 🎯 Competition Overview & The Challenge
In real-world data science, datasets are rarely pristine. The core challenge of this hackathon is to train a binary classifier that predicts whether a respondent has diabetes (`1`) or not (`0`) using a training dataset that has been **deliberately corrupted** with:
* ⚠️ **Malicious Label Noise:** A significant portion of the target labels (`Diabetes_binary`) have been flipped or misrecorded.
* ⚖️ **Extreme Class Imbalance:** Severe skewness between diabetic and non-diabetic records.
* 🛑 **Anomalous Sensor Readings:** Out-of-range or impossible values injected into the features.

While the training data is highly noisy, the **test data remains perfectly clean**. The objective is to construct a denoising and training pipeline that generalises effectively without overfitting to the artificial noise.

---

## 📊 Evaluation Metric
Submissions are evaluated using the **Macro F1-score**, which calculates the unweighted mean of the F1-score for both classes:

$$\text{Macro F1} = \frac{F1_0 + F1_1}{2}$$

### Why this matters:
Standard accuracy metrics are highly misleading under extreme class imbalances. A naive model that blindly predicts the majority class (`0`) could achieve high overall accuracy but will yield a terrible Macro F1-score. This project prioritizes robust minority-class identification.

---

## ⚙️ Core Pipeline Architecture
Everything is implemented entirely within a single, end-to-end Jupyter Notebook (`diabetes_classifier.ipynb`) broken into distinct operational phases:

### 1. Exploratory Data Analysis (EDA)
* Profiling the 21 health survey features (demographics, lifestyle factors, and clinical history).
* Visualizing class distributions and isolating anomalies in sensor configurations.

### 2. Denoising & Feature Engineering
* Identifying and cleaning or masking corrupted data labels.
* Handling anomalous out-of-bounds metrics.
* Addressing class imbalance using localized sampling techniques or class-weighted loss adjustments.

### 3. Model Training & Cross-Validation
* Utilizing robust tree-based ensembles (such as XGBoost, LightGBM, or CatBoost) which are inherently more resilient to feature anomalies.
* Executing a strict stratified cross-validation strategy to mirror the clean test set distribution and prevent noise leakage.

---

## 📂 Repository Structure
```text

├── diabetes_classifier.ipynb     # Complete end-to-end pipeline (EDA, Denoising, Modeling)
├── submission.csv                # Generated predictions ready for Kaggle upload
├── .gitignore                    # Configured to ignore large CSV datasets and virtual envs
├── LICENSE                       # MIT License
└── README.md                     # Documentation
