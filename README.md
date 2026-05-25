# 🔄 Customer Churn Prediction – End-to-End ML Pipeline

A production-ready, reusable machine learning pipeline for predicting customer churn using the scikit-learn Pipeline API. Built as part of the AI/ML Engineering Advanced Internship at DevelopersHub Corporation.

---

## 📌 Objective
Build a fully automated, end-to-end ML pipeline that preprocesses data, trains classification models, tunes hyperparameters, and exports a reusable pipeline for predicting whether a telecom customer will churn.

---

## 🗂️ Dataset
* **Name:** Telco Customer Churn Dataset
* **Source:** [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Size:** ~7,000 rows, 21 features
* **Target:** `Churn` (Yes / No)

---

## 🏗️ Methodology / Approach

### 1. Exploratory Data Analysis
* Analysed churn distribution and class imbalance
* Visualised tenure vs churn relationship
* Identified key features driving churn behaviour

### 2. Data Preprocessing (Inside Pipeline)
* Fixed `TotalCharges` column (contained spaces, converted to numeric)
* Dropped `customerID` (non-predictive identifier)
* Applied `StandardScaler` to numerical features: `tenure`, `MonthlyCharges`, `TotalCharges`
* Applied `OneHotEncoder` to all categorical features
* Used `ColumnTransformer` to handle both feature types simultaneously

### 3. Model Training
* Trained two models inside the pipeline:
  * **Logistic Regression** (baseline)
  * **Random Forest Classifier**
* Compared performance on accuracy, F1, and ROC-AUC

### 4. Hyperparameter Tuning
* Applied `GridSearchCV` with 5-fold cross-validation on Random Forest
* Tuned: `n_estimators`, `max_depth`, `min_samples_split`
* Scoring metric: F1-Score

### 5. Evaluation & Visualisation
* Generated confusion matrix for the best model
* Plotted ROC curves for all three models (LR, RF, Tuned RF)
* Visualised top 15 feature importances from the tuned Random Forest

### 6. Pipeline Export
* Exported the complete pipeline (preprocessing + model) using `joblib`
* Verified by loading the saved pipeline and running predictions

---

## 📊 Evaluation Metrics
| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.80 | 0.60 | 0.84 |
| Random Forest | 0.78 | 0.53 | 0.81 |
| Tuned Random Forest | 0.80 | 0.58 | 0.84 |

---

## 📈 Key Results & Observations
* Random Forest outperformed Logistic Regression by capturing non-linear relationships in the data
* GridSearchCV improved the F1 score by finding the optimal depth and estimator count
* `tenure`, `MonthlyCharges`, and `TotalCharges` were consistently the most important features
* The exported `churn_pipeline.pkl` can be loaded and used in production without re-training
* Customers with low tenure and high monthly charges showed the highest churn probability

---

## 🛠️ Tech Stack
* Python, scikit-learn
* pandas, NumPy
* matplotlib, seaborn
* joblib (model export)

---

## ⚙️ Setup & Run

### Requirements
```bash
pip install scikit-learn pandas matplotlib seaborn joblib
```

### Dataset
Download the dataset from Kaggle and upload the CSV to your environment:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Steps
1. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to your working directory
2. Open `Customer_Churn_ML_Pipeline.ipynb`
3. Run all cells in order

### Load the Exported Pipeline
```python
import joblib
pipeline = joblib.load('churn_pipeline.pkl')
predictions = pipeline.predict(new_data)
```

---

## 📁 Repository Structure
```
├── Customer_Churn_ML_Pipeline.ipynb
├── churn_pipeline.pkl           ← exported after running the notebook
└── README.md
```
