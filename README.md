<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Random%20Forest-Best%20Model-228B22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Accuracy-100%25-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Domain-Healthcare%20AI-red?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Deployment-Pickle-lightgrey?style=for-the-badge"/>

<br/><br/>

# 🦷 Oral Cancer Prediction — Classification

### *Early diagnosis through machine learning — MedTech Insights Pvt. Ltd.*

<br/>

> **Why it matters:** Oral cancer is one of the most prevalent cancers in developing countries, yet highly treatable when caught early. This project builds a clinical classification system to predict oral cancer diagnosis from demographic, lifestyle, and medical variables — putting ML to work where it counts.

<br/>

---

</div>

## 📌 Table of Contents

- [Project Background](#-project-background)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models & Results](#-models--results)
- [Feature Importance](#-feature-importance)
- [Key Insights](#-key-insights)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)

---

## 🏢 Project Background

**Role:** Junior Data Scientist  
**Department:** Healthcare Analytics  
**Company:** MedTech Insights Pvt. Ltd.

Oral cancer is driven by a combination of lifestyle factors — tobacco use, alcohol consumption, betel quid chewing — and underlying medical conditions. MedTech Insights tasked this project with building a **predictive classification model** that flags high-risk patients based on their profile, enabling earlier clinical intervention and better survival outcomes.

---

## 📂 Dataset

**File:** `oral_cancer_prediction_dataset.csv`  
**Target Variable:** `Oral Cancer (Diagnosis)` — Binary (Yes / No)

| Feature | Description |
|---|---|
| `Age` | Patient age |
| `Gender` | Male / Female |
| `Country` | Patient's country of origin (17 countries) |
| `Tobacco Use` | Yes / No |
| `Alcohol Consumption` | Yes / No |
| `HPV Infection` | Yes / No |
| `Betel Quid Use` | Yes / No |
| `Chronic Sun Exposure` | Yes / No |
| `Poor Oral Hygiene` | Yes / No |
| `Diet (Fruits & Vegetables)` | Low / Moderate / High |
| `Family History of Cancer` | Yes / No |
| `Compromised Immune System` | Yes / No |
| `Oral Lesions` | Yes / No |
| `Unexplained Bleeding` | Yes / No |
| `Difficulty Swallowing` | Yes / No |
| `White or Red Patches in Mouth` | Yes / No |
| `Tumor Size (cm)` | Continuous |
| `Cancer Stage` | Ordinal (1–4) |
| `Treatment Type` | Surgery / Radiation / Chemotherapy / Targeted / None |
| `Survival Rate (5-Year, %)` | Continuous |
| `Cost of Treatment (USD)` | Continuous |
| `Economic Burden (Lost Workdays/Year)` | Continuous |
| `Early Diagnosis` | Yes / No |
| **`Oral Cancer (Diagnosis)`** | 🎯 **Target** |

---

## 🔬 Project Workflow

```
Raw Dataset
   │
   ├── Step 1: Data Loading & Understanding
   │       ├── Verified dtypes, null values, duplicates
   │       ├── Dropped ID column (non-predictive)
   │       └── Renamed all columns to clean snake_case
   │
   ├── Step 2: Exploratory Data Analysis (EDA)
   │       ├── Class distribution of target variable
   │       ├── KDE + histogram distributions per feature
   │       ├── Skewness & kurtosis analysis
   │       ├── Transformation experiments (log, sqrt, Yeo-Johnson)
   │       ├── Pairplots across numeric features
   │       ├── Age vs Diagnosis line analysis
   │       ├── Tobacco Use × Diagnosis crosstab
   │       └── Cancer Stage vs Treatment Cost trend
   │
   ├── Step 3: Outlier Detection
   │       ├── IQR method across all numeric columns
   │       └── Age outliers retained (valid range 0–100)
   │
   ├── Step 4: Data Preprocessing
   │       ├── Ordinal & binary encoding via manual mapping
   │       ├── Train-test split (70/30)
   │       └── StandardScaler normalization
   │
   ├── Step 5–6: Model Building & Evaluation
   │       └── 5 classifiers trained and compared
   │
   ├── Step 7: Model Selection & Finalization
   │       └── Random Forest selected (interpretability + perfect score)
   │
   ├── Step 8: Feature Importance
   │       └── Top 10 features extracted → leaner model built
   │
   └── Step 9: Deployment
           ├── Model saved with Pickle
           └── Load & predict pipeline demonstrated
```

---

## 📈 Models & Results

Five classification algorithms evaluated on a 70/30 train-test split with StandardScaler:

| Model | Accuracy | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Decision Tree** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest** ✅ | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| K-Nearest Neighbors | 0.9957 | 1.0000 | 0.9914 | 0.9957 | 0.9914 |

### 🏆 Selected Model: Random Forest Classifier

Four of five models achieved perfect scores — a result driven by strong, clean separability in the dataset. **Random Forest** was selected as the final model for its:
- **Interpretability** via feature importances
- **Robustness** to overfitting through ensemble averaging
- **Deployment readiness** with a compact, exportable pickle file

The leaner **top-10 feature RFC model** also achieved perfect classification — confirming that feature selection doesn't compromise performance here.

---

## 🔍 Feature Importance

Top 10 features ranked by Random Forest importance scores:

| Rank | Feature | Clinical Relevance |
|---|---|---|
| 1 | `tumor_size` | Directly indicates cancer progression |
| 2 | `cancer_stage` | Core clinical staging metric |
| 3 | `survival_rate` | Proxy for disease severity |
| 4 | `economic_burden` | Correlates with treatment intensity |
| 5 | `treatment_cost` | Reflects severity of required care |
| 6 | `treatment_type` | Indicates how aggressively disease was treated |
| 7 | `age` | Older patients carry higher risk |
| 8 | `country` | Reflects regional prevalence & healthcare access |
| 9 | `diet_intake` | Protective factor (fruits & vegetables) |
| 10 | `immune_status` | Compromised immunity elevates cancer risk |

---

## 💡 Key Insights

- **Clinical markers dominate** — tumor size, cancer stage, and survival rate are the most predictive features, which aligns with medical literature
- **Age is a meaningful predictor** — cancer-positive patients tend to be older on average
- **Tobacco use** shows a clear elevated diagnosis rate in crosstab analysis — consistent with known epidemiology
- **Diet (fruits & vegetables)** ranks as a top-10 protective feature — reinforcing the preventive value of nutrition
- **Compromised immune system** amplifies cancer risk significantly across the dataset
- The model's high accuracy suggests the dataset has strong signal — making it a reliable candidate for clinical decision support tooling

---

## 🚀 Deployment

The final model is serialized and ready for integration into clinical workflows:

```python
import pickle

# Load the model
with open('rfc_10.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on a new patient
# Features: tumor_size, cancer_stage, survival_rate, economic_burden,
#           treatment_cost, treatment_type, age, country, diet_intake, immune_status
new_patient = [[2.59, 4, 28.88, 94, 72950.0, 1, 41, 6, 0, 1]]
new_patient_scaled = scaler.transform(new_patient)

prediction = model.predict(new_patient_scaled)
# Output: [1] → Oral Cancer Diagnosed
```

---

## 🛠 Tech Stack

```
Language      Python 3.10+
ML Library    scikit-learn, XGBoost
Data          pandas, NumPy
Viz           matplotlib, seaborn
Deployment    Pickle
Environment   Google Colab / Jupyter Notebook
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Pfuglo1/oral-cancer-prediction.git
cd oral-cancer-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# 3. Launch the notebook
jupyter notebook Project_3_Oral_Cancer_Prediction_Dataset_with_Deployment.ipynb
```

> 📂 Place `oral_cancer_prediction_dataset.csv` in the same directory before running.

---

## 🗂 Project Structure

```
oral-cancer-prediction/
│
├── Project_3_Oral_Cancer_Prediction_Dataset_with_Deployment.ipynb  # Full pipeline notebook
├── Project_3__Oral_Cancer_Prediction.pdf                           # Project brief
├── rfc_10.pkl                                                       # Saved model (Pickle)
├── README.md                                                        # You are here
└── oral_cancer_prediction_dataset.csv                              # Dataset
```

---

<div align="center">

**Built at MedTech Insights Pvt. Ltd. | Healthcare Analytics Division 🏥**

*Machine learning in service of early diagnosis — because early detection saves lives.*

*Found this useful? Give it a ⭐!*

</div>
