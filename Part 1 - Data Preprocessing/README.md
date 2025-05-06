# Data Preprocessing in Machine Learning

This repository contains two essential Python scripts for preprocessing data in machine learning workflows. Proper data preprocessing is a crucial step to ensure your model performs well and learns from high-quality, well-structured data.

---

## 📁 Folder Contents

```
datapreprocessing/
├── Data.csv                         # Sample dataset used for demonstration
├── Data Preprocessing Template.py  # Basic setup: importing, loading, splitting
├── Data Preprocessing Tools.py     # Full preprocessing pipeline
└── README.md                        # You're here!
```

---

## 🧠 Purpose of This Repository

The goal of this repository is to show how to:
- Prepare raw datasets for machine learning models
- Handle missing values
- Encode categorical variables
- Normalize numerical values (feature scaling)
- Split datasets into training and test sets

---

## 📌 1. Data Preprocessing Template

### 🔹 File: `Data Preprocessing Template.py`

This script performs basic preprocessing:
- **Imports libraries**: NumPy, Matplotlib, Pandas
- **Loads dataset** from `Data.csv`
- **Separates features and target**
- **Splits** the dataset into training and test sets

### 🔧 Use When:
You already have a clean dataset (no missing values, properly encoded).

### ✅ Output:
- `X_train`, `X_test`, `y_train`, `y_test` split and ready for model training and evaluation.

---

## 📌 2. Data Preprocessing Tools

### 🔹 File: `Data Preprocessing Tools.py`

This script builds a complete preprocessing pipeline, step-by-step.

### 📋 Steps Explained:

#### 1. **Importing Libraries**
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
Used for numerical computation, data manipulation, and visual inspection.

---

#### 2. **Importing the Dataset**
```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # All columns except last (features)
y = dataset.iloc[:, -1].values  # Last column (target)
```

---

#### 3. **Handling Missing Data**
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])  # Apply to numerical columns
X[:, 1:3] = imputer.transform(X[:, 1:3])
```
Replaces missing numerical values with the column mean.

> ℹ️ Why? Most ML algorithms can't handle `NaN` values. This fills in missing values in a statistically sound way.

---

#### 4. **Encoding Categorical Variables**

##### a. Encoding Independent Variables
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```
Uses OneHotEncoding for non-numerical columns (like country names).

> ℹ️ Why? ML models require numerical inputs. One-hot encoding avoids giving an ordinal relationship to categorical data.

##### b. Encoding Dependent Variable
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```
Converts categorical output (e.g., "Yes"/"No") into 1s and 0s.

---

#### 5. **Splitting Dataset**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
```
Splits 80% of data into training and 20% into test sets.

> 🔄 `random_state=1` ensures reproducibility

---

#### 6. **Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
Normalizes features by removing mean and scaling to unit variance.

> ℹ️ Why? Ensures that all features contribute equally to the model's performance (especially for distance-based algorithms like KNN or SVM).

---

## 📊 Example Dataset Format (`Data.csv`)

```csv
Country,Age,Salary,Purchased
France,44,72000,No
Spain,27,48000,Yes
Germany,30,54000,No
Spain,38,61000,No
Germany,40,,Yes
France,35,58000,Yes
Spain,,52000,No
France,48,79000,Yes
Germany,50,83000,No
France,37,67000,Yes
```

---

## ▶️ How to Run the Scripts

1. Make sure Python 3.x is installed.
2. Install required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
3. Run either script:
   ```bash
   python "Data Preprocessing Template.py"
   ```
   or
   ```bash
   python "Data Preprocessing Tools.py"
   ```

You will see printed outputs showing the intermediate steps like the transformed `X`, `y`, scaled values, etc.

---

## 📎 Key Learnings

| Step                  | Purpose                                                  |
|-----------------------|-----------------------------------------------------------|
| Missing Value Handling| Ensures no NaNs interrupt model training                  |
| Encoding              | Converts categorical → numerical                          |
| Splitting             | Separates training and evaluation phases                  |
| Scaling               | Prevents bias from feature value ranges                   |

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙋‍♂️ Questions?

Feel free to open an issue or reach out if you have any questions or need more preprocessing techniques like:
- Text processing
- Outlier removal
- Feature selection
