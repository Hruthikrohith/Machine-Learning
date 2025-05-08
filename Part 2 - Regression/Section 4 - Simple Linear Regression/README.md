Here's a detailed and professional README file you can use for your **Simple Linear Regression** project. It includes sections like project description, prerequisites, setup, usage, and visual output explanation.

---

# 📈 Simple Linear Regression: Salary vs Experience

This project implements a simple linear regression model using Python to predict salaries based on years of experience. The dataset used is a small set of salary data, and the model demonstrates how linear relationships between variables can be used to make predictions.

---

## 📁 Project Structure

```
Simple_Linear_Regression/
├── Salary_Data.csv
├── simple_linear_regression.py
└── README.md
```

---

## 📌 Objective

To build a regression model that predicts a person’s salary based on their years of experience using the **Simple Linear Regression** technique.

---

## 📊 Dataset

**File**: `Salary_Data.csv`
This CSV file contains two columns:

* `YearsExperience`: Number of years a person has worked
* `Salary`: Salary corresponding to the experience

> 📦 Total Records: 30 rows
> 📥 Source: Synthetic dataset commonly used in machine learning examples

---

## 🔧 Technologies & Libraries Used

* Python 3.x
* NumPy
* Pandas
* Matplotlib
* scikit-learn

---

## ⚙️ Installation and Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Hruthikrohith/Machine-Learning/simple_linear_regression.git
   cd imple_Linear_Regression
   ```

2. **Install required libraries**

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

3. **Run the script**

   ```bash
   python simple_linear_regression.py
   ```

---

## 🧠 How It Works

### Step 1: Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### Step 2: Loading the Dataset

```python
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### Step 3: Splitting the Data

Split into training (2/3) and testing (1/3) sets.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
```

### Step 4: Training the Model

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### Step 5: Making Predictions

```python
y_pred = regressor.predict(X_test)
```

### Step 6: Visualization

* **Training Set**
  Shows how the regression line fits the training data.
* **Test Set**
  Compares predicted salaries with actual test data.

Both plots display:

* Red dots = actual data
* Blue line = regression prediction

---


## ✅ Key Learnings

* How to use `LinearRegression` from scikit-learn
* Basic data preprocessing and train-test splitting
* Visualizing regression lines with matplotlib

---

## 📝 License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute.

---


