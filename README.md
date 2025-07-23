# 📉 Customer Churn Prediction Dashboard

This Streamlit app predicts whether a telecom customer is likely to churn based on their service usage, billing details, and support history. The prediction is powered by a trained Random Forest machine learning model with scaled numeric features and label-encoded categorical variables.

---

## 🚀 Features

- 🎯 Predicts customer churn probability in real-time
- 📋 Input customer attributes with sliders and dropdowns
- 📊 Confusion matrix and performance metrics (accuracy, precision, recall, F1)
- 🌐 Easy-to-use web dashboard built with Streamlit
- 🔐 Clean and secure code with proper input processing

---

## 🧠 Model Info

- Algorithm: `RandomForestClassifier` (sklearn)
- 
- Feature scaling: `StandardScaler` on tenure, monthly & total charges
- Categorical encoding: Label encoding (No = 0, Yes = 1, etc.)
- Trained on historical customer data with ~76% accuracy

---
## 🔍 Hyperparameter Tuning

The Random Forest model was optimized using `GridSearchCV` to improve performance and generalization.

- **Tuned parameters**:
  - `n_estimators` (number of trees)
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`
- **Cross-validation**: 5-fold (`cv=5`)
- **Scoring metric**: F1 score (weighted)

This tuning helped balance the bias-variance trade-off and improve the recall score on the minority class (churners).


## 🛠 Tech Stack

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- matplotlib & seaborn

---

## 📦 How to Run

```bash
git clone https://github.com/Samad296/Churn_prediction_Dashboard.git
cd customer-churn-app
pip install -r requirements.txt
streamlit run app.py
