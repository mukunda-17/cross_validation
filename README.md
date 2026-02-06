# 🩺 Breast Cancer Classification using Cross-Validation & Random Forest

A clean and complete machine learning workflow to compare multiple classification models using cross-validation, tune a Random Forest model using GridSearchCV, and evaluate performance using standard classification metrics.

---

## 📌 Project Overview

This project demonstrates how different machine learning models perform on a medical classification problem using proper validation techniques.

The notebook performs:

- Loading the Breast Cancer dataset from `scikit-learn`
- Training and comparing:
  - Decision Tree
  - Support Vector Machine (SVM – RBF kernel)
  - Random Forest
- Evaluating models using:
  - K-Fold Cross-Validation
  - Stratified K-Fold Cross-Validation
- Hyperparameter tuning using GridSearchCV
- Final evaluation using Accuracy, Precision, Recall and F1-Score

---

## 🧠 Dataset

Breast Cancer Wisconsin dataset from `sklearn.datasets`.

- Total samples: **569**
- Total features: **30**
- Task: **Binary classification** (malignant / benign)

---

## ⚙️ Models Used

- Decision Tree Classifier  
- Support Vector Machine (RBF kernel)  
- Random Forest Classifier  

---

## 🔁 Cross-Validation Strategy

Two validation strategies are used:

### ✔ K-Fold Cross-Validation
- Number of folds: **5**
- Data is shuffled before splitting

### ✔ Stratified K-Fold Cross-Validation
- Preserves class distribution in each fold
- More suitable for classification problems

Both strategies are applied to compare Decision Tree and SVM models.

---

## 🌲 Random Forest with Hyperparameter Tuning

Random Forest is optimized using **GridSearchCV**.

### Tuned parameters

- `n_estimators`: [100, 200]  
- `max_depth`: [None, 10, 20]  
- `min_samples_split`: [2, 5]  
- `min_samples_leaf`: [1, 2]

- Validation strategy: **Stratified K-Fold**
- Scoring metric: **Accuracy**

---

## 🧪 Final Evaluation

After selecting the best Random Forest model:

- Train-test split is performed
- Predictions are generated on the test set
- The following metrics are computed:

- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## 📊 Model Comparison

| Model         | Cross-Validated Accuracy |
|--------------|---------------------------|
| Decision Tree | ~0.910 |
| SVM           | ~0.914 |
| Random Forest | ~0.956 |

---

## 🏆 Key Observations

- Random Forest achieves the highest cross-validated accuracy.
- Stratified K-Fold gives more reliable evaluation for this dataset.
- Hyperparameter tuning improves model performance and stability.

---

## 🛠️ Tech Stack

- Python
- Jupyter Notebook / Google Colab
- NumPy
- Pandas
- scikit-learn

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone <your-repository-url>
