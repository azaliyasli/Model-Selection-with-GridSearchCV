# 🔍 Best SVM Model Selection with Grid Search

## 📌 Overview
This project uses **Support Vector Machines (SVMs)** to predict whether a user will purchase a product based on their **Age** and **Estimated Salary**. The main focus is on **Hyperparameter Tuning with Grid Search**, which is an essential technique for finding the **best model settings** automatically.

---

## 🧩 What is Grid Search?
**Grid Search** is a brute-force method that tries **all possible combinations of hyperparameters** you specify.  

- Example: 
  - `C = [0.25, 0.5, 0.75, 1]`  
  - `kernel = ['linear', 'rbf']`  

  → Grid Search will train and evaluate the model **6 times** (3×2) using every combination.

- It uses **cross-validation** internally (`cv=10` here) to measure accuracy for each combination, ensuring results are not biased by a single train/test split.  

- Finally, it reports:  
  ✅ The **best accuracy**  
  ✅ The **best hyperparameter combination**

---

## ✨ Features
- Load and preprocess dataset
- Feature scaling with `StandardScaler`
- Train an SVM classifier
- Apply **Grid Search** to tune:
  - `C` → Regularization strength  
  - `kernel` → Type of SVM (linear or RBF)  
  - `gamma` → Controls flexibility of RBF decision boundaries  
- Report **best accuracy** and **best parameters**

---

## 📂 Dataset
The `Social_Network_Ads.csv` file contains:
- **Age**  
- **Estimated Salary**  
- **Purchased** (target: 0 = No, 1 = Yes)

---

## 🛠 Requirements
- Python 3.x
- pandas
- scikit-learn
