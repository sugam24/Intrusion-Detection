---

```markdown
# NSL-KDD Intrusion Detection using Machine Learning

This project implements and compares various machine learning models for network intrusion detection using the **NSL-KDD** dataset. It includes steps for automatic dataset download, preprocessing, dimensionality reduction (PCA), model training, evaluation, and visualisation.

---

## 🔍 Overview

- Automatically downloads the NSL-KDD dataset from Kaggle.
- Preprocesses data with robust scaling and one-hot encoding.
- Applies PCA for optional dimensionality reduction.
- Trains multiple classification models including:
  - Logistic Regression
  - K-Nearest Neighbours
  - Naive Bayes
  - Linear SVM
  - Decision Tree
  - Random Forest
  - XGBoost
- Evaluates models using Accuracy, Precision, Recall, and Confusion Matrix.
- Visualises:
  - Feature importance (Random Forest & XGBoost)
  - Decision Tree (depth-limited)
  - Model comparison metrics

---

## 📁 Project Structure

```

.
├── dataextraction.py         # Script to download and set up dataset from Kaggle
├── main.py                   # Core pipeline: preprocessing, training, evaluation
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore config
└── README.md                 # Project documentation (this file)

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sugam24/nsl-kdd-ml.git
cd nsl-kdd-ml
````

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Authenticate with Kaggle

Ensure you have your Kaggle API credentials set up (i.e., a valid `~/.kaggle/kaggle.json` file). Learn more: [Kaggle API Docs](https://www.kaggle.com/docs/api)

### 5. Run the Dataset Setup

```bash
python dataextraction.py
```

### 6. Run the Main Pipeline

```bash
python main.py
```

---

## 🧠 Models Compared

| Model               | Accuracy | Precision | Recall |
| ------------------- | -------- | --------- | ------ |
| Logistic Regression | ✅        | ✅         | ✅      |
| KNN                 | ✅        | ✅         | ✅      |
| Naive Bayes         | ✅        | ✅         | ✅      |
| Linear SVM          | ✅        | ✅         | ✅      |
| Decision Tree       | ✅        | ✅         | ✅      |
| Random Forest       | ✅        | ✅         | ✅      |
| XGBoost             | ✅        | ✅         | ✅      |

*Exact metrics will be shown during execution via confusion matrices and bar plots.*

---

## 📊 Visualisations Included

* Confusion matrices for each model
* Top 20 feature importances for:

  * Random Forest
  * XGBoost
* Decision tree visual (depth=3)
* Bar plots comparing model performance metrics

---

## 📦 Dataset

* **NSL-KDD** (hosted on Kaggle): A refined version of the KDD Cup 1999 dataset, widely used for intrusion detection research.
* Dataset link: [https://www.kaggle.com/datasets/hassan06/nslkdd](https://www.kaggle.com/datasets/hassan06/nslkdd)

---

## 📑 Requirements

Make sure the following libraries are installed (included in `requirements.txt`):

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `kagglehub`

---

👥 Authors
Sugam Dahal
📍 Kathmandu, Nepal
🔗 GitHub: @sugam24

Pranil Parajuli
📍 Kathmandu, Nepal
🔗 GitHub: @praniil

---

## 📜 License

This project is for educational purposes and research use. You are free to use and modify it with proper attribution.

```

---

Let me know if you'd like badges, interactive demo links, or Docker setup included!
```
