---

# 🚨 NSL-KDD Intrusion Detection using Machine Learning

This project implements and compares several machine learning models for detecting network intrusions using the **NSL-KDD** dataset. It automates dataset setup, applies preprocessing techniques, trains models, and evaluates them with visual metrics.

---

## ✨ Highlights

- Automatic dataset download from Kaggle  
- Categorical encoding + robust feature scaling  
- Optional dimensionality reduction with PCA  
- Multiple classifiers tested and compared  
- Confusion matrix, feature importance, and decision tree visualisations


---

## 📂 Project Structure

```
.
├── dataextraction.py     # Downloads and extracts the NSL-KDD dataset
├── main.py               # Preprocessing, training, evaluation, visualisation
├── requirements.txt      # Python dependencies
├── .gitignore            # Files/folders to ignore in version control
└── README.md             # Project overview and documentation
```

---

## ⚙️ How to Run

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sugam24/Intrusion-Detection.git
   cd Intrusion-Detection
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install Required Libraries**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Kaggle API Access**

   * Download `kaggle.json` from your Kaggle account.
   * Place it inside `~/.kaggle/` directory.

5. **Download the Dataset**

   ```bash
   python dataextraction.py
   ```

6. **Run the ML Pipeline**

   ```bash
   python main.py
   ```

---

## 🧠 Models Trained

* Logistic Regression
* K-Nearest Neighbours (KNN)
* Naive Bayes
* Linear SVM
* Decision Tree
* Random Forest
* XGBoost

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* Confusion Matrix

---

## 📊 Visual Outputs

* 🔷 Confusion matrix for each model
* 📈 Feature importance plots (Random Forest, XGBoost)
* 🌳 Decision tree visual (max depth = 3)
* 📊 Comparison bar charts for accuracy, precision, and recall

---

## 🗂 Dataset Used

**NSL-KDD** – A cleaned and improved version of the KDD'99 dataset
📥 Source: [Kaggle - hassan06/nslkdd](https://www.kaggle.com/datasets/hassan06/nslkdd)

---

## 🧾 Dependencies

* `numpy`, `pandas` – data manipulation
* `matplotlib`, `seaborn` – plotting
* `scikit-learn` – machine learning models & metrics
* `xgboost` – gradient boosting classifier
* `kagglehub` – easy dataset download from Kaggle

(Everything is listed in `requirements.txt`)

---

## 👨‍💻 Authors

**Sugam Dahal**
📍 Kathmandu, Nepal
🔗 [github.com/sugam24](https://github.com/sugam24)

**Pranil Parajuli**
📍 Kathmandu, Nepal
🔗 [github.com/praniil](https://github.com/praniil)

---

## 📜 License

This project is for educational and research purposes.
Feel free to use, adapt, and cite with proper credit.

---
