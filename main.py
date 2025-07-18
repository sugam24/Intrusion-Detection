import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# === Load dataset ===
columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
           'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
           'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
           'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
           'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
           'dst_host_srv_rerror_rate','outcome','level']

data = pd.read_csv("dataset/KDDTrain+.txt", names=columns)

# === Binary label ===
data['outcome'] = data['outcome'].apply(lambda x: 0 if x.strip() == 'normal' else 1)

# === Preprocessing ===
cat_cols = ['protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login']
X = data.drop(columns=['outcome', 'level'])
y = data['outcome']

# One-hot encoding for categorical features
X = pd.get_dummies(X, columns=cat_cols)

# Robust Scaling for numeric features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# === PCA (optional) ===
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)
print(f"Original: {X.shape[1]}, After PCA: {X_pca.shape[1]}")

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# === Evaluation Function ===
results = {}

def evaluate_model(model, name, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    results[name] = [acc, prec, rec]
    print(f"\n{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack'])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.grid(False)
    plt.show()

# === Train Models ===
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=20),
    "Naive Bayes": GaussianNB(),
    "Linear SVM": LinearSVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    evaluate_model(model, name, X_train, X_test, y_train, y_test)

# === Feature Importance from XGBoost and RandomForest ===
def plot_feature_importance(model, X, title, top_n=20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-top_n:]
        names = np.array(X.columns)[idx]
        plt.figure(figsize=(10, 6))
        plt.barh(names, importances[idx])
        plt.title(f"{title} - Top {top_n} Features")
        plt.show()

plot_feature_importance(models["Random Forest"], pd.DataFrame(X_scaled, columns=X.columns), "Random Forest")
plot_feature_importance(models["XGBoost"], pd.DataFrame(X_scaled, columns=X.columns), "XGBoost")

# === Decision Tree Visualization ===
plt.figure(figsize=(16, 10))
plot_tree(models["Decision Tree"], filled=True, feature_names=X.columns, class_names=["Normal", "Attack"], max_depth=3)
plt.title("Decision Tree (depth=3)")
plt.show()

# === Model Comparison Plot ===
def plot_comparison(metric_idx, title):
    names = list(results.keys())
    vals = [metrics[metric_idx] for metrics in results.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(names, vals, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel(title)
    plt.title(f"Model Comparison - {title}")
    plt.tight_layout()
    plt.show()

plot_comparison(0, "Accuracy")
plot_comparison(1, "Precision")
plot_comparison(2, "Recall")
