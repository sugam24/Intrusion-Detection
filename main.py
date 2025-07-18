import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# === Column Names ===
columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
           'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
           'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
           'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
           'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
           'dst_host_srv_rerror_rate','outcome','level']

# === Load Datasets ===
train_data = pd.read_csv("dataset/KDDTrain+.txt", names=columns)
test_data = pd.read_csv("dataset/KDDTest+.txt", names=columns)

# === Label Encoding: Binary (normal=0, attack=1) ===
train_data['outcome'] = train_data['outcome'].apply(lambda x: 0 if x.strip() == 'normal' else 1)
test_data['outcome'] = test_data['outcome'].apply(lambda x: 0 if x.strip() == 'normal' else 1)

# === Categorical Columns ===
cat_cols = ['protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login']

# === Separate Features and Labels ===
X_train = train_data.drop(columns=['outcome', 'level'])
y_train = train_data['outcome']
X_test = test_data.drop(columns=['outcome', 'level'])
y_test = test_data['outcome']

# === One-hot Encoding ===
X_train = pd.get_dummies(X_train, columns=cat_cols)
X_test = pd.get_dummies(X_test, columns=cat_cols)

# === Align Test Set Columns ===
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# === Robust Scaling ===
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === PCA (optional, for analysis & comparison) ===
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original Features: {X_train.shape[1]}, After PCA: {X_train_pca.shape[1]}")

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
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# === Define Models ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=20),
    "Naive Bayes": GaussianNB(),
    "Linear SVM": LinearSVC(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# === Train & Evaluate ===
for name, model in models.items():
    evaluate_model(model, name, X_train_scaled, X_test_scaled, y_train, y_test)

# === Feature Importance for Tree-Based Models ===
def plot_feature_importance(model, X, title, top_n=20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-top_n:]
        names = np.array(X.columns)[idx]
        plt.figure(figsize=(10, 6))
        plt.barh(names, importances[idx])
        plt.title(f"{title} - Top {top_n} Features")
        plt.xlabel("Importance")
        plt.show(block=False)
        plt.pause(2)
        plt.close()

plot_feature_importance(models["Random Forest"], pd.DataFrame(X_train_scaled, columns=X_train.columns), "Random Forest")
plot_feature_importance(models["XGBoost"], pd.DataFrame(X_train_scaled, columns=X_train.columns), "XGBoost")

# === Decision Tree Visualization ===
plt.figure(figsize=(16, 10))
plot_tree(models["Decision Tree"], filled=True, feature_names=X_train.columns, class_names=["Normal", "Attack"], max_depth=3)
plt.title("Decision Tree (depth=3)")
plt.show(block=False)
plt.pause(2)
plt.close()

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
    plt.show(block=False)
    plt.pause(2)
    plt.close()

plot_comparison(0, "Accuracy")
plot_comparison(1, "Precision")
plot_comparison(2, "Recall")
