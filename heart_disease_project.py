import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lime import lime_tabular

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
print("📥 Loading Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load data, treat '?' as NaN
df = pd.read_csv(url, names=columns, na_values="?")

# Fill missing values with median (Best practice for medical data)
df.fillna(df.median(), inplace=True)

# Convert target to binary: 0 = Healthy, 1 = Heart Disease
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# ==========================================
# 2. FEATURE ENGINEERING & SPLITTING
# ==========================================
X = df.drop('target', axis=1)
y = df['target']

# Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Standardizing data to mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING (XGBoost)
# ==========================================
print("🧠 Training XGBoost Model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model.predict(X_test_scaled)
print("\n" + "="*30)
print(f"✅ MODEL ACCURACY: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("="*30)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizing Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix: Heart Disease Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# 5. XAI: INTERPRETABILITY WITH LIME
# ==========================================
print("\n🔍 Generating LIME Explanation for a single patient...")

# Initialize LIME Explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_scaled),
    feature_names=X.columns,
    class_names=['Healthy', 'Heart Disease'],
    mode='classification'
)

# Explain the prediction for the first patient in the test set
idx = 0
exp = explainer.explain_instance(
    data_row=X_test_scaled[idx], 
    predict_fn=model.predict_proba
)

print(f"\nPrediction for Patient {idx}: {'Heart Disease' if y_pred[idx] == 1 else 'Healthy'}")
print("Explanation (Local Feature Contribution):")
for feature, importance in exp.as_list():
    print(f"{feature}: {importance:.4f}")

# Visualize LIME output
exp.as_pyplot_figure()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lime import lime_tabular

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
print("📥 Loading Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load data, treat '?' as NaN
df = pd.read_csv(url, names=columns, na_values="?")

# Fill missing values with median (Best practice for medical data)
df.fillna(df.median(), inplace=True)

# Convert target to binary: 0 = Healthy, 1 = Heart Disease
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# ==========================================
# 2. FEATURE ENGINEERING & SPLITTING
# ==========================================
X = df.drop('target', axis=1)
y = df['target']

# Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Standardizing data to mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING (XGBoost)
# ==========================================
print("🧠 Training XGBoost Model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model.predict(X_test_scaled)
print("\n" + "="*30)
print(f"✅ MODEL ACCURACY: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("="*30)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizing Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix: Heart Disease Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# 5. XAI: INTERPRETABILITY WITH LIME
# ==========================================
print("\n🔍 Generating LIME Explanation for a single patient...")

# Initialize LIME Explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_scaled),
    feature_names=X.columns,
    class_names=['Healthy', 'Heart Disease'],
    mode='classification'
)

# Explain the prediction for the first patient in the test set
idx = 0
exp = explainer.explain_instance(
    data_row=X_test_scaled[idx], 
    predict_fn=model.predict_proba
)

print(f"\nPrediction for Patient {idx}: {'Heart Disease' if y_pred[idx] == 1 else 'Healthy'}")
print("Explanation (Local Feature Contribution):")
for feature, importance in exp.as_list():
    print(f"{feature}: {importance:.4f}")

# Visualize LIME output
exp.as_pyplot_figure()
plt.tight_layout()
plt.show()

import joblib

# This saves the 'Brain' of your project so the UI can use it
joblib.dump(model, 'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train_scaled, 'X_train_scaled.pkl')

print("\n✅ SUCCESS: Files 'heart_model.pkl', 'scaler.pkl', and 'X_train_scaled.pkl' created!")