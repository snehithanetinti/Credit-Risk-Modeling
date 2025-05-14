import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Load dataset
df = pd.read_excel("small_credit_risk_dataset.xlsx")

# Separate features and target
X = df.drop(columns=["default"])
y = df["default"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

if len(set(y_test)) == 2:
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n🎯 ROC AUC Score: {auc:.2f}")
else:
    print("\n⚠️ ROC AUC Score cannot be computed (only one class in test set).")
