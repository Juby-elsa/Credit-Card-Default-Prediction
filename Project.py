import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sns.set_theme(style="whitegrid")

df = pd.read_csv("UCI_Credit_Card.csv")
df.drop("ID", axis=1, inplace=True)

X = df.drop("default.payment.next.month", axis=1)
y = df["default.payment.next.month"]

print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

plt.figure(figsize=(6,4))
sns.countplot(x=y, hue=y, palette=["#8ECAE6", "#FFB703"], legend=False)
plt.xticks([0,1], ["No Default", "Default"])
plt.title("Default vs Non-Default Customers")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(data=df, x="AGE", hue="default.payment.next.month", bins=30, kde=True)
plt.title("Age Distribution by Default Status")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(data=df, x="LIMIT_BAL", hue="default.payment.next.month", bins=40)
plt.title("Credit Limit Distribution by Default Status")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="EDUCATION", hue="default.payment.next.month", data=df)
plt.title("Education Level vs Default")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="MARRIAGE", hue="default.payment.next.month", data=df)
plt.title("Marital Status vs Default")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.drop("default.payment.next.month", axis=1).corr(), cmap="Spectral", linewidths=0.3)
plt.title("Feature Correlation Heatmap")
plt.show()

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:,1]

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:,1]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

models = {
    "Logistic Regression": (lr_pred, lr_prob),
    "Decision Tree": (dt_pred, dt_prob),
    "Random Forest": (rf_pred, rf_prob)
}

plt.figure(figsize=(7,6))
colors = ["#4361EE", "#F72585", "#4CC9F0"]
for (name, (pred, prob)), color in zip(models.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=name, linewidth=2, color=color)

plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

for name, (pred, prob) in models.items():
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

importances = rf.feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feat_df,
    hue="Feature",
    palette="rocket",
    legend=False
)
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

print("\nBest model based on Accuracy, ROC-AUC and stability: Random Forest")

import joblib

joblib.dump(rf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
