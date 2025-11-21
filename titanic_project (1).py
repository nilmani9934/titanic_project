# ============================================================
# TITANIC SURVIVAL PREDICTION PROJECT - COMPLETE IMPLEMENTATION
# ============================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load Dataset
data = pd.read_csv("Titanic-Dataset.csv")
print("Dataset loaded successfully ✅")
print("Shape:", data.shape)
print(data.head())

# Step 3: Handle Missing Values
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

# Step 4: Feature Engineering
# Title extraction from Name
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
                                       'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Family size and Alone flag
data['FamilySize'] = data['SibSp'] + data['Parch']
data['IsAlone'] = (data['FamilySize'] == 0).astype(int)

# Drop unused columns
data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])
data['Title'] = le.fit_transform(data['Title'])

# Step 5: Feature Selection
X = data.drop('Survived', axis=1)
y = data['Survived']

# Step 6: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Model Training & Comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\nModel: {name}")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred))

# Step 9: Model Comparison Visualization
plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# Step 10: Choose Best Model (Random Forest)
best_model = max(results, key=results.get)
model = models[best_model]
print(f"\n✅ Best model selected: {best_model}")

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title(f"Confusion Matrix - {best_model}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 12: ROC–AUC Curve
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title(f"ROC Curve - {best_model}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Step 13: Feature Importance (for tree models)
if hasattr(model, "feature_importances_"):
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance.sort_values().plot(kind='barh', color='teal')
    plt.title("Feature Importance in Survival Prediction")
    plt.show()

# Step 14: Cross-Validation for Robustness
cv_score = cross_val_score(model, X_scaled, y, cv=5).mean()
print(f"\nCross-Validation Accuracy: {cv_score:.4f}")

# Step 15: Save Model for Deployment
joblib.dump(model, "titanic_model.pkl")
print("\nModel saved as titanic_model.pkl ✅")

# Step 16: Exploratory Visualizations
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=data)
plt.title("Survival Rate by Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title("Survival Rate by Passenger Class")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Title', y='Survived', data=data)
plt.title("Survival Rate by Passenger Title")
plt.show()

# Step 17 (Optional): Streamlit Web App Template
# Save this in a separate file app.py if you want to run it as a web app
"""
import streamlit as st
import joblib

st.title("🚢 Titanic Survival Prediction")

# Load model
model = joblib.load("titanic_model.pkl")

Pclass = st.selectbox("Passenger Class (1, 2, 3)", [1, 2, 3])
Sex = st.selectbox("Sex", ["Female", "Male"])
Age = st.slider("Age", 1, 80, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10)
Parch = st.number_input("Parents/Children Aboard", 0, 10)
Fare = st.number_input("Fare", 0.0, 500.0)
Embarked = st.selectbox("Embarked (0=C, 1=Q, 2=S)", [0,1,2])
Title = st.selectbox("Title (0-4)", [0,1,2,3,4])
FamilySize = SibSp + Parch
IsAlone = 1 if FamilySize == 0 else 0

# Scale manually (not shown here; use saved scaler if needed)
data = [[Pclass, 1 if Sex=="Male" else 0, Age, SibSp, Parch, Fare, Embarked, Title, FamilySize, IsAlone]]

if st.button("Predict"):
    pred = model.predict(data)[0]
    st.success("✅ Survived" if pred==1 else "❌ Did not survive")
"""
# ============================================================
print("\n🎯 Project completed successfully — ready for presentation!")
