# -------------------------------------------------------------
# Practical 6: Ensemble Learning - Bagging and Boosting
# Project: Student Performance Prediction
# -------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1️⃣ Create Dataset
data = {
    'Student': ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13'],
    'Hours_Studied': [2,3,4,5,6,7,8,9,10,11,4,6,8],
    'Attendance': [60,65,70,75,80,85,90,95,98,99,85,60,70],
    'Sleep_Hours': [6,7,6,7,7,8,7,8,7,8,6,7,8],
    'Marks': [50,55,60,65,70,80,85,90,95,97,58,59,75]
}

df = pd.DataFrame(data)
df['Result'] = df['Marks'].apply(lambda x: 1 if x >= 60 else 0)
print("Dataset:\n", df, "\n")

# 2️⃣ Features & Target
X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours']]
y = df['Result']

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------------------------------
# 🔹 Base Model: Decision Tree
# -------------------------------------------------------
base_model = DecisionTreeClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)

base_acc = accuracy_score(y_test, y_pred_base)
print("Base Decision Tree Accuracy:", base_acc)

# -------------------------------------------------------
# 🔹 Bagging Model
# -------------------------------------------------------
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,  # Number of trees
    random_state=42
)
bagging_model.fit(X_train, y_train)
y_pred_bag = bagging_model.predict(X_test)

bag_acc = accuracy_score(y_test, y_pred_bag)
print("Bagging Accuracy:", bag_acc)

# -------------------------------------------------------
# 🔹 Boosting Model (AdaBoost)
# -------------------------------------------------------
boost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=20,  # Number of weak learners
    learning_rate=0.5,
    random_state=42
)
boost_model.fit(X_train, y_train)
y_pred_boost = boost_model.predict(X_test)

boost_acc = accuracy_score(y_test, y_pred_boost)
print("Boosting (AdaBoost) Accuracy:", boost_acc)

# -------------------------------------------------------
# 🔹 Confusion Matrices
# -------------------------------------------------------
print("\nDecision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_base))
print("\nBagging Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bag))
print("\nBoosting Confusion Matrix:\n", confusion_matrix(y_test, y_pred_boost))

# -------------------------------------------------------
# 🔹 Classification Reports (with warning fix)
# -------------------------------------------------------
print("\nClassification Report (Boosting):\n", classification_report(y_test, y_pred_boost, zero_division=0))

# -------------------------------------------------------
# 🔹 Compare Accuracies Visually
# -------------------------------------------------------
models = ['Decision Tree', 'Bagging', 'Boosting']
accuracies = [base_acc, bag_acc, boost_acc]

plt.bar(models, accuracies, color=['red', 'green', 'blue'])
plt.title('Comparison of Ensemble Techniques')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# -------------------------------------------------------
# 🔹 Observations and Interpretation
# -------------------------------------------------------
print(f"""
Observations:
-------------
1. Base Decision Tree Accuracy : {base_acc:.2f}
2. Bagging Accuracy             : {bag_acc:.2f}
3. Boosting Accuracy            : {boost_acc:.2f}

Explanation:
------------
- All models achieved identical accuracy because of the small and slightly imbalanced dataset.
- Only one sample belonged to class ‘0’ in the test set, and none of the models predicted it correctly.
- The warning message indicates zero precision for class ‘0’ since no samples were predicted in that class.
- This demonstrates how data imbalance affects model evaluation.
- In larger datasets, Bagging and Boosting typically outperform a single Decision Tree by improving generalization and reducing overfitting.
""")
