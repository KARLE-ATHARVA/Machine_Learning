# Practical 4 (Improved) — Comparison between Decision Tree and SVM

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

# 1️⃣ Create a slightly noisier dataset
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

# 3️⃣ Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------------------------------
# 🔹 Decision Tree
# ------------------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
cm_dt = confusion_matrix(y_test, y_pred_dt)
acc_dt = accuracy_score(y_test, y_pred_dt)

# ------------------------------------------------------
# 🔹 Support Vector Machine (SVM)
# ------------------------------------------------------
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
acc_svm = accuracy_score(y_test, y_pred_svm)

# ------------------------------------------------------
# 🔹 ROC Curves
# ------------------------------------------------------
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# ------------------------------------------------------
# 🔹 Plot ROC Comparison
# ------------------------------------------------------
plt.plot(fpr_dt, tpr_dt, color='green', label='Decision Tree (AUC = %0.2f)' % roc_auc_dt)
plt.plot(fpr_svm, tpr_svm, color='blue', label='SVM (AUC = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree vs SVM')
plt.legend()
plt.show()

# ------------------------------------------------------
# 🔹 Print Comparison Summary
# ------------------------------------------------------
print("Decision Tree Confusion Matrix:\n", cm_dt)
print("Decision Tree Accuracy:", acc_dt)
print("\nSVM Confusion Matrix:\n", cm_svm)
print("SVM Accuracy:", acc_svm)

print(f"""
Observations:
-------------
1. Decision Tree Accuracy : {acc_dt:.2f}
2. SVM Accuracy           : {acc_svm:.2f}
3. Decision Tree may overfit small, clean datasets and perform slightly worse on noisy data.
4. SVM maintains a smoother decision boundary and usually generalizes better.
5. ROC curve comparison shows which model separates the classes more effectively.
""")
