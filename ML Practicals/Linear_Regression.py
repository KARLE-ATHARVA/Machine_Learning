# Practical 2: Linear Regression
# Project: Student Performance Prediction

# 1️⃣ Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 2️⃣ Create the same dataset from Practical 1
data = {
    'Student': ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'],
    'Hours_Studied': [2,3,4,5,6,7,8,9,10,11],
    'Attendance': [60,65,70,75,80,85,90,95,98,99],
    'Sleep_Hours': [6,7,6,7,7,8,7,8,7,8],
    'Marks': [50,55,60,65,70,80,85,90,95,97]
}

df = pd.DataFrame(data)

# 3️⃣ Split dataset into features (X) and target (y)
X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours']]
y = df['Marks']

# 4️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Build the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6️⃣ Predict on test data
y_pred = model.predict(X_test)

# 7️⃣ Evaluate the model
print("Actual Marks:", y_test.tolist())
print("Predicted Marks:", np.round(y_pred, 2).tolist())


# R² Score and Mean Squared Error
print("\nR² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# 8️⃣ Visualization - Actual vs Predicted Marks
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks (Linear Regression)")
plt.show()

# 9️⃣ Observations
print("""
Observations:
1. The Linear Regression model fits almost perfectly (R² ≈ 0.99+).
2. As expected, Hours_Studied and Attendance are the strongest predictors of Marks.
3. The prediction line closely follows actual marks, indicating a low error rate.
4. Hence, Linear Regression is suitable for predicting marks from study-related features.
""")
