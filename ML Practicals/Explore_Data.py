# Practical 1: Exploratory Data Analysis (EDA)
# Mini Project: Student Performance Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Create custom dataset
data = {
    'Student': ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'],
    'Hours_Studied': [2,3,4,5,6,7,8,9,10,11],
    'Attendance': [60,65,70,75,80,85,90,95,98,99],
    'Sleep_Hours': [6,7,6,7,7,8,7,8,7,8],
    'Marks': [50,55,60,65,70,80,85,90,95,97]
}

df = pd.DataFrame(data)

# 2️⃣ Basic dataset checks
print("Shape of Dataset:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())

# 3️⃣ Check for missing or duplicate values
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Entries:", df.duplicated().sum())

# 4️⃣ Correlation analysis (only numeric columns)
numeric_df = df.select_dtypes(include=['int64', 'float64'])
print("\nCorrelation Matrix:\n", numeric_df.corr())

# 5️⃣ Visualization

# Histogram - Distribution of Marks
plt.hist(df['Marks'], bins=5, edgecolor='black')
plt.title('Distribution of Marks')
plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot - Hours vs Marks
plt.scatter(df['Hours_Studied'], df['Marks'], color='blue')
plt.title('Hours Studied vs Marks')
plt.xlabel('Hours Studied')
plt.ylabel('Marks')
plt.show()

# Correlation Heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 6️⃣ Observations
print("""
Observations:
1. Marks increase as Hours_Studied and Attendance increase.
2. Sleep_Hours has a moderate positive relation with Marks.
3. No missing or duplicate values found.
4. Hours_Studied has the strongest correlation with Marks.
5. Dataset is clean and ready for further modeling.
""")
