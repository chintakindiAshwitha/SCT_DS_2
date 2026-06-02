import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Basic Information
print("Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head())

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Remove Cabin column because it contains many missing values
df.drop('Cabin', axis=1, inplace=True)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# --------------------------
# EDA VISUALIZATIONS
# --------------------------

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.savefig("1_survival_count.png")
plt.close()

# 2. Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.savefig("2_gender_distribution.png")
plt.close()

# 3. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.savefig("3_survival_by_gender.png")
plt.close()

# 4. Passenger Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Distribution')
plt.savefig("4_passenger_class.png")
plt.close()

# 5. Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.savefig("5_age_distribution.png")
plt.close()

# 6. Correlation Heatmap
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.savefig("6_correlation_heatmap.png")
plt.close()

print("All charts generated successfully!")