
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/raw/ds_salaries.csv")

print("Original Shape:", df.shape)

# Display first five rows
print(df.head())
df = df.drop(
    columns=["salary", "salary_currency"],
    errors="ignore"
)
print("Missing values:")
print(df.isnull().sum())

# Remove rows with missing target
df = df.dropna(subset=["salary_in_usd"])

# Fill categorical missing values
categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

print("Missing values handled!")
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after cleaning:", df.shape)
X = df.drop("salary_in_usd", axis=1)

y = df["salary_in_usd"]

print("Features:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)
import os

os.makedirs("data/processed", exist_ok=True)

cleaned_df = X.copy()
cleaned_df["salary_in_usd"] = y

cleaned_df.to_csv(
    "data/processed/cleaned_salary.csv",
    index=False
)

print("Cleaned dataset saved successfully!")