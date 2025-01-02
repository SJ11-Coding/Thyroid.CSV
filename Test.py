import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
# Ensure the file path is correct
data_path = "C:/Users/jhasa/Downloads/Thyroid_Diff (1).csv"

# Check if file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The dataset file was not found at the specified path: {data_path}")

# Data load
data = pd.read_csv(data_path)

# Check if there are missing values
if data.isnull().values.any():
    print("Warning: Dataset contains missing values. These will be dropped.")

# Drop rows with missing values
data = data.dropna()

# Display basic dataset info
print("Dataset head:")
print(data.head())
print("\nDataset info:")
data.info()

# Encode categorical columns (assuming some categorical columns are present)
categorical_cols = ['Gender', 'Thyroid Function', 'Pathology', 'Focality', 'Risk', 'Stage', 'Response']
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# One-hot encode 'T', 'N', 'M' columns if they exist
if all(col in data.columns for col in ['T', 'N', 'M']):
    data = pd.get_dummies(data, columns=['T', 'N', 'M'], drop_first=True)

# Separate features and target
X = data.drop(columns=['Recurred'])
y = data['Recurred']

# Convert all feature columns to numeric where possible (while keeping categorical columns as is)
for col in X.columns:
    if X[col].dtype == 'object':  # Handle categorical columns
        X[col] = LabelEncoder().fit_transform(X[col])

# Ensure that X contains only numeric values before scaling
print("Data types of X before scaling:")
print(X.dtypes)

# Drop rows with any NaN values in features or target after conversion
X = X.dropna()
y = y.dropna()

# Ensure X isn't empty
if X.empty:
    raise ValueError("The feature set X is empty after preprocessing. Please check the data.")

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- Cross-validation setup --- #

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression()

# Track results
accuracies = []

# Perform cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    print(f"\nTraining fold {fold}...")

    # Split the data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Print when each fold completes
    print(f"Fold {fold} complete.")

# Print when all folds have been completed
print("\nAll folds completed!")

# Analyze Age
print("Highest value in Age:", data['Age'].max())
print("Smallest value in Age:", data['Age'].min())

# Binning Age
# --- Bar Graphs --- #

# 1. Plot Age Group Distribution
bins = [15, 20, 30, 40, 50, 60, 70, 80, 90]
labels = ['15-20', '20-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)

# Plot Age Group Distribution
age_group_counts = data['Age_Group'].value_counts().sort_index()

# Plot the Age Group distribution bar graph
age_group_counts.plot(kind='bar', color=('lightblue', '#ffd238'), edgecolor='black')

# Customize the plot
plt.title('Count of People in Each Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Gender Distribution
gender_counts = data['Gender'].value_counts()

# 2. Plot Gender Distribution
gender_counts.plot(kind='bar', color=['#ff799f', '#76b5f8'], edgecolor='black')

# Customize the plot
plt.title('Count of Males and Females')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Convert binary categorical columns to numeric
binary_columns = ['Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Adenopathy', 'Recurred']
mapping = {'No': 0, 'Yes': 1}
for column in binary_columns:
    data[column + '_numeric'] = data[column].map(mapping).fillna(0).astype(int)

# Display numeric conversions
print("Converted binary columns:")
print(data[[column + '_numeric' for column in binary_columns]].head())

# Plot Thyroid Function Distribution
# 3. Plot Thyroid Function Distribution
thyroid_function_counts = data['Thyroid Function'].value_counts().sort_index()

# Plot Thyroid Function distribution bar graph
thyroid_function_counts.plot(kind='bar', color=('#1a267b', '#ffd03c'), edgecolor='white')

# Rotate x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right')  # Rotate labels by 45 degrees and adjust alignment
plt.title('Thyroid Function Distribution')
plt.tight_layout()
plt.show()

# EDA
# Data load
data = pd.read_csv(data_path)

# Display basic information
print("Dataset head:")
print(data.head())
print("\nDataset info:")
data.info()

# --- 1. Check for missing values ---
missing_values = data.isnull().sum()
print(f"\nMissing Values:\n{missing_values}")

# --- 2. Summary Statistics ---
print("\nSummary Statistics:")
print(data.describe())

# --- 3. Distribution of Numerical Features ---
# We will focus on numerical features such as Age
print("\nDistribution of 'Age' feature:")
age_min = data['Age'].min()
age_max = data['Age'].max()
age_mean = data['Age'].mean()
age_std = data['Age'].std()
print(f"Min: {age_min}, Max: {age_max}, Mean: {age_mean:.2f}, Standard Deviation: {age_std:.2f}")

# --- 4. Categorical Data Distribution ---
# Gender distribution
print("\nGender Distribution:")
gender_counts = data['Gender'].value_counts()
print(gender_counts)

# Thyroid Function distribution
print("\nThyroid Function Distribution:")
thyroid_function_counts = data['Thyroid Function'].value_counts()
print(thyroid_function_counts)

# --- 5. Correlation Matrix ---
# Only select numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation = numeric_data.corr()
print("\nCorrelation Matrix:")
print(correlation)

# --- 6. Outliers Detection ---
# Check for outliers in Age using a basic range method (min, max, 1.5*IQR rule)
q1 = data['Age'].quantile(0.25)
q3 = data['Age'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print("\nOutliers Detection for Age:")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# Detect outliers
age_outliers = data[(data['Age'] < lower_bound) | (data['Age'] > upper_bound)]
print(f"Number of outliers in Age: {len(age_outliers)}")

# --- 7. Target Variable Distribution ---
# Distribution of the target variable 'Recurred'
print("\nRecurred Distribution:")
recurred_counts = data['Recurred'].value_counts()
print(recurred_counts)