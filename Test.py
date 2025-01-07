import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data_path = "C:/Users/jhasa/Downloads/Thyroid_Diff (1).csv"
data = pd.read_csv(data_path)

# Display basic dataset info (EDA)
print("Dataset head:")
print(data.head())
print("\nDataset info:")
print(data.info())

# Encode categorical columns
categorical_cols = ['Gender', 'Thyroid Function', 'Pathology', 'Focality', 'Risk', 'Stage', 'Response']
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Separate features and target
X = data.drop(columns=['Recurred'])
y = data['Recurred']

# Check for missing values in features before dropping them
print("Missing values in each column:\n", X.isnull().sum())

# Convert all feature columns to numeric where possible
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Check the shape of X after preprocessing
print("Shape of X after preprocessing:", X.shape)

# Drop rows with any NaN values in features or target after conversion
X = X.dropna()
y = y.dropna()

# Ensure X isn't empty after dropping rows with missing values
if X.empty:
    raise ValueError("The feature set X is empty after preprocessing. Please check the data.")

# Check the data types of columns after encoding
print("Data types of columns:\n", X.dtypes)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now check if X_scaled is empty after scaling
if X_scaled.size == 0:
    raise ValueError("The feature set X is empty after scaling. Please check the data.")

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression()

# Track results
accuracies = []

# Perform cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), start=1):
    print(f"\nTraining fold {fold}...")

    # Split the data
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
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
bins = [15, 20, 30, 40, 50, 60, 70, 80, 90]
labels = ['15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)

# Plot Age Group Distribution
age_group_counts = data['Age_Group'].value_counts().sort_index()
age_group_counts.plot(kind='bar', color=('lightblue', '#ffd238'), edgecolor='black')
plt.title('Count of People in Each Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Gender Distribution
gender_counts = data['Gender'].value_counts()
gender_counts.plot(kind='bar', color=['#ff799f', '#76b5f8'], edgecolor='black')
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
thyroid_function_counts = data['Thyroid Function'].value_counts().sort_index()
thyroid_function_counts.plot(kind='bar', color=('#1a267b', '#ffd03c'), edgecolor='white')
plt.xticks(rotation=45, ha='right')
plt.title('Thyroid Function Distribution')
plt.tight_layout()
plt.show()

# --- 1. Summary Statistics ---
print("\nSummary Statistics:")
print(data.describe())

# --- 2. Categorical Data Distribution ---
print("\nFemale/Male Count:")
print(data['Gender'].value_counts())

print("\nThyroid Function Distribution:")
print(data['Thyroid Function'].value_counts())

# --- 3. Correlation Matrix ---
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
print("\nCorrelation Matrix:")
print(correlation)

# --- 4. Target Variable Distribution ---
recurred_counts = data['Recurred'].value_counts()
print("\nRecurrence:")
print(recurred_counts)
recurred_counts.plot(kind='bar', color=['#34a5cf', '#ffdf40'], edgecolor='black')
plt.title('Recurred Distribution')
plt.xlabel('Recurrence')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)

# Random Forest
y = data['Recurred'].map({'No': 0, 'Yes': 1})
X = data.drop(columns=['Recurred'])
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()
y = data['Recurred'].dropna()

print(X.shape)  # Check the shape of X
print(X.head())  # If X is a DataFrame
print(X)  # If X is a numpy array
X = X.dropna()  # For pandas DataFrame

scaler = StandardScaler()

try:
    # Assuming X has the appropriate features for scaling
    X_scaled = scaler.fit_transform(X)  # Scale the data
    print("\nData scaled successfully")
except Exception as e:
    print(f"Error during scaling: {e}")

# Convert y_pred to numeric, handling non-numeric values (coerce turns invalid values to NaN)
try:
    y_pred = pd.to_numeric(y_pred, errors='coerce')  # This ensures y_pred is numeric
except Exception as e:
    print(f"Error converting y_pred to numeric: {e}")
    exit()

# Check for any NaN values after conversion
if np.isnan(y_pred).any():
    print("Warning: y_pred contains NaN values after conversion.")
else:
    print("y_pred is numeric.")

# Now round the predictions and convert them to integers
# Round the predictions to binary and convert them to 'Yes'/'No'
y_pred_binary = np.round(y_pred).astype(int)  # Ensures that predictions are integers (0 or 1)

# Map the numeric predictions (0/1) to the actual labels 'No'/'Yes'
y_pred_labels = ['Yes' if pred == 1 else 'No' for pred in y_pred_binary]

# Map the true labels (y_test) to the same format as y_pred_labels ('Yes'/'No')
y_true_labels = ['Yes' if label == 1 else 'No' for label in y_test]

# Calculate the accuracy
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"Accuracy: {accuracy}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_true_labels, y_pred_labels)
print("\nClassification Report:")
print(class_report)

# Final evaluations after cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores, r2_scores, accuracies = [], [], []

# Separate features and target
X = data.drop(columns=['Recurred'])
y = data['Recurred']

# Convert all feature columns to numeric where possible
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Drop rows with any NaN values in features or target after conversion
X = X.dropna()
y = y.dropna()

# Ensure X isn't empty after dropping rows with missing values
if X.empty:
    raise ValueError("The feature set X is empty after preprocessing. Please check the data.")
# Ensure the target variable (y) is numeric
y = data['Recurred'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)  # Map 'Yes'/'No' to 1/0

# Cross-validation and model evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores, r2_scores, accuracies = [], [], []

for train_idx, test_idx in kf.split(X_scaled):
    # Split into train and test sets for this fold
    X_fold_train, X_fold_test = X_scaled[train_idx], X_scaled[test_idx]
    y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train the model
    model.fit(X_fold_train, y_fold_train)

    # Predict on the test set
    y_fold_pred = model.predict(X_fold_test)

    # Ensure predictions are numeric and binary
    y_fold_pred = np.round(y_fold_pred).astype(int)

    # Append scores for this fold
    mse_scores.append(mean_squared_error(y_fold_test, y_fold_pred))
    r2_scores.append(r2_score(y_fold_test, y_fold_pred))
    accuracies.append(accuracy_score(y_fold_test, y_fold_pred))

# Final evaluation metrics
print("\nCross-validation results:")
print(f"Mean Squared Errors: {mse_scores}")
print(f"R-squared Scores: {r2_scores}")
print(f"Accuracies: {accuracies}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ensure test data is scaled correctly
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model on the scaled training set
model.fit(X_train_scaled, y_train)

# Predict on the scaled test set
y_test_pred = model.predict(X_test_scaled)

# Convert predictions to binary labels if necessary
y_test_pred_binary = np.round(y_test_pred)

# Evaluation
print("\nTest Set Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_test_pred))
print("R-squared:", r2_score(y_test, y_test_pred))
print("Accuracy:", accuracy_score(y_test, y_test_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_test_pred_binary))
print("\nTest Set Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_test_pred))
print("R-squared:", r2_score(y_test, y_test_pred))
print("Accuracy:", accuracy_score(y_test, y_test_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_test_pred_binary))
