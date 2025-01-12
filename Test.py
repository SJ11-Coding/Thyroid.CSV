from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

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

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Handle missing values using SimpleImputer (fill with median for numeric, most_frequent for categorical)
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputers
X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

# Check if X is still empty
if X.empty:
    raise ValueError("The feature set X is empty after imputation. Please check the data.")

# Ensure the target variable (y) is numeric
y = y.map({'No': 0, 'Yes': 1})

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Handle missing values using SimpleImputer
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputers
if numeric_cols:  # Check if there are numeric columns
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

if categorical_cols:  # Check if there are categorical columns
    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

# Encode categorical columns (Label Encoding or OneHotEncoding)
if categorical_cols:
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Convert all data in X to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Check if any non-numeric values remain
if X.isnull().values.any():
    print("Warning: Missing or non-numeric data found after preprocessing!")
    print(X.isnull().sum())

# Drop any rows with NaN values (optional, depending on your approach)
X = X.dropna()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape of X after preprocessing:", X.shape)
print("First few rows of X:\n", X.head())


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
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Ensure train and test sets are not empty
if X_train.shape[0] == 0 or X_test.shape[0] == 0:
    raise ValueError("Train or test set is empty. Please check data splitting!")

# Ensure test data is scaled correctly
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model on the scaled training set
model.fit(X_train_scaled, y_train)

# Predict on the scaled test set
y_test_pred = model.predict(X_test_scaled)

# Ensure y_test_pred is numerical and convert to binary labels
if isinstance(y_test_pred[0], str):
    # In case predictions are strings, make sure they are numerical
    y_test_pred = np.array([0 if val == 'No' else 1 for val in y_test_pred])

# Ensure y_test is numeric (0 or 1)
y_test = y_test.map({'No': 0, 'Yes': 1})

# Ensure y_test_pred_binary is numeric (0 or 1)
y_test_pred_binary = np.round(y_test_pred).astype(int)

# Classification Evaluation
print("\nTest Set Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_test_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_test_pred_binary))


# Testing all Folds

# Define the top three risk factors based on domain knowledge
top_three_features = ['Age', 'Risk', 'T']  # Replace with actual top 3 features from your analysis

# Define target variable
target_column = 'Recurred'

# Split data for the two feature sets
X_top_three = data[top_three_features]
X_all = data.drop(columns=[target_column])

# Ensure target variable is numeric
y = data[target_column].map({'No': 0, 'Yes': 1})

# Initialize KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results
top_three_roc_aucs = []
all_features_roc_aucs = []


# Function to get the preprocessor
def get_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


# Perform cross-validation for both feature sets
for fold, (train_idx, test_idx) in enumerate(kf.split(X_all), start=1):
    print(f"\n--- Fold {fold} ---")

    # Train-test split for top three features
    X_train_top, X_test_top = X_top_three.iloc[train_idx], X_top_three.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train-test split for all features
    X_train_all, X_test_all = X_all.iloc[train_idx], X_all.iloc[test_idx]

    # Define pipelines
    preprocessor_top = get_preprocessor(X_train_top)
    preprocessor_all = get_preprocessor(X_train_all)

    pipeline_top = Pipeline(steps=[
        ('preprocessor', preprocessor_top),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    pipeline_all = Pipeline(steps=[
        ('preprocessor', preprocessor_all),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Train models
    pipeline_top.fit(X_train_top, y_train)
    pipeline_all.fit(X_train_all, y_train)

    # Predict probabilities
    y_pred_proba_top = pipeline_top.predict_proba(X_test_top)[:, 1]
    y_pred_proba_all = pipeline_all.predict_proba(X_test_all)[:, 1]

    # Calculate ROC-AUC
    roc_auc_top = roc_auc_score(y_test, y_pred_proba_top)
    roc_auc_all = roc_auc_score(y_test, y_pred_proba_all)

    # Append scores
    top_three_roc_aucs.append(roc_auc_top)
    all_features_roc_aucs.append(roc_auc_all)

    print(f"Fold {fold} ROC-AUC (Top Three): {roc_auc_top:.4f}")
    print(f"Fold {fold} ROC-AUC (All Features): {roc_auc_all:.4f}")

# Print average ROC-AUC scores
print("\n--- Cross-Validation Results ---")
print(f"Average ROC-AUC (Top Three Features): {np.mean(top_three_roc_aucs):.4f}")
print(f"Average ROC-AUC (All Features): {np.mean(all_features_roc_aucs):.4f}")
