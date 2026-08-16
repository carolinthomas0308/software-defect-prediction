!pip install imbalanced-learn

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

print("All libraries imported successfully.")

import os

print("Files currently uploaded to Colab:")

for file_name in os.listdir("/content"):
    if file_name.endswith(".csv"):
        print(file_name)

# Load all datasets
datasets = {
    "KC1": pd.read_csv("/content/kc1.csv"),
    "KC2": pd.read_csv("/content/kc2.csv"),
    "PC1": pd.read_csv("/content/pc1.csv"),
    "CM1": pd.read_csv("/content/cm1.csv"),
    "JM1": pd.read_csv("/content/jm1.csv")
}

print("Datasets loaded successfully!")

for name, df in datasets.items():
    print(f"{name}: {df.shape}")

# Create a summary table for all datasets

summary = []

for name, df in datasets.items():

    # Identify the target column
    if "defects" in df.columns:
        target = "defects"
    else:
        target = "problems"

    # Count defective and non-defective modules
    class_counts = df[target].value_counts()

    summary.append({
        "Dataset": name,
        "Rows": df.shape[0],
        "Features": df.shape[1] - 1,
        "Target Column": target,
        "Defective": class_counts.iloc[1] if len(class_counts) > 1 else 0,
        "Non-Defective": class_counts.iloc[0],
        "Defect %": round(class_counts.iloc[1] / df.shape[0] * 100, 2) if len(class_counts) > 1 else 0
    })

summary_df = pd.DataFrame(summary)

summary_df

# Function to prepare each dataset

def prepare_dataset(df):
    """
    Separates features (X) and target (y) for any PROMISE dataset.
    Converts target labels to binary values (0 and 1).
    """

    # Identify the target column
    if "defects" in df.columns:
        target = "defects"
    else:
        target = "problems"

    # Features
    X = df.drop(columns=[target])

    # Target
    y = df[target]

    # Convert target labels to binary
    if y.dtype == "object":
        y = y.map({"no": 0, "yes": 1})
    else:
        y = y.astype(int)

    return X, y

# Test the function on all datasets

for name, df in datasets.items():
    X, y = prepare_dataset(df)

    print(f"\n{name}")
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print("Class counts:")
    print(y.value_counts())

# Stratified 10-Fold Cross Validation

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

print(cv)

# Evaluation metrics

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print(scoring)

# Logistic Regression pipeline for the original data

lr_original = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])

print(lr_original)

# Function to evaluate one model on one dataset

def evaluate_model(dataset_name, X, y, model, model_name, sampling_method):
    scores = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Sampling": sampling_method,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Accuracy SD": scores["test_accuracy"].std(),
        "Precision Mean": scores["test_precision"].mean(),
        "Precision SD": scores["test_precision"].std(),
        "Recall Mean": scores["test_recall"].mean(),
        "Recall SD": scores["test_recall"].std(),
        "F1 Mean": scores["test_f1"].mean(),
        "F1 SD": scores["test_f1"].std(),
        "AUC Mean": scores["test_roc_auc"].mean(),
        "AUC SD": scores["test_roc_auc"].std()
    }

    return result

# Run original Logistic Regression on all five datasets

lr_results = []

for name, df in datasets.items():
    X, y = prepare_dataset(df)

    result = evaluate_model(
        dataset_name=name,
        X=X,
        y=y,
        model=lr_original,
        model_name="Logistic Regression",
        sampling_method="Original"
    )

    lr_results.append(result)
    print(f"{name} completed")

print("All original Logistic Regression experiments completed.")

lr_original_results_df = pd.DataFrame(lr_results)

lr_original_results_df.round(4)

# Save Logistic Regression (Original) results

lr_original_results_df.to_csv(
    "lr_original_results.csv",
    index=False
)

print("Logistic Regression (Original) results saved successfully.")

# Master list to store all experiment results

all_results = []

# Add the original Logistic Regression results
all_results.extend(lr_results)

print(f"Current experiments completed: {len(all_results)}")

# Logistic Regression with Random Over Sampling (ROS)

lr_ros = Pipeline([
    ("scaler", StandardScaler()),
    ("sampler", RandomOverSampler(random_state=42)),
    ("classifier", LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])

# Run Logistic Regression with ROS

for name, df in datasets.items():

    X, y = prepare_dataset(df)

    result = evaluate_model(
        dataset_name=name,
        X=X,
        y=y,
        model=lr_ros,
        model_name="Logistic Regression",
        sampling_method="ROS"
    )

    all_results.append(result)

    print(f"{name} completed")

print("ROS experiments completed.")

results_df = pd.DataFrame(all_results)

results_df.round(4)

# Function to run experiments on all datasets

def run_experiment(model, model_name, sampling_method):

    for name, df in datasets.items():

        X, y = prepare_dataset(df)

        result = evaluate_model(
            dataset_name=name,
            X=X,
            y=y,
            model=model,
            model_name=model_name,
            sampling_method=sampling_method
        )

        all_results.append(result)

        print(f"{name} completed")

    print(f"\n{sampling_method} experiments completed.\n")

lr_rus = Pipeline([
    ("scaler", StandardScaler()),
    ("sampler", RandomUnderSampler(random_state=42)),
    ("classifier", LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])

run_experiment(
    lr_rus,
    "Logistic Regression",
    "RUS"
)

lr_smote = Pipeline([
    ("scaler", StandardScaler()),
    ("sampler", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])

run_experiment(
    lr_smote,
    "Logistic Regression",
    "SMOTE"
)

results_df = pd.DataFrame(all_results)

lr_all_results = results_df[
    results_df["Model"] == "Logistic Regression"
].reset_index(drop=True)

lr_all_results.round(4)

lr_all_results.to_csv(
    "logistic_regression_all_results.csv",
    index=False
)

print("All Logistic Regression results saved successfully.")

# Random Forest - Original

rf_original = Pipeline([
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

run_experiment(
    rf_original,
    "Random Forest",
    "Original"
)

# Random Forest - ROS

rf_ros = Pipeline([
    ("sampler", RandomOverSampler(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

run_experiment(
    rf_ros,
    "Random Forest",
    "ROS"
)

# Random Forest - RUS

rf_rus = Pipeline([
    ("sampler", RandomUnderSampler(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

run_experiment(
    rf_rus,
    "Random Forest",
    "RUS"
)

# Random Forest - SMOTE

rf_smote = Pipeline([
    ("sampler", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

run_experiment(
    rf_smote,
    "Random Forest",
    "SMOTE"
)

results_df = pd.DataFrame(all_results)

results_df.round(4)

results_df.to_csv("Final_Results.csv", index=False)

print("Final results saved successfully!")

summary_df.to_csv("Dataset_Summary.csv", index=False)

summary_df

comparison = results_df.pivot_table(
    index=["Dataset", "Model"],
    columns="Sampling",
    values=["Accuracy Mean", "Precision Mean", "Recall Mean", "F1 Mean", "AUC Mean"]
)

comparison.round(4)

# Recreate the final results DataFrame
results_df = pd.DataFrame(all_results)

# Save separate CSV files
results_df.to_csv(
    "/content/Final_SDP_Results.csv",
    index=False
)

summary_df.to_csv(
    "/content/Dataset_Summary.csv",
    index=False
)

print("CSV files saved successfully.")

import os

print(os.listdir("/content"))

!pip install openpyxl

import os

print("Current working directory:", os.getcwd())
print("\nFiles in current directory:")
print(os.listdir())

plot_df = results_df.copy()

plot_df["Sampling"] = pd.Categorical(
    plot_df["Sampling"],
    categories=["Original", "ROS", "RUS", "SMOTE"],
    ordered=True
)

plot_df = plot_df.sort_values(
    by=["Dataset", "Model", "Sampling"]
)

plot_df.head()
