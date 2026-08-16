import pandas as pd

!ls

import pandas as pd

df = pd.read_csv("kc1.csv")

print("Dataset shape:", df.shape)
df.head()

print(df.columns)

print(df["defects"].value_counts())

print(df.columns)

print(df["defects"].value_counts())

total = len(df)

non_defective = 1783
defective = 326

print("Non-defective %:", round(non_defective/total*100,2))
print("Defective %:", round(defective/total*100,2))

X = df.drop(columns=["defects"])
y = df["defects"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

baseline_model = Pipeline([
    ("classifier", RandomForestClassifier(random_state=42))
])

baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
y_prob = baseline_model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob), 4))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("Random Forest - Confusion Matrix")
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
print("True Positives (TP):", TP)

!pip install imbalanced-learn

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

rf_ros_model = ImbPipeline([
    ("sampler", RandomOverSampler(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_ros_model.fit(X_train, y_train)

y_pred_ros = rf_ros_model.predict(X_test)
y_prob_ros = rf_ros_model.predict_proba(X_test)[:, 1]

print("Random Forest with ROS")
print("Accuracy:", round(accuracy_score(y_test, y_pred_ros), 4))
print("Precision:", round(precision_score(y_test, y_pred_ros), 4))
print("Recall:", round(recall_score(y_test, y_pred_ros), 4))
print("F1 Score:", round(f1_score(y_test, y_pred_ros), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob_ros), 4))

cm_ros = confusion_matrix(y_test, y_pred_ros)

TN, FP, FN, TP = cm_ros.ravel()

print("\nRandom Forest with ROS - Confusion Matrix")
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
print("True Positives (TP):", TP)

rf_rus_model = ImbPipeline([
    ("sampler", RandomUnderSampler(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_rus_model.fit(X_train, y_train)

y_pred_rus = rf_rus_model.predict(X_test)
y_prob_rus = rf_rus_model.predict_proba(X_test)[:, 1]

print("Random Forest with RUS")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rus), 4))
print("Precision:", round(precision_score(y_test, y_pred_rus), 4))
print("Recall:", round(recall_score(y_test, y_pred_rus), 4))
print("F1 Score:", round(f1_score(y_test, y_pred_rus), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob_rus), 4))

cm_rus = confusion_matrix(y_test, y_pred_rus)

TN, FP, FN, TP = cm_rus.ravel()

print("\nRandom Forest with RUS - Confusion Matrix")
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
print("True Positives (TP):", TP)

rf_smote_model = ImbPipeline([
    ("sampler", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_smote_model.fit(X_train, y_train)

y_pred_smote = rf_smote_model.predict(X_test)
y_prob_smote = rf_smote_model.predict_proba(X_test)[:, 1]

print("Random Forest with SMOTE")
print("Accuracy:", round(accuracy_score(y_test, y_pred_smote), 4))
print("Precision:", round(precision_score(y_test, y_pred_smote), 4))
print("Recall:", round(recall_score(y_test, y_pred_smote), 4))
print("F1 Score:", round(f1_score(y_test, y_pred_smote), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob_smote), 4))

cm_smote = confusion_matrix(y_test, y_pred_smote)

TN, FP, FN, TP = cm_smote.ravel()

print("\nRandom Forest with SMOTE - Confusion Matrix")
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
print("True Positives (TP):", TP)
