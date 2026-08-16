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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
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

print("TN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)

!pip install imbalanced-learn

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

oversample_model = ImbPipeline([
    ("sampler", RandomOverSampler(random_state=42)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

oversample_model.fit(X_train, y_train)

y_pred = oversample_model.predict(X_test)
y_prob = oversample_model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1:", round(f1_score(y_test, y_pred), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob), 4))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("TN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)

undersample_model = ImbPipeline([
    ("sampler", RandomUnderSampler(random_state=42)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

undersample_model.fit(X_train, y_train)

y_pred = undersample_model.predict(X_test)
y_prob = undersample_model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1:", round(f1_score(y_test, y_pred), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob), 4))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("TN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)

smote_model = ImbPipeline([
    ("sampler", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

smote_model.fit(X_train, y_train)

y_pred = smote_model.predict(X_test)
y_prob = smote_model.predict_proba(X_test)[:,1]

print("Accuracy:", round(accuracy_score(y_test, y_pred),4))
print("Precision:", round(precision_score(y_test, y_pred),4))
print("Recall:", round(recall_score(y_test, y_pred),4))
print("F1:", round(f1_score(y_test, y_pred),4))
print("AUC:", round(roc_auc_score(y_test, y_prob),4))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("TN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)
