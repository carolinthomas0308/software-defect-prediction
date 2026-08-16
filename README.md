# Software Defect Prediction

Code and notebooks used for the MSc dissertation experiments on class imbalance handling in Software Defect Prediction.

## Contents

### Notebooks
- `Phase_1_Logistic_Regression.ipynb` – initial Logistic Regression experiment.
- `Phase_2_Random_Forest.ipynb` – initial Random Forest experiment.
- `Phase_3_Final_SDP_Experiments.ipynb` – final experiment using Stratified 10-fold Cross-Validation across the five datasets.

### Python files
The `code` directory contains Python versions of the code from the corresponding notebooks.

## Experimental setup

The study evaluates Logistic Regression and Random Forest under four class-distribution conditions:

- Original
- Random Over-Sampling (ROS)
- Random Under-Sampling (RUS)
- SMOTE

The final experiment evaluates KC1, KC2, CM1, JM1 and PC1 using Accuracy, Precision, Recall, F1-score and ROC-AUC.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets are not redistributed in this repository. The experiments use the KC1, KC2, CM1, JM1 and PC1 benchmark datasets described in the dissertation.

