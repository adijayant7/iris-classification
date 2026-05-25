# 🌸 Iris Species Classification

## Overview
End-to-end ML pipeline to classify Iris flowers into three species
(**setosa**, **versicolor**, **virginica**) using sepal and petal measurements.

---

## Repository Structure
```
iris-classification/
├── Iris_Classification_Notebook.ipynb   # Full notebook (EDA + training + results)
├── iris_best_model.joblib               # Saved best model bundle
├── inference.py                         # Standalone inference script
├── IRIS.csv                             # Dataset (150 rows × 5 cols)
├── README.md                            # This file
└── plots/
    ├── 01_class_dist.png
    ├── 02_violin.png
    ├── 03_pairplot.png
    ├── 04_corr.png
    ├── 05_confusion.png
    ├── 06_comparison.png
    └── 07_tree.png
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Run inference (command-line)
```bash
# Demo sample (auto-predicts Iris-setosa)
python inference.py

# Custom input: sepal_length sepal_width petal_length petal_width
python inference.py 6.3 3.3 6.0 2.5
```

### 3. Python API
```python
from inference import predict

result = predict(
    sepal_length=5.1,
    sepal_width=3.5,
    petal_length=1.4,
    petal_width=0.2
)
print(result["predicted_species"])      # → Iris-setosa
print(result["probabilities"])          # → {'Iris-setosa': 1.0, ...}
```

### 4. Load model directly
```python
import joblib, pandas as pd

bundle  = joblib.load("iris_best_model.joblib")
clf     = bundle["model"]          # KNeighborsClassifier
scaler  = bundle["scaler"]         # StandardScaler
le      = bundle["label_encoder"]  # LabelEncoder (maps int→species)

sample  = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                        columns=bundle["feature_names"])
scaled  = scaler.transform(sample)
pred    = clf.predict(scaled)
species = le.inverse_transform(pred)[0]   # → 'Iris-setosa'
```

---

## Model Performance Summary

| Model               | Test Accuracy | CV Accuracy (5-fold) |
|---------------------|:------------:|:--------------------:|
| K-Nearest Neighbors | **93.3 %**   | **95.8 % ± 2.6 %**   |
| Logistic Regression | 93.3 %       | 95.8 % ± 2.6 %       |
| Decision Tree       | 90.0 %       | 94.2 % ± 2.0 %       |

**Best model saved:** K-Nearest Neighbours (k=5)
> KNN and Logistic Regression tied on both test and CV accuracy;
> KNN was selected as it yielded perfect Setosa precision with no false positives.

---

## Dataset
- **Source:** [Kaggle – Iris Classification Dataset](https://www.kaggle.com/datasets/bhanupratapbiswas/iris-classification-dataset)
- 150 samples · 3 balanced classes · 4 numerical features · 0 missing values

---

## Author
InternSpark Internship Task Submission
