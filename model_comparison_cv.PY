import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv(r"data/shg_loan_eligibility.csv")

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["eligible_for_loan"].value_counts())


# ==========================================
# 2. Separate X and y
# ==========================================

X = df.drop("eligible_for_loan", axis=1)
y = df["eligible_for_loan"]

# Remove ID column
X = X.drop("group_id", axis=1)


# ==========================================
# 3. Identify categorical/numerical columns
# ==========================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns


# ==========================================
# 4. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        ),
        (
            "num",
            "passthrough",
            numerical_cols
        )
    ]
)


# ==========================================
# 5. Random Forest
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", rf_model)
])


# ==========================================
# 6. XGBoost
# ==========================================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


xgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", xgb_model)
])


# ==========================================
# 7. Stratified 5-Fold
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 8. Evaluation metrics
# ==========================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "macro_f1": "f1_macro"
}


# ==========================================
# 9. Random Forest CV
# ==========================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf_scores = cross_validate(
    rf_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


print("Accuracy:",
      rf_scores["test_accuracy"].mean())

print("Precision:",
      rf_scores["test_precision"].mean())

print("Recall:",
      rf_scores["test_recall"].mean())

print("F1:",
      rf_scores["test_f1"].mean())

print("Macro F1:",
      rf_scores["test_macro_f1"].mean())


# ==========================================
# 10. XGBoost CV
# ==========================================

print("\n==============================")
print("XGBOOST")
print("==============================")

xgb_scores = cross_validate(
    xgb_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


print("Accuracy:",
      xgb_scores["test_accuracy"].mean())

print("Precision:",
      xgb_scores["test_precision"].mean())

print("Recall:",
      xgb_scores["test_recall"].mean())

print("F1:",
      xgb_scores["test_f1"].mean())

print("Macro F1:",
      xgb_scores["test_macro_f1"].mean())


# ==========================================
# 11. Comparison table
# ==========================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Macro F1"
    ],

    "Random Forest": [
        rf_scores["test_accuracy"].mean(),
        rf_scores["test_precision"].mean(),
        rf_scores["test_recall"].mean(),
        rf_scores["test_f1"].mean(),
        rf_scores["test_macro_f1"].mean()
    ],

    "XGBoost": [
        xgb_scores["test_accuracy"].mean(),
        xgb_scores["test_precision"].mean(),
        xgb_scores["test_recall"].mean(),
        xgb_scores["test_f1"].mean(),
        xgb_scores["test_macro_f1"].mean()
    ]
})


print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(results.to_string(index=False))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
import numpy as np


def evaluate_model(model, X, y, cv):

    class0_precision = []
    class0_recall = []
    class0_f1 = []

    class1_precision = []
    class1_recall = []
    class1_f1 = []

    for train_index, test_index in cv.split(X, y):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Class 0
        class0_precision.append(
            precision_score(
                y_test,
                y_pred,
                pos_label=0
            )
        )

        class0_recall.append(
            recall_score(
                y_test,
                y_pred,
                pos_label=0
            )
        )

        class0_f1.append(
            f1_score(
                y_test,
                y_pred,
                pos_label=0
            )
        )

        # Class 1
        class1_precision.append(
            precision_score(
                y_test,
                y_pred,
                pos_label=1
            )
        )

        class1_recall.append(
            recall_score(
                y_test,
                y_pred,
                pos_label=1
            )
        )

        class1_f1.append(
            f1_score(
                y_test,
                y_pred,
                pos_label=1
            )
        )

    return {
        "Class 0 Precision": np.mean(class0_precision),
        "Class 0 Recall": np.mean(class0_recall),
        "Class 0 F1": np.mean(class0_f1),

        "Class 1 Precision": np.mean(class1_precision),
        "Class 1 Recall": np.mean(class1_recall),
        "Class 1 F1": np.mean(class1_f1)
    }


rf_result = evaluate_model(
    rf_pipeline,
    X,
    y,
    cv
)

xgb_result = evaluate_model(
    xgb_pipeline,
    X,
    y,
    cv
)


print("\n==============================")
print("RANDOM FOREST CLASS-WISE")
print("==============================")

for key, value in rf_result.items():
    print(f"{key}: {value:.4f}")


print("\n==============================")
print("XGBOOST CLASS-WISE")
print("==============================")

for key, value in xgb_result.items():
    print(f"{key}: {value:.4f}")