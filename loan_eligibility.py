import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("data/shg_loan_eligibility.csv")
df["previous_loan_repayment_ontime_pct"] = df["previous_loan_repayment_ontime_pct"].fillna(0)

# Separate features and target
X = df.drop("eligible_for_loan", axis=1)
y = df["eligible_for_loan"]

# Remove ID
X = X.drop("group_id", axis=1)

# Identify columns
categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns


# Preprocessing
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


# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


# Complete pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


# Train
pipeline.fit(X, y)


# Save complete pipeline
joblib.dump(
    pipeline,
    "shg_loan_model.joblib"
)

print("Model saved successfully!")
print("File: shg_loan_model.joblib")