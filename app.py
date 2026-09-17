from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(
    title="SHG Loan Eligibility API",
    description="API for predicting SHG loan eligibility",
    version="1.0"
)

# Load trained model
model = joblib.load("shg_loan_model.joblib")


@app.get("/")
def home():
    return {
        "message": "SHG Loan Eligibility API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
 

@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([data])

    # Remove group_id if it is sent
    if "group_id" in input_data.columns:
        input_data = input_data.drop("group_id", axis=1)

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0]

    result = "Eligible" if prediction == 1 else "Not Eligible"

    return {
        "prediction": int(prediction),
        "result": result,
        "probability_not_eligible": round(float(probability[0]), 4),
        "probability_eligible": round(float(probability[1]), 4)
    }