# SHG Loan Eligibility Prediction API

An end-to-end Machine Learning project that predicts **loan eligibility** using customer/application information. The trained model is deployed as a **FastAPI REST API** and packaged using **Docker**.

The project covers the complete machine learning lifecycle:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Missing-value handling
* Categorical feature encoding
* Feature engineering
* Class imbalance analysis
* SMOTE
* Train/test splitting
* Stratified Cross-Validation
* Multiple ML model comparison
* Hyperparameter tuning
* Model evaluation
* XGBoost model training
* Model serialization
* FastAPI deployment
* Docker containerization
* REST API prediction with probabilities

---

## Project Architecture

```text
Dataset
   |
   v
Data Cleaning
   |
   v
EDA & Data Analysis
   |
   v
Feature Engineering
   |
   v
Categorical Encoding
   |
   v
Class Imbalance Check
   |
   +---- SMOTE
   |
   v
Train/Test Split
   |
   v
Stratified Cross-Validation
   |
   +-----------------------------+
   |                             |
   v                             v
Logistic Regression       Random Forest
   |                             |
   +-------------+---------------+
                 |
                 v
              XGBoost
                 |
                 v
        Hyperparameter Tuning
                 |
                 v
          Model Evaluation
                 |
                 v
       Save Trained Model
                 |
                 v
             FastAPI
                 |
                 v
              Docker
                 |
                 v
          REST API Endpoint
```

---

## Technologies Used

### Programming

* Python 3.11
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### Data Processing

* StandardScaler
* OneHotEncoder
* Feature Engineering
* Missing-value handling
* SMOTE

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* Cross-Validation

### Deployment

* FastAPI
* Uvicorn
* Docker
* Joblib

---

# 1. Problem Statement

Loan eligibility prediction is a binary classification problem.

The objective is to predict whether an applicant is:

```text
0 → Not Eligible
1 → Eligible
```

The model uses customer/application attributes such as:

* Gender
* Senior citizen status
* Partner
* Dependents
* Tenure
* Phone service
* Internet service
* Online security
* Online backup
* Device protection
* Technical support
* Streaming services
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

The final trained model is exposed through a REST API.

---

# 2. Exploratory Data Analysis

Before training the model, the dataset was analyzed to understand:

* Dataset shape
* Data types
* Missing values
* Duplicate records
* Numerical feature distributions
* Categorical feature distributions
* Target-class distribution
* Relationships between important features and the target

Example:

```python
df.shape
```

```python
df.info()
```

```python
df.describe()
```

Check missing values:

```python
df.isnull().sum()
```

Check duplicate records:

```python
df.duplicated().sum()
```

Target distribution:

```python
df["target"].value_counts()
```

Target percentage:

```python
df["target"].value_counts(normalize=True) * 100
```

---

# 3. Data Preprocessing

The following preprocessing steps were performed:

### Missing Values

Missing values were identified using:

```python
df.isnull().sum()
```

Numerical and categorical columns were handled according to their data type.

### Categorical Features

Categorical features were converted into numerical representations using encoding techniques such as One-Hot Encoding.

Example:

```python
from sklearn.preprocessing import OneHotEncoder
```

### Numerical Features

Numerical features were scaled where required.

Example:

```python
from sklearn.preprocessing import StandardScaler
```

---

# 4. Feature Engineering

Feature engineering was performed to make the input data suitable for machine learning.

Examples include:

* Converting categorical variables into numerical features
* Converting numeric columns to appropriate data types
* Handling missing/blank values
* Creating meaningful derived features where required
* Removing unnecessary columns
* Maintaining consistent feature columns between training and prediction

The same preprocessing logic must be applied during API prediction.

---

# 5. Handling Class Imbalance

Class imbalance was checked before model training.

For example:

```python
df["target"].value_counts()
```

If one class contains significantly more samples than the other, the model may become biased toward the majority class.

Therefore, class distribution was analyzed before selecting the final model.

---

# 6. SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** was used to address class imbalance.

SMOTE creates synthetic examples of the minority class instead of simply duplicating existing records.

Example:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

### Important

SMOTE should be applied **only to the training data**.

It should not be applied before splitting the dataset because that can cause data leakage.

Correct workflow:

```text
Original Dataset
       |
       v
Train/Test Split
       |
       +-------- Test Data
       |
       v
Training Data
       |
       v
SMOTE
       |
       v
Balanced Training Data
```

The test set remains untouched so that evaluation represents unseen data.

---

# 7. Train/Test Split

The dataset was divided into training and testing sets.

A stratified split was used to preserve the target-class distribution.

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

# 8. Cross-Validation

Stratified K-Fold Cross-Validation was used to obtain a more reliable estimate of model performance.

Example:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Cross-validation evaluates the model across multiple train/validation splits instead of depending on a single split.

### Why Stratified K-Fold?

For classification problems, stratification helps maintain a similar class distribution in each fold.

```text
Fold 1 → Train + Validation
Fold 2 → Train + Validation
Fold 3 → Train + Validation
Fold 4 → Train + Validation
Fold 5 → Train + Validation
```

The average validation performance is then used to compare models.

---

# 9. Machine Learning Models

Multiple classification algorithms were evaluated.

### Logistic Regression

Used as a baseline classification model.

```python
from sklearn.linear_model import LogisticRegression
```

Advantages:

* Simple
* Fast
* Easy to interpret
* Good baseline model

---

### Decision Tree

A tree-based classification algorithm.

Advantages:

* Easy to understand
* Handles non-linear relationships
* Does not require feature scaling

---

### Random Forest

An ensemble of multiple decision trees.

```python
from sklearn.ensemble import RandomForestClassifier
```

Advantages:

* Handles non-linear relationships
* More robust than a single decision tree
* Can capture feature interactions
* Provides feature importance

---

### Gradient Boosting

Gradient Boosting builds models sequentially, where each new model attempts to improve the errors of previous models.

---

### XGBoost

XGBoost was evaluated as a powerful gradient-boosting algorithm.

```python
from xgboost import XGBClassifier
```

XGBoost is useful for structured/tabular datasets and can capture complex non-linear relationships.

---

# 10. Model Evaluation

The models were evaluated using multiple metrics instead of relying only on accuracy.

### Accuracy

Measures the percentage of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive cases were actually positive.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual positive cases were correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1 Score

The harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

---

# 11. Confusion Matrix

The confusion matrix was used to understand classification errors.

```text
                    Predicted
                 0          1

Actual 0        TN         FP

Actual 1        FN         TP
```

Where:

* TN = True Negative
* TP = True Positive
* FP = False Positive
* FN = False Negative

Example:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

---

# 12. Model Selection

The models were compared using:

* Cross-validation performance
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

The final model was selected based on the project's classification requirements and validation results rather than relying on a single metric.

For this implementation, **XGBoost** was used for the deployed API.

---

# 13. Hyperparameter Tuning

Hyperparameter tuning was performed to improve model performance.

Examples of XGBoost hyperparameters include:

```python
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
```

Search techniques such as:

```python
GridSearchCV
```

or:

```python
RandomizedSearchCV
```

can be used to identify suitable parameter combinations.

Example:

```python
from sklearn.model_selection import RandomizedSearchCV
```

Cross-validation is used during tuning to reduce dependence on a single validation split.

---

# 14. Model Saving

After training, the final model was serialized using Joblib.

```python
import joblib

joblib.dump(model, "loan_eligibility_model.joblib")
```

The saved model is loaded by the FastAPI application.

```python
model = joblib.load("loan_eligibility_model.joblib")
```

---

# 15. FastAPI Deployment

The trained model was exposed through a FastAPI REST API.

Example endpoint:

```text
POST /predict
```

The API accepts customer information as JSON.

Example request:

```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "No",
  "dependents": "No",
  "tenure": 2,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 85.5,
  "total_charges": 171.0
}
```

---

# 16. API Response

The API returns the prediction and class probabilities.

Example:

```json
{
  "prediction": 1,
  "result": "Eligible",
  "probability_not_eligible": 0.0546,
  "probability_eligible": 0.9454
}
```

This means:

```text
Prediction                → 1
Result                    → Eligible
Probability Not Eligible  → 5.46%
Probability Eligible      → 94.54%
```

---

# 17. Docker Deployment

The FastAPI application was containerized using Docker.

Build the Docker image:

```bash
docker build -t shg-loan-api .
```

Run the container:

```bash
docker run -d \
  --name shg-loan-api \
  -p 8000:8000 \
  shg-loan-api
```

Check running containers:

```bash
docker ps
```

Check logs:

```bash
docker logs shg-loan-api
```

---

# 18. API Documentation

Once the Docker container is running, FastAPI provides interactive Swagger documentation.

Open:

```text
http://localhost:8000/docs
```

The `/docs` page can be used to:

1. Select `POST /predict`
2. Click **Try it out**
3. Enter JSON input
4. Click **Execute**
5. View the prediction and probabilities

---

# 19. Health Check

The API also provides a health endpoint.

Example:

```text
GET /health
```

Expected response:

```json
{
  "status": "healthy"
}
```

This can be used to verify that the API container is running correctly.

---

# 20. Project Structure

```text
shg-loan-eligibility-api/
│
├── app/
│   └── main.py
│
├── model/
│   └── loan_eligibility_model.joblib
│
├── sample_request.json
│
├── requirements.txt
│
├── Dockerfile
│
├── .dockerignore
│
├── .gitignore
│
└── README.md
```

---

# 21. Installation Without Docker

Create a virtual environment:

```bash
python -m venv my_env
```

Activate it on Windows:

```bash
my_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run FastAPI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

---

# 22. Docker Workflow

The complete deployment workflow is:

```text
Train ML Model
      ↓
Evaluate Model
      ↓
Save Model
      ↓
Create FastAPI Application
      ↓
Create Dockerfile
      ↓
Build Docker Image
      ↓
Run Docker Container
      ↓
Test REST API
      ↓
Return Prediction
```

---

# 23. Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Supervised Learning
* Binary Classification
* Exploratory Data Analysis
* Data Cleaning
* Missing Value Handling
* Feature Engineering
* Categorical Encoding
* Feature Scaling
* Class Imbalance
* SMOTE
* Train/Test Split
* Stratified Sampling
* Stratified K-Fold Cross-Validation
* Model Comparison
* Hyperparameter Tuning
* Confusion Matrix
* Precision
* Recall
* F1-score
* ROC-AUC
* XGBoost
* Model Serialization
* REST API
* FastAPI
* Docker
* ML Model Deployment

---

# 24. Important ML Practices

### Avoiding Data Leakage

SMOTE is applied only to the training data.

The test dataset is kept separate and untouched.

### Consistent Preprocessing

The preprocessing used during prediction must match the preprocessing used during model training.

### Cross-Validation

Stratified cross-validation is used to obtain a more reliable estimate of model performance.

### Multiple Evaluation Metrics

Accuracy alone is not sufficient for evaluating an imbalanced classification problem. Precision, recall, F1-score, confusion matrix, and ROC-AUC are also considered.

---

# 25. Future Improvements

Possible future improvements include:

* Add CI/CD pipeline
* Deploy API to AWS/Azure
* Add model versioning
* Add MLflow experiment tracking
* Add automated model monitoring
* Add authentication to API
* Add logging and monitoring
* Add automated retraining pipeline
* Add frontend dashboard
* Add prediction history database
* Add unit and integration tests

---

# 26. Author

**Gomathi Selvakumar**

Android Developer transitioning into **AI/ML and Generative AI**, with experience in mobile application development, machine learning, deep learning, and ML model deployment.

---

## Disclaimer

This project is developed for educational and portfolio purposes. Model predictions should not be treated as the sole basis for real-world lending or financial decisions.
