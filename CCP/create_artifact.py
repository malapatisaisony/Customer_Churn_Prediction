"""Create a simple LogisticRegression pipeline from the provided CSV
and save it as churn_model.joblib (artifact with model and model_name).
This is a helper to produce a runnable artifact if you don't already have one.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_PATH = "churn_model.joblib"

def load_data(path=CSV_PATH):
    return pd.read_csv(path)

def build_pipeline(df):
    # drop customerID and make TotalCharges numeric
    X = df.copy()
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    # target
    if "Churn" in X.columns:
        y = X["Churn"].map({"Yes":1, "No":0})
        X = X.drop(columns=["Churn"])
    else:
        raise RuntimeError("CSV must include Churn column for artifact creation")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    preproc = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)])

    pipe = Pipeline([("preproc", preproc), ("clf", LogisticRegression(max_iter=1000))])

    return pipe, X, y

def train_and_save():
    df = load_data()
    pipe, X, y = build_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    artifact = {"model": pipe, "model_name": "Logistic Regression"}
    joblib.dump(artifact, OUT_PATH)
    print(f"Saved artifact to {OUT_PATH}")

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH} — place your dataset in the workspace.")
    train_and_save()
