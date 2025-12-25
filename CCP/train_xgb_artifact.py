"""Train an XGBoost pipeline on the Telco CSV and save churn_model.joblib
This replaces any existing artifact with an XGBoost-based pipeline so the Streamlit app
can compute SHAP explanations.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import joblib
import os

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_PATH = "churn_model.joblib"

def load_data(path=CSV_PATH):
    return pd.read_csv(path)

def build_pipeline(df):
    X = df.copy()
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

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

    clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, random_state=42)
    pipe = Pipeline([("preproc", preproc), ("clf", clf)])
    return pipe, X, y

def train_and_save():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    df = load_data()
    pipe, X, y = build_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training XGBoost pipeline (may take a short while)...")
    pipe.fit(X_train, y_train)
    artifact = {"model": pipe, "model_name": "XGBoost"}
    joblib.dump(artifact, OUT_PATH)
    print(f"Saved XGBoost artifact to {OUT_PATH}")

if __name__ == "__main__":
    train_and_save()
