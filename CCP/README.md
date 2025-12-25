Customer Churn Prediction - Streamlit Dashboard

This workspace contains a Jupyter notebook and a Streamlit dashboard that loads a saved model artifact (churn_model.joblib).

Files:
- customer-churn-prediction.ipynb
- app.py
- create_artifact.py
- train_xgb_artifact.py
- verify_artifact.py
- requirements.txt

Run instructions:
1) Create and activate a virtual environment.
2) Install dependencies: pip install -r requirements.txt
3) Create artifact (optional): python create_artifact.py  OR  python train_xgb_artifact.py
4) Run: streamlit run app.py

Outputs saved to workspace:
- shap_global.png
- shap_local_<index>.png
- coef_global.png
- coef_local_<index>.png
- top50_high_risk.csv (downloaded via app)

If you want these changes pushed to a remote git repo, provide the remote URL.
