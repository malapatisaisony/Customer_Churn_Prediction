import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from matplotlib import pyplot as plt

st.set_page_config(layout="wide", page_title="Churn Risk Dashboard")

@st.cache_resource
def load_artifact(path="churn_model.joblib"):
    return joblib.load(path)

def load_data(uploaded_file):
    if uploaded_file is None:
        if os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv"):
            return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        return None
    return pd.read_csv(uploaded_file)

def preprocess(df):
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", ""), errors="coerce")
    return df

def find_estimator(model, module_prefixes=("xgboost",)):
    if hasattr(model, "named_steps"):
        for v in model.named_steps.values():
            mod = getattr(v.__class__, "__module__", "")
            for p in module_prefixes:
                if mod.startswith(p):
                    return v
    return model

st.title("Customer Churn Risk Dashboard")

with st.sidebar:
    st.header("Data & Model")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader")
    artifact_path = st.text_input("Artifact path", value="churn_model.joblib", key="artifact_path")
    save_imgs = st.checkbox("Save explainability PNGs to workspace", value=True, key="save_imgs")
    threshold = st.slider("Churn threshold", 0.0, 1.0, 0.5, 0.01, key="threshold")
    run_btn = st.button("Run predictions", key="run_btn")

try:
    art = load_artifact(artifact_path)
    model = art["model"] if isinstance(art, dict) and "model" in art else art
    model_name = art.get("model_name") if isinstance(art, dict) else str(getattr(model, "__class__", "model"))
    st.sidebar.success(f"Loaded artifact: {model_name}")
except Exception as e:
    st.sidebar.error(f"Failed to load artifact: {e}")
    model = None

df = load_data(uploaded)
if df is None:
    st.info("No data loaded — upload a CSV or place WA_Fn-UseC_-Telco-Customer-Churn.csv in the workspace.")
    st.stop()

st.subheader("Data sample")
st.dataframe(df.head())

if model is None:
    st.stop()

df_proc = preprocess(df)

with st.sidebar.expander("Filters"):
    filters = {}
    for col in ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "Partner", "Dependents"]:
        if col in df_proc.columns:
            vals = sorted(df_proc[col].dropna().unique().tolist())
            if vals:
                sel = st.multiselect(f"{col}", options=vals, default=vals, key=f"filter_{col}")
                filters[col] = sel

if run_btn:
    mask = pd.Series(True, index=df_proc.index)
    for k, v in filters.items():
        if v:
            mask &= df_proc[k].isin(v)
    X = df_proc.loc[mask].copy()
    features = X.drop(columns=[c for c in ["Churn"] if c in X.columns], errors="ignore")

    try:
        probs = model.predict_proba(features)[:, 1]
    except Exception:
        probs = model.predict_proba(features.values)[:, 1]
    X["churn_prob"] = probs

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows considered", f"{len(X)}")
    with c2:
        st.metric("Mean churn probability", f"{X['churn_prob'].mean():.3f}")
    with c3:
        st.metric("Predicted churners", f"{(X['churn_prob']>=threshold).sum()}")

    st.plotly_chart(px.histogram(X, x="churn_prob", nbins=40, title="Churn Probability Distribution"), use_container_width=True)

    seg = None
    for c in ["Contract", "InternetService", "PaymentMethod"]:
        if c in X.columns:
            seg = c
            break
    if seg:
        seg_df = X.groupby(seg)["churn_prob"].mean().reset_index()
        st.plotly_chart(px.bar(seg_df, x=seg, y="churn_prob", title=f"Mean churn by {seg}"), use_container_width=True)

    top50 = X.sort_values("churn_prob", ascending=False).head(50)
    st.subheader("Top 50 high-risk customers")
    st.dataframe(top50)
    st.download_button("Download top 50 CSV", top50.to_csv(index=False).encode("utf-8"), file_name="top50_high_risk.csv", mime="text/csv", key="download_top50")

    st.header("Explainability")
    model_name_l = str(model_name).lower()
    if "xgboost" in model_name_l or "xgb" in model_name_l:
        try:
            import shap
            from sklearn.preprocessing import LabelEncoder
            
            # Use a sample for SHAP computation
            sample_size = min(100, len(features))
            sample = features.sample(sample_size, random_state=42)
            
            # Encode categorical variables for SHAP
            sample_encoded = sample.copy()
            encoders = {}
            for col in sample_encoded.columns:
                if sample_encoded[col].dtype == 'object':
                    encoders[col] = LabelEncoder()
                    sample_encoded[col] = encoders[col].fit_transform(sample_encoded[col].astype(str))
            
            # Create a wrapper that handles both encoded and original data
            def model_predict_encoded(data_encoded):
                # Convert encoded data back to original format for the model
                data_original = pd.DataFrame(data_encoded, columns=sample_encoded.columns)
                data_decoded = data_original.copy()
                
                for col, encoder in encoders.items():
                    if col in data_decoded.columns:
                        # Decode back to original categorical values
                        data_decoded[col] = encoder.inverse_transform(data_decoded[col].astype(int))
                
                # Ensure column order matches original
                data_decoded = data_decoded[sample.columns]
                return model.predict_proba(data_decoded)[:, 1]
            
            # Create explainer using encoded data
            background = shap.sample(sample_encoded, min(50, len(sample_encoded)))
            explainer = shap.Explainer(model_predict_encoded, background)
            shap_vals = explainer(sample_encoded)
            
            # Extract SHAP values
            if hasattr(shap_vals, 'values'):
                sv = shap_vals.values
            else:
                sv = shap_vals
            
            mean_abs = np.abs(sv).mean(axis=0)
            feat_names = sample.columns.tolist()
            df_shap = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).head(20)
            st.plotly_chart(px.bar(df_shap, x="mean_abs_shap", y="feature", orientation="h", title="Top SHAP features"), use_container_width=True)
            
            if save_imgs:
                outp = os.path.join(os.getcwd(), "shap_global.png")
                try:
                    fig, ax = plt.subplots(figsize=(8, max(4, len(df_shap) * 0.3)))
                    ax.barh(df_shap['feature'][::-1], df_shap['mean_abs_shap'][::-1], color='C0')
                    ax.set_xlabel('mean |SHAP|')
                    ax.set_title('Top 20 features by mean |SHAP|')
                    plt.tight_layout()
                    fig.savefig(outp)
                    plt.close(fig)
                    st.write(f"Saved {outp}")
                except Exception as e:
                    st.warning(f"Could not save SHAP global image: {e}")

            idx = st.selectbox("Select index for local SHAP", options=features.index.tolist(), key="shap_local_idx")
            if idx is not None:
                row = features.loc[[idx]]
                # Encode the row
                row_encoded = row.copy()
                for col, encoder in encoders.items():
                    if col in row_encoded.columns:
                        try:
                            row_encoded[col] = encoder.transform(row_encoded[col].astype(str))
                        except:
                            # If value not seen during training, use the first class
                            row_encoded[col] = 0
                
                sv_local = explainer(row_encoded)
                if hasattr(sv_local, 'values'):
                    s_local = sv_local.values[0]
                else:
                    s_local = sv_local[0]
                df_local = pd.DataFrame({"feature": row.columns.tolist(), "shap_value": s_local}).sort_values("shap_value")
                st.plotly_chart(px.bar(df_local, x="shap_value", y="feature", orientation="h", title=f"Local SHAP for index {idx}"), use_container_width=True)
                
                if save_imgs:
                    outl = os.path.join(os.getcwd(), f"shap_local_{idx}.png")
                    try:
                        fig_local, ax_local = plt.subplots(figsize=(8, max(4, len(df_local) * 0.3)))
                        ax_local.barh(df_local['feature'][::-1], df_local['shap_value'][::-1], color='C1')
                        ax_local.set_xlabel('SHAP value')
                        ax_local.set_title(f'Local SHAP for index {idx}')
                        plt.tight_layout()
                        fig_local.savefig(outl)
                        plt.close(fig_local)
                        st.write(f"Saved {outl}")
                    except Exception as e:
                        st.warning(f"Could not save local SHAP image: {e}")
        except Exception as e:
            st.warning(f"SHAP explainability failed: {e}")
    else:
        try:
            est = find_estimator(model, ())
            coef = getattr(est, "coef_", None)
            if coef is None:
                st.info("No coefficients available for model.")
            else:
                coef_arr = coef[0] if coef.ndim > 1 else coef
                feat_names = None
                if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                    try:
                        feat_names = model.named_steps["preprocessor"].get_feature_names_out()
                    except Exception:
                        feat_names = None
                if feat_names is None:
                    feat_names = features.columns.tolist()
                df_coef = pd.DataFrame({"feature": feat_names, "coef": coef_arr})
                df_coef["abs_coef"] = df_coef["coef"].abs()
                df_top = df_coef.sort_values("abs_coef", ascending=False).head(20)
                st.plotly_chart(px.bar(df_top, x="abs_coef", y="feature", orientation="h", title="Top coefficients"), use_container_width=True)
                if save_imgs:
                    outc = os.path.join(os.getcwd(), "coef_global.png")
                    try:
                        figc, axc = plt.subplots(figsize=(8, max(4, len(df_top) * 0.3)))
                        axc.barh(df_top['feature'][::-1], df_top['abs_coef'][::-1], color='C2')
                        axc.set_xlabel('|coefficient|')
                        axc.set_title('Top 20 features by |coefficient|')
                        plt.tight_layout()
                        figc.savefig(outc)
                        plt.close(figc)
                        st.write(f"Saved {outc}")
                    except Exception as e:
                        st.warning(f"Could not save coefficient global image: {e}")

                idx = st.selectbox("Select index for local contributions", options=features.index.tolist(), key="coef_local_idx")
                if idx is not None:
                    row = features.loc[[idx]]
                    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                        try:
                            transformed = model.named_steps["preprocessor"].transform(row)
                            arr = transformed.toarray().ravel() if hasattr(transformed, "toarray") else transformed.ravel()
                        except Exception:
                            arr = row.values.ravel()
                    else:
                        arr = row.values.ravel()
                    contribs = arr * coef_arr
                    df_loc = pd.DataFrame({"feature": feat_names, "contrib": contribs}).sort_values("contrib")
                    st.plotly_chart(px.bar(df_loc.tail(10), x="contrib", y="feature", orientation="h", title=f"Top local positive contributions for {idx}"), use_container_width=True)
                    if save_imgs:
                        outl = os.path.join(os.getcwd(), f"coef_local_{idx}.png")
                        try:
                            figcl, axcl = plt.subplots(figsize=(8, max(3, min(20, df_loc.shape[0]) * 0.25)))
                            subset = df_loc.tail(10)
                            axcl.barh(subset['feature'][::-1], subset['contrib'][::-1], color='C3')
                            axcl.set_xlabel('contribution')
                            axcl.set_title(f'Local linear contributions for index {idx}')
                            plt.tight_layout()
                            figcl.savefig(outl)
                            plt.close(figcl)
                            st.write(f"Saved {outl}")
                        except Exception as e:
                            st.warning(f"Could not save local coefficient image: {e}")
        except Exception as e:
            st.warning(f"Coefficient explainability failed: {e}")

else:
    st.info("Use the sidebar and press 'Run predictions' to compute churn scores and explainability.")
