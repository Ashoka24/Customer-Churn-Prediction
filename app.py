from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customer_churn.csv"
OUT = ROOT / "outputs"

st.set_page_config(page_title="Customer Churn Analysis", layout="wide")
st.title("Customer Churn Analysis")
st.caption("Synthetic portfolio data only; results do not describe a real provider.")

df = pd.read_csv(DATA)
contract = st.selectbox("Contract type", ["All", *sorted(df["contract"].dropna().unique())])
filtered = df if contract == "All" else df[df["contract"] == contract]
a, b, c = st.columns(3)
a.metric("Customers in selection", f"{len(filtered):,}")
b.metric("Churn rate", f"{(filtered['churn'].eq('Yes').mean() if len(filtered) else 0):.1%}")
c.metric("Mean tenure", f"{(filtered['tenure_months'].mean() if len(filtered) else 0):.1f} months")

left, right = st.columns(2)
with left:
    st.subheader("Churn rate by contract")
    by_contract = df.assign(churned=df["churn"].eq("Yes").astype(int)).groupby("contract")["churned"].mean()
    st.bar_chart(by_contract)
with right:
    st.subheader("Churn rate by tenure")
    bands = pd.cut(df["tenure_months"], bins=[-1, 5, 11, 23, 47, 100], labels=["0–5", "6–11", "12–23", "24–47", "48+"])
    by_tenure = df.assign(tenure_band=bands, churned=df["churn"].eq("Yes").astype(int)).groupby("tenure_band", observed=False)["churned"].mean()
    st.bar_chart(by_tenure)

metrics_path = OUT / "model_metrics.csv"
if metrics_path.exists():
    st.subheader("Held-out model metrics")
    st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
    plot = OUT / "roc_curves.png"
    if plot.exists():
        st.image(str(plot), caption="ROC curves on the held-out test set")
else:
    st.info("Run python scripts/run_analysis.py to generate the model evaluation files.")
