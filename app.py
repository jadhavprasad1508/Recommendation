"""
app.py - Streamlit frontend.

Features:
- Login (Admin/Vengro@2025)
- Dataset selector: use current bundled dataset OR upload new dataset
- Validation + insights for uploaded dataset
- Generate recommendations using the Recommender class
- Explanations via OpenRouter if API key is available via environment or .streamlit/secrets.toml
"""

import streamlit as st
import pandas as pd
import os
import time

from recommender import Recommender, validate_dataframe

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="THE BEST AI Recommender", layout="wide")

# ---------------------------
# Login
# ---------------------------
def login_ui():
    st.header("Login")
    username = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    dataset_choice = st.selectbox("Dataset Source", ["Use Current Dataset", "Upload New Dataset"])
    upload_file = None
    if dataset_choice == "Upload New Dataset":
        upload_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        st.write("Minimum recommended columns: `customer_id`, `product_id` or `product_name`, `qty`, `list_price`, `txn_timestamp`.")
    if st.button("Login"):
        if username == "Admin" and password == "Vengro@2025":
            st.session_state["authenticated"] = True
            st.session_state["dataset_choice"] = dataset_choice
            if upload_file is not None:
                st.session_state["uploaded_file_bytes"] = upload_file.getvalue()
                st.session_state["uploaded_file_name"] = upload_file.name
            st.experimental_rerun()
        else:
            st.error("Invalid credentials.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_ui()
    st.stop()

# ---------------------------
# Load dataset (current or uploaded)
# ---------------------------
@st.cache_data
def load_bundled_dataset(path="recommendation_dataset_60k_with_names.xlsx"):
    df = pd.read_excel(path)
    return df

dataset_choice = st.session_state.get("dataset_choice", "Use Current Dataset")
df = None
uploaded = False

if dataset_choice == "Use Current Dataset":
    df = load_bundled_dataset()
else:
    # uploaded dataset handling
    uploaded_bytes = st.session_state.get("uploaded_file_bytes")
    uploaded_name = st.session_state.get("uploaded_file_name", None)
    if uploaded_bytes is None:
        st.warning("You selected 'Upload New Dataset' but didn't upload a file. Please upload a CSV or Excel file and re-click Login.")
        st.stop()
    # load it into pandas
    try:
        if uploaded_name.endswith(".csv"):
            df = pd.read_csv(pd.io.common.BytesIO(uploaded_bytes))
        else:
            df = pd.read_excel(pd.io.common.BytesIO(uploaded_bytes))
        uploaded = True
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

# Validate dataset
ok, msg = validate_dataframe(df)
if not ok:
    st.error(msg)
    st.stop()

# ---------------------------
# Create/instantiate recommender (lazy)
# ---------------------------
# try to get API key for explainability
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", None) if "secrets" in dir(st) else None

@st.cache_resource
def build_recommender(dataframe, api_key):
    return Recommender(dataframe, openrouter_api_key=api_key)

reco = build_recommender(df, OPENROUTER_API_KEY)

# ---------------------------
# Sidebar: controls
# ---------------------------
with st.sidebar:
    st.header("Recommendation Settings")
    customer_list = df["customer_id"].astype(str).unique().tolist()
    selected_customer = st.selectbox("Customer ID", customer_list)
    mode = st.selectbox("Recommendation Mode", ["Item Type", "Customer Type", "Hybrid"])
    top_n = st.slider("Number of Recommendations", 1, 10, 5)
    run_button = st.button("Generate Recommendations")
    st.markdown("---")
    st.subheader("Dataset Insights")
    if st.button("Show Insights"):
        insights = reco.get_insights()
        st.write(insights)

# ---------------------------
# Main area
# ---------------------------
st.title("THE BEST — Dynamic Recommender")

# show whether current vs uploaded
if uploaded:
    st.info(f"Using uploaded dataset: {uploaded_name}")
else:
    st.info("Using bundled dataset (internal)")

# quick top-level insights
ins = reco.get_insights()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Customers", ins["unique_customers"])
c2.metric("Unique Products", ins["unique_products"])
c3.metric("Total Purchases", ins["total_rows"])
c4.metric("Repeated Purchase Ratio", f"{ins['repeated_purchase_ratio']:.2f}")

# show top locations / demos if exist
if ins["top_locations"]:
    st.write("Top Locations:", ins["top_locations"])
if ins["demographics"]:
    st.write("Top Demographics:", ins["demographics"])

# purchase history (last 3)
st.subheader("Recent Purchases")
hist = reco.getPurchaseHistory(selected_customer, n=3)
for p in hist:
    st.write(f"- {p['product_name']} (Qty: {p['qty']}) — ₹{p['price']:.2f}")

# generate recommendations
if run_button:
    start_time = time.time()
    recs = reco.recommend(selected_customer, method=mode, top_n=top_n)
    elapsed = time.time() - start_time

    st.subheader("Recommendations")
    if len(recs) == 0:
        st.warning("No recommendations found for this customer (maybe new or no purchases).")
    else:
        # display recommendations with bars
        for name, score in recs:
            pct = int(max(0, min(100, score * 100))) if score is not None else 0
            cols = st.columns([4, 1])
            cols[0].write(f"**{name}**")
            cols[1].progress(pct)

    # explanation
    rec_names = [n for n, _ in recs]
    explanation = reco.explain(selected_customer, rec_names, method=mode)
    st.markdown("---")
    st.subheader("Why these recommendations?")
    st.info(explanation)

    # additional dataset-level insight after generation
    st.markdown("---")
    st.write(f"Generated in {elapsed:.2f}s")

else:
    st.info("Choose settings on the sidebar and click Generate Recommendations.")
