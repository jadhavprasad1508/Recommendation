import streamlit as st
import pandas as pd
import time
from recommender import (
    generate_recommendations,
    generate_hybrid_recommendations,
    generate_recommendations_from_upload
)

# ──────────────────────────────────────────────
# 🔐 Login Gate with dataset selection
# ──────────────────────────────────────────────

def login():
    st.title("🔒 Login Portal")

    # Center the login form with smaller width
    login_col = st.columns([1, 1, 1])[1]

    with login_col:
        st.markdown("### Please log in to continue")
        username = st.text_input("User ID", key="user_id", max_chars=20)
        password = st.text_input("Password", type="password", key="password", max_chars=20)

        # Dataset selection dropdown
        st.markdown("### Select Dataset Source")
        dataset_choice = st.selectbox(
            "Choose your dataset option",
            ["Current Dataset", "Upload New Dataset"],
            key="dataset_choice"
        )

        uploaded_df = None
        if dataset_choice == "Upload New Dataset":
            st.info("📂 Please upload an Excel file containing your transaction data.")
            uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"], key="uploaded_dataset")

            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_excel(uploaded_file)
                    st.success("✅ File uploaded successfully!")
                    st.write("### Dataset Preview")
                    st.dataframe(uploaded_df.head())
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")

        if st.button("Login"):
            if username == "Admin" and password == "Vengro@2025":
                st.session_state.authenticated = True
                st.session_state.dataset_choice = dataset_choice
                st.session_state.uploaded_df = uploaded_df
                st.rerun()
            else:
                st.error("Invalid credentials.")

# ──────────────────────────────────────────────
# 🏠 App Entry Point
# ──────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# ──────────────────────────────────────────────
# ✅ Load Appropriate Dataset
# ──────────────────────────────────────────────

@st.cache_data
def load_default_data():
    df = pd.read_excel("recommendation_dataset_60k_with_names.xlsx")
    df['list_price'] = pd.to_numeric(df['list_price'], errors='coerce')
    return df

if st.session_state.get("dataset_choice") == "Upload New Dataset" and st.session_state.get("uploaded_df") is not None:
    df = st.session_state.uploaded_df
    using_uploaded_data = True
else:
    df = load_default_data()
    using_uploaded_data = False

# ──────────────────────────────────────────────
# 🧭 Main Dashboard
# ──────────────────────────────────────────────

st.set_page_config(page_title="AI Recommender", layout="wide")
st.title("🚀 Dynamic AI-Powered Product Recommendation System")

if using_uploaded_data:
    st.markdown("### Using your uploaded dataset 📊")
else:
    st.markdown("### Using the default recommendation dataset")

# Show basic insights for uploaded data
if using_uploaded_data:
    try:
        st.subheader("📈 Dataset Insights")
        st.write(f"**Unique Customers:** {df['customer_id'].nunique()}")
        st.write(f"**Unique Products:** {df['product_id'].nunique()}")
        st.write(f"**Repeated Purchases:** {(df.duplicated(['customer_id','product_id']).sum())}")
        st.write(f"**Total Transactions:** {len(df)}")
        if 'customer_city' in df.columns:
            city_counts = df['customer_city'].value_counts().head(5)
            st.bar_chart(city_counts)
    except Exception as e:
        st.warning(f"Could not generate insights: {e}")

# Sidebar Controls
with st.sidebar:
    st.header("🔍 Recommendation Filters")
    customer_ids = df['customer_id'].unique()
    customer_id = st.selectbox("Select Customer ID", customer_ids)

    mode = st.selectbox("Recommendation Mode", ["Item Type", "Customer Type", "Hybrid"])
    top_n = st.slider("Number of Recommendations", 1, 10, 5)
    generate = st.button("Generate Recommendations")

# Main Section
if generate:
    start_time = time.time()

    if using_uploaded_data:
        recs_data = generate_recommendations_from_upload(df, customer_id, mode, top_n)
    elif mode == "Hybrid":
        recs_data = generate_hybrid_recommendations(customer_id, top_n)
    else:
        names, recs, explanation = generate_recommendations(customer_id, method=mode, top_n=top_n)
        recs_data = {"purchase_history": names, "recommendations": recs, "explanation": explanation}

    # Display Recommendations
    st.subheader("🛒 Recommendations")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Recent Purchases")
        for prod, qty in recs_data["purchase_history"]:
            st.markdown(f"- **{prod}** — Quantity: {qty}")

    with col2:
        st.markdown("### Recommended Products")
        for prod, score in recs_data["recommendations"]:
            st.markdown(f"- **{prod}** — Confidence: {score:.2f}")

    st.markdown("---")
    st.subheader("💡 Explanation")
    st.info(recs_data["explanation"])
