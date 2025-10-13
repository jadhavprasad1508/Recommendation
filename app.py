import streamlit as st
import pandas as pd
import time
from recommender import (
    generate_recommendations,
    generate_hybrid_recommendations
)

# ──────────────────────────────────────────────
# 🔐 Login Gate
# ──────────────────────────────────────────────

def login():
    if st.session_state.get("authenticated") != True:
        username = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "Admin" and password == "Vengro@2025":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# ──────────────────────────────────────────────
# ✅ Logged In - Load Dashboard
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_excel("recommendation_dataset_60k_with_names.xlsx")
    df['list_price'] = pd.to_numeric(df['list_price'], errors='coerce')
    return df

df = load_data()

# Map product name → price
price_map = (
    df[['product_name','list_price']]
    .drop_duplicates()
    .set_index('product_name')['list_price']
    .to_dict()
)

customer_ids = df['customer_id'].unique()
min_price, max_price = min(price_map.values()), max(price_map.values())

# ─────────────── Page Setup ───────────────
st.set_page_config(page_title="THE BEST AI Recommender", layout="wide")
st.title("🚀 THE BEST AI-Powered Product Recommendation Engine")
st.markdown(
    "Filter, sort, and explore recommendations with confidence bars and insights—all powered by advanced AI."
)

# ─────────────── Sidebar ───────────────
with st.sidebar:
    st.header("Filters & Settings")

    customer_id = st.selectbox("Select Customer ID", customer_ids)

    price_range = st.slider(
        "Price range (₹)",
        min_value=float(min_price),
        max_value=float(max_price),
        value=(float(min_price), float(max_price)),
        format="%.2f"
    )

    mode = st.selectbox(
        "Recommendation Mode",
        ["Item Type", "Customer Type", "Hybrid"]
    )

    descriptions = {
        "Item Type": "Suggests products similar to what you've bought.",
        "Customer Type": "Recommends items popular among customers like you.",
        "Hybrid": "Combines similarity, peer behavior, recency, and price fit."
    }
    st.caption(descriptions[mode])

    top_n = st.slider("Number of Recommendations", 1, 10, 5)
    sort_by = st.selectbox("Sort by", ["Score ⬇️", "Alphabetical ⬆️"])

    st.markdown("---")
    st.subheader("Dashboard Metrics")

    recs_served = st.empty()
    avg_score = st.empty()

    generate = st.button("Generate Recommendations")

# ─────────────── Main Area ───────────────
if generate:
    start_time = time.time()

    if mode == "Hybrid":
        result = generate_hybrid_recommendations(customer_id, top_n)
        history = result["purchase_history"]
        recs = result["recommendations"]
        explanation = result["explanation"]
    else:
        history, recs, explanation = generate_recommendations(
            customer_id, method=mode, top_n=top_n
        )

    # Apply price filter
    filtered = []
    for name, score in recs:
        price = price_map.get(name, None)
        if price is None or (price_range[0] <= price <= price_range[1]):
            filtered.append((name, score))
    recs = filtered

    # Sort
    if sort_by == "Score ⬇️":
        recs.sort(key=lambda x: x[1] or 0, reverse=True)
    else:
        recs.sort(key=lambda x: x[0])

    # Update Metrics
    recs_served.metric("Recs Served", len(recs))
    numeric_scores = [s for _, s in recs if s is not None]
    avg = (sum(numeric_scores)/len(numeric_scores)) if numeric_scores else 0
    avg_score.metric("Avg Confidence", f"{avg*100:.1f}%")

    # Columns: Purchases | Recommendations
    col1, col2 = st.columns([1,1], gap="large")

    # 🛍️ Show Purchase History with Quantity
    with col1:
        st.subheader("🛍️ Your Last 3 Purchases")
        recent_purchases = df[df['customer_id'] == customer_id].sort_values('txn_timestamp', ascending=False).head(3)
        for _, row in recent_purchases.iterrows():
            prod = row['product_name']
            qty = row['qty']
            price = row['list_price']
            st.markdown(f"- **{prod}** (Qty: {qty}) — ₹{price:.2f}")

    # ✨ Recommendations (without Add to Cart)
    with col2:
        st.subheader(f"✨ Top {len(recs)} Recommendations")
        for i, (prod, score) in enumerate(recs):
            c1, c2 = st.columns([4,2], gap="small")
            price = price_map.get(prod, 0)
            c1.markdown(f"**{prod}** — ₹{price:.2f}")
            if score is not None:
                pct = max(0, min(int(score*100), 100))
                c2.progress(pct)
                c2.caption(f"{pct}%")
            else:
                c2.write("—")

    # 💡 Explanation
    st.markdown("---")
    st.subheader("💡 Why These Recommendations?")
    st.info(explanation)

else:
    st.info("Adjust filters & settings, then click **Generate Recommendations**.")
