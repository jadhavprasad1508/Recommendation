import streamlit as st
import pandas as pd
import time
from recommender import generate_recommendations, generate_hybrid_recommendations

# ──────────────── 1. Load Data & Precompute Maps ────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("recommendation_dataset_60k_with_names.xlsx")
    df['list_price'] = pd.to_numeric(df['list_price'], errors='coerce')
    return df

df = load_data()

# Map product name → price for fast lookup
price_map = (
    df[['product_name','list_price']]
    .drop_duplicates()
    .set_index('product_name')['list_price']
    .to_dict()
)

customer_ids = df['customer_id'].unique()
min_price, max_price = min(price_map.values()), max(price_map.values())

# ──────────────── 2. Page Setup ────────────────
st.set_page_config(page_title="THE BEST AI Recommender", layout="wide")
st.title("🚀 THE BEST AI‑Powered Product Recommendation Engine")
st.markdown(
    "Filter, sort, and explore recommendations with confidence bars, live metrics, "
    "and mock cart buttons—all powered by advanced AI."
)

# ──────────────── 3. Sidebar Controls & Metrics Placeholders ────────────────
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
    avg_score   = st.empty()
    latency_ms  = st.empty()

    generate = st.button("Generate Recommendations")

# ──────────────── 4. Main Display ────────────────
if generate:
    start_time = time.time()

    # 4a) Fetch recommendations
    if mode == "Hybrid":
        result = generate_hybrid_recommendations(customer_id, top_n)
        history   = result["purchase_history"]
        recs      = result["recommendations"]
        explanation = result["explanation"]
    else:
        history, recs, explanation = generate_recommendations(
            customer_id, method=mode, top_n=top_n
        )

    # 4b) Apply price filter via price_map
    filtered = []
    for name, score in recs:
        price = price_map.get(name, None)
        if price is None or (price_range[0] <= price <= price_range[1]):
            filtered.append((name, score))
    recs = filtered

    # 4c) Sort
    if sort_by == "Score ⬇️":
        recs.sort(key=lambda x: x[1] or 0, reverse=True)
    else:
        recs.sort(key=lambda x: x[0])

    # 4d) Update Metrics
    recs_served.metric("Recs Served", len(recs))
    numeric_scores = [s for _,s in recs if s is not None]
    avg = (sum(numeric_scores)/len(numeric_scores)) if numeric_scores else 0
    avg_score.metric("Avg Confidence", f"{avg*100:.1f}%")
    latency = int((time.time() - start_time)*1000)
    latency_ms.metric("API Latency (ms)", latency)

    # 4e) Layout: Purchases on Left, Recs on Right
    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.subheader("🛍️ Your Last 3 Purchases")
        for prod in history[:3]:
            price = price_map.get(prod, 0)
            st.markdown(f"- **{prod}** — ₹{price:.2f}")

    with col2:
        st.subheader(f"✨ Top {len(recs)} Recommendations")
        for i, (prod, score) in enumerate(recs):
            c1, c2, c3 = st.columns([4,2,2], gap="small")
            price = price_map.get(prod, 0)
            c1.markdown(f"**{prod}** — ₹{price:.2f}")

            if score is not None:
                pct = max(0, min(int(score*100), 100))
                c2.progress(pct)
                c2.caption(f"{pct}%")
            else:
                c2.write("—")

            if c3.button("Add to Cart", key=f"{mode}_{i}_{prod}"):
                st.toast(f"✔️ Added {prod} to cart!")

    # 4f) Explanation
    st.markdown("---")
    st.subheader("💡 Why These Recommendations?")
    st.info(explanation)

else:
    st.info("Adjust filters & settings, then click **Generate Recommendations**.")
