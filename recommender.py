import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

# ====== Configuration ======
OPENROUTER_API_KEY = " "
DATA_PATH = "recommendation_dataset_60k_with_names.xlsx"

# ====== Load & Preprocess Data ======
_df = pd.read_excel(DATA_PATH)
_df['txn_timestamp'] = pd.to_datetime(_df['txn_timestamp'])
_df = _df.fillna({
    'recommended_product_id': 'NA',
    'recommended_product_name': 'NA',
    'bundle_id': 'NA',
    'bundle_products': ''
})

# Mapping and matrices
prod_map = _df[['product_id','product_name']].drop_duplicates().set_index('product_id')['product_name'].to_dict()
cust_prod = _df.pivot_table(index='customer_id', columns='product_id', values='qty', aggfunc='sum').fillna(0)
prod_sim_df = pd.DataFrame(cosine_similarity(cust_prod.T), index=cust_prod.columns, columns=cust_prod.columns)
user_sim_df = pd.DataFrame(cosine_similarity(cust_prod), index=cust_prod.index, columns=cust_prod.index)

# Recency matrix
grp = _df.groupby(['customer_id','product_id'])['txn_timestamp'].max().reset_index()
grp['days_since'] = ( _df['txn_timestamp'].max() - grp['txn_timestamp'] ).dt.days
grp['recency_weight'] = np.exp(-0.01 * grp['days_since'])
recency_mat = grp.pivot_table(index='customer_id', columns='product_id', values='recency_weight', fill_value=0)

# Price sensitivity matrix
_df['total_price'] = _df['qty'] * _df['list_price']
cust_spend = _df.groupby('customer_id')['total_price'].mean().rename('avg_spend')
prod_price = _df.groupby('product_id')['list_price'].mean().rename('mean_price')
cp = (
    cust_spend.reset_index().assign(key=1)
    .merge(prod_price.reset_index().assign(key=1), on='key')
    .drop(columns='key')
)
cp['price_weight'] = 1 / (1 + (cp['mean_price'] - cp['avg_spend']).abs() / cp['avg_spend'])
price_mat = cp.pivot_table(index='customer_id', columns='product_id', values='price_weight', fill_value=0)

# ====== Recommendation Functions ======

def recommend_items(customer_id, top_n=5):
    if customer_id not in cust_prod.index:
        return []
    bought = cust_prod.loc[customer_id]
    bought_ids = bought[bought>0].index.tolist()
    scores = {}
    for pid in bought_ids:
        for rec_pid, sc in prod_sim_df[pid].sort_values(ascending=False)[1:top_n+1].items():
            if rec_pid not in bought_ids:
                scores[rec_pid] = max(scores.get(rec_pid, 0), sc)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

def recommend_users(customer_id, top_n=5):
    if customer_id not in user_sim_df.index:
        return []
    neighbors = user_sim_df[customer_id].sort_values(ascending=False)[1:top_n+1].index
    agg = cust_prod.loc[neighbors].sum()
    bought = cust_prod.loc[customer_id]
    recs = agg[bought==0].sort_values(ascending=False).head(top_n)
    return list(zip(recs.index, recs.values))

def hybrid_recommendations(customer_id, top_n=5, w_item=0.4, w_user=0.3, w_recency=0.2, w_price=0.1):
    if customer_id not in cust_prod.index:
        return []
    purchased = cust_prod.loc[customer_id]
    bought_ids = purchased[purchased>0].index.tolist()
    scores = {}
    for pid in cust_prod.columns:
        if pid in bought_ids: continue
        item_score = max((prod_sim_df.at[pid,b] for b in bought_ids), default=0)
        buyers = cust_prod[cust_prod[pid]>0].index
        user_score = user_sim_df.loc[customer_id, buyers].mean() if len(buyers) else 0
        rec_score = recency_mat.at[customer_id, pid] if pid in recency_mat.columns else 0
        price_score = price_mat.at[customer_id, pid] if pid in price_mat.columns else 0
        scores[pid] = w_item*item_score + w_user*user_score + w_recency*rec_score + w_price*price_score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

# ====== Explainability ======

PROMPT_TEMPLATES = {
    "Item Type": """You are a helpful shopping assistant.
The customer recently purchased: {purchased}.
We recommend: {recommended}.
Explain why these items are similar and beneficial in 2 sentences.""",
    "Customer Type": """You are a savvy shopping advisor.
Customers similar to this one bought: {purchased}.
We recommend: {recommended}.
Explain peer influence and value in 2 sentences.""",
    "Hybrid": """You are an AI shopping concierge.
Customer history: {purchased}.
Using a hybrid score of similarity, peers, recency, and price.
Recommend: {recommended}.
Explain how each factor contributed in 3 sentences."""
}

def get_rich_explanation(purchased, recommended, mode="Hybrid"):
    prompt = PROMPT_TEMPLATES[mode].format(
        purchased=", ".join(purchased),
        recommended=", ".join(recommended)
    )
    payload = {
        "model": "meta-llama/llama-3.2-3b-instruct",
        "messages": [{"role":"user","content":prompt}]
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers=headers, data=json.dumps(payload), timeout=20)
    if not resp.ok:
        return f"[Error {resp.status_code}] {resp.text}"
    return resp.json()["choices"][0]["message"]["content"].strip()

# ====== Public API ======

def generate_recommendations(customer_id, method='Item Type', top_n=5):
    recent = _df[_df['customer_id']==customer_id].sort_values('txn_timestamp',ascending=False).head(5)['product_id'].tolist()
    names = [prod_map[p] for p in recent]
    if method=="Item Type":
        recs = recommend_items(customer_id, top_n)
    else:
        recs = recommend_users(customer_id, top_n)
    rec_ids, scores = zip(*recs) if recs else ([],[])
    rec_names = [prod_map.get(pid,pid) for pid in rec_ids]
    expl = get_rich_explanation(names, rec_names, mode=method)
    return names, list(zip(rec_names, scores)), expl

def generate_hybrid_recommendations(customer_id, top_n=5):
    recent = _df[_df['customer_id']==customer_id].sort_values('txn_timestamp',ascending=False).head(5)['product_id'].tolist()
    names = [prod_map[p] for p in recent]
    recs = hybrid_recommendations(customer_id, top_n)
    rec_ids, scores = zip(*recs) if recs else ([],[])
    rec_names = [prod_map.get(pid,pid) for pid in rec_ids]
    expl = get_rich_explanation(names, rec_names, mode="Hybrid")
    return {
        "purchase_history": names,
        "recommendations": list(zip(rec_names, scores)),
        "explanation": expl
    }
