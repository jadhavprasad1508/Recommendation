"""
recommender.py

Provides a lightweight, memory-conscious Recommender class that can be
instantiated with any pandas DataFrame that has the required columns.

Public API:
- Recommender(df, openrouter_api_key=None)
- recommender.get_insights()
- recommender.recommend(customer_id, method='Item Type', top_n=5)
- recommender.recommend_hybrid(customer_id, top_n=5)

Notes:
- Required minimal columns: 'customer_id', 'product_id' OR 'product_name',
  'qty', 'list_price', 'txn_timestamp'.
- If dataset has 'product_id' use it; otherwise product_name will be used as id.
- For explainability the OpenRouter key is optional. If missing, a short local
  explanation text will be returned.
"""

import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests
import json

# Constants - tuneable
MIN_REQUIRED_COLUMNS = {"customer_id", "qty", "list_price", "txn_timestamp"}
# product identifier can be product_id OR product_name
RECOMMENDER_PRODUCT_ID_COLS = ["product_id", "product_name"]

# Limits to avoid OOM on small VMs
MAX_PRODUCTS_FOR_FULL_SIM = 5000   # if > this, we sample or reduce
MAX_USERS_FOR_USER_SIM = 3000      # for user-based neighbor search we sample

# Helper: safe read of API key (prefer environment; Streamlit secrets maps to env via deployment)
def _get_openrouter_key(provided_key=None):
    if provided_key:
        return provided_key
    # prefer env var
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    # try common Streamlit secret environment mapping
    # (when using streamlit secrets.toml, you can also expose as env var during deploy)
    return None


class Recommender:
    def __init__(self, df: pd.DataFrame, openrouter_api_key: str = None):
        """
        Initialize recommender with a pandas DataFrame.
        We do minimal preprocessing here and build lightweight indices on demand.
        """
        self.df_raw = df.copy()
        self.openrouter_api_key = _get_openrouter_key(openrouter_api_key)

        # Basic clean / canonicalization
        self._prepare_dataframe()
        # lazy attributes
        self._product_index = None
        self._cust_index = None
        self._cust_prod_matrix = None
        self._nn_for_products = None  # NearestNeighbors model for items
        self._product_popularity = None
        # recency and price matrices cached as dicts
        self._recency = None
        self._price_fit = None

    def _prepare_dataframe(self):
        df = self.df_raw
        # ensure timestamp
        df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], errors="coerce")
        # normalize product id column
        for col in RECOMMENDER_PRODUCT_ID_COLS:
            if col in df.columns:
                self.df_raw["product_id_norm"] = df[col].astype(str)
                break
        else:
            # if no product columns at all, try to create from product_name or product_id missing
            raise ValueError(f"Input DataFrame must include one of: {RECOMMENDER_PRODUCT_ID_COLS}")

        # qty and list_price numeric
        self.df_raw["qty"] = pd.to_numeric(self.df_raw["qty"], errors="coerce").fillna(0).astype(int)
        self.df_raw["list_price"] = pd.to_numeric(self.df_raw["list_price"], errors="coerce").fillna(0.0)

        # Fill missing locations/demos if absent - no harm
        if "location" not in self.df_raw.columns:
            self.df_raw["location"] = None
        # create small product name map
        prod_map = (
            self.df_raw[["product_id_norm"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "prod_idx"})
        )
        # map product -> display name (prefer product_name if present)
        if "product_name" in self.df_raw.columns:
            name_map = (
                self.df_raw[["product_id_norm", "product_name"]]
                .drop_duplicates()
                .set_index("product_id_norm")["product_name"]
                .to_dict()
            )
        else:
            # fallback display = id
            name_map = {pid: pid for pid in prod_map["product_id_norm"].tolist()}

        self._display_name_map = name_map

    def _ensure_indices_and_matrix(self, sample_product_limit=MAX_PRODUCTS_FOR_FULL_SIM):
        """Build indexes and sparse customer-product matrix lazily."""
        if self._cust_prod_matrix is not None:
            return

        df = self.df_raw
        # Build product and customer index mapping
        products = df["product_id_norm"].astype(str).unique().tolist()
        customers = df["customer_id"].astype(str).unique().tolist()

        # If too many products, keep top-k by popularity
        prod_counts = df["product_id_norm"].value_counts()
        if len(prod_counts) > sample_product_limit:
            top_products = prod_counts.head(sample_product_limit).index.tolist()
            df = df[df["product_id_norm"].isin(top_products)].copy()

        # rebuild mappings after sampling
        products = df["product_id_norm"].astype(str).unique().tolist()
        customers = df["customer_id"].astype(str).unique().tolist()

        self._product_index = {pid: i for i, pid in enumerate(products)}
        self._product_index_rev = {i: pid for pid, i in self._product_index.items()}
        self._cust_index = {cid: i for i, cid in enumerate(customers)}
        self._cust_index_rev = {i: cid for cid, i in self._cust_index.items()}

        # Build sparse matrix: rows=customers, cols=products
        rows = []
        cols = []
        vals = []
        grouped = df.groupby(["customer_id", "product_id_norm"])["qty"].sum().reset_index()
        for _, r in grouped.iterrows():
            cid = str(r["customer_id"])
            pid = str(r["product_id_norm"])
            if cid not in self._cust_index or pid not in self._product_index:
                continue
            rows.append(self._cust_index[cid])
            cols.append(self._product_index[pid])
            vals.append(float(r["qty"]))
        if len(rows) == 0:
            # empty dataset edge-case
            self._cust_prod_matrix = csr_matrix((0, 0))
        else:
            shape = (len(self._cust_index), len(self._product_index))
            self._cust_prod_matrix = csr_matrix((vals, (rows, cols)), shape=shape)

        # Precompute product popularity (total qty)
        product_sum = np.array(self._cust_prod_matrix.sum(axis=0)).ravel()
        self._product_popularity = {self._product_index_rev[i]: float(product_sum[i]) for i in range(len(product_sum))}

    def _ensure_product_nn(self, n_neighbors=10):
        """Create NearestNeighbors model for products using product vectors (columns of cust-prod)."""
        if self._nn_for_products is not None:
            return

        self._ensure_indices_and_matrix()
        if self._cust_prod_matrix.shape[1] == 0:
            self._nn_for_products = None
            return

        # Build product vectors = columns of cust_prod_matrix (transpose)
        product_matrix = self._cust_prod_matrix.T.tocsr()  # shape: (n_products, n_customers)
        # Use NearestNeighbors with cosine metric; it works with sparse input
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine", algorithm="brute")
        nn.fit(product_matrix)
        self._nn_for_products = nn
        self._product_matrix = product_matrix

    def _compute_recency(self, decay_lambda=0.01):
        if self._recency is not None:
            return
        df = self.df_raw
        # reference date = latest txn_timestamp
        ref_date = df["txn_timestamp"].max()
        grp = df.groupby(["customer_id", "product_id_norm"])["txn_timestamp"].max().reset_index()
        grp["days_since"] = (ref_date - grp["txn_timestamp"]).dt.days.fillna(9999).astype(int)
        grp["recency_weight"] = np.exp(-decay_lambda * grp["days_since"])
        # store as dict keyed (cust,prod) -> weight
        self._recency = {(str(r["customer_id"]), str(r["product_id_norm"])): float(r["recency_weight"]) for _, r in grp.iterrows()}

    def _compute_price_fit(self):
        if self._price_fit is not None:
            return
        df = self.df_raw
        df["total_price"] = df["qty"] * df["list_price"]
        cust_spend = df.groupby("customer_id")["total_price"].mean().rename("avg_spend")
        prod_price = df.groupby("product_id_norm")["list_price"].mean().rename("mean_price")
        # cross join implemented using product of indices, but keep it as dict (costly otherwise)
        price_fit = {}
        for cust, avg_sp in cust_spend.items():
            for prod, mean_p in prod_price.items():
                # avoid division by zero
                if avg_sp == 0:
                    w = 0.5
                else:
                    w = 1 / (1 + abs(mean_p - avg_sp) / (avg_sp))
                price_fit[(str(cust), str(prod))] = float(w)
        self._price_fit = price_fit

    # --------------------------
    # Public utilities & insights
    # --------------------------
    def get_insights(self):
        """Return basic dataset insights used in the app UI."""
        df = self.df_raw
        insights = {}
        insights["total_rows"] = int(len(df))
        insights["unique_customers"] = int(df["customer_id"].nunique())
        insights["unique_products"] = int(df["product_id_norm"].nunique())
        insights["total_qty"] = int(df["qty"].sum())
        insights["avg_order_value"] = float((df["qty"] * df["list_price"]).sum() / max(1, len(df)))

        # repeated purchase: fraction of (cust,product) combinations that occurred more than once
        gp = df.groupby(["customer_id", "product_id_norm"]).size()
        if len(gp) > 0:
            insights["repeated_purchase_ratio"] = float((gp > 1).sum() / len(gp))
        else:
            insights["repeated_purchase_ratio"] = 0.0

        # locations (if exists)
        if "location" in df.columns and df["location"].notna().any():
            insights["top_locations"] = df["location"].value_counts().head(5).to_dict()
        else:
            insights["top_locations"] = {}

        # demographics (if columns exist e.g., 'age', 'gender' or 'city')
        demos = {}
        for col in ["gender", "age", "city", "state"]:
            if col in df.columns:
                demos[col] = df[col].dropna().value_counts().head(5).to_dict()
        insights["demographics"] = demos

        return insights

    # --------------------------
    # Recommendation methods
    # --------------------------
    def recommend_item_type(self, customer_id: str, top_n: int = 5):
        """
        Item-type recommendations: find nearest neighbor products to the
        products this customer has purchased using product vectors.
        """
        self._ensure_product_nn(n_neighbors=top_n + 5)
        if self._nn_for_products is None:
            return []

        cid = str(customer_id)
        if cid not in self._cust_index:
            return []

        # products the customer bought
        cust_row_idx = self._cust_index[cid]
        cust_vector = self._cust_prod_matrix.getrow(cust_row_idx).toarray().ravel()
        bought_idx = np.where(cust_vector > 0)[0].tolist()
        if len(bought_idx) == 0:
            return []

        candidate_scores = defaultdict(float)
        # for each bought product, query nn to get similar products
        for bidx in bought_idx:
            # use product matrix stored earlier
            distances, indices = self._nn_for_products.kneighbors(self._product_matrix[bidx], n_neighbors=self._nn_for_products.n_neighbors)
            # distances are cosine distances (0=identical). convert to similarity
            for dist, idx in zip(distances.ravel(), indices.ravel()):
                pid = self._product_index_rev.get(int(idx))
                if pid is None:
                    continue
                sim = 1 - float(dist)  # approximate similarity
                # skip the same product
                if int(idx) == int(bidx):
                    continue
                candidate_scores[pid] = max(candidate_scores[pid], sim)
        # sort by score and return top_n
        sorted_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return sorted_items

    def recommend_customer_type(self, customer_id: str, top_n: int = 5):
        """
        Customer-type recommendations: find customers similar to this one (based on purchase vector)
        and recommend their popular products that the target hasn't bought.
        To keep memory low we sample other users if there are many.
        """
        self._ensure_indices_and_matrix()
        cid = str(customer_id)
        if cid not in self._cust_index:
            return []

        # sample neighbors if too many users
        num_users = len(self._cust_index)
        user_indices = list(range(num_users))
        if num_users > MAX_USERS_FOR_USER_SIM:
            # pick top users by activity
            user_activity = np.array(self._cust_prod_matrix.sum(axis=1)).ravel()
            top_user_idx = np.argsort(user_activity)[-MAX_USERS_FOR_USER_SIM:]
            user_indices = top_user_idx.tolist()

        # compute simple cosine similarity between target and sampled users (dense but limited)
        target_vec = self._cust_prod_matrix.getrow(self._cust_index[cid]).toarray()
        sample_mat = self._cust_prod_matrix[user_indices].toarray()
        # small dot-product cosine
        denom = (np.linalg.norm(target_vec, axis=1) * np.linalg.norm(sample_mat, axis=1))
        # when denom zero, similarity = 0
        sims = (sample_mat @ target_vec.T).ravel()
        norms = np.linalg.norm(sample_mat, axis=1) * np.linalg.norm(target_vec)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim_scores = np.where(norms > 0, sims / norms, 0.0)
        # map back to user ids
        user_idx_to_id = {idx: uid for uid, idx in self._cust_index.items() if idx in user_indices}
        # combine top neighbors
        top_neighbor_idx = np.argsort(sim_scores)[-5:]  # top 5
        neighbor_user_ids = [user_indices[i] for i in top_neighbor_idx]
        # aggregate their purchases
        agg = self._cust_prod_matrix[neighbor_user_ids].sum(axis=0).A1
        # exclude products the target already bought
        target_bought = set(np.where(self._cust_prod_matrix.getrow(self._cust_index[cid]).toarray().ravel() > 0)[0])
        candidate_idx = [i for i, qty in enumerate(agg) if qty > 0 and i not in target_bought]
        # turn into (pid,score) sorted by aggregated qty
        items = sorted(((self._product_index_rev[i], float(agg[i])) for i in candidate_idx), key=lambda x: x[1], reverse=True)[:top_n]
        return items

    def recommend_hybrid(self, customer_id: str, top_n: int = 5, w_item=0.5, w_customer=0.2, w_recency=0.2, w_price=0.1):
        """Combine item and customer and recency and price fit into a hybrid score."""
        self._ensure_indices_and_matrix()
        self._compute_recency()
        self._compute_price_fit()

        cid = str(customer_id)
        if cid not in self._cust_index:
            return []

        # get candidate products = all products except those already bought
        cust_row_idx = self._cust_index[cid]
        cust_vec = self._cust_prod_matrix.getrow(cust_row_idx).toarray().ravel()
        bought_idx = set(np.where(cust_vec > 0)[0])

        # item-based nominations
        item_recs = dict(self.recommend_item_type(customer_id, top_n=top_n * 5))
        # customer-based nominations
        cust_recs = dict(self.recommend_customer_type(customer_id, top_n=top_n * 5))

        # combine candidate set
        candidate_pids = set(list(item_recs.keys()) + list(cust_recs.keys()))
        scores = {}
        for pid in candidate_pids:
            pid_s = str(pid)
            # item score
            item_score = item_recs.get(pid_s, 0.0)
            # customer score
            cust_score = cust_recs.get(pid_s, 0.0)
            # recency
            recency_score = self._recency.get((cid, pid_s), 0.0)
            # price fit
            price_score = self._price_fit.get((cid, pid_s), 0.5)
            hybrid = w_item * item_score + w_customer * cust_score + w_recency * recency_score + w_price * price_score
            scores[pid_s] = hybrid

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        # return list of (pid, score)
        return top

    # --------------------------
    # user facing wrapper functions
    # --------------------------
    def recommend(self, customer_id: str, method: str = "Item Type", top_n: int = 5):
        """Unified recommendation endpoint; returns display names with scores."""
        # lazy build
        self._ensure_indices_and_matrix()
        # choose function
        if method == "Item Type":
            raw = self.recommend_item_type(customer_id, top_n)
        elif method == "Customer Type":
            raw = self.recommend_customer_type(customer_id, top_n)
        else:
            raw = self.recommend_hybrid(customer_id, top_n)
        # map product id -> display name
        out = []
        for pid, score in raw:
            name = self._display_name_map.get(pid, pid)
            out.append((name, float(score)))
        return out

    def getPurchaseHistory(self, customer_id: str, n=5):
        """Return last n purchased product display names and qty."""
        df = self.df_raw
        cdf = df[df["customer_id"].astype(str) == str(customer_id)].sort_values("txn_timestamp", ascending=False).head(n)
        out = []
        for _, r in cdf.iterrows():
            pname = self._display_name_map.get(str(r["product_id_norm"]), str(r["product_id_norm"]))
            out.append({"product_name": pname, "qty": int(r["qty"]), "price": float(r["list_price"])})
        return out

    # --------------------------
    # Explainability
    # --------------------------
    def _call_openrouter(self, prompt, max_tokens=120):
        key = self.openrouter_api_key
        if not key:
            return None
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-3.2-3b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            if not resp.ok:
                return None
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def explain(self, customer_id: str, recommended_names: list, method="Item Type"):
        """
        Try to produce a helpful explanation. Uses OpenRouter if API key present,
        else produce a concise deterministic explanation.
        """
        purchased = [p["product_name"] for p in self.getPurchaseHistory(customer_id, n=5)]
        prompt = ""
        if method == "Item Type":
            prompt = f"You are a helpful shopping assistant. Customer recently purchased: {', '.join(purchased)}. Recommend: {', '.join(recommended_names)}. Explain why these items are similar and beneficial in 2 sentences."
        elif method == "Customer Type":
            prompt = f"You are a savvy shopping advisor. Customers similar to this one bought: {', '.join(purchased)}. We recommend: {', '.join(recommended_names)}. Explain peer influence and value in 2 sentences."
        else:
            prompt = f"You are an AI shopping concierge. Customer history: {', '.join(purchased)}. Using a hybrid score of similarity, peers, recency, and price. Recommend: {', '.join(recommended_names)}. Explain how each factor contributed in 3 sentences."

        resp = self._call_openrouter(prompt)
        if resp:
            return resp
        # fallback basic explanation
        return f"We recommended {', '.join(recommended_names)} based on recent purchases ({', '.join(purchased[:3])}), similarity and popularity among similar customers."

# Simple helper to validate uploaded dataframe
def validate_dataframe(df: pd.DataFrame):
    cols = set(df.columns)
    # require at least one product id column
    if not any(c in cols for c in RECOMMENDER_PRODUCT_ID_COLS):
        return False, f"Uploaded dataset must include one of: {RECOMMENDER_PRODUCT_ID_COLS}"
    # require base columns
    missing = MIN_REQUIRED_COLUMNS - cols
    if missing:
        return False, f"Missing required columns: {missing}. Required minimal: {MIN_REQUIRED_COLUMNS} plus product id column."
    return True, "OK"


# --------------------------
# Compatibility wrappers for app.py
# --------------------------
def _ensure_recommender_instance(df=None, openrouter_api_key=None):
    """Create a Recommender instance from df or raise if df missing."""
    if df is None:
        raise ValueError("DataFrame must be provided to build recommendations.")
    return Recommender(df, openrouter_api_key=openrouter_api_key)


def generate_recommendations(customer_id: str, method: str = "Item Type", top_n: int = 5, df: pd.DataFrame = None):
    """Return (purchase_history_names, recommendations, explanation) for compatibility with app.py.

    - purchase_history_names: list of tuples (product_display_name, qty)
    - recommendations: list of tuples (product_display_name, score)
    - explanation: short text
    """
    if df is None:
        # try to load the app's default dataset if available
        default_path = os.path.join(os.path.dirname(__file__), "recommendation_dataset_60k_with_names.xlsx")
        if os.path.exists(default_path):
            df = pd.read_excel(default_path)
            df['list_price'] = pd.to_numeric(df.get('list_price', pd.Series()), errors='coerce')
        else:
            raise ValueError("generate_recommendations requires a DataFrame 'df' argument or the default dataset file must be present")

    rec = _ensure_recommender_instance(df=df)
    purchase_history = [(p['product_name'], p['qty']) for p in rec.getPurchaseHistory(customer_id, n=5)]
    recs = rec.recommend(customer_id, method=method, top_n=top_n)
    # rec.recommend returns list of (display_name, score)
    recommended_names = [name for name, _ in recs]
    explanation = rec.explain(customer_id, recommended_names, method=method)
    return purchase_history, recs, explanation


def generate_hybrid_recommendations(customer_id: str, top_n: int = 5, df: pd.DataFrame = None):
    """Return dict with keys purchase_history, recommendations, explanation for app.py when using default dataset."""
    if df is None:
        default_path = os.path.join(os.path.dirname(__file__), "recommendation_dataset_60k_with_names.xlsx")
        if os.path.exists(default_path):
            df = pd.read_excel(default_path)
            df['list_price'] = pd.to_numeric(df.get('list_price', pd.Series()), errors='coerce')
        else:
            raise ValueError("generate_hybrid_recommendations requires a DataFrame 'df' argument or the default dataset file must be present")
    rec = _ensure_recommender_instance(df=df)
    purchase_history = [(p['product_name'], p['qty']) for p in rec.getPurchaseHistory(customer_id, n=5)]
    recs = rec.recommend_hybrid(customer_id, top_n=top_n)
    # map ids to display names
    recs_named = []
    for pid, score in recs:
        name = rec._display_name_map.get(str(pid), str(pid))
        recs_named.append((name, float(score)))
    explanation = rec.explain(customer_id, [n for n, _ in recs_named], method="Hybrid")
    return {"purchase_history": purchase_history, "recommendations": recs_named, "explanation": explanation}


def generate_recommendations_from_upload(df_uploaded: pd.DataFrame, customer_id: str, method: str, top_n: int = 5):
    """Helper used by the app when user uploads a dataset.

    Returns a dict with purchase_history, recommendations, explanation.
    """
    valid, msg = validate_dataframe(df_uploaded)
    if not valid:
        raise ValueError(f"Uploaded dataframe invalid: {msg}")
    rec = Recommender(df_uploaded)
    purchase_history = [(p['product_name'], p['qty']) for p in rec.getPurchaseHistory(customer_id, n=5)]
    if method == 'Hybrid':
        recs = rec.recommend_hybrid(customer_id, top_n=top_n)
    else:
        recs = rec.recommend(customer_id, method=method, top_n=top_n)
    # rec.recommend returns display names already
    recommended_names = [name for name, _ in recs]
    explanation = rec.explain(customer_id, recommended_names, method=method)
    return {"purchase_history": purchase_history, "recommendations": recs, "explanation": explanation}
