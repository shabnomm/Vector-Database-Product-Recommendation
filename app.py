# app.py
# Streamlit demo for: Product Recommendation + Hybrid Search + Product Comparison + ANN Trade-off (Review IVF)
# Works with your downloaded folder: streamlit_artifacts/

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Vector DB Product Recommender", layout="wide")
st.title("🧠 Vector Database Product Recommendation (Electronics)")
st.caption("Free-text semantic search • Product-ID similarity • Hybrid TF-IDF+Vector • Product comparison • ANN trade-off (FAISS IVF)")

# -----------------------------
# Paths
# -----------------------------
ART_DIR = "streamlit_artifacts"  # place your downloaded folder next to app.py

PATH_PRODUCTS = os.path.join(ART_DIR, "products.csv")
PATH_PROD_EMB = os.path.join(ART_DIR, "product_embeddings.npy")
PATH_FAISS_PROD_FLAT = os.path.join(ART_DIR, "faiss_products_flat.index")

PATH_TFIDF_VEC = os.path.join(ART_DIR, "tfidf_vectorizer.joblib")
PATH_TFIDF_MAT = os.path.join(ART_DIR, "tfidf_matrix.joblib")

PATH_REVIEWS = os.path.join(ART_DIR, "reviews.csv")
PATH_REVIEW_EMB = os.path.join(ART_DIR, "review_embeddings.npy")
PATH_FAISS_REV_FLAT = os.path.join(ART_DIR, "faiss_reviews_flat.index")
PATH_FAISS_REV_IVF = os.path.join(ART_DIR, "faiss_reviews_ivf.index")
PATH_ANN_TABLE = os.path.join(ART_DIR, "ann_tradeoff_table.csv")

# -----------------------------
# Helpers
# -----------------------------
def file_ok(p: str) -> bool:
    return os.path.exists(p) and os.path.getsize(p) > 0

def require_files(paths):
    missing = [p for p in paths if not file_ok(p)]
    if missing:
        st.error("Missing required artifact files. Please ensure this folder exists next to app.py: `streamlit_artifacts/`")
        st.code("\n".join(missing))
        st.stop()

# Validate required files for core tabs
require_files([PATH_PRODUCTS, PATH_PROD_EMB, PATH_FAISS_PROD_FLAT, PATH_TFIDF_VEC, PATH_TFIDF_MAT])

# ANN tab is optional (but you have it)
ann_available = all(file_ok(p) for p in [PATH_REVIEWS, PATH_REVIEW_EMB, PATH_FAISS_REV_FLAT, PATH_FAISS_REV_IVF, PATH_ANN_TABLE])

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_products():
    return pd.read_csv(PATH_PRODUCTS)

@st.cache_data
def load_prod_emb():
    return np.load(PATH_PROD_EMB).astype("float32")

@st.cache_resource
def load_faiss_prod_flat():
    return faiss.read_index(PATH_FAISS_PROD_FLAT)

@st.cache_resource
def load_tfidf():
    vec = joblib.load(PATH_TFIDF_VEC)
    mat = joblib.load(PATH_TFIDF_MAT)
    return vec, mat

@st.cache_resource
def load_model():
    # must match Kaggle embedding model
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_reviews():
    return pd.read_csv(PATH_REVIEWS)

@st.cache_data
def load_review_emb():
    return np.load(PATH_REVIEW_EMB).astype("float32")

@st.cache_resource
def load_faiss_reviews():
    idx_flat = faiss.read_index(PATH_FAISS_REV_FLAT)
    idx_ivf = faiss.read_index(PATH_FAISS_REV_IVF)
    return idx_flat, idx_ivf

@st.cache_data
def load_ann_table():
    return pd.read_csv(PATH_ANN_TABLE)

# -----------------------------
# Load core artifacts
# -----------------------------
prod_meta = load_products()
prod_emb = load_prod_emb()
idx_prod_flat = load_faiss_prod_flat()
tfidf_vec, tfidf_mat = load_tfidf()
model = load_model()

# Column guesses
COL_PRODUCT = "id" if "id" in prod_meta.columns else prod_meta.columns[0]
COL_BRAND = "brand" if "brand" in prod_meta.columns else None
COL_NREV = "n_reviews" if "n_reviews" in prod_meta.columns else None
COL_RATING = "avg_rating" if "avg_rating" in prod_meta.columns else None
COL_ASIN = "asins" if "asins" in prod_meta.columns else None
COL_CATEGORIES = "categories" if "categories" in prod_meta.columns else None


# Map product_id to index
pid_list = prod_meta[COL_PRODUCT].astype(str).tolist()
pid_to_idx = {pid: i for i, pid in enumerate(pid_list)}

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

top_k = st.sidebar.slider("Top-K results", 3, 20, 10, 1)
preview_len = st.sidebar.slider("Preview length (chars)", 80, 400, 180, 20)

# Filters for two-stage retrieval
st.sidebar.subheader("🔎 Two-stage Filters (metadata)")
brand = "All"
if COL_BRAND is not None:
    brand_values = sorted(prod_meta[COL_BRAND].astype(str).fillna("unknown").str.lower().unique().tolist())
    brand = st.sidebar.selectbox("Brand", ["All"] + brand_values, index=0)

min_reviews = 1
if COL_NREV is not None:
    min_reviews = st.sidebar.slider("Minimum reviews", 1, int(prod_meta[COL_NREV].max()), min(2, int(prod_meta[COL_NREV].max())), 1)

min_rating = None
if COL_RATING is not None and not prod_meta[COL_RATING].isna().all():
    min_rating = st.sidebar.slider("Minimum avg rating", 1.0, 5.0, 3.0, 0.1)

# Hybrid weight
st.sidebar.subheader("🧩 Hybrid Search")
alpha = st.sidebar.slider("Alpha (vector weight)", 0.0, 1.0, 0.7, 0.05)

# -----------------------------
# Filtering helper
# -----------------------------
def candidate_indices():
    mask = np.ones(len(prod_meta), dtype=bool)
    if COL_BRAND is not None and brand.lower() != "all":
        mask &= prod_meta[COL_BRAND].astype(str).str.lower().eq(brand.lower())
    if COL_RATING is not None and min_rating is not None:
        mask &= prod_meta[COL_RATING].fillna(-1) >= float(min_rating)
    if COL_NREV is not None:
        mask &= prod_meta[COL_NREV].fillna(0) >= int(min_reviews)
    return np.where(mask)[0]

# -----------------------------
# Search functions
# -----------------------------
def embed_query(text: str) -> np.ndarray:
    v = model.encode([text], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

def search_products_vector(query: str, k: int):
    cand = candidate_indices()
    if len(cand) == 0:
        return pd.DataFrame(), 0.0

    qv = embed_query(query)
    # For filtered set, do fast dot product on subset (since product count is small)
    t0 = time.time()
    scores = (prod_emb[cand] @ qv[0]).astype(np.float32)
    top_local = np.argsort(-scores)[:k]
    t1 = time.time()

    idxs = cand[top_local]
    out = prod_meta.iloc[idxs].copy()
    out["score"] = scores[top_local]
    return out, (t1 - t0) * 1000

def search_products_hybrid(query: str, k: int):
    cand = candidate_indices()
    if len(cand) == 0:
        return pd.DataFrame(), 0.0

    qv = embed_query(query)
    t0 = time.time()

    # vector score
    vec_scores = (prod_emb[cand] @ qv[0]).astype(np.float32)

    # tfidf score
    q_t = tfidf_vec.transform([query])
    tf_full = cosine_similarity(q_t, tfidf_mat).ravel().astype(np.float32)
    tf_scores = tf_full[cand]

    # normalize tfidf scores to 0..1 for mixing
    if tf_scores.max() > tf_scores.min():
        tf_scores = (tf_scores - tf_scores.min()) / (tf_scores.max() - tf_scores.min() + 1e-12)

    final = alpha * vec_scores + (1 - alpha) * tf_scores
    top_local = np.argsort(-final)[:k]
    t1 = time.time()

    idxs = cand[top_local]
    out = prod_meta.iloc[idxs].copy()
    out["hybrid_score"] = final[top_local]
    out["vector_score"] = vec_scores[top_local]
    out["tfidf_score"] = tf_scores[top_local]
    return out, (t1 - t0) * 1000

def search_by_product_id(pid: str, k: int):
    pid = str(pid)
    if pid not in pid_to_idx:
        return pd.DataFrame(), 0.0

    cand = candidate_indices()
    qidx = pid_to_idx[pid]
    cand = cand[cand != qidx]
    if len(cand) == 0:
        return pd.DataFrame(), 0.0

    qv = prod_emb[qidx:qidx+1]
    t0 = time.time()
    scores = (prod_emb[cand] @ qv[0]).astype(np.float32)
    top_local = np.argsort(-scores)[:k]
    t1 = time.time()

    idxs = cand[top_local]
    out = prod_meta.iloc[idxs].copy()
    out["score"] = scores[top_local]
    return out, (t1 - t0) * 1000

def format_results(df_res: pd.DataFrame, score_col: str):
    if df_res.empty:
        st.warning("No results found for current filters.")
        return

    show_cols = [COL_PRODUCT]
    if COL_ASIN in df_res.columns: show_cols.append(COL_ASIN)
    if COL_BRAND in df_res.columns: show_cols.append(COL_BRAND)
    if COL_RATING in df_res.columns: show_cols.append(COL_RATING)
    if COL_NREV in df_res.columns: show_cols.append(COL_NREV)
    # Include example_text preview
    if "example_text" in df_res.columns:
        df_res["preview"] = df_res["example_text"].astype(str).str.slice(0, preview_len) + "..."
        show_cols.append("preview")
    show_cols.append(score_col)

    st.dataframe(df_res[show_cols], use_container_width=True)

    # Expanders
    for _, row in df_res.iterrows():
        pid = str(row[COL_PRODUCT])
        title = f"Product {pid} | {score_col}={float(row[score_col]):.4f}"
        if COL_BRAND in df_res.columns:
            title += f" | brand={row.get(COL_BRAND, '')}"
        with st.expander(title):
            for c in [COL_ASIN, COL_BRAND, COL_CATEGORIES, COL_RATING, COL_NREV]:
                if c is not None and c in row.index:
                    st.write(f"**{c}**: {row[c]}")
            if "example_text" in row.index:
                st.write("**Example text:**")
                st.write(str(row["example_text"]))

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔎 Search (Free text)", "🆔 Search (By Product ID)", "📊 Compare & ANN"])

# ---- Tab 1: Free text search
with tab1:
    st.subheader("Free-text semantic search")
    mode = st.radio("Retrieval mode", ["Vector only (SBERT + Flat)", "Hybrid (Vector + TF-IDF)"], horizontal=True)
    query = st.text_input("Type your query", value="kindle paperwhite glare free screen battery life")

    colA, colB = st.columns([1, 2])
    with colA:
        run = st.button("Search", type="primary")
    with colB:
        st.caption("Tip: Adjust filters and alpha from sidebar.")

    if run:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            if mode.startswith("Vector"):
                res, ms = search_products_vector(query, top_k)
                st.success(f"Done • Method: Vector • Time: {ms:.3f} ms")
                format_results(res, "score")
            else:
                res, ms = search_products_hybrid(query, top_k)
                st.success(f"Done • Method: Hybrid • Time: {ms:.3f} ms • alpha={alpha:.2f}")
                format_results(res, "hybrid_score")

# ---- Tab 2: Product ID search
with tab2:
    st.subheader("Product-ID similarity search (item-to-item recommendation)")

    # pick ID from dropdown to avoid typing mistakes
    pid = st.selectbox("Choose a product ID", options=pid_list, index=0)
    run2 = st.button("Find Similar Products", type="primary")

    if run2:
        res, ms = search_by_product_id(pid, top_k)
        st.success(f"Done • Time: {ms:.3f} ms")
        format_results(res, "score")

# ---- Tab 3: Compare & ANN
with tab3:
    st.subheader("Product comparison + ANN trade-off")

    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("### 🧮 Compare 2–4 products")
        selected = st.multiselect("Select product IDs", options=pid_list, default=pid_list[:2])
        if st.button("Compare Now"):
            if len(selected) < 2:
                st.warning("Select at least 2 products.")
            elif len(selected) > 4:
                st.warning("Select at most 4 products.")
            else:
                idxs = [pid_to_idx[str(p)] for p in selected]
                details_cols = [COL_PRODUCT]
                if COL_ASIN in prod_meta.columns:
                    details_cols.append(COL_ASIN)
                if COL_BRAND in prod_meta.columns:
                    details_cols.append(COL_BRAND)
                if COL_CATEGORIES in prod_meta.columns:
                    details_cols.append(COL_CATEGORIES)
                if COL_RATING in prod_meta.columns:
                    details_cols.append(COL_RATING)
                if COL_NREV in prod_meta.columns:
                    details_cols.append(COL_NREV)
                    details_df = prod_meta.iloc[idxs][details_cols].copy()
                st.markdown("### 📦 Selected Product Details")
                st.dataframe(details_df, use_container_width=True)
                V = prod_emb[idxs]  # normalized embeddings
                sim = V @ V.T
                labels = [
                f"{row[COL_PRODUCT]} ({row.get(COL_BRAND, '')})"
                for _, row in details_df.iterrows()
                ]
                sim_df = pd.DataFrame(sim, index=labels, columns=labels)
                st.markdown("### 🔗 Semantic Similarity Matrix")
                st.dataframe(sim_df, use_container_width=True)
                st.caption("Higher value = more semantically similar (cosine similarity)")


    with colR:
        st.markdown("### ⚡ ANN Trade-off (Review-level IVF)")
        if not ann_available:
            st.info("ANN artifacts not found. If you exported review IVF files, place them in streamlit_artifacts/.")
        else:
            ann_table = load_ann_table()
            st.dataframe(ann_table, use_container_width=True)

            # Interactive nprobe demo
            reviews = load_reviews()
            review_emb = load_review_emb()
            idx_flat, idx_ivf = load_faiss_reviews()

            # pick a sample query review (from dropdown)
            if "id" in reviews.columns:
                review_pid_col = "id"
            else:
                review_pid_col = reviews.columns[0]

            review_text_col = "combined_text" if "combined_text" in reviews.columns else ("reviews.text" if "reviews.text" in reviews.columns else None)
            if review_text_col is None:
                review_text_col = reviews.columns[-1]

            nprobe = st.slider("IVF nprobe", 1, 30, 5, 1)
            idx_ivf.nprobe = int(nprobe)

            q_idx = st.slider("Pick a query review row index", 0, len(reviews) - 1, 0, 1)

            k = st.slider("Top-K similar reviews", 3, 20, 10, 1)

            if st.button("Run ANN Demo"):
                qv = review_emb[q_idx:q_idx+1]

                t0 = time.time()
                s_flat, i_flat = idx_flat.search(qv, k+1)
                t1 = time.time()

                t2 = time.time()
                s_ivf, i_ivf = idx_ivf.search(qv, k+1)
                t3 = time.time()

                # remove self if present
                flat_ids = i_flat[0].tolist()
                flat_scores = s_flat[0].tolist()
                if q_idx in flat_ids:
                    j = flat_ids.index(q_idx)
                    flat_ids.pop(j); flat_scores.pop(j)
                flat_ids, flat_scores = flat_ids[:k], flat_scores[:k]

                ivf_ids = i_ivf[0].tolist()
                ivf_scores = s_ivf[0].tolist()
                if q_idx in ivf_ids:
                    j = ivf_ids.index(q_idx)
                    ivf_ids.pop(j); ivf_scores.pop(j)
                ivf_ids, ivf_scores = ivf_ids[:k], ivf_scores[:k]

                st.markdown(f"**Flat time:** {(t1 - t0)*1000:.3f} ms  |  **IVF time:** {(t3 - t2)*1000:.3f} ms (nprobe={nprobe})")

                st.markdown("**Query review preview:**")
                st.write(str(reviews.iloc[q_idx].get(review_text_col, ""))[:400] + "...")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Flat (Exact) results**")
                    df_flat = reviews.iloc[flat_ids].copy()
                    df_flat["score"] = flat_scores
                    cols_show = [review_pid_col, "score", review_text_col]
                    cols_show = [c for c in cols_show if c in df_flat.columns]
                    st.dataframe(df_flat[cols_show].assign(
                        preview=df_flat[review_text_col].astype(str).str.slice(0, preview_len) + "..."
                    )[[review_pid_col, "score", "preview"]], use_container_width=True)

                with c2:
                    st.markdown("**IVF (ANN) results**")
                    df_ivf = reviews.iloc[ivf_ids].copy()
                    df_ivf["score"] = ivf_scores
                    cols_show = [review_pid_col, "score", review_text_col]
                    cols_show = [c for c in cols_show if c in df_ivf.columns]
                    st.dataframe(df_ivf[cols_show].assign(
                        preview=df_ivf[review_text_col].astype(str).str.slice(0, preview_len) + "..."
                    )[[review_pid_col, "score", "preview"]], use_container_width=True)

st.markdown("---")
st.caption(
    "Notes: Product retrieval uses Flat (exact) because product count is small. ANN trade-off is demonstrated on review vectors where scale makes ANN meaningful."
)
