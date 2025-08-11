# -*- coding: utf-8 -*-
# streamlit_app.py
"""
Streamlit dashboard for clustering Greek job ads into 5 buckets and predicting buckets for new ads.
Fixes:
- Uses session_state so bucket selectors never "reset to homepage".
- Every bucket selector filters live without re-running the pipeline.
- Bigger, centered PNG icons in KPI cards and on the bucket distribution chart.

Features kept:
- HDBSCAN / KMeans / Agglomerative clustering
- Hard routing rules + optional supervised router for 5 buckets
- Top terms per chosen bucket
- Word-frequency bars per chosen bucket
- 2D scatter of ads
- Save/Load router model, CSV downloads
"""

import re, json, random, unicodedata, hashlib, logging, base64
from io import BytesIO
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import streamlit as st
from joblib import dump, load

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import plotly.express as px
from PIL import Image

# ---------- Page ----------
st.set_page_config(page_title="Clustering Αγγελιών", page_icon="🧭", layout="wide")
logging.getLogger("numba").setLevel(logging.WARNING)
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---------- Buckets ----------
BUCKETS = ["hotel","food","retail","logistics","office"]
BUCKET_TITLES = {
    "hotel": "ΞΕΝΟΔΟΧΕΙΑ / HOTELS",
    "food": "ΕΣΤΙΑΣΗ / FOOD",
    "retail": "ΛΙΑΝΙΚΗ / RETAIL",
    "logistics": "ΜΕΤΑΦΟΡΕΣ / LOGISTICS",
    "office": "ΓΡΑΦΕΙΟ-ΛΟΓΙΣΤΗΡΙΟ-ΤΕΧΝΙΚΑ / OFFICE-TECH"
}
BUCKET_COLORS = {
    "hotel": "#4DB6AC",
    "food": "#FF7043",
    "retail": "#7E57C2",
    "logistics": "#29B6F6",
    "office": "#66BB6A",
}
EMOJI_FALLBACK = {"hotel":"🏨","food":"🍽️","retail":"🛍️","logistics":"🚚","office":"🧾"}

# ---------- Text utils ----------
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\+?\d[\d/\-\s]{6,}\d")
NON_ALNUM_RE = re.compile(r"[^0-9a-zα-ωάέίόύήώϊϋΐΰ\s]", re.IGNORECASE)

CUSTOM_STOP = {"αγια","μαρινα","boutique"}
GREEK_STOP = {
    "και","για","στην","στη","στο","στον","των","με","απο","σε","της","τον",
    "ζητα","ζηταει","ζητειται","ζηταμε","πληρ","τηλ","βιογραφικα","αποστολη","ωραριο",
    "θεσεις","ατομο","ατομα","θεση","εργασια","πληρης","μονιμη","σεζον","βιογραφικων","στα","στα χανια", ""
}
ALL_STOP = set(GREEK_STOP) | set(CUSTOM_STOP)

FORCE_RULES = {
    "hotel": {"αγια μαρινα","μαρινα","boutique","hotel","ξενοδοχ","ρεσεψ","reception","housekeeping","υποδοχ","front office","guest relations","night auditor","κρατησ"},
    "food": {"restaurant","σερβιτ","σερβιτο","κουζιν","μαγειρ","μπαρ","pool bar","μπαρμαν","barista","sommelier","steward","λάντζ","λαντζ","paso","kitchen","chef","commis","demi"},
    "retail": {"καταστημα","πωλητ","stylists","super market","σουπερ","market","lidl","πωληση","πωλησεων","προωθητ","ready","καλλυντικ","outlet","vodafone","κινητης"},
    "logistics": {"οδηγ","διανομ","αποθηκ","φορτοεκφορ","χρηματαποστο","van","β κατηγορια","forklift"},
    "office": {"λογιστ","μισθοδοσ","γραμματει","assistant","προσωπικου","operation manager","store manager","marketing","real estate","δικτυ","μηχανικ","ηλεκτρολογ","υδραυλ","κηπουρ","απεντομ","συντηρητ","maintenance","office","spa therapist","kids club","it","network"}
}

def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def normalize_text(s: str) -> str:
    s = strip_accents(s.lower())
    s = EMAIL_RE.sub(" ", s)
    s = PHONE_RE.sub(" ", s)
    s = re.sub(r"\d+([.,:/-]\d+)*", " ", s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = re.sub(r"\b(αγια|μαρινα|boutique)\b", " ", s)
    s = re.sub(r"\b(πληρ|τηλ|αποστολη|βιογραφικα|ωραριο|τηλεφωνο|email)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def data_signature(cleaned: list[str]) -> str:
    import hashlib
    h = hashlib.sha1()
    for s in cleaned: h.update(b"\n"); h.update(s.encode("utf-8","ignore"))
    return h.hexdigest()

def route_bucket_raw(raw_text: str) -> str | None:
    t = strip_accents(raw_text.lower())
    if "αγια μαρινα" in t: return "hotel"
    for b, keys in FORCE_RULES.items():
        for k in keys:
            if k in t: return b
    return None

def guess_bucket_clean(cleaned_text: str) -> str:
    scores = {b:0 for b in BUCKETS}
    for b, keys in FORCE_RULES.items():
        for k in keys:
            if k in cleaned_text: scores[b]+=1
    return max(scores, key=scores.get)

def load_items_from_text(txt: str):
    t = txt.strip()
    if not t: return []
    if t.startswith("["):
        try: return [str(x) for x in json.loads(t)]
        except Exception: pass
    quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', t, flags=re.DOTALL)
    if quoted:
        items = [bytes(s,"utf-8").decode("unicode_escape") for s in quoted]
        return [x.strip() for x in items if x.strip()]
    return [ln for ln in t.splitlines() if ln.strip()]

def top_terms_by_subset(texts: list[str], topn: int = 20):
    if not texts: return pd.DataFrame(columns=["term","score"])
    cln = [normalize_text(x) for x in texts]
    vec = TfidfVectorizer(max_df=0.85, min_df=1, ngram_range=(1,2), stop_words=list(ALL_STOP))
    X = vec.fit_transform(cln)
    centroid = X.mean(axis=0).A1
    terms = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(centroid)[::-1][:topn]
    return pd.DataFrame({"term": terms[top_idx], "score": centroid[top_idx]})

def top_terms_per_cluster(tfidf_matrix, labels, vectorizer, topn=12):
    terms = np.array(vectorizer.get_feature_names_out()); rows=[]
    for c in sorted(set(labels)):
        idx = np.where(labels==c)[0]
        if not len(idx): continue
        centroid = tfidf_matrix[idx].mean(axis=0).A1
        top_idx = np.argsort(centroid)[::-1][:topn]
        for rank,j in enumerate(top_idx,1):
            rows.append({"cluster": int(c), "rank": rank, "term": terms[j], "score": float(centroid[j])})
    return pd.DataFrame(rows)

def dedup_indices(X_tfidf, thr=0.90):
    from sklearn.metrics.pairwise import cosine_similarity
    S = cosine_similarity(X_tfidf); keep=[]
    for i in range(S.shape[0]):
        if any(S[i,j]>thr for j in keep): continue
        keep.append(i)
    return keep

# ---------- Icons ----------
@st.cache_data(show_spinner=False)
def icon_b64(path: str, size: int) -> str|None:
    try:
        im = Image.open(path).convert("RGBA")
        im.thumbnail((size,size), Image.LANCZOS)
        buf = BytesIO(); im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None

def get_icon_map(icons_dir: Path, size: int):
    mp={}
    for b in BUCKETS:
        p = icons_dir / f"{b}.png"
        mp[b] = icon_b64(str(p), size) if p.exists() else None
    return mp

# ---------- Caches ----------
@st.cache_data(show_spinner=False)
def embed_or_tfidf(items, cleaned, sig, outdir: Path):
    tfidf = TfidfVectorizer(max_df=0.85, min_df=1, ngram_range=(1,2), stop_words=list(ALL_STOP))
    X_tfidf = tfidf.fit_transform(cleaned)
    emb_p = outdir/"embeddings.npy"; meta_p = outdir/"embed_meta.json"
    X_for=None; backend="tfidf"
    if emb_p.exists() and meta_p.exists():
        try:
            meta=json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("sig")==sig:
                X_for=np.load(emb_p); backend=meta.get("backend","tfidf")
        except Exception: X_for=None
    if X_for is None:
        try:
            from sentence_transformers import SentenceTransformer
            st.info("Φόρτωση sentence-transformers…")
            m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            X_embed=m.encode(items,batch_size=32,show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True)
            X_for=X_embed; backend="st"
            np.save(emb_p,X_for); meta_p.write_text(json.dumps({"backend":"st","n":len(items),"sig":sig}),encoding="utf-8")
        except Exception as e:
            st.warning(f"Χωρίς sentence-transformers, χρήση TF-IDF. {e}")
            X_for=X_tfidf.toarray()
            np.save(emb_p,X_for); meta_p.write_text(json.dumps({"backend":"tfidf","n":len(items),"sig":sig}),encoding="utf-8")
    return tfidf, X_tfidf, X_for, backend

def do_cluster(X_for, method, force_k, min_k, max_k):
    sil, inertia={}, {}; labels=None; Y2=None; best_k=None
    if method=="HDBSCAN":
        try:
            import hdbscan
            from umap import UMAP
        except Exception as e:
            st.error(f"umap-learn+hdbscan απαιτούνται. {e}"); return None,None,None,None,None
        um=UMAP(n_neighbors=12,n_components=15,metric="cosine",random_state=SEED)
        X_um=um.fit_transform(X_for)
        h=hdbscan.HDBSCAN(min_cluster_size=3,min_samples=1,metric='euclidean',cluster_selection_method="leaf")
        labels=h.fit_predict(X_um)
        if (labels!=-1).sum()>1 and len(set(labels[labels!=-1]))>1:
            sil["hdbscan_nonnoise"]=silhouette_score(X_um[labels!=-1], labels[labels!=-1], metric="euclidean")
        Y2=X_um[:,:2]
    elif method=="KMeans":
        from sklearn.cluster import KMeans
        if force_k and force_k>0: best_k=force_k
        else:
            for k in range(min_k,max_k+1):
                km_try=KMeans(n_clusters=k,random_state=SEED,n_init=10)
                lab=km_try.fit_predict(X_for); inertia[k]=km_try.inertia_
                try: sil[k]=silhouette_score(X_for,lab)
                except Exception: sil[k]=np.nan
            best_k=max(sil,key=lambda k: sil[k] if not np.isnan(sil[k]) else -1e9)
        km=KMeans(n_clusters=best_k or force_k,random_state=SEED,n_init=20)
        labels=km.fit_predict(X_for)
        from sklearn.manifold import TSNE
        Xv=StandardScaler().fit_transform(X_for)
        Y2 = TSNE(n_components=2,perplexity=min(30,max(5,len(labels)//3)),random_state=SEED,init="pca",learning_rate="auto").fit_transform(Xv)
    else:
        from sklearn.cluster import AgglomerativeClustering
        Xc=X_for
        if force_k and force_k>0: best_k=force_k
        else:
            for k in range(min_k,max_k+1):
                ac_try=AgglomerativeClustering(n_clusters=k,metric="cosine",linkage="average")
                lab=ac_try.fit_predict(Xc)
                try: sil[k]=silhouette_score(Xc,lab,metric="cosine")
                except Exception: sil[k]=np.nan
            best_k=max(sil,key=lambda k: sil[k] if not np.isnan(sil[k]) else -1e9)
        ac=AgglomerativeClustering(n_clusters=best_k,metric="cosine",linkage="average")
        labels=ac.fit_predict(Xc)
        from sklearn.manifold import TSNE
        Xv=StandardScaler().fit_transform(Xc)
        Y2 = TSNE(n_components=2,perplexity=min(30,max(5,len(labels)//3)),random_state=SEED,init="pca",learning_rate="auto").fit_transform(Xv)
    return labels, Y2, sil, inertia, best_k

def train_router(raw_list, cleaned_list):
    y=[]
    for raw,cln in zip(raw_list,cleaned_list):
        b=route_bucket_raw(raw) or guess_bucket_clean(cln); y.append(b)
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5))
    X=vec.fit_transform([normalize_text(t) for t in raw_list])
    clf=LogisticRegression(max_iter=2000)
    if len(set(y))>1:
        try:
            skf=StratifiedKFold(n_splits=min(5,max(2,len(raw_list)//5)))
            scores=cross_val_score(clf,X,y,cv=skf,scoring="f1_macro")
            st.caption(f"Router CV f1_macro: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception: pass
    clf.fit(X,y)
    return {"vectorizer":vec,"model":clf,"labels":BUCKETS,"rules":FORCE_RULES}

def predict_router(model, texts):
    res=[]; vec=model["vectorizer"]; clf=model["model"]
    for raw in texts:
        forced=route_bucket_raw(raw)
        if forced: res.append(forced); continue
        res.append(clf.predict(vec.transform([normalize_text(raw)]))[0])
    return res

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Ρυθμίσεις")
method = st.sidebar.selectbox("Μέθοδος clustering", ["HDBSCAN","KMeans","Agglomerative"], index=1)
force_k = st.sidebar.number_input("Force K (KMeans/Agglo)", 0, 20, 5, 1)
min_k   = st.sidebar.number_input("Min K", 2, 20, 2, 1)
max_k   = st.sidebar.number_input("Max K", 2, 50, 8, 1)
dedup   = st.sidebar.checkbox("Αφαίρεση διπλότυπων (cosine>0.90)", value=False)
topn_words_global = st.sidebar.slider("Top λέξεις (global)", 10, 100, 30, 5)
save_model_flag   = st.sidebar.checkbox("Αποθήκευση router μοντέλου", value=True)
model_path = st.sidebar.text_input("Μονοπάτι μοντέλου", "out/router_model.joblib")

st.sidebar.divider()
icons_dir_input = st.sidebar.text_input("Φάκελος icons (PNG)", "icons")
icons_dir = Path(icons_dir_input).expanduser()

uploaded = st.sidebar.file_uploader("📄 Φόρτωση αρχείου (txt/json/quoted)", type=["txt","json"])
sample_btn = st.sidebar.button("Φόρτωση demo")

# ---------- Input area ----------
st.title("🧭 Clustering Αγγελιών & Προβλέψεις Buckets")

if "df" not in st.session_state:
    st.session_state.update({
        "df": None, "Y2": None, "backend": None, "best_k": None,
        "inertias": {}, "sil_scores": {}, "items": [], "cleaned": []
    })

default_text = ""
if sample_btn:
    default_text = '\n'.join([
        "Σερβιτόροι σε Ξενοδοχειακή μονάδα 4* στα Χανιά",
        "Personal Assistant to General Manager",
        "Διανομέας (Moto) - Χανιά",
        "Οδηγός VAN - Χανιά",
        "Πωλητές - Πωλήτριες",
        "Groom για 4* Ξενοδοχειακή Μονάδα στα Χανιά",
        "Α' Σερβιτόρος",
        "Barman για ξενοδοχειακή μονάδα 4* στα Χανιά",
        "Συντηρητής",
        "Λογιστής / Βοηθός Λογιστή",
        "Store Manager Χανιά",
        "Υπάλληλος Αποθήκης",
        "Sous Chef",
        "Commis Chef",
        "Spa Therapist"
    ])

txt_input = uploaded.read().decode("utf-8","ignore") if uploaded else st.text_area(
    "Επικολλήστε αγγελίες (μία ανά γραμμή):", value=default_text, height=200
)

c_run, c_reset = st.columns([1,1])
run_clicked = c_run.button("🚀 Εκτέλεση clustering")
reset_clicked = c_reset.button("♻️ Reset προβολής")

if reset_clicked:
    st.session_state["df"]=None

if run_clicked:
    items = load_items_from_text(txt_input)
    if not items:
        st.warning("Δεν βρέθηκαν αγγελίες.")
    else:
        cleaned = [normalize_text(x) for x in items]
        sig = data_signature(cleaned)
        outdir = Path("out"); outdir.mkdir(parents=True, exist_ok=True)

        tfidf, X_tfidf_all, X_for, backend = embed_or_tfidf(items, cleaned, sig, outdir)

        if dedup:
            keep = dedup_indices(X_tfidf_all, 0.90)
            items  = [items[i] for i in keep]
            cleaned= [cleaned[i] for i in keep]
            X_tfidf_all = X_tfidf_all[keep]
            X_for = X_for[keep]

        labels, Y2, sil, inert, best_k = do_cluster(X_for, method, force_k if method!="HDBSCAN" else 0, min_k, max_k)
        if labels is not None:
            # bucket routing
            buckets = [route_bucket_raw(r) or guess_bucket_clean(c) for r,c in zip(items, cleaned)]
            df = pd.DataFrame({"id": range(len(items)), "description": items, "cluster": labels, "bucket": buckets})
            # store in session for persistent filtering
            st.session_state.update({
                "df": df, "Y2": Y2, "backend": backend, "best_k": best_k,
                "inertias": inert, "sil_scores": sil, "items": items, "cleaned": cleaned
            })
            # write CSVs
            df.to_csv(outdir/"clustered_with_buckets.csv", index=False, encoding="utf-8")
            terms_df = top_terms_per_cluster(X_tfidf_all, labels, tfidf, topn=12)
            terms_df.to_csv(outdir/"cluster_top_terms.csv", index=False, encoding="utf-8")

# ---------- Show results if available ----------
if st.session_state["df"] is not None:
    df = st.session_state["df"]; Y2 = st.session_state["Y2"]

    # Load icons once
    ICONS_BIG  = get_icon_map(icons_dir, size=72)   # KPI size bigger and centered
    ICONS_BAR  = get_icon_map(icons_dir, size=44)   # for bar top-of-bars
    ICONS_SMALL= get_icon_map(icons_dir, size=22)   # expander headers

    # KPI cards with big, centered icons
    cols = st.columns(5)
    for i,b in enumerate(BUCKETS):
        cnt = int((df["bucket"]==b).sum())
        with cols[i]:
            if ICONS_BIG[b]:
                st.markdown(f"<div style='text-align:center'><img src='{ICONS_BIG[b]}' width='64'></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center;font-size:42px'>{EMOJI_FALLBACK[b]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;font-weight:600'>{BUCKET_TITLES[b]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;font-size:28px'>{cnt}</div>", unsafe_allow_html=True)

    # Stats area
    left, right = st.columns([1.2,1])

    with left:
        st.subheader("📊 Κατανομή Buckets")
        bucket_counts = df["bucket"].value_counts().reindex(BUCKETS).fillna(0).astype(int)
        fig = px.bar(bucket_counts, labels={"index":"Bucket","value":"Count"},
                     color=bucket_counts.index, color_discrete_map=BUCKET_COLORS,
                     text=bucket_counts.values, height=420)
        fig.update_layout(showlegend=False, bargap=0.25)
        max_y = int(bucket_counts.max()) if len(bucket_counts) else 1
        for b in BUCKETS:
            if ICONS_BAR[b]:
                fig.add_layout_image(dict(
                    source=ICONS_BAR[b], x=b, y=max_y*1.08, xref="x", yref="y",
                    sizex=0.45, sizey=0.45, xanchor="center", yanchor="bottom", layer="above"
                ))
        fig.update_yaxes(range=[0, max_y*1.25])
        st.plotly_chart(fig, use_container_width=True)

        if Y2 is not None:
            st.subheader("🗺️ 2D Scatter των αγγελιών")
            d2 = pd.DataFrame({"x":Y2[:,0], "y":Y2[:,1], "bucket":df["bucket"], "cluster":df["cluster"], "text":df["description"]})
            fig2 = px.scatter(d2, x="x", y="y", color="bucket",
                              color_discrete_map=BUCKET_COLORS,
                              hover_data={"cluster":True,"bucket":True,"x":False,"y":False},
                              hover_name="text", height=540)
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        # Top terms per chosen bucket
        st.subheader("🔝 Top όροι ανά bucket")
        pick_bucket = st.selectbox("Διάλεξε bucket", BUCKETS, index=0, key="top_terms_bucket")
        pick_topn   = st.slider("Πλήθος όρων", 5, 40, 20, 1, key="top_terms_n")
        subset_texts = df[df["bucket"]==pick_bucket]["description"].tolist()
        top_terms_df = top_terms_by_subset(subset_texts, topn=pick_topn)
        if len(top_terms_df):
            fig_top = px.bar(top_terms_df.sort_values("score", ascending=True), x="score", y="term",
                             orientation="h", height=420, labels={"score":"TF-IDF","term":"Όρος"},
                             color_discrete_sequence=[BUCKET_COLORS[pick_bucket]])
            st.plotly_chart(fig_top, use_container_width=True)
            fig_top.update_layout(
                font=dict(size=26),
                xaxis=dict(title_font=dict(size=26), tickfont=dict(size=27)),
                yaxis=dict(title_font=dict(size=18), tickfont=dict(size=16))
            )
        else:
            st.info("Δεν υπάρχουν κείμενα σε αυτό το bucket.")


        st.subheader("ℹ️ Diagnostics")
        backend = st.session_state["backend"]; best_k = st.session_state["best_k"]
        inertias = st.session_state["inertias"]; sil_scores = st.session_state["sil_scores"]
        st.write(f"Μέθοδος: **{method}** | Backend: **{backend}** | Clusters: **{len(set(df['cluster']))}**")
        if best_k: st.write(f"Best K: **{best_k}**")
        if inertias: st.write("Inertia (KMeans):", inertias)
        if sil_scores: st.write("Silhouette:", sil_scores)

    # Word frequency bars per chosen bucket
    st.subheader("📚 Συχνότητα λέξεων ανά bucket")
    freq_cols = st.columns(3)
    freq_bucket = freq_cols[0].selectbox("Bucket", BUCKETS, index=0, key="freq_bucket")
    freq_topn   = freq_cols[1].number_input("Πλήθος λέξεων", 5, 50, 20, 1, key="freq_topn")
    tok_min     = freq_cols[2].number_input("Ελάχιστο μήκος", 1, 6, 2, 1, key="freq_tokmin")

    texts_fb = df[df["bucket"]==freq_bucket]["description"].tolist()
    tokens=[]
    for t in [normalize_text(x) for x in texts_fb]:
        tokens.extend([w for w in t.split() if w not in ALL_STOP and len(w)>=tok_min])
    freq = Counter(tokens).most_common(int(freq_topn))
    if freq:
        freq_df = pd.DataFrame(freq, columns=["term","count"])
        fig_freq = px.bar(freq_df, x="term", y="count", height=420, color_discrete_sequence=[BUCKET_COLORS[freq_bucket]])
        fig_freq.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("Δεν βρέθηκαν λέξεις για εμφάνιση.")

    # Listings per bucket
    st.subheader("📜 Αγγελίες ανά bucket")
    for b in BUCKETS:
        sub = df[df["bucket"]==b].reset_index(drop=True)
        with st.expander(f"{BUCKET_TITLES[b]} ({len(sub)})", expanded=False):
            if ICONS_SMALL[b]:
                st.markdown(f"<img src='{ICONS_SMALL[b]}' width='20'>", unsafe_allow_html=True)
            else:
                st.markdown(f"**{EMOJI_FALLBACK[b]}**")
            st.dataframe(sub[["description","cluster","bucket"]], use_container_width=True, height=260)

    # Save/Load router + predict
    st.divider()
    st.subheader("🧠 Router μοντέλο (5 buckets) & Προβλέψεις")
    colA, colB = st.columns(2)

    with colA:
        if save_model_flag and st.button("💾 Εκπαίδευση + Αποθήκευση Router"):
            model = train_router(st.session_state["items"], st.session_state["cleaned"])
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            dump(model, model_path)
            st.success(f"Αποθηκεύτηκε: {model_path}")

        upl = st.file_uploader("Φόρτωση router (.joblib)", type=["joblib"], key="router_upl")
        if upl and st.button("📥 Φόρτωσε Router"):
            tmp = Path("out")/"router_loaded.joblib"; tmp.write_bytes(upl.read())
            st.session_state["router_model_path"]=str(tmp); st.success(f"Φορτώθηκε: {tmp}")

    with colB:
        model_used = st.session_state.get("router_model_path", model_path if Path(model_path).exists() else None)
        st.write(f"Μοντέλο σε χρήση: **{model_used or 'κανένα'}**")
        new_jobs = st.text_area("Νέες αγγελίες (μία ανά γραμμή):", height=120)
        if st.button("🔮 Πρόβλεψη"):
            if not new_jobs.strip():
                st.warning("Δώστε κείμενα.")
            else:
                if not model_used or not Path(model_used).exists():
                    st.error("Δεν βρέθηκε αποθηκευμένο μοντέλο. Αποθηκεύστε ή φορτώστε πρώτα.")
                else:
                    model = load(model_used)
                    new_list = load_items_from_text(new_jobs)
                    preds = predict_router(model, new_list)
                    pred_df = pd.DataFrame({"description": new_list, "bucket": preds})
                    st.dataframe(pred_df, use_container_width=True)

    # Downloads
    st.divider()
    outdir = Path("out")
    c1, c2 = st.columns(2)
    if (outdir/"clustered_with_buckets.csv").exists():
        c1.download_button("⬇️ CSV clustered_with_buckets",
                           data=(outdir/"clustered_with_buckets.csv").read_bytes(),
                           file_name="clustered_with_buckets.csv", mime="text/csv")
    if (outdir/"cluster_top_terms.csv").exists():
        c2.download_button("⬇️ CSV cluster_top_terms",
                           data=(outdir/"cluster_top_terms.csv").read_bytes(),
                           file_name="cluster_top_terms.csv", mime="text/csv")

# ---------- Sidebar help ----------
st.sidebar.divider()
st.sidebar.code("pip install streamlit scikit-learn pandas numpy plotly joblib umap-learn hdbscan sentence-transformers pillow", language="cmd")
st.sidebar.code("streamlit run streamlit_app.py", language="cmd")
st.sidebar.markdown("**Icons:** βάλτε PNG αρχεία στον φάκελο που δηλώσατε, ονόματα:\n" + "\n".join([f"{icons_dir_input}/{b}.png" for b in BUCKETS]))
