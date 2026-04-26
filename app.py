"""
Tokopedia Review Analyzer — Streamlit App
==========================================
Install:
    pip install streamlit playwright beautifulsoup4 pandas plotly
                scikit-learn transformers torch
    playwright install chromium

Jalankan:
    streamlit run app.py
"""

import asyncio
import sys
import re
from collections import Counter
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ── Stopwords Indonesia ───────────────────────────────────
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "tidak", "ada", "juga", "sudah", "saya", "nya", "aja", "tapi", "karena",
    "bisa", "lebih", "atau", "dalam", "pada", "akan", "tp", "si", "ya",
    "banget", "sangat", "pake", "pakai", "masih", "jadi", "kalau", "kalo",
    "beli", "barang", "produk", "sih", "nih", "deh", "lah", "dong", "saja",
    "oke", "ok", "udah", "udh", "lg", "lagi", "dgn", "yg", "dr", "sy",
    "msh", "blm", "sdh", "krn", "utk", "pd", "dlm", "jg", "hrs", "bs",
    "bgt", "gak", "ga", "ngga", "nggak", "kok", "dah", "lho", "kan",
    "pun", "per", "mau", "harus", "semua", "hal", "satu", "dua", "kali",
    "karna", "memang", "emang", "saat", "setelah", "sebelum", "waktu",
    "sama", "seperti", "pas", "lumayan", "cukup", "sekali", "paling",
    "kurang", "harga", "seller", "toko", "pengiriman", "ongkir", "kirim",
    "sampai", "tiba", "datang", "segel", "unit", "baru",
}

SENTIMENT_COLOR = {
    "Positif": ("#dcfce7", "#16a34a"),
    "Netral":  ("#fef9c3", "#ca8a04"),
    "Negatif": ("#fee2e2", "#dc2626"),
}
SENTIMENT_EMOJI = {"Positif": "😊", "Netral": "😐", "Negatif": "😞"}


# ══════════════════════════════════════════════════════════
# SCRAPING HELPERS
# ══════════════════════════════════════════════════════════

def clean_url(raw: str) -> str:
    parsed = urlparse(raw.strip())
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


async def scroll_slowly(page, steps: int = 12):
    for i in range(1, steps + 1):
        pct = i / steps
        await page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {pct})")
        await page.wait_for_timeout(400)


async def click_review_tab(page) -> bool:
    for sel in ["text=Ulasan", "button:has-text('Ulasan')", "a:has-text('Ulasan')",
                "[data-testid*='ulasan']", "[data-testid*='review']", "text=Penilaian"]:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=2000):
                await el.scroll_into_view_if_needed()
                await el.click()
                await page.wait_for_timeout(2000)
                return True
        except Exception:
            continue
    return False


async def wait_for_reviews(page, timeout: int = 15000):
    for sel in ["[data-testid='lblItemUlasan']", "#review-feed article",
                "article.css-15m2bcr", ".css-15m2bcr"]:
        try:
            await page.wait_for_selector(sel, timeout=timeout)
            return sel
        except Exception:
            continue
    return None


async def click_next_page(page) -> bool:
    for sel in ["button[aria-label='Laman berikutnya']", "button[aria-label='Next']",
                "[data-testid='btnNextPage']", "button:has-text('›')"]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=2000):
                await btn.scroll_into_view_if_needed()
                await btn.click()
                await page.wait_for_timeout(2500)
                return True
        except Exception:
            continue
    return False


def parse_product_info(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    info = {"name": "", "description": "", "avg_rating": "", "total_reviews": ""}
    for sel in ["h1[data-testid='lblPDPDetailProductName']", "h1.css-1320bma", "h1"]:
        el = soup.select_one(sel)
        if el and el.text.strip():
            info["name"] = el.text.strip()
            break
    for sel in ["[data-testid='lblPDPDescriptionProduk']", ".css-buqman"]:
        el = soup.select_one(sel)
        if el and el.text.strip():
            raw = el.text.strip()
            info["description"] = raw[:400] + ("..." if len(raw) > 400 else "")
            break
    for sel in ["[data-testid='lblPDPDetailProductRatingNumber']", "[class*='rating'] span"]:
        el = soup.select_one(sel)
        if el and el.text.strip():
            d = re.search(r'[\d.]+', el.text)
            if d:
                info["avg_rating"] = d.group()
                break
    for sel in ["[data-testid='lblPDPDetailProductRatingCounter']"]:
        el = soup.select_one(sel)
        if el and el.text.strip():
            d = re.search(r'[\d,\.]+', el.text)
            if d:
                info["total_reviews"] = d.group().replace(",", "").replace(".", "")
                break
    return info


def parse_reviews(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    articles = (
        soup.select("article.css-15m2bcr")
        or soup.select("#review-feed article")
        or soup.select("[data-testid='review-feed'] article")
    )
    results = []
    for art in articles:
        name_el = art.select_one("span.name")
        name = name_el.text.strip() if name_el else "Anonim"
        rating_el = art.select_one("[data-testid='icnStarRating']")
        if rating_el:
            aria = rating_el.get("aria-label", "")
            digits = "".join(filter(str.isdigit, aria))
            rating = int(digits) if digits else len(art.select("[data-testid='icnStarRating'] svg"))
        else:
            rating = 0
        time_el = art.select_one("[class*='css-1rpz5os']")
        review_time = time_el.text.strip() if time_el else ""
        variant_el = art.select_one("[data-testid='lblVarian']")
        variant = variant_el.text.strip() if variant_el else ""
        comment_el = art.select_one("[data-testid='lblItemUlasan']")
        comment = comment_el.text.strip() if comment_el else ""
        like_el = art.select_one("[class*='css-q2y3yl']")
        like_text = like_el.text.strip() if like_el else ""
        digits_like = "".join(filter(str.isdigit, like_text))
        like_count = int(digits_like) if digits_like else 0
        if comment:
            results.append({"name": name, "rating": rating, "review_time": review_time,
                            "variant": variant, "comment": comment, "like_count": like_count})
    return results


async def run_scraper(url: str, max_pages: int, status_fn, scrape_all: bool = False):
    all_data = []
    product_info = {}
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                channel="chrome", headless=False,
                args=["--disable-blink-features=AutomationControlled",
                      "--disable-dev-shm-usage", "--no-sandbox"])
        except Exception:
            browser = await p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled",
                      "--disable-dev-shm-usage", "--no-sandbox"])

        context = await browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
            locale="id-ID", viewport={"width": 1280, "height": 900},
            ignore_https_errors=True,
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
            Object.defineProperty(navigator, 'languages', {get: () => ['id-ID','id','en-US']});
            window.chrome = { runtime: {} };
        """)
        page = await context.new_page()

        status_fn("🌐 Membuka halaman produk...")
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        status_fn("📦 Mengambil info produk...")
        product_info = parse_product_info(await page.content())

        status_fn("📜 Scrolling halaman...")
        await scroll_slowly(page, steps=12)

        status_fn("🔍 Mencari tab Ulasan...")
        found = await click_review_tab(page)
        if not found:
            await scroll_slowly(page, steps=8)
            await click_review_tab(page)

        await scroll_slowly(page, steps=6)

        status_fn("⏳ Menunggu review muncul...")
        working_sel = await wait_for_reviews(page)
        if not working_sel:
            status_fn("❌ Review tidak ditemukan.")
            await browser.close()
            return [], product_info

        page_limit = 9999 if scrape_all else max_pages
        for pg in range(1, page_limit + 1):
            label = f"halaman {pg}" if not scrape_all else f"halaman {pg} (total: {len(all_data)})"
            status_fn(f"📄 Scraping {label}...")
            try:
                await page.wait_for_selector(working_sel, timeout=10000)
            except Exception:
                break
            await page.wait_for_timeout(1500)
            batch = parse_reviews(await page.content())
            if not batch:
                break
            all_data.extend(batch)
            status_fn(f"✅ +{len(batch)} review (total: {len(all_data)})")
            if pg < page_limit:
                if not await click_next_page(page):
                    break

        await browser.close()
    return all_data, product_info


def scrape_sync(url, max_pages, status_fn, scrape_all=False):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.run(run_scraper(url, max_pages, status_fn, scrape_all))


# ══════════════════════════════════════════════════════════
# ML FUNCTIONS
# ══════════════════════════════════════════════════════════

def preprocess_for_lda(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split()
               if t not in STOPWORDS_ID and len(t) > 2 and not t.isdigit()]
    return " ".join(tokens)


@st.cache_data
def run_lda(df_hash: str, rating: int, n_topics: int, n_words: int,
            _df: pd.DataFrame) -> tuple:
    subset = _df[_df["rating"] == rating]["comment"].tolist()
    if len(subset) < max(5, n_topics):
        return None, None
    docs = [preprocess_for_lda(c) for c in subset]
    docs = [d for d in docs if len(d.split()) >= 2]
    if len(docs) < n_topics:
        return None, None
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=500)
    try:
        dtm = vectorizer.fit_transform(docs)
    except Exception:
        return None, None
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42,
        max_iter=20, learning_method="online")
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-n_words:][::-1]
        words = [(feature_names[i], round(float(topic[i]), 2)) for i in top_idx]
        topics.append({"topic": idx + 1, "words": words})
    doc_topics = lda.transform(dtm)
    dominant = doc_topics.argmax(axis=1)
    topic_counts = [int((dominant == i).sum()) for i in range(n_topics)]
    return topics, topic_counts


@st.cache_resource
def load_sentiment_model():
    """Load model IndoBERT — dicache di memory, hanya load sekali."""
    from transformers import (AutoModelForSequenceClassification,
                               AutoTokenizer, pipeline)
    # Model dengan nama yang benar (ada -classification di akhir)
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model,
                    tokenizer=tokenizer, truncation=True, max_length=512)


# Label mapping untuk mdhugol model
LABEL_MAP = {"LABEL_0": "Positif", "LABEL_1": "Netral", "LABEL_2": "Negatif"}


def run_sentiment(texts: list[str]) -> list[dict]:
    clf = load_sentiment_model()
    raw = clf(texts, batch_size=8)
    return [{"label": LABEL_MAP.get(r["label"], r["label"]),
             "score": round(r["score"] * 100, 1)} for r in raw]


@st.cache_data
def compute_top_words(df_hash: str, rating: int, n: int,
                      _df: pd.DataFrame) -> pd.DataFrame:
    tokens = []
    for comment in _df[_df["rating"] == rating]["comment"]:
        text = comment.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens.extend([t for t in text.split()
                       if t not in STOPWORDS_ID and len(t) > 2 and not t.isdigit()])
    counted = Counter(tokens).most_common(n)
    return pd.DataFrame(counted, columns=["kata", "frekuensi"])


# ══════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════

st.set_page_config(page_title="Tokopedia Review Analyzer",
                   page_icon="🛒", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #f0f5ff; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important; border-right: 1px solid #dde8ff;
    box-shadow: 2px 0 12px rgba(37,99,235,0.06);
}

/* Navbar */
.navbar {
    background: #ffffff; border: 1px solid #dde8ff; border-radius: 14px;
    padding: 0.6rem 1rem; margin-bottom: 1.5rem;
    box-shadow: 0 2px 10px rgba(37,99,235,0.07);
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
}
.nav-btn {
    display: inline-block; padding: 6px 16px;
    background: #f0f5ff; color: #1d4ed8;
    border-radius: 8px; font-size: 0.8rem; font-weight: 700;
    text-decoration: none; border: 1px solid #bfdbfe;
    cursor: pointer; transition: all 0.15s;
}
.nav-btn:hover { background: #1d4ed8; color: #fff; }
.nav-divider { color: #c7d9f8; margin: 0 2px; font-size: 0.8rem; }

/* Product card */
.product-card {
    background: #ffffff; border: 1px solid #dde8ff; border-radius: 16px;
    padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(37,99,235,0.07);
}
.product-name { font-size: 1.1rem; font-weight: 800; color: #1e3a5f; line-height: 1.3; }
.product-desc { font-size: 0.81rem; color: #6b7280; line-height: 1.6; margin-top: 6px; }
.product-badge {
    display: inline-block; background: #dbeafe; color: #1d4ed8;
    font-size: 0.72rem; font-weight: 700; padding: 3px 10px;
    border-radius: 20px; margin-right: 6px; margin-top: 6px;
}

/* Metric card */
.metric-card {
    background: #ffffff; border: 1px solid #dde8ff; border-radius: 16px;
    padding: 1.3rem 1.2rem; text-align: center;
    box-shadow: 0 2px 12px rgba(37,99,235,0.07);
}
.metric-card .value { font-size: 1.9rem; font-weight: 800; color: #1d4ed8; line-height: 1; }
.metric-card .label {
    font-size: 0.72rem; color: #6b8cba; margin-top: 5px;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
}

/* Star bar */
.star-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.star-label { font-size: 0.83rem; color: #1d4ed8; font-weight: 700; min-width: 60px; }
.star-bar-bg { flex:1; background: #e8f0fe; border-radius: 6px; height: 12px; overflow: hidden; }
.star-bar-fill { height:100%; background:linear-gradient(90deg,#3b82f6,#1d4ed8); border-radius:6px; }
.star-count { font-size:0.8rem; color:#6b8cba; min-width:35px; text-align:right; font-weight:600; }

/* Section title */
.section-title {
    font-size: 1rem; font-weight: 800; color: #1e3a5f;
    border-bottom: 3px solid #3b82f6; padding-bottom: 5px;
    margin-bottom: 16px; display: inline-block;
}
.section-anchor { scroll-margin-top: 70px; }

/* Review card */
.review-card {
    background:#ffffff; border:1px solid #dde8ff; border-radius:12px;
    padding:1rem 1.2rem; margin-bottom:10px;
    box-shadow:0 1px 6px rgba(37,99,235,0.05);
}
.review-card .rc-name { font-weight:700; font-size:0.88rem; color:#1e3a5f; }
.review-card .rc-meta { font-size:0.73rem; color:#8faac8; margin-top:3px; }
.review-card .rc-comment { font-size:0.85rem; color:#374151; margin-top:8px; line-height:1.6; }
.star-badge {
    display:inline-block; background:#dbeafe; color:#1d4ed8;
    font-size:0.72rem; font-weight:700; padding:2px 8px;
    border-radius:20px; margin-left:6px;
}

/* Topic card */
.topic-card {
    background:#ffffff; border:1px solid #dde8ff; border-radius:12px;
    padding:0.9rem 1.1rem; margin-bottom:8px;
    box-shadow:0 1px 6px rgba(37,99,235,0.05);
}
.topic-title { font-size:0.8rem; font-weight:800; color:#1d4ed8;
    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px; }
.topic-pill {
    display:inline-block; background:#eff6ff; color:#1e40af;
    font-size:0.71rem; font-weight:600; padding:3px 10px;
    border-radius:20px; margin:2px 3px 2px 0; border:1px solid #bfdbfe;
}

/* Sentiment card */
.sent-card { border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:8px; border:1px solid; }
.sent-comment { font-size:0.83rem; line-height:1.55; margin-bottom:6px; }
.sent-label { font-size:0.75rem; font-weight:700; }

/* Buttons */
div[data-testid="stButton"] button {
    background:linear-gradient(135deg,#3b82f6,#1d4ed8) !important;
    color:#fff !important; font-weight:700 !important; border:none !important;
    border-radius:10px !important; box-shadow:0 4px 14px rgba(29,78,216,0.25) !important;
}
.stTextInput input {
    background:#f8faff !important; border:1.5px solid #c7d9f8 !important;
    color:#1e3a5f !important; border-radius:10px !important;
}
div[data-testid="stDownloadButton"] button {
    background:#fff !important; color:#1d4ed8 !important;
    border:2px solid #3b82f6 !important; font-weight:700 !important; border-radius:10px !important;
}
span[data-baseweb="tag"] { background-color:#dbeafe !important; color:#1d4ed8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:1rem;">
  <div style="background:linear-gradient(135deg,#3b82f6,#1d4ed8);width:48px;height:48px;
    border-radius:14px;display:flex;align-items:center;justify-content:center;
    font-size:1.5rem;box-shadow:0 4px 14px rgba(29,78,216,0.3);">🛒</div>
  <div>
    <h1 style="font-size:1.65rem;font-weight:800;color:#1e3a5f;margin:0;line-height:1.1;">
      Tokopedia Review Analyzer</h1>
    <p style="color:#6b8cba;font-size:0.8rem;margin:0;font-weight:500;">
      Scrape, analisis, dan visualisasi ulasan produk dengan Machine Learning</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Detect cloud environment ──────────────────────────────
import os
IS_CLOUD = (
    os.environ.get("STREAMLIT_SHARING_MODE") is not None or
    os.environ.get("IS_STREAMLIT_CLOUD") is not None or
    "appuser" in os.path.expanduser("~")
)

# ── Session state ─────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "product_info" not in st.session_state:
    st.session_state.product_info = {}

df    = st.session_state.df
pinfo = st.session_state.product_info

# ── Navbar ────────────────────────────────────────────────
if df is not None and not df.empty:
    st.markdown("""
    <div class="navbar">
        <span style="font-size:0.75rem;font-weight:700;color:#6b8cba;margin-right:4px;">Navigasi:</span>
        <a class="nav-btn" href="#produk">🛍️ Produk</a>
        <span class="nav-divider">›</span>
        <a class="nav-btn" href="#statistik">📊 Statistik</a>
        <span class="nav-divider">›</span>
        <a class="nav-btn" href="#sentimen">🤖 Sentimen</a>
        <span class="nav-divider">›</span>
        <a class="nav-btn" href="#topik">🧠 Topic Modeling</a>
        <span class="nav-divider">›</span>
        <a class="nav-btn" href="#topkata">🔤 Top Kata</a>
        <span class="nav-divider">›</span>
        <a class="nav-btn" href="#ulasan">💬 Ulasan</a>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:

    if IS_CLOUD:
        # ── CLOUD MODE: Upload CSV ─────────────────────────
        st.markdown("**📂 Upload Data Review**")
        st.info(
            "Cara pakai di cloud: "
            "(1) Jalankan scrape_tokopedia.py di komputer lokal, "
            "(2) Upload file reviews_tokopedia.csv di bawah.",
            icon="ℹ️",
        )
        uploaded = st.file_uploader("Upload CSV hasil scraping", type=["csv"])
        product_name = st.text_input("Nama Produk (opsional)",
                                     placeholder="misal: POCO Pad M1 8GB")

        # Preview info jika file sudah dipilih
        if uploaded is not None:
            # Tampilkan info file sebelum diproses
            st.markdown(f"""
            <div style="background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
                padding:10px 14px;margin:8px 0;font-size:0.8rem;">
                <b style="color:#0369a1;">📄 File siap diproses:</b><br>
                <span style="color:#374151;">{uploaded.name}</span>
                <span style="color:#6b7280;margin-left:8px;">
                    ({uploaded.size/1024:.1f} KB)
                </span>
            </div>
            """, unsafe_allow_html=True)

            proses_btn = st.button(
                "⚡ Proses & Muat Data",
                use_container_width=True,
                type="primary",
                key="proses_cloud",
            )

            if proses_btn:
                with st.spinner("⏳ Memproses data CSV..."):
                    try:
                        df_up = pd.read_csv(uploaded)
                        required = {"name", "rating", "comment"}
                        if not required.issubset(df_up.columns):
                            st.error(f"❌ CSV harus punya kolom: {required}")
                            st.stop()

                        for col in ["review_time", "variant", "like_count"]:
                            if col not in df_up.columns:
                                df_up[col] = "" if col != "like_count" else 0
                        df_up["rating"] = pd.to_numeric(
                            df_up["rating"], errors="coerce").fillna(0).astype(int)
                        df_up["like_count"] = pd.to_numeric(
                            df_up["like_count"], errors="coerce").fillna(0).astype(int)
                        df_up = df_up[df_up["comment"].notna() & (df_up["comment"] != "")]

                        n_reviews = len(df_up)
                        avg_r = df_up["rating"].mean()
                        n_r5  = (df_up["rating"] == 5).sum()

                        st.session_state.df = df_up
                        st.session_state.product_info = {
                            "name": product_name or uploaded.name.replace(".csv", ""),
                            "description": "",
                            "avg_rating": f"{avg_r:.2f}",
                            "total_reviews": str(n_reviews),
                        }
                        compute_top_words.clear()
                        run_lda.clear()

                        # Konfirmasi sukses dengan ringkasan
                        st.success(f"✅ Berhasil memuat **{n_reviews} review**!")
                        st.markdown(f"""
                        <div style="background:#f0fdf4;border:1px solid #86efac;
                            border-radius:10px;padding:12px 16px;font-size:0.82rem;">
                            <b style="color:#15803d;">📊 Ringkasan Data:</b><br>
                            <span style="color:#374151;">
                            🗂 Total review : <b>{n_reviews}</b><br>
                            ⭐ Rata-rata    : <b>{avg_r:.2f} ★</b><br>
                            💚 Bintang 5   : <b>{n_r5} review ({n_r5/n_reviews*100:.0f}%)</b>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("👆 Dashboard akan muncul di atas — scroll ke atas!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Gagal memproses CSV: {e}")
        else:
            # Belum ada file — tampilkan panduan
            st.markdown("""
            <div style="background:#fafafa;border:1.5px dashed #c7d9f8;border-radius:10px;
                padding:14px;text-align:center;color:#8faac8;font-size:0.8rem;margin-top:8px;">
                ⬆️ Pilih file CSV di atas untuk mulai
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── LOCAL MODE: Scraping + Upload ─────────────────
        mode = st.radio("Mode Input", ["🔍 Scraping Langsung", "📂 Upload CSV"],
                        horizontal=True)

        if mode == "🔍 Scraping Langsung":
            st.markdown("**⚙️ Konfigurasi Scraping**")
            url_input = st.text_input("URL Produk Tokopedia",
                                      placeholder="https://www.tokopedia.com/...",
                                      help="Query string (?t_id=...) otomatis dihapus.")
            if url_input and "tokopedia.com" in url_input:
                c = clean_url(url_input)
                if c != url_input.strip():
                    st.caption("✅ Query params dihapus")
            scrape_all = st.checkbox("📥 Ambil SEMUA review", value=False)
            if scrape_all:
                st.warning("⚠️ Proses bisa sangat lama.")
                max_pages = 999
            else:
                max_pages = st.slider("Jumlah Halaman", 1, 20, 5)
            scrape_btn = st.button("🚀 Mulai Scraping", use_container_width=True)
        else:
            url_input  = ""
            scrape_btn = False
            scrape_all = False
            max_pages  = 5
            uploaded_local = st.file_uploader("Upload CSV review", type=["csv"])
            if uploaded_local:
                st.markdown(f"""
                <div style="background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
                    padding:10px 14px;margin:8px 0;font-size:0.8rem;">
                    <b style="color:#0369a1;">📄 File siap diproses:</b><br>
                    <span style="color:#374151;">{uploaded_local.name}</span>
                    <span style="color:#6b7280;margin-left:8px;">
                        ({uploaded_local.size/1024:.1f} KB)
                    </span>
                </div>
                """, unsafe_allow_html=True)

                proses_local_btn = st.button(
                    "⚡ Proses & Muat Data",
                    use_container_width=True,
                    type="primary",
                    key="proses_local",
                )
                if proses_local_btn:
                    with st.spinner("⏳ Memproses data CSV..."):
                        try:
                            df_loc = pd.read_csv(uploaded_local)
                            for col in ["review_time", "variant", "like_count"]:
                                if col not in df_loc.columns:
                                    df_loc[col] = "" if col != "like_count" else 0
                            df_loc["rating"] = pd.to_numeric(
                                df_loc["rating"], errors="coerce").fillna(0).astype(int)
                            df_loc["like_count"] = pd.to_numeric(
                                df_loc["like_count"], errors="coerce").fillna(0).astype(int)
                            df_loc = df_loc[df_loc["comment"].notna() & (df_loc["comment"] != "")]
                            n_reviews = len(df_loc)
                            avg_r = df_loc["rating"].mean()
                            st.session_state.df = df_loc
                            st.session_state.product_info = {
                                "name": uploaded_local.name.replace(".csv", ""),
                                "description": "",
                                "avg_rating": f"{avg_r:.2f}",
                                "total_reviews": str(n_reviews),
                            }
                            compute_top_words.clear()
                            run_lda.clear()
                            st.success(f"✅ Berhasil memuat **{n_reviews} review**!")
                            st.info("👆 Dashboard akan muncul — scroll ke atas!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Gagal memproses CSV: {e}")
            else:
                st.markdown("""
                <div style="background:#fafafa;border:1.5px dashed #c7d9f8;border-radius:10px;
                    padding:14px;text-align:center;color:#8faac8;font-size:0.8rem;margin-top:8px;">
                    ⬆️ Pilih file CSV di atas untuk mulai
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🔤 Pengaturan Analisis**")
    top_n    = st.slider("Top N Kata per Rating", 5, 30, 15)
    n_topics = st.slider("Jumlah Topik LDA", 2, 6, 3)
    n_words  = st.slider("Kata per Topik LDA", 5, 15, 8)

    st.markdown("---")
    st.markdown("""
    <p style="font-size:0.72rem;color:#8faac8;">
    💡 Slider analisis bisa diubah bebas tanpa upload ulang.
    </p>""", unsafe_allow_html=True)

# ── Scraping trigger (lokal only) ─────────────────────────
if (not IS_CLOUD and "scrape_btn" in locals() and scrape_btn):
    if not url_input or "tokopedia.com" not in url_input:
        st.error("⚠️ Masukkan URL produk Tokopedia yang valid.")
    else:
        final_url = clean_url(url_input)
        status_box = st.empty()
        prog = st.progress(0)

        def update_status(msg):
            status_box.info(msg)

        with st.spinner("Scraping sedang berjalan..."):
            try:
                data, pinf = scrape_sync(final_url, max_pages, update_status, scrape_all)
                if data:
                    st.session_state.df = pd.DataFrame(data)
                    st.session_state.product_info = pinf
                    compute_top_words.clear()
                    run_lda.clear()
                    prog.progress(100)
                    status_box.success(f"✅ Berhasil mengambil **{len(data)} review**!")
                    st.rerun()
                else:
                    status_box.error("❌ Review tidak ditemukan.")
            except Exception as e:
                status_box.error(f"❌ Error: {e}")

# ══════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════

if df is not None and not df.empty:
    total      = len(df)
    avg_r      = df["rating"].mean()
    total_lk   = df["like_count"].sum()
    pct5       = (df["rating"] == 5).sum() / total * 100
    rc         = df["rating"].value_counts().sort_index(ascending=False)
    avail_r    = sorted(df["rating"].unique(), reverse=True)
    df_hash    = str(total) + str(df["comment"].iloc[0][:20])

    # ── 1. PRODUK ─────────────────────────────────────────
    st.markdown('<div id="produk" class="section-anchor"></div>', unsafe_allow_html=True)
    if pinfo.get("name"):
        badges = ""
        if pinfo.get("avg_rating"):
            badges += f'<span class="product-badge">⭐ {pinfo["avg_rating"]}</span>'
        if pinfo.get("total_reviews"):
            badges += f'<span class="product-badge">💬 {pinfo["total_reviews"]} rating</span>'
        badges += f'<span class="product-badge">📋 {total} review terkumpul</span>'
        desc_html = (f'<div class="product-desc">{pinfo["description"]}</div>'
                     if pinfo.get("description") else "")
        st.markdown(f"""
        <div class="product-card">
            <div class="product-name">🛍️ {pinfo["name"]}</div>
            <div>{badges}</div>{desc_html}
        </div>""", unsafe_allow_html=True)

    # ── 2. STATISTIK ──────────────────────────────────────
    st.markdown('<div id="statistik" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-title">📊 Statistik Ulasan</span>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, str(total),       "Review Terkumpul"),
        (c2, f"{avg_r:.2f} ★", "Rata-rata Rating"),
        (c3, f"{pct5:.0f}%",   "Review Bintang 5"),
        (c4, str(total_lk),    "Total Likes"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2, gap="large")
    with cl:
        for star in range(5, 0, -1):
            cnt = rc.get(star, 0)
            pct = cnt / total * 100
            st.markdown(f"""
            <div class="star-row">
                <span class="star-label">{"★"*star}</span>
                <div class="star-bar-bg">
                    <div class="star-bar-fill" style="width:{pct}%"></div>
                </div>
                <span class="star-count">{cnt}</span>
            </div>""", unsafe_allow_html=True)

    with cr:
        rdf = pd.DataFrame({"Rating": [f"★{i}" for i in range(5,0,-1)],
                             "Jumlah": [rc.get(i,0) for i in range(5,0,-1)]})
        fig_bar = px.bar(rdf, x="Rating", y="Jumlah", color="Jumlah",
                         color_continuous_scale=[[0,"#bfdbfe"],[1,"#1d4ed8"]], text="Jumlah")
        fig_bar.update_traces(textposition="outside",
                              textfont=dict(color="#1e3a5f", size=12), marker_line_width=0)
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#1e3a5f", showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=20,b=10,l=10,r=10),
                              xaxis=dict(gridcolor="#e8f0fe", title=""),
                              yaxis=dict(gridcolor="#e8f0fe", title="Jumlah"))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<hr style="border:none;border-top:2px solid #dde8ff;margin:1.5rem 0;">', unsafe_allow_html=True)

    # ── 3. SENTIMENT ANALYSIS ─────────────────────────────
    st.markdown('<div id="sentimen" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-title">🤖 Sentiment Analysis (IndoBERT)</span>', unsafe_allow_html=True)
    st.caption("Model: `mdhugol/indonesia-bert-sentiment-classification` — fine-tuned IndoBERT bahasa Indonesia (3 kelas: Positif, Netral, Negatif)")

    all_comments = df["comment"].tolist()
    all_labels = [
        f"[★{int(df.iloc[i]['rating'])}] {df.iloc[i]['name']} — "
        f"{c[:80]}{'...' if len(c)>80 else ''}"
        for i, c in enumerate(all_comments)
    ]

    sa1, sa2 = st.columns([3, 1])
    with sa1:
        selected_labels = st.multiselect(
            "Pilih komentar yang ingin dianalisis (maks. 10)",
            options=all_labels,
            default=all_labels[:min(5, len(all_labels))],
            max_selections=10,
            key="sent_select",
        )
    with sa2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_sent = st.button("🔍 Analisis Sentimen", use_container_width=True)

    if run_sent and selected_labels:
        sel_idx   = [all_labels.index(l) for l in selected_labels]
        sel_texts = [all_comments[i] for i in sel_idx]
        sel_rows  = [df.iloc[i] for i in sel_idx]

        with st.spinner("Memuat model IndoBERT & menganalisis..."):
            try:
                results = run_sentiment(sel_texts)

                # Ringkasan 3 kolom
                counts = Counter(r["label"] for r in results)
                s1, s2, s3 = st.columns(3)
                for col, lbl, emoji in [(s1,"Positif","😊"),(s2,"Netral","😐"),(s3,"Negatif","😞")]:
                    with col:
                        bg, fg = SENTIMENT_COLOR[lbl]
                        st.markdown(f"""
                        <div style="background:{bg};border:1px solid {fg}44;border-radius:14px;
                            padding:1.1rem;text-align:center;">
                            <div style="font-size:1.8rem;">{emoji}</div>
                            <div style="font-size:1.8rem;font-weight:800;color:{fg};">{counts.get(lbl,0)}</div>
                            <div style="font-size:0.72rem;color:{fg};font-weight:700;
                                text-transform:uppercase;letter-spacing:.05em;">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Detail per komentar
                for row_data, text, res in zip(sel_rows, sel_texts, results):
                    lbl   = res["label"]
                    score = res["score"]
                    bg, fg = SENTIMENT_COLOR[lbl]
                    emoji  = SENTIMENT_EMOJI[lbl]
                    st.markdown(f"""
                    <div class="sent-card" style="background:{bg}22;border-color:{fg}44;">
                        <div style="display:flex;justify-content:space-between;
                            align-items:flex-start;margin-bottom:6px;">
                            <span style="font-size:0.82rem;font-weight:700;color:#1e3a5f;">
                                {"★"*int(row_data["rating"])} {row_data["name"]}
                                <span style="color:#8faac8;font-weight:400;font-size:0.72rem;
                                    margin-left:6px;">{row_data["review_time"]}</span>
                            </span>
                            <span class="sent-label" style="background:{bg};color:{fg};
                                border:1px solid {fg}55;padding:3px 12px;border-radius:20px;
                                white-space:nowrap;">
                                {emoji} {lbl} · {score:.0f}%
                            </span>
                        </div>
                        <div class="sent-comment" style="color:#374151;">{text}</div>
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Gagal load model: {e}\n\n"
                         "Pastikan sudah install: `pip install transformers torch`")
    elif run_sent:
        st.warning("⚠️ Pilih minimal 1 komentar terlebih dahulu.")

    st.markdown('<hr style="border:none;border-top:2px solid #dde8ff;margin:1.5rem 0;">', unsafe_allow_html=True)

    # ── 4. TOPIC MODELING ─────────────────────────────────
    st.markdown('<div id="topik" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-title">🧠 Topic Modeling per Rating (LDA)</span>', unsafe_allow_html=True)
    st.caption(f"Jumlah topik: **{n_topics}** · Kata per topik: **{n_words}** · Ubah via sidebar tanpa scraping ulang")

    lda_tabs = st.tabs([f"★{r}  ({(df['rating']==r).sum()} review)" for r in avail_r])
    DONUT_COLORS = ["#3b82f6","#60a5fa","#93c5fd","#bfdbfe","#dbeafe","#eff6ff"]

    for ltab, rating in zip(lda_tabs, avail_r):
        with ltab:
            n_docs = (df["rating"] == rating).sum()
            if n_docs < 5:
                st.info(f"Review ★{rating} terlalu sedikit ({n_docs}) untuk topic modeling.")
                continue

            topics, topic_counts = run_lda(df_hash, rating, n_topics, n_words, df)
            if topics is None:
                st.info("Data tidak cukup untuk topic modeling pada rating ini.")
                continue

            lda_l, lda_r = st.columns([2, 1])
            with lda_l:
                for t in topics:
                    doc_cnt = topic_counts[t["topic"] - 1]
                    pills = "".join(
                        f'<span class="topic-pill">{w}'
                        f'<span style="opacity:.55;font-size:.62rem;margin-left:3px;">({s:.0f})</span>'
                        f'</span>'
                        for w, s in t["words"]
                    )
                    st.markdown(f"""
                    <div class="topic-card">
                        <div class="topic-title">Topik {t["topic"]}
                            <span style="font-size:.68rem;font-weight:500;color:#6b8cba;margin-left:8px;">
                                ~{doc_cnt} dokumen
                            </span>
                        </div>
                        <div>{pills}</div>
                    </div>""", unsafe_allow_html=True)

            with lda_r:
                fig_dn = go.Figure(go.Pie(
                    labels=[f"Topik {t['topic']}" for t in topics],
                    values=topic_counts, hole=0.55,
                    marker=dict(colors=DONUT_COLORS[:n_topics]),
                    textinfo="label+percent",
                    textfont=dict(size=11, color="#1e3a5f"),
                ))
                fig_dn.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#1e3a5f",
                    showlegend=False, margin=dict(t=10,b=10,l=10,r=10), height=260)
                st.plotly_chart(fig_dn, use_container_width=True)

    st.markdown('<hr style="border:none;border-top:2px solid #dde8ff;margin:1.5rem 0;">', unsafe_allow_html=True)

    # ── 5. TOP KATA ───────────────────────────────────────
    st.markdown('<div id="topkata" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="section-title">🔤 Top {top_n} Kata per Rating</span>'
        '<span style="font-size:.72rem;color:#6b8cba;margin-left:10px;">'
        '✨ cached — ubah slider bebas</span>',
        unsafe_allow_html=True,
    )

    word_tabs = st.tabs([f"★{r}  ({(df['rating']==r).sum()})" for r in avail_r])
    for wtab, rating in zip(word_tabs, avail_r):
        with wtab:
            wdf = compute_top_words(df_hash, rating, top_n, df)
            if wdf.empty:
                st.info("Tidak cukup data."); continue
            mf = wdf["frekuensi"].max()
            colors = [f"rgba(59,130,246,{0.3+0.7*(v/mf):.2f})" for v in wdf["frekuensi"]]
            fig_w = go.Figure(go.Bar(
                x=wdf["frekuensi"], y=wdf["kata"], orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=wdf["frekuensi"], textposition="outside",
                textfont=dict(color="#1e3a5f", size=11),
            ))
            fig_w.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#1e3a5f", margin=dict(t=10,b=10,l=10,r=60),
                height=max(320, top_n*28),
                xaxis=dict(gridcolor="#e8f0fe", title="Frekuensi"),
                yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_w, use_container_width=True)

    st.markdown('<hr style="border:none;border-top:2px solid #dde8ff;margin:1.5rem 0;">', unsafe_allow_html=True)

    # ── 6. ULASAN ─────────────────────────────────────────
    st.markdown('<div id="ulasan" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-title">💬 Semua Ulasan</span>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1, 1, 2], gap="medium")
    with f1:
        rating_filter = st.multiselect("Filter Rating", [5,4,3,2,1], default=[5,4,3,2,1],
                                       format_func=lambda x: f"★{x}")
    with f2:
        sort_by = st.selectbox("Urutkan", ["Default","Rating ↑","Rating ↓","Likes ↓"])
    with f3:
        keyword = st.text_input("🔍 Cari kata dalam komentar", placeholder="ketik kata kunci...")

    filtered = df[df["rating"].isin(rating_filter)].copy()
    if keyword:
        filtered = filtered[filtered["comment"].str.contains(keyword, case=False, na=False)]
    if sort_by == "Rating ↑":
        filtered = filtered.sort_values("rating", ascending=True)
    elif sort_by == "Rating ↓":
        filtered = filtered.sort_values("rating", ascending=False)
    elif sort_by == "Likes ↓":
        filtered = filtered.sort_values("like_count", ascending=False)

    st.caption(f"Menampilkan **{len(filtered)}** dari **{total}** review")

    for _, row in filtered.iterrows():
        st.markdown(f"""
        <div class="review-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <span class="rc-name">{row['name']}</span>
                    <span class="star-badge">{"★"*int(row['rating'])} {row['rating']}</span>
                </div>
                <span style="font-size:.73rem;color:#8faac8;">👍 {row['like_count']}</span>
            </div>
            <div class="rc-meta">{row['review_time']} · {row['variant']}</div>
            <div class="rc-comment">{row['comment']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("⬇️ Download CSV",
                       data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="reviews_tokopedia.csv", mime="text/csv")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;">
        <div style="background:linear-gradient(135deg,#dbeafe,#bfdbfe);width:80px;height:80px;
            border-radius:24px;display:inline-flex;align-items:center;justify-content:center;
            font-size:2.2rem;margin-bottom:1.2rem;
            box-shadow:0 4px 20px rgba(59,130,246,0.2);">🔍</div>
        <h3 style="font-weight:800;color:#1e3a5f;margin-bottom:0.5rem;">Belum ada data</h3>
        <p style="color:#6b8cba;font-size:0.88rem;max-width:380px;margin:0 auto;">
            Masukkan URL produk Tokopedia di sidebar,<br>
            atur konfigurasi, lalu klik <strong>Mulai Scraping</strong>.<br><br>
            <span style="font-size:0.75rem;color:#93c5fd;">
            Fitur: Statistik · Sentiment Analysis (IndoBERT) · Topic Modeling (LDA) · Top Kata
            </span>
        </p>
    </div>""", unsafe_allow_html=True)