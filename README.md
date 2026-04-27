# 🛒 Tokopedia Review Analyzer

A powerful **end-to-end review analysis tool** that scrapes product reviews from Tokopedia and transforms them into actionable insights using **Machine Learning & NLP**.

Built with **Streamlit**, this app allows you to:

* Scrape reviews automatically
* Analyze sentiment (IndoBERT)
* Discover hidden topics (LDA)
* Visualize insights interactively

---

## 🚀 Features

### 🔍 Review Scraping

* Scrape langsung dari URL produk Tokopedia
* Auto-detect review section & pagination
* Support:

  * Multiple pages
  * Full scraping (all reviews)
* Extract:

  * Username
  * Rating ⭐
  * Comment 💬
  * Likes 👍
  * Variant & timestamp

---

### 📊 Review Analytics Dashboard

* Total reviews & average rating
* Rating distribution (1–5 stars)
* Total likes
* Interactive filtering & search

---

### 🤖 Sentiment Analysis (IndoBERT)

* Model: `mdhugol/indonesia-bert-sentiment-classification`
* Bahasa Indonesia (native NLP model)
* Output:

  * Positif 😊
  * Netral 😐
  * Negatif 😞
* Confidence score (%)
* Multi-select review analysis

---

### 🧠 Topic Modeling (LDA)

* Automatic topic extraction dari review
* Per rating (★1–★5)
* Menampilkan:

  * Top words per topic
  * Distribution of topics
* Menggunakan:

  * `CountVectorizer`
  * `LatentDirichletAllocation`

---

### 🔤 Top Keywords Analysis

* Kata paling sering muncul per rating
* Stopword Indonesia custom
* Visualisasi bar chart interaktif

---

### 💬 Review Explorer

* Filter:

  * Rating
  * Keyword
  * Sorting (rating / likes)
* Clean UI card-based display

---

### 📥 Export Data

* Download hasil scraping ke CSV

---

## 🧠 Tech Stack

* **Python**
* **Streamlit** → UI & dashboard
* **Playwright** → web scraping automation
* **BeautifulSoup** → HTML parsing
* **Pandas & NumPy** → data processing
* **Plotly** → interactive visualization
* **Scikit-learn** → LDA topic modeling
* **Transformers (HuggingFace)** → IndoBERT sentiment model
* **PyTorch** → backend ML model 

---

## ⚙️ Installation

### 1. Clone Repo

```bash id="b1b8ot"
git clone https://github.com/your-username/tokopedia-review-analyzer.git
cd tokopedia-review-analyzer
```

---

### 2. Create Virtual Environment

```bash id="q5g2lw"
python -m venv venv
```

### Activate:

**Windows:**

```bash id="x4x4rz"
venv\Scripts\activate
```

**Mac/Linux:**

```bash id="0fxk7x"
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash id="j2s8p3"
pip install -r requirements.txt
```

---

### 4. Install Playwright Browser

```bash id="m7z9h1"
playwright install chromium
```

---

### 5. Run App

```bash id="2d9f7k"
streamlit run app.py
```

---

## 🖥️ Usage

### Mode 1: Scraping Langsung (Local)

1. Masukkan URL produk Tokopedia
2. Tentukan jumlah halaman / scrape all
3. Klik **Mulai Scraping**

---

### Mode 2: Upload CSV (Cloud-Friendly)

1. Jalankan scraping di lokal
2. Upload file CSV ke app
3. Langsung analisis

---

## 📂 Output Data Format

CSV hasil scraping memiliki struktur:

| Column      | Description   |
| ----------- | ------------- |
| name        | Nama user     |
| rating      | Rating (1–5)  |
| comment     | Isi review    |
| like_count  | Jumlah like   |
| variant     | Varian produk |
| review_time | Waktu review  |

---

## 🎯 Use Cases

* 📊 Data Analyst Portfolio
* 🛒 E-commerce Insight Analysis
* 🧠 NLP & Text Mining Project
* 📈 Customer Feedback Analysis
* 🤖 Machine Learning Showcase

---

## 🔥 Highlights

* End-to-end pipeline:

  * Scraping → Processing → NLP → Visualization
* Bahasa Indonesia NLP (IndoBERT)
* UI modern & interaktif
* Bisa jalan lokal & cloud
* Real-world use case (Tokopedia)

---

## ⚠️ Notes

* Scraping hanya bisa dijalankan di **local environment**
* Untuk deployment (Streamlit Cloud), gunakan mode **upload CSV**
* Pastikan install:

  ```bash id="8r3t7k"
  playwright install chromium
  ```

---

## 👨‍💻 Author

**Rizki Widianto**

---

## 📬 Future Improvements

* Auto scraping scheduler
* Wordcloud visualization
* Sentiment per aspect (aspect-based sentiment)
* Deploy API version (FastAPI)
* Dashboard multi-product comparison

---
