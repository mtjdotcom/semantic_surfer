# 🏄 Semantic Surfer

An AI-powered portfolio search tool for Isomer Capital. This app uses **Semantic Search (Vector Embeddings)** to find companies in the portfolio based on "concept" rather than just keywords.

## 🚀 Features
* **Semantic Search:** "Uber for Dogs" finds relevant marketplace companies.
* **Bulk Analysis:** Upload a CSV to match hundreds of companies against the portfolio instantly.
* **Smart Caching:** Remembers previous searches in Google Sheets to save AI costs and speed up results.
* **Live Research:** Uses Gemini 2.0 + Google Search to research new companies on the fly.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **AI/Embeddings:** Google Gemini (GenAI)
* **Database:** Google Sheets (via `streamlit-gsheets`)
* **Vector Search:** Cosine Similarity (Scikit-Learn)

## 📦 Installation

1.  Clone the repo:
    ```bash
    git clone [https://github.com/mtjdotcom/semantic_surfer.git](https://github.com/mtjdotcom/semantic_surfer.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run searcher3.py
    ```

## 🔒 Secrets
This app requires a `.streamlit/secrets.toml` file with:
* `GOOGLE_API_KEY`: For Gemini & Embeddings.
* `[connections.gsheets]`: For Portfolio & Cache access.