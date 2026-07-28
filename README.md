# 🏄 Semantic Surfer

An AI-powered portfolio search tool for Isomer Capital. This app uses **Semantic Search (Vector Embeddings)** to find companies in the portfolio based on "concept" rather than just keywords.

## 🚀 Features
* **Semantic Search:** "Uber for Dogs" finds relevant marketplace companies.
* **Bulk Analysis:** Upload a CSV to match hundreds of companies against the portfolio instantly.
* **Smart Caching:** Remembers previous searches in Google Sheets to save AI costs and speed up results.
* **Live Research:** Uses Gemini 3.5 Flash-Lite + Google Search to research new companies on the fly.

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

## 🔄 Quarterly Data Updates

When you drop new portfolio data into the `portfolio` tab (e.g. Q1 2026), the
embeddings need rebuilding — the app searches the stored vectors, not the text,
so **an edited description with a stale vector will keep matching on its old
meaning**.

Run this after every upload:

```bash
python backfill_embeddings.py --dry-run   # see what it plans to do
python backfill_embeddings.py             # do it
```

The script works out what changed on its own. It re-embeds a row when the row
is new, when its description has been edited, when the model or dimensionality
changes, or when the stored vector is corrupt — and skips everything else, so a
quarterly update only pays for the rows that actually moved.

It writes **only** the `Embedding` and `Embedding Hash` columns, cell by cell.
Formatting, formulas and every other column are left untouched.

### Before you upload

* Keep the header row exactly as-is. The script matches columns **by name, not
  by position**, so reordering columns is fine but renaming one is not.
* Put descriptions in `New Long Description`. If a row hasn't got one, the
  script falls back to `New One Line Description`, then to `Name`.
* A row with none of those three is skipped entirely and stays invisible to
  search — the dry run lists them.
* Add new companies as **new rows**. Don't reorder existing rows mid-update.
* Don't hand-edit the `Embedding` or `Embedding Hash` columns.

### Useful flags

| Flag | What it does |
|---|---|
| `--dry-run` | Reports the plan. No API calls, no writes. Always start here. |
| `--limit 10` | Embeds only the first 10 rows needing work — good for a cheap sanity check. |
| `--force --yes` | Re-embeds **every** row. Only needed if you change the model or suspect the column is bad. |
| `--worksheet NAME` | Reads a different tab (default `portfolio`). |
| `--batch-size N` | Texts per API request (default 50). Lower it if you hit rate limits. |

### Afterwards

The app caches the sheet for 10 minutes, so wait for that to expire (or
restart the app) before checking results. The load banner shows the company
count — if it dropped, some rows failed to embed and were filtered out.

If a run dies partway through, just run it again. Finished rows are recorded as
they go and get skipped on the next pass.

## 🔒 Secrets
This app requires a `.streamlit/secrets.toml` file with:
* `GOOGLE_API_KEY`: For Gemini & Embeddings.
* `[connections.gsheets]`: For Portfolio & Cache access.