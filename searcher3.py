import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import gspread
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_gsheets import GSheetsConnection
import time
from google_auth_oauthlib.flow import Flow
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="Semantic Surfer", layout="wide")

# --- MODELS ---
# Research model. Safe to change on its own - it only affects the wording of
# the AI summaries, not the stored portfolio vectors.
RESEARCH_MODEL = "gemini-3.5-flash-lite"

# Embedding model. These MUST stay in step with backfill_embeddings.py.
# Queries are compared against the vectors stored in the sheet, and vectors
# from two different models are not comparable - the maths still runs, it just
# returns meaningless similarities. If you change either value, re-run
# backfill_embeddings.py to rebuild the whole sheet before trusting results.
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768

# --- PORTFOLIO SHEET SCHEMA ---
# Column headers on the 'portfolio' tab. Note these differ from the headers on
# the 'cache' tab and from the CSV the Bulk tab expects, both of which use
# 'Company Name' - don't unify them without checking all three.
PORTFOLIO_NAME_COL = "Name"
PORTFOLIO_DESC_COL = "New One Line Description"

# Tag columns offered as filters on the By Sector tab. Any that are missing
# from the sheet are simply not shown.
SECTOR_FILTER_COLS = ["Sector", "Granular Tag", "Technology Tag", "Business Model"]

# --- AUTHENTICATION LOGIC (THE GATEKEEPER) ---
ALLOWED_DOMAIN = "@isomercapital.com"


def check_authentication():
    """
    Gate the app behind a Google sign-in restricted to ALLOWED_DOMAIN.

    Every path out of here either returns True for an authenticated session or
    ends in st.stop(). Returning False is not enough: the caller runs at module
    level, so anything short of stopping lets the rest of the page render to
    whoever triggered the failure.
    """
    # 1. If already authenticated in this session, pass
    if st.session_state.get("auth_status") == "authenticated":
        return True

    # 2. Setup Google OAuth Flow
    try:
        client_config = {
            "web": {
                "client_id": st.secrets["oauth"]["client_id"],
                "client_secret": st.secrets["oauth"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["oauth"]["redirect_uri"]],
            }
        }
        flow = Flow.from_client_config(
            client_config,
            scopes=["https://www.googleapis.com/auth/userinfo.email", "openid"],
            redirect_uri=st.secrets["oauth"]["redirect_uri"],
        )

        # Disable PKCE. The library generates a code_verifier while building
        # the sign-in URL and needs the same value back when exchanging the
        # code - but those happen in two different script runs, and the trip
        # out to Google returns as a fresh page load with new session state.
        # The verifier is therefore always gone by the time we exchange, and
        # Google rejects it with 'invalid_grant: Missing code verifier'.
        # This is a confidential client, so the client_secret already
        # authenticates the exchange. Set on the instance rather than passed
        # as a constructor argument so it holds across library versions.
        flow.autogenerate_code_verifier = False
        flow.code_verifier = None
    except Exception as e:
        # Misconfigured secrets - we cannot even offer a login, so stop dead
        # rather than falling through to the app.
        st.error(f"Sign-in is not configured correctly: {e}")
        st.stop()

    # 3. Handle the Return Trip (Exchange Code for Token)
    email = ""
    auth_code = st.query_params.get("code")

    if auth_code:
        try:
            flow.fetch_token(code=auth_code)
            user_info = requests.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                headers={"Authorization": f"Bearer {flow.credentials.token}"},
                timeout=10,
            ).json()
            email = user_info.get("email", "")
        except Exception as e:
            # An authorisation code can only be exchanged once. The usual cause
            # is a page refresh or an app restart replaying a code that has
            # already been used, which surfaces as 'invalid_grant'. Drop the
            # stale code so the retry below starts a clean sign-in.
            st.query_params.clear()
            st.warning(f"Sign-in could not be completed: {e}")
            st.caption("This usually means the sign-in link was already used. "
                       "Please sign in again.")

        # 4. THE DOMAIN CHECK
        if email:
            if email.endswith(ALLOWED_DOMAIN):
                st.session_state["auth_status"] = "authenticated"
                st.session_state["user_email"] = email
                st.query_params.clear()
                st.rerun()
            else:
                st.session_state["auth_status"] = "failed"
                st.query_params.clear()
                st.error(f"Access Denied. {email} is not an Isomer Capital account.")

    # 5. Not authenticated. Offer sign-in and HALT - nothing below this runs.
    auth_url, _ = flow.authorization_url(prompt="consent")

    st.title("🔒 Semantic Surfer Access")
    st.markdown("Please sign in with your **Isomer Capital** Google account.")

    st.link_button("Sign in with Google", auth_url, type="primary")
    st.stop()


# --- RUN THE CHECK IMMEDIATELY ---
# check_authentication() either returns True or halts the script, so nothing
# below runs for an unauthenticated visitor.
check_authentication()

# Display the logged-in user (Optional nice touch)
st.sidebar.caption(f"Logged in as: {st.session_state.get('user_email')}")
if st.sidebar.button("Logout"):
    st.session_state["auth_status"] = None
    st.rerun()

if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("GOOGLE_API_KEY not found in secrets.")
    st.stop()

# --- OPTIMIZED DATA LOADING ---
@st.cache_data(ttl=600)
def load_portfolio():
    # 1. Load Data from Google Sheets
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet="portfolio") # Or "Sheet1" if that's what you used
    
    # 2. Check if Embeddings exist
    if 'Embedding' not in df.columns:
        st.error("Column 'Embedding' not found! Did you run the backfill script?")
        st.stop()

    # 3. Parse JSON Strings back to Vectors
    # The sheet stores "[0.1, -0.5]" as a string. We need to turn it back into a list.
    try:
        # Fill empty cells to avoid errors
        df['Embedding'] = df['Embedding'].fillna("[]")
        
        # Filter out rows with empty brackets or invalid data
        valid_mask = df['Embedding'].str.len() > 5 
        df = df[valid_mask].copy()
        
        # Apply JSON parsing
        df['Vector'] = df['Embedding'].apply(json.loads)
        
        # Convert to Numpy Matrix for Math
        embeddings_matrix = np.array(df['Vector'].tolist())
        
        return df, embeddings_matrix
        
    except Exception as e:
        st.error(f"Error parsing embeddings from Sheet: {e}")
        st.stop()

# --- CACHE SYSTEM ---
@st.cache_data(ttl=600)
def load_cache():
    """Loads previous research to avoid re-running Gemini."""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # We use a separate tab called 'Cache'
        df = conn.read(worksheet="cache")
        
        # If empty or just created, return empty structures
        if df.empty or 'Embedding' not in df.columns:
            return pd.DataFrame(), np.array([])
            
        # Parse Vectors (Assuming they are stored as JSON strings)
        df['Embedding'] = df['Embedding'].fillna("[]")
        df['Vector'] = df['Embedding'].apply(json.loads)
        
        # Filter invalid rows
        valid_mask = df['Vector'].apply(len) > 0
        df = df[valid_mask]
        
        if not df.empty:
            vectors = np.array(df['Vector'].tolist())
            return df, vectors
            
        return pd.DataFrame(), np.array([])
        
    except Exception:
        # Fail silently if Cache tab doesn't exist yet
        return pd.DataFrame(), np.array([])

def save_to_cache(company_name, research_text, query_vector):
    """Saves new research to the Google Sheet using atomic append."""
    try:
        # 1. Connect directly to GSpread (This bypasses Streamlit's read cache)
        # We reuse the credentials already inside st.secrets
        gc = gspread.service_account_from_dict(st.secrets["connections"]["gsheets"])
        sh = gc.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        worksheet = sh.worksheet("cache")
        
        # 2. Prepare the data
        # We must convert the numpy vector -> list -> JSON string
        vector_str = json.dumps(query_vector.tolist()[0])
        
        # 3. Append the row to the bottom (Atomic & Safe)
        # This will never overwrite existing data
        worksheet.append_row([company_name, research_text, vector_str])
        
        # 4. Clear Streamlit's RAM cache 
        # This forces the app to re-download the Cache tab next time you search
        load_cache.clear()
        
    except Exception as e:
        st.warning(f"Could not save to cache: {e}")


def check_semantic_cache(new_query_name, new_query_vector, cache_df, cache_vectors, threshold=0.92):
    """Checks cache for exact name match FIRST, then falls back to vector similarity."""
    
    # 1. Handle empty cache immediately
    if cache_df.empty:
        return None
        
    # --- CHECK 1: EXACT NAME MATCH (Fast & 100% Accurate) ---
    # This prevents duplicates like "Ark Robotics" vs "Ark Robotics "
    clean_query = str(new_query_name).lower().strip()
    
    if 'Company Name' in cache_df.columns:
        # Check if any existing row matches our query
        matches = cache_df[cache_df['Company Name'].str.lower().str.strip() == clean_query]
        if not matches.empty:
            # Return the research from the most recent entry
            return matches.iloc[-1]['Research']

    # --- CHECK 2: SEMANTIC VECTOR MATCH (Fuzzy Backup) ---
    # If the name was spelled differently, use vectors
    if cache_vectors.size > 0:
        # We must validate that new_query_vector is actually an array, not a string
        if isinstance(new_query_vector, (str, type(None))):
            return None
            
        scores = cosine_similarity(new_query_vector, cache_vectors)[0]
        best_idx = np.argmax(scores)
        
        if scores[best_idx] > threshold:
            return cache_df.iloc[best_idx]['Research']
        
    return None

def analyze_deal(company_name, company_url, portfolio_df, portfolio_vectors, precomputed_research=None, top_n=3):
    """
    Analyzes a deal using Hybrid Search (Vector + Keyword) and Caching.

    Args:
        precomputed_research (str, optional): If provided (from Cache), we skip the Gemini API call.
        top_n (int): How many portfolio matches to return (clamped to the portfolio size).
    """
    
    # --- PHASE 1: RESEARCH ---
    # If we found this in the Cache, use it and save $$
    if precomputed_research:
        profile_text = precomputed_research
    else:
        # Otherwise, run the expensive Gemini API call
        prompt = f"""
        You are a Venture Capital Analyst. Research {company_name}: (website: {company_url}).
        1. Summarize what they do in 2 sentences.
        2. Identify their sector.
        """
        
        try:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            response = client.models.generate_content(
                model=RESEARCH_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"]
                )
            )
            profile_text = response.text
        except Exception as e:
            return {"error": f"Research failed: {e}"}

    # --- PHASE 2: EMBEDDING ---
    # We always need to embed the profile text (whether cached or new) to do the math
    try:
        embed_response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=profile_text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIMS)
        )
        new_vector = np.array(embed_response.embeddings[0].values).reshape(1, -1)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    # --- PHASE 3: HYBRID SCORING (Math + Keyword Boost) ---
    # 1. Base Score: Cosine Similarity
    scores = cosine_similarity(new_vector, portfolio_vectors)[0]
    
    # 2. Boost Score: Exact Keyword Match
    # If the user searches "Stripe", and we have "Stripe" in the portfolio, 
    # we force the score to be higher, even if the description is vague.
    search_clean = str(company_name).lower().strip()
    
    for idx, row in portfolio_df.iterrows():
        # Handle cases where index might not match row number due to filtering
        numeric_idx = portfolio_df.index.get_loc(idx)
        
        # safely get name
        p_name = str(row.get(PORTFOLIO_NAME_COL, '')).lower().strip()
        
        # Rule A: Exact Match -> Max Score
        if search_clean == p_name and p_name != "":
            scores[numeric_idx] = 1.0 
            
        # Rule B: Partial Match -> 15% Boost (e.g. "Stripe" vs "Stripe Inc")
        elif (search_clean in p_name or p_name in search_clean) and len(search_clean) > 3:
            scores[numeric_idx] = min(scores[numeric_idx] + 0.15, 0.99)

    # --- PHASE 4: RANKING ---
    # Get indices of the top N scores, sorted descending
    top_n = max(1, min(int(top_n), len(scores)))
    top_indices = np.argsort(scores)[-top_n:][::-1]
    
    matches = []
    for idx in top_indices:
        row = portfolio_df.iloc[idx]
        matches.append({
            "Company": row.get(PORTFOLIO_NAME_COL, 'Unknown Company'),
            "Similarity": scores[idx],
            "Status": row.get('Status', 'Unknown'),
            "Multiple": row.get('Multiple', '-'),
            "Partner": row.get('Partner VC - CList', 'N/A'),
            "Fund": row.get('Isomer Fund', 'N/A'),
            "Website": row.get('Website', ''),
            "Description": row.get(PORTFOLIO_DESC_COL, '')
        })

    return {
        "Company": company_name,
        "Research": profile_text,
        "Matches": matches,
        "error": None
    }


def display_match_cards(results):
    """Helper to display the top matches consistently across tabs."""
    matches = results.get('Matches')
    if not matches:
        st.warning("No matches found.")
        return

    st.markdown(f"### 🎯 Top {len(matches)} Portfolio Matches")

    # Medals for the podium, plain numbering from 4th place on
    medals = ["🥇", "🥈", "🥉"] + [f"#{n}" for n in range(4, len(matches) + 1)]

    for i, match in enumerate(matches):
        with st.container(border=True):
            
            # --- NEW "ONE-LINER" LOGIC ---
            # 1. Prepare the Website Link Variable
            website_html = "" # Default is empty
            url = match.get('Website')
            
            if url and isinstance(url, str) and len(url.strip()) > 0:
                clean_url = url.strip()
                if not clean_url.startswith('http'):
                    clean_url = f"https://{clean_url}"
                
                # We create a small, clickable HTML link that sits right next to the name
                # target='_blank' ensures it opens in a new tab
                website_html = f"&nbsp; <a href='{clean_url}' target='_blank' style='font-size: 0.9rem; font-weight: normal; vertical-align: middle;'>🌐 Visit Site</a>"
            
            # 2. Display Name + Link together
            # "###" makes the name big. The HTML link stays small next to it.
            st.markdown(f"### {medals[i]} {match['Company']}{website_html}", unsafe_allow_html=True)
            # -----------------------------

            # Row 1: Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Similarity", f"{match['Similarity']*100:.1f}%")
            with c2:
                st.metric("Status", match['Status'])
            with c3:
                st.metric("Multiple", match['Multiple'])
            
            st.divider()
            
            # Row 2: Attribution
            c4, c5 = st.columns(2)
            with c4:
                st.metric("Partner VC", match['Partner'])
            with c5:
                st.metric("Isomer Fund", match['Fund'])

def tag_tokens(cell):
    """
    Splits a tag cell into individual tags.

    Cells may hold one value or several ("Fintech; Payments"), so we split on
    common delimiters and compare whole tags, never substrings - selecting
    'AI' must not match 'Retail'.
    """
    if not isinstance(cell, str):
        return []
    return [p.strip() for p in re.split(r"[;,/|]", cell) if p.strip()]


def build_tag_vocabulary(df):
    """Unique tags per filter column, for the multiselect options."""
    vocab = {}
    for col in SECTOR_FILTER_COLS:
        if col in df.columns:
            tags = {t for cell in df[col] for t in tag_tokens(cell)}
            if tags:
                vocab[col] = sorted(tags, key=str.lower)
    return vocab


# --- UI ---
st.title("🏄🏄‍♀️ Semantic Surfer 🔍︎")
st.caption("The semantic surfer will surf through the Isomer portfolio to find companies that are most similar to the company or companies you are researching. You can search one company at a time on the Single Screen tab. Move to the Custom Search tab to search by concept. Or use the bulk analysis tab to search through a CSV of names and URLs. An AI-generated summary is produced for each company searched (i.e., it's not free, but it's very cheap)")

# Load Data
with st.spinner("Loading Databases..."):
    # 1. CRITICAL: Load Portfolio (Must succeed)
    try:
        df_portfolio, portfolio_vectors = load_portfolio()
        st.success(f"Loaded {len(df_portfolio)} companies.")
    except Exception as e:
        st.error(f"CRITICAL ERROR: Could not load Portfolio. {e}")
        st.stop() # Stop app only if portfolio fails

    # 2. OPTIONAL: Load Cache (Can fail gracefully)
    try:
        df_cache, cache_vectors = load_cache()
        # Optional: Show a small popup toast instead of a big green bar
        if not df_cache.empty:
            st.toast(f"Memory loaded: {len(df_cache)} previous searches", icon="🧠")
    except Exception as e:
        # If cache fails, just use empty data and keep going
        st.warning(f"Cache unavailable ({e}). Running in fresh mode.")
        df_cache = pd.DataFrame()
        cache_vectors = np.array([])

tab_single, tab_custom, tab_sector, tab_bulk = st.tabs(
    ["🔎 Single Screen", "📝 Custom Search", "🏷️ By Sector", "📂 Bulk Upload"])

# TAB 1: Single Search
with tab_single:
    st.header("Search by Company")
    st.caption("Add a company name and company URL below. Our semantic surfers will search to find the most similar Isomer portfolio companies. ")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        s_company = st.text_input("Company Name", placeholder="e.g. Stripe")
    with col2:
        s_url = st.text_input("URL", placeholder="e.g. stripe.com")
    with col3:
        s_top_n = st.number_input("Results", min_value=1, max_value=25, value=3, step=1,
                                  help="How many portfolio matches to show", key="single_top_n")

    if st.button("Search Isomer Portfolio", type="primary"):
        if not s_company:
            st.warning("Please enter a company name.")
        else:
            with st.spinner(f"Analyzing {s_company}..."):
                
                # --- STEP 1: EMBED THE QUERY (Required for Cache Check) ---
                try:
                    # We embed the company name to see if it matches previous searches
                    q_resp = client.models.embed_content(
                        model=EMBED_MODEL,
                        contents=s_company,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIMS)
                    )
                    query_vector = np.array(q_resp.embeddings[0].values).reshape(1, -1)
                except Exception as e:
                    st.error(f"Embedding error: {e}")
                    st.stop()

                # --- STEP 2: CHECK CACHE ---
                # Check if we have researched this company before
                cached_research = check_semantic_cache(s_company, query_vector, df_cache, cache_vectors)
                
                from_cache = False
                research_text = None

                if cached_research:
                    st.success(f"⚡ Cache Hit! Loaded previous research for {s_company}")
                    research_text = cached_research
                    from_cache = True
                else:
                    st.caption("🤖 New company detected. Agent researching live...")

                # --- STEP 3: RUN ANALYSIS ---
                # We pass 'research_text' (if found) to skip the Gemini generation step
                res = analyze_deal(s_company, s_url, df_portfolio, portfolio_vectors, precomputed_research=research_text, top_n=s_top_n)
                
                # --- STEP 4: SAVE TO CACHE (If it was new) ---
                if not from_cache and not res.get('error'):
                    # Save the new research and vector to the Google Sheet
                    save_to_cache(s_company, res['Research'], query_vector)

                if res.get('error'):
                    st.error(res['error'])
                else:
                    if not from_cache:
                        st.success("Analysis Complete")
                    
                    # USE THE HELPER FUNCTION (This activates your Website Link logic)
                    display_match_cards(res)

                    # --- Research Section (At the bottom) ---
                    st.markdown("### 📝 AI Research Summary")
                    st.markdown(res['Research'])                            


# --- TAB 2: CUSTOM DESCRIPTION (NEW) ---
with tab_custom:
    st.header("Search by Concept")
    st.caption("Paste a pitch, a thesis, or a raw description to find similar existing companies in the portfolio.")
    
    # Simple inputs
    c_left, c_right = st.columns([4, 1])
    with c_left:
        custom_name = st.text_input("Project Label (Optional)", placeholder="e.g. 'Uber for Dogs'")
    with c_right:
        c_top_n = st.number_input("Results", min_value=1, max_value=25, value=3, step=1,
                                  help="How many portfolio matches to show", key="custom_top_n")
    custom_desc = st.text_area("Description / Thesis", height=150,
                              placeholder="A marketplace connecting pet owners with walkers on demand...")

    if st.button("Find Matches", type="primary"):
        if not custom_desc:
            st.warning("Please enter a description.")
        else:
            with st.spinner("Embedding and matching..."):
                # We reuse analyze_deal!
                # By passing 'precomputed_research', we skip the Gemini research step 
                # and go straight to embedding your text.
                res = analyze_deal(
                    company_name=custom_name if custom_name else "Custom Search",
                    company_url="",
                    portfolio_df=df_portfolio,
                    portfolio_vectors=portfolio_vectors,
                    precomputed_research=custom_desc, # <--- MAGIC TRICK
                    top_n=c_top_n
                )
                
                if res.get('error'):
                    st.error(res['error'])
                else:
                    st.success("Search Complete")
                    display_match_cards(res)

# --- TAB 3: BY SECTOR ---
with tab_sector:
    st.header("Browse by Sector")
    st.caption("Pick one or more tags to list every matching portfolio company - "
               "this searches the sheet's own tags, so it is exact and free. "
               "Optionally add a concept to rank the results by similarity.")

    vocab = build_tag_vocabulary(df_portfolio)

    if not vocab:
        st.warning("None of the tag columns "
                   f"({', '.join(SECTOR_FILTER_COLS)}) were found on the sheet.")
    else:
        selections = {}
        for widget_col, col in zip(st.columns(len(vocab)), vocab):
            with widget_col:
                chosen = st.multiselect(col, vocab[col], key=f"sector_{col}")
            if chosen:
                selections[col] = set(chosen)

        sector_concept = st.text_input(
            "Rank by concept (optional)",
            placeholder="e.g. applying LLMs to drug discovery",
            help="Leave blank for an alphabetical list. Fill in to sort the "
                 "filtered companies by semantic similarity (one AI call).")

        if st.button("List Companies", type="primary"):
            if not selections and not sector_concept.strip():
                st.warning("Pick at least one tag, or enter a concept to rank by.")
            else:
                # Within a column, any selected tag matches (OR). Across
                # columns, all filters must hold (AND).
                mask = pd.Series(True, index=df_portfolio.index)
                for col, wanted in selections.items():
                    mask &= df_portfolio[col].apply(
                        lambda cell, w=wanted: bool(w & set(tag_tokens(cell))))

                filtered = df_portfolio[mask]

                if filtered.empty:
                    st.warning("No companies match that combination of tags. "
                               "Try removing a filter.")
                else:
                    # Optional semantic ranking of the filtered set
                    if sector_concept.strip():
                        try:
                            q_resp = client.models.embed_content(
                                model=EMBED_MODEL,
                                contents=sector_concept,
                                config=types.EmbedContentConfig(
                                    task_type="RETRIEVAL_QUERY",
                                    output_dimensionality=EMBED_DIMS)
                            )
                            q_vec = np.array(q_resp.embeddings[0].values).reshape(1, -1)
                            # portfolio_vectors rows align positionally with
                            # df_portfolio rows, so a positional mask is safe
                            sub_vectors = portfolio_vectors[mask.to_numpy()]
                            sims = cosine_similarity(q_vec, sub_vectors)[0]
                            filtered = filtered.assign(Similarity=sims)
                            filtered = filtered.sort_values("Similarity", ascending=False)
                        except Exception as e:
                            st.warning(f"Ranking failed ({e}) - showing an "
                                       "alphabetical list instead.")
                            filtered = filtered.sort_values(PORTFOLIO_NAME_COL)
                    else:
                        filtered = filtered.sort_values(PORTFOLIO_NAME_COL)

                    st.success(f"{len(filtered)} matching companies.")

                    # Assemble the display table from whatever columns exist
                    out = pd.DataFrame(index=filtered.index)
                    out["Name"] = filtered[PORTFOLIO_NAME_COL]
                    if "Similarity" in filtered.columns:
                        out["Similarity"] = filtered["Similarity"].apply(
                            lambda x: f"{x*100:.1f}%")
                    for col in [PORTFOLIO_DESC_COL] + SECTOR_FILTER_COLS + [
                            "Status", "Multiple", "Isomer Fund",
                            "Partner VC - CList"]:
                        if col in filtered.columns:
                            out[col] = filtered[col]
                    if "Website" in filtered.columns:
                        out["Website"] = filtered["Website"].apply(
                            lambda u: (u.strip() if str(u).strip().startswith("http")
                                       else f"https://{str(u).strip()}")
                            if isinstance(u, str) and str(u).strip() else "")

                    st.dataframe(
                        out,
                        column_config={
                            "Website": st.column_config.LinkColumn("Website"),
                        },
                        hide_index=True,
                        width="stretch",
                    )

                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="isomer_sector_results.csv",
                        mime="text/csv",
                        type="primary",
                    )

# --- TAB 4: BULK UPLOAD ---
with tab_bulk:
    st.header("📂 Bulk Analysis")
    st.caption("Upload a CSV with columns: 'Company Name' and 'URL'.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        
        # Clean column names
        df_upload.columns = df_upload.columns.str.strip()
        
        if "Company Name" in df_upload.columns and "URL" in df_upload.columns:
            
            if st.button("Run Bulk Analysis", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                # --- NEW: SESSION MEMORY ---
                # Create a set of names we ALREADY know from the loaded cache
                # We normalize them (lowercase, stripped) to ensure matching works
                if not df_cache.empty and 'Company Name' in df_cache.columns:
                    session_known_names = set(df_cache['Company Name'].str.lower().str.strip().tolist())
                else:
                    session_known_names = set()
                # ---------------------------

                # Iterate through the uploaded companies
                for i, row in df_upload.iterrows():
                    company_input = row['Company Name']
                    url_input = row['URL']
                    
                    # Normalize input name for checking
                    clean_input_name = str(company_input).lower().strip()
                    
                    # --- CACHE LOGIC START ---
                    
                    # 1. Embed the Company Name (Needed for Vector Check)
                    query_vector = None
                    try:
                        q_resp = client.models.embed_content(
                            model=EMBED_MODEL,
                            contents=company_input,
                            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIMS)
                        )
                        query_vector = np.array(q_resp.embeddings[0].values).reshape(1, -1)
                    except Exception:
                        pass

                    # 2. Check Cache (Did we research this before?)
                    cached_research = None
                    
                    # FAST CHECK: Have we seen this name in this session or the loaded cache?
                    if clean_input_name in session_known_names:
                        # If we know the name, we try to grab the research from the existing dataframe
                        # (Note: If it was added *just now* in the loop, we might not have the text handy, 
                        # so we might still skip the save but might need to re-run analysis if we didn't store the text.
                        # For simplicity, if it's a duplicate name, we assume we don't need to re-save it.)
                        
                        # Use the function to get the actual text if available
                        if query_vector is not None:
                            cached_research = check_semantic_cache(company_input, query_vector, df_cache, cache_vectors)
                            
                        # If check_semantic_cache returned None (because the dataframe is stale),
                        # but 'clean_input_name' IS in 'session_known_names', it means we processed it 
                        # moments ago. We treat this as a "Soft Hit" - we won't save it again.
                        already_processed_in_session = True
                    else:
                        # Regular check for old data
                        if query_vector is not None:
                            cached_research = check_semantic_cache(company_input, query_vector, df_cache, cache_vectors)
                        already_processed_in_session = False
                    
                    # --- CACHE LOGIC END ---

                    # 3. Run Analysis
                    res = analyze_deal(
                        company_name=company_input, 
                        company_url=url_input, 
                        portfolio_df=df_portfolio, 
                        portfolio_vectors=portfolio_vectors,
                        precomputed_research=cached_research 
                    )
                    
                    # 4. Save to Cache (ONLY if it's new AND we haven't processed it this session)
                    if not cached_research and not res.get('error') and query_vector is not None:
                        if not already_processed_in_session:
                            save_to_cache(company_input, res['Research'], query_vector)
                            
                            # CRITICAL: Add to session memory so next loop iteration knows!
                            session_known_names.add(clean_input_name)
                    
                    # 5. Handle Errors & Results
                    if res.get('error'):
                        results.append({
                            "Uploaded Name": company_input,
                            "Match Status": "Error",
                            "Error Details": res['error']
                        })
                    else:
                        best_match = res['Matches'][0] if res['Matches'] else {}
                        
                        # Clean the URL
                        raw_url = best_match.get('Website', '')
                        clean_url = ""
                        if raw_url and isinstance(raw_url, str) and len(raw_url.strip()) > 0:
                            clean_url = raw_url.strip()
                            if not clean_url.startswith('http'):
                                clean_url = f"https://{clean_url}"

                        results.append({
                            "Uploaded Name": company_input,           
                            "Top Match": best_match.get('Company', 'None'), 
                            "Website": clean_url, 
                            "Similarity": best_match.get('Similarity', 0.0),
                            "Status": best_match.get('Status', '-'),
                            "Multiple": best_match.get('Multiple', '-'),
                            "Partner VC": best_match.get('Partner', '-'),
                            "Isomer Fund": best_match.get('Fund', '-')
                        })
                    
                    progress_bar.progress((i + 1) / len(df_upload))
                
                # 6. Display Results
                st.success("Bulk Analysis Complete!")
                
                result_df = pd.DataFrame(results)
                
                display_df = result_df.copy()
                if "Similarity" in display_df.columns:
                    display_df['Similarity'] = display_df['Similarity'].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else x)
                
                st.dataframe(
                    display_df,
                    column_config={
                        "Website": st.column_config.LinkColumn("Website"),
                    },
                    width="stretch"
                )
                
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="isomer_bulk_results.csv",
                    mime="text/csv",
                    type="primary"
                )
        else:
            st.error("CSV Error: Your file must have columns named exactly 'Company Name' and 'URL'.")
            st.write("Found columns:", list(df_upload.columns))                        