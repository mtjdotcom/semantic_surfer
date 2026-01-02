import streamlit as st
import pandas as pd
import numpy as np
import json
import gspread
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_gsheets import GSheetsConnection
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Semantic Surfer", layout="wide")

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

def analyze_deal(company_name, company_url, portfolio_df, portfolio_vectors, precomputed_research=None):
    """
    Analyzes a deal using Hybrid Search (Vector + Keyword) and Caching.
    
    Args:
        precomputed_research (str, optional): If provided (from Cache), we skip the Gemini API call.
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
                model='gemini-2.0-flash', 
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
            model='text-embedding-004',
            contents=profile_text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
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
        p_name = str(row.get('Company Name', '')).lower().strip()
        
        # Rule A: Exact Match -> Max Score
        if search_clean == p_name and p_name != "":
            scores[numeric_idx] = 1.0 
            
        # Rule B: Partial Match -> 15% Boost (e.g. "Stripe" vs "Stripe Inc")
        elif (search_clean in p_name or p_name in search_clean) and len(search_clean) > 3:
            scores[numeric_idx] = min(scores[numeric_idx] + 0.15, 0.99)

    # --- PHASE 4: RANKING ---
    # Get indices of top 3 scores, sorted descending
    top_indices = np.argsort(scores)[-3:][::-1]
    
    matches = []
    for idx in top_indices:
        row = portfolio_df.iloc[idx]
        matches.append({
            # Note: Switched back to 'Company Name' to match your sheet schema
            "Company": row.get('Company Name', 'Unknown Company'),
            "Similarity": scores[idx],
            "Status": row.get('Status', 'Unknown'),
            "Multiple": row.get('Multiple', '-'),
            "Partner": row.get('Partner VC - CList', 'N/A'),
            "Fund": row.get('Isomer Fund', 'N/A'),
            "Website": row.get('Website', ''),
            "Description": row.get('Description', '')
        })

    return {
        "Company": company_name,
        "Research": profile_text,
        "Matches": matches,
        "error": None
    }


def display_match_cards(results):
    """Helper to display the Top 3 matches consistently across tabs."""
    if not results.get('Matches'):
        st.warning("No matches found.")
        return

    st.markdown("### 🎯 Top 3 Portfolio Matches")
    
    medals = ["🥇", "🥈", "🥉"]
    
    for i, match in enumerate(results['Matches']):
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

# tab_single, tab_bulk = st.tabs(["🔎 Single Screen", "📂 Bulk Upload"])
tab_single, tab_custom, tab_bulk = st.tabs(["🔎 Single Screen", "📝 Custom Search", "📂 Bulk Upload"])

# TAB 1: Single Search
with tab_single:
    st.header("Search by Company")
    st.caption("Add a company name and company URL below. Our semantic surfers will search to find the three most similar Isomer portfolio companies. ")

    col1, col2 = st.columns(2)
    with col1:
        s_company = st.text_input("Company Name", placeholder="e.g. Stripe")
    with col2:
        s_url = st.text_input("URL", placeholder="e.g. stripe.com")       

    if st.button("Search Isomer Portfolio", type="primary"):
        if not s_company:
            st.warning("Please enter a company name.")
        else:
            with st.spinner(f"Analyzing {s_company}..."):
                
                # --- STEP 1: EMBED THE QUERY (Required for Cache Check) ---
                try:
                    # We embed the company name to see if it matches previous searches
                    q_resp = client.models.embed_content(
                        model='text-embedding-004',
                        contents=s_company,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
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
                res = analyze_deal(s_company, s_url, df_portfolio, portfolio_vectors, precomputed_research=research_text)
                
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
    custom_name = st.text_input("Project Label (Optional)", placeholder="e.g. 'Uber for Dogs'")
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
                    precomputed_research=custom_desc # <--- MAGIC TRICK
                )
                
                if res.get('error'):
                    st.error(res['error'])
                else:
                    st.success("Search Complete")
                    display_match_cards(res)

# --- TAB 3: BULK UPLOAD ---
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
                            model='text-embedding-004',
                            contents=company_input,
                            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
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