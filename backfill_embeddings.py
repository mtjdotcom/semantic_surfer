import pandas as pd
import numpy as np
import json
import time
from google import genai
from google.genai import types
import streamlit as st
import gspread
from gspread.utils import rowcol_to_a1 # Import the helper to find "AD1"

# 1. Setup Client
if "GOOGLE_API_KEY" not in st.secrets:
    raise ValueError("Secrets not found. Make sure .streamlit/secrets.toml exists.")

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

def batch_embed(texts, batch_size=50):
    """Embeds a list of texts in batches to respect API limits."""
    results = []
    total = len(texts)
    
    print(f"Starting embedding for {total} items...")
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        print(f"Processing batch {i} to {i+batch_size}...")
        
        try:
            # New SDK Batch Embedding
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            
            # Extract values
            for embedding in response.embeddings:
                results.append(json.dumps(embedding.values))
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Fill failed batch with empty strings to keep alignment
            results.extend([""] * len(batch))
            
        time.sleep(1) # Rate limit safety
        
    return results

def run_backfill():
    # ... (Connection logic remains the same) ...
    print("Connecting to Google Sheets...")
    gc = gspread.service_account_from_dict(st.secrets["connections"]["gsheets"])
    sh = gc.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
    
    try:
        worksheet = sh.worksheet("portfolio")
    except gspread.WorksheetNotFound:
        worksheet = sh.worksheet("Sheet1")

    # Load data
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows.")

    # Check if 'Embedding' exists in the DataFrame
    if 'Embedding' not in df.columns:
        # Create it as a new column
        df['Embedding'] = ""
        # The index (position) of this new column is the length of existing columns + 1
        # e.g., if you have 29 columns, this becomes column 30
        col_index = len(df.columns) 
    else:
        # If it exists, find its index (add 1 because Sheets is 1-indexed, Python is 0-indexed)
        col_index = df.columns.get_loc("Embedding") + 1

    # --- LOGIC TO CALCULATE EMBEDDINGS (Same as before) ---
    mask = (df['Embedding'] == "") | (df['Embedding'].isna())
    missing_df = df[mask]
    
    if missing_df.empty:
        print("✅ No missing embeddings found!")
        return

    print(f"Found {len(missing_df)} rows missing embeddings.")
    
    descriptions = missing_df['New Long Description'].fillna("Company").replace("", "Company").tolist()
    new_vectors = batch_embed(descriptions, batch_size=50)
    
    # Update DataFrame
    df.loc[mask, 'Embedding'] = new_vectors
    df = df.fillna("") # Clean NaNs
    
    # --- SAFER SAVE LOGIC ---
    
    # 1. Prepare the data for JUST the embedding column
    # We create a list of lists: [["Embedding"], ["[0.1, 0.2]"], ["[0.5, -0.1]"]...]
    # This includes the Header at the top
    column_values = [["Embedding"]] + df[['Embedding']].values.tolist()
    
    # 2. Calculate the specific cell to start at (e.g., "AD1")
    start_cell = rowcol_to_a1(1, col_index) 
    
    print(f"Writing ONLY the 'Embedding' column to {start_cell}...")
    
    # 3. Update ONLY that column range
    worksheet.update(range_name=start_cell, values=column_values)
    
    print("✅ Done! Formatting and formulas in other columns are safe.")

if __name__ == "__main__":
    run_backfill()