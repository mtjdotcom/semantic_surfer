#!/usr/bin/env python3
"""
Rebuild the 'Embedding' column on the portfolio sheet.

Run this after every quarterly data drop (e.g. Q1 2026). It works out which
rows actually need embedding and only rewrites those cells - everything else
on the sheet (formatting, formulas, other columns) is left untouched.

A row is re-embedded when:
  * it has no embedding yet                     (new companies)
  * its description changed since last time     (edited companies)
  * the model / dimensionality changed          (whole sheet re-embeds)
  * the stored vector is the wrong length       (corrupt or legacy data)

Change detection works off an 'Embedding Hash' column that this script
manages for you. The first run after adding it will re-embed everything
once (there are no hashes to compare against yet), then settle down.

Usage:
    python backfill_embeddings.py                  # embed new + changed rows
    python backfill_embeddings.py --dry-run        # show the plan, call nothing
    python backfill_embeddings.py --limit 10       # try 10 rows first
    python backfill_embeddings.py --force --yes    # re-embed every single row
"""

import argparse
import hashlib
import json
import sys
import time

import streamlit as st
import gspread
from gspread.utils import rowcol_to_a1
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# These three values feed the content hash. Change any of them and the whole
# sheet correctly re-embeds on the next run.
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768
TASK_TYPE = "RETRIEVAL_DOCUMENT"

EMBEDDING_COL = "Embedding"
HASH_COL = "Embedding Hash"

# Description columns, best first. We use the first one that has real text, so
# a new Q1 row carrying only a one-liner still gets a usable vector instead of
# being dropped. Override with --text-columns.
DEFAULT_TEXT_COLUMNS = ["New Long Description", "New One Line Description"]

# Last resort when a row has no description at all. Override with --name-column.
NAME_COLUMN = "Name"

# Floats are rounded before storage. 6dp is well below the noise floor of the
# model and keeps each cell around 8KB instead of 15KB (Sheets caps at 50k).
ROUND_DP = 6

# Rows per Sheets write request. Two ~8KB cells per row, so this keeps each
# request comfortably small.
WRITE_CHUNK_ROWS = 100


# --- SHEET HELPERS ---
def open_worksheet(name):
    """Connects with the same service-account creds the Streamlit app uses."""
    print("Connecting to Google Sheets...")
    gc = gspread.service_account_from_dict(st.secrets["connections"]["gsheets"])
    sh = gc.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])

    try:
        return sh.worksheet(name)
    except gspread.WorksheetNotFound:
        if name == "portfolio":
            print("No 'portfolio' tab, falling back to 'Sheet1'.")
            return sh.worksheet("Sheet1")
        raise


def read_sheet(worksheet):
    """
    Reads raw cell values rather than get_all_records().

    This matters: get_all_records() throws on duplicate headers and can shift
    rows around, and we are about to write back by absolute row number. Raw
    values guarantee that data row i is always sheet row i + 2.
    """
    values = worksheet.get_all_values()
    if not values:
        raise RuntimeError("Sheet is empty - nothing to embed.")

    header = values[0]
    # The API trims trailing empties; drop them so appends land in the right place.
    while header and header[-1].strip() == "":
        header.pop()

    width = len(header)
    rows = []
    for raw in values[1:]:
        row = list(raw[:width])
        row.extend([""] * (width - len(row)))
        rows.append(row)

    return header, rows


def column_lookup(header):
    """Maps header name -> 0-based index. First occurrence wins."""
    lookup = {}
    for i, name in enumerate(header):
        key = name.strip()
        if key and key not in lookup:
            lookup[key] = i
    return lookup


def ensure_column(worksheet, header, rows, name, dry_run=False):
    """Returns the 0-based index of `name`, appending the column if missing."""
    lookup = column_lookup(header)
    if name in lookup:
        return lookup[name]

    index = len(header)
    print(f"Column '{name}' not found - appending it as column {index + 1}.")

    if not dry_run:
        if index + 1 > worksheet.col_count:
            worksheet.add_cols(index + 1 - worksheet.col_count)
        worksheet.update(
            range_name=rowcol_to_a1(1, index + 1),
            values=[[name]],
            value_input_option="RAW",
        )

    header.append(name)
    for row in rows:
        row.append("")
    return index


# --- TEXT + HASHING ---
def build_text(row, lookup, text_columns, name_column):
    """
    Picks the best available text for a row.

    Returns (text, quality) where quality is 'description', 'name' or 'none'.
    A name-only vector is weak - it still lets the company be found, and the
    keyword boost in the app covers exact name hits - but the caller reports
    those rows so you know which ones need a description written.
    """
    for col in text_columns:
        if col in lookup:
            text = row[lookup[col]].strip()
            if text:
                return text, "description"

    if name_column in lookup:
        name = row[lookup[name_column]].strip()
        if name:
            return name, "name"

    return "", "none"


def content_hash(text):
    """Hash covers the model config too, so a model swap invalidates everything."""
    payload = f"{EMBED_MODEL}|{EMBED_DIMS}|{TASK_TYPE}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def stored_vector_is_valid(cell):
    """True only if the cell holds a JSON list of the expected dimensionality."""
    cell = cell.strip()
    if not cell:
        return False
    try:
        vector = json.loads(cell)
    except (ValueError, TypeError):
        return False
    return isinstance(vector, list) and len(vector) == EMBED_DIMS


# --- EMBEDDING ---
def embed_batch(client, texts, retries=4):
    """
    Embeds one batch, with backoff. On persistent failure falls back to
    embedding each text on its own, so a single bad row cannot take out 49
    good ones. Returns a list the same length as `texts`, with None for
    anything that could not be embedded.
    """
    delay = 2
    for attempt in range(retries):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=TASK_TYPE, output_dimensionality=EMBED_DIMS
                ),
            )
            return [list(e.values) for e in response.embeddings]
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  Batch failed after {retries} attempts ({exc}).")
                print("  Retrying one row at a time...")
                break
            print(f"  Attempt {attempt + 1} failed ({exc}); retrying in {delay}s.")
            time.sleep(delay)
            delay *= 2

    results = []
    for text in texts:
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type=TASK_TYPE, output_dimensionality=EMBED_DIMS
                ),
            )
            results.append(list(response.embeddings[0].values))
        except Exception as exc:
            print(f"  Skipping a row - embedding failed: {exc}")
            results.append(None)
    return results


def encode_vector(vector):
    return json.dumps([round(float(v), ROUND_DP) for v in vector])


# --- WRITING ---
def contiguous_runs(numbers):
    """Groups sorted row numbers into [start, end] runs so we write few ranges."""
    runs = []
    for n in numbers:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return runs


def write_updates(worksheet, updates, embed_index, hash_index):
    """
    Writes only the Embedding and Embedding Hash cells for the given rows.

    `updates` maps sheet row number -> (embedding_json, hash). Every other
    cell on the sheet is left completely alone.
    """
    if not updates:
        return

    embed_col = embed_index + 1
    hash_col = hash_index + 1

    for start, end in contiguous_runs(sorted(updates)):
        for chunk_start in range(start, end + 1, WRITE_CHUNK_ROWS):
            chunk_end = min(chunk_start + WRITE_CHUNK_ROWS - 1, end)
            rows = range(chunk_start, chunk_end + 1)

            requests = [
                {
                    "range": f"{rowcol_to_a1(chunk_start, embed_col)}:"
                             f"{rowcol_to_a1(chunk_end, embed_col)}",
                    "values": [[updates[r][0]] for r in rows],
                },
                {
                    "range": f"{rowcol_to_a1(chunk_start, hash_col)}:"
                             f"{rowcol_to_a1(chunk_end, hash_col)}",
                    "values": [[updates[r][1]] for r in rows],
                },
            ]
            worksheet.batch_update(requests, value_input_option="RAW")
            print(f"  Wrote rows {chunk_start}-{chunk_end}.")


# --- PLANNING ---
def build_plan(rows, lookup, embed_index, hash_index, text_columns, force,
               name_column, require_description=False):
    """
    Decides what each row needs.

    Returns (work, skipped, empty, name_only, clear). `clear` holds rows whose
    stored vector should be wiped - only used with require_description, where
    a name-only row must not merely be left alone but actively removed from
    the index, since the app treats any parseable vector as searchable.
    """
    work, skipped, empty, name_only, clear = [], 0, [], [], []

    for i, row in enumerate(rows):
        sheet_row = i + 2
        text, quality = build_text(row, lookup, text_columns, name_column)

        if quality == "none":
            empty.append(sheet_row)
            continue

        if quality == "name":
            name_only.append(row[lookup[name_column]].strip())
            if require_description:
                # Drop any vector it already carries, otherwise it stays
                # searchable on the strength of its name alone.
                if stored_vector_is_valid(row[embed_index]) or row[hash_index].strip():
                    clear.append(sheet_row)
                continue

        digest = content_hash(text)
        has_vector = stored_vector_is_valid(row[embed_index])
        hash_matches = row[hash_index].strip() == digest

        if force:
            reason = "forced"
        elif not has_vector:
            reason = "missing"
        elif not row[hash_index].strip():
            reason = "unhashed"
        elif not hash_matches:
            reason = "changed"
        else:
            skipped += 1
            continue

        work.append({"row": sheet_row, "text": text, "hash": digest, "reason": reason})

    return work, skipped, empty, name_only, clear


def summarise(work, skipped, empty, name_only, clear, total, require_description):
    print("")
    print("--- PLAN ---")
    print(f"Rows on sheet:      {total}")
    print(f"Already up to date: {skipped}")
    print(f"Unusable (no text): {len(empty)}")
    print(f"To embed:           {len(work)}")
    if require_description:
        print(f"Name-only, excluded: {len(name_only)} "
              f"({len(clear)} with a stale vector to clear)")

    if work:
        counts = {}
        for item in work:
            counts[item["reason"]] = counts.get(item["reason"], 0) + 1
        labels = {
            "missing": "never embedded",
            "changed": "description changed",
            "unhashed": "no hash on record",
            "forced": "forced rebuild",
        }
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  - {count} {labels.get(reason, reason)}")

        chars = sum(len(item["text"]) for item in work)
        print(f"Characters to send: {chars:,} (~{chars // 4:,} tokens, rough)")

    if name_only:
        preview = ", ".join(name_only[:10])
        more = f" (+{len(name_only) - 10} more)" if len(name_only) > 10 else ""
        print("")
        noun = "company has" if len(name_only) == 1 else "companies have"
        if require_description:
            print(f"NOTE: {len(name_only)} {noun} no description and will be left OUT")
            print(f"      of search entirely (--require-description).")
        else:
            print(f"NOTE: {len(name_only)} {noun} no description, so they are embedded")
            print(f"      on their name alone and WILL still appear in search results,")
            print(f"      matched on the name alone. Pass --require-description to")
            print(f"      exclude them instead.")
        print(f"      Affected: {preview}{more}")

    if empty:
        preview = ", ".join(str(r) for r in empty[:10])
        more = f" (+{len(empty) - 10} more)" if len(empty) > 10 else ""
        print("")
        print(f"WARNING: rows {preview}{more} have no name or description at all")
        print(f"         and will be invisible to search.")
    print("")


# --- MAIN ---
def run(args):
    # Printed up front so it is always obvious which script and which model
    # are actually running - a stale checkout is otherwise easy to miss.
    print(f"backfill_embeddings.py | model={EMBED_MODEL} dims={EMBED_DIMS}")
    if args.dry_run:
        print("DRY RUN - no embeddings will be requested, nothing will be written.")
    print("")

    worksheet = open_worksheet(args.worksheet)
    header, rows = read_sheet(worksheet)
    print(f"Loaded {len(rows)} rows x {len(header)} columns from '{worksheet.title}'.")

    embed_index = ensure_column(worksheet, header, rows, EMBEDDING_COL, args.dry_run)
    hash_index = ensure_column(worksheet, header, rows, HASH_COL, args.dry_run)
    lookup = column_lookup(header)

    text_columns = args.text_columns or DEFAULT_TEXT_COLUMNS
    missing_cols = [c for c in text_columns if c not in lookup]
    if missing_cols:
        print(f"Note: description column(s) not on this sheet: {', '.join(missing_cols)}")
    if not any(c in lookup for c in text_columns):
        raise RuntimeError(
            f"None of {text_columns} exist on the sheet. Columns are: {header}"
        )

    # Without a name column, any row lacking a description is dropped entirely
    # rather than falling back to its name - which can silently exclude a large
    # chunk of the sheet. Make that loud, and show what is actually there.
    name_column = args.name_column
    if name_column not in lookup:
        print("")
        print(f"WARNING: name column '{name_column}' is not on this sheet, so rows")
        print( "         with no description cannot fall back to their name and")
        print( "         will be skipped. Columns found:")
        for name in header:
            if name.strip():
                print(f"           {name}")
        print("         Re-run with --name-column 'Your Column' if one of these is it.")

    work, skipped, empty, name_only, clear = build_plan(
        rows, lookup, embed_index, hash_index, text_columns, args.force,
        name_column, args.require_description
    )
    summarise(work, skipped, empty, name_only, clear, len(rows),
              args.require_description)

    if not work and not clear:
        print("Nothing to do.")
        return

    if args.limit and args.limit < len(work):
        work = work[: args.limit]
        print(f"--limit set: only processing the first {len(work)} rows.\n")

    if args.dry_run:
        print("Dry run - no API calls made, nothing written.")
        return

    if args.force and not args.yes:
        if not sys.stdin.isatty():
            raise RuntimeError("--force needs --yes when running non-interactively.")
        answer = input(f"Re-embed all {len(work)} rows? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # Clearing needs no API call, so do it first and get those rows out of the
    # index even if the embedding run later fails partway.
    if clear:
        print(f"Clearing stale vectors on {len(clear)} name-only rows...")
        write_updates(worksheet, {r: ("", "") for r in clear},
                      embed_index, hash_index)

    if not work:
        print("Nothing left to embed.")
        return

    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

    pending, failed, done = {}, 0, 0
    for start in range(0, len(work), args.batch_size):
        batch = work[start : start + args.batch_size]
        print(f"Embedding rows {batch[0]['row']}-{batch[-1]['row']} "
              f"({start + len(batch)}/{len(work)})...")

        for item, vector in zip(batch, embed_batch(client, [b["text"] for b in batch])):
            if vector is None:
                failed += 1
                continue
            pending[item["row"]] = (encode_vector(vector), item["hash"])
            done += 1

        # Flush periodically so an interrupted run keeps the work it paid for.
        if len(pending) >= args.flush_every:
            write_updates(worksheet, pending, embed_index, hash_index)
            pending = {}

        if start + args.batch_size < len(work):
            time.sleep(args.sleep)

    write_updates(worksheet, pending, embed_index, hash_index)

    print("")
    print(f"Done. Embedded {done} rows" + (f", {failed} failed." if failed else "."))
    if failed:
        print("Re-run the script to retry the failures - finished rows are skipped.")
    print("The app caches the sheet for 10 minutes, so give it a moment to refresh.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worksheet", default="portfolio",
                        help="Tab to read (default: portfolio)")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed every row, ignoring hashes")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt for --force")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would happen without calling the API")
    parser.add_argument("--limit", type=int,
                        help="Only process the first N rows needing work")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Texts per embedding request (default: 50)")
    parser.add_argument("--flush-every", type=int, default=250,
                        help="Write to the sheet every N embedded rows (default: 250)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to pause between batches (default: 1.0)")
    parser.add_argument("--text-columns", nargs="+",
                        help=f"Description columns, best first "
                             f"(default: {' '.join(DEFAULT_TEXT_COLUMNS)})")
    parser.add_argument("--name-column", default=NAME_COLUMN,
                        help=f"Column to fall back to when a row has no "
                             f"description (default: {NAME_COLUMN})")
    parser.add_argument("--require-description", action="store_true",
                        help="Leave rows with no description out of search "
                             "entirely, clearing any vector they already have, "
                             "instead of embedding them on their name")
    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in st.secrets:
        raise ValueError("Secrets not found. Make sure .streamlit/secrets.toml exists.")

    run(args)


if __name__ == "__main__":
    main()
