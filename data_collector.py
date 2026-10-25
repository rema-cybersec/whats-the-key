"""
data_collector.py
-----------------
Reads the Google Form responses spreadsheet and batch-processes all songs
through the preprocessor, appending rows to data/features.csv.

The spreadsheet uses local key names (C-major, Gb-minor, Db-major, etc.)
which are normalized to our canonical format ("C major", "F# major", etc.)
before being passed to the preprocessor.

Usage:
    # Process all songs not yet in features.csv
    python data_collector.py --sheet <SPREADSHEET_CSV_URL_OR_PATH>

    # Dry run — print what would be processed without downloading anything
    python data_collector.py --sheet <URL> --dry-run

    # Re-process a specific row range (1-indexed, header excluded)
    python data_collector.py --sheet <URL> --rows 5-20

Export the sheet as CSV from Google Sheets:
    File → Download → Comma Separated Values
Or use the direct CSV export URL:
    https://docs.google.com/spreadsheets/d/<ID>/export?format=csv
"""

import os
import sys
import argparse
import time
import pandas as pd

# ── Key name normalization ────────────────────────────────────────────────────
# The form uses names like "C-major", "Gb-minor", "Db-major".
# Our canonical format is "C major", "F# major", "C# minor", etc.
# Enharmonic equivalents are collapsed: Gb = F#, Db = C#, etc.

ENHARMONIC_MAP = {
    "Db": "C#",
    "Eb": "D#",
    "Fb": "E",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
}

def normalize_key(raw: str) -> str | None:
    """
    Convert spreadsheet key name to canonical form.

    Examples:
        "C-major"  → "C major"
        "Gb-minor" → "F# minor"
        "Db-major" → "C# major"

    Returns None if the key cannot be parsed — that row will be skipped
    with a warning so one bad entry doesn't abort the whole batch.
    """
    raw = raw.strip()
    if "-" not in raw:
        return None

    parts = raw.split("-", 1)
    if len(parts) != 2:
        return None

    note, quality = parts[0].strip(), parts[1].strip().lower()
    note = ENHARMONIC_MAP.get(note, note)   # Gb → F#, etc.

    if quality not in ("major", "minor"):
        return None

    canonical = f"{note} {quality}"

    # Validate against the known 24-key list
    from preprocessor import KEY_LABELS
    if canonical not in KEY_LABELS:
        return None

    return canonical


# ── Already-processed URL tracking ───────────────────────────────────────────

def already_processed_urls(csv_path: str) -> set[str]:
    """
    Return the set of YouTube URLs already present in features.csv.
    Used to skip songs that have already been downloaded and processed.
    """
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path, usecols=["youtube_url"] if "youtube_url" in
                     pd.read_csv(csv_path, nrows=0).columns else [])
    if "youtube_url" not in df.columns:
        # features.csv doesn't store the URL — use song_id count as proxy
        # (nothing to skip, safer to let the user manage duplicates manually)
        return set()
    return set(df["youtube_url"].dropna())


# ── URL cleanup ───────────────────────────────────────────────────────────────

def clean_youtube_url(url: str) -> str:
    """
    Strip playlist/radio parameters from YouTube URLs so yt-dlp
    downloads only the specific video, not the whole playlist.

    https://www.youtube.com/watch?v=ABC&list=RD...&start_radio=1
    → https://www.youtube.com/watch?v=ABC
    """
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    parsed = urlparse(url.strip())
    params = parse_qs(parsed.query)
    # Keep only the video id
    clean_params = {"v": params["v"][0]} if "v" in params else {}
    # Handle youtu.be short links — no query params needed
    if parsed.netloc == "youtu.be":
        return urlunparse(parsed._replace(query=""))
    return urlunparse(parsed._replace(query=urlencode(clean_params)))


# ── Main batch processor ──────────────────────────────────────────────────────

def run(sheet_source: str, dry_run: bool, row_range: tuple[int, int] | None):
    # Load spreadsheet (URL or local file)
    print(f"Loading spreadsheet from: {sheet_source}")
    df = pd.read_csv(sheet_source)

    # Keep only the columns we need; rename to internal names
    # Spreadsheet columns: Timestamp | link | Tonalitet | Početno vreme | Krajnje vreme
    df = df.rename(columns={
        df.columns[1]: "url",
        df.columns[2]: "key_raw",
        df.columns[3]: "start_time",
        df.columns[4]: "end_time",
    })[["url", "key_raw", "start_time", "end_time"]]

    # Apply row range filter (1-indexed, header excluded)
    if row_range:
        lo, hi = row_range
        df = df.iloc[lo - 1 : hi]

    total = len(df)
    print(f"Found {total} rows to consider\n")

    # Import preprocessor here so KEY_LABELS is available for validation
    from preprocessor import preprocess, CSV_PATH

    skipped_key    = 0
    skipped_done   = 0
    processed      = 0
    failed         = 0

    done_urls = already_processed_urls(CSV_PATH)

    for i, row in df.iterrows():
        raw_url  = str(row["url"]).strip()
        key_raw  = str(row["key_raw"]).strip()
        start    = float(row["start_time"]) if pd.notna(row["start_time"]) else None
        end      = float(row["end_time"])   if pd.notna(row["end_time"])   else None

        canonical_key = normalize_key(key_raw)

        print(f"[{i+1}/{total}] {raw_url[:60]}...")
        print(f"         key={key_raw!r} → {canonical_key!r}  "
              f"start={start}  end={end}")

        # Skip unrecognised keys
        if canonical_key is None:
            print(f"  ⚠ Skipping — could not parse key '{key_raw}'\n")
            skipped_key += 1
            continue

        url = clean_youtube_url(raw_url)

        # Skip already-processed songs
        if url in done_urls:
            print(f"  ↩ Already processed — skipping\n")
            skipped_done += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] Would process: key={canonical_key}  url={url}\n")
            processed += 1
            continue

        try:
            preprocess(url, canonical_key, start, end)
            done_urls.add(url)
            processed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            failed += 1

        # Small pause between downloads to be polite to YouTube
        time.sleep(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  Batch complete")
    print("="*50)
    print(f"  Processed : {processed}")
    print(f"  Skipped (already done) : {skipped_done}")
    print(f"  Skipped (bad key name) : {skipped_key}")
    print(f"  Failed    : {failed}")
    print(f"  Total     : {total}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-process Google Form responses into features.csv"
    )
    parser.add_argument(
        "--sheet", required=True,
        help="Path or URL to the spreadsheet CSV. "
             "Get the URL from: File → Share → Publish to web → CSV"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be processed without downloading anything"
    )
    parser.add_argument(
        "--rows", default=None,
        help="Process only a row range, e.g. --rows 1-50 (1-indexed, header excluded)"
    )
    args = parser.parse_args()

    row_range = None
    if args.rows:
        lo, hi = args.rows.split("-")
        row_range = (int(lo), int(hi))

    run(args.sheet, args.dry_run, row_range)
