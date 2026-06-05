"""
preprocessor.py
---------------
Downloads a song from YouTube and extracts chromagram features
for key signature detection.

Output: one or more rows appended to data/features.csv
Each row: song_id + 12 chroma values (C through B) + a key label

Usage:
    python preprocessor.py --url "https://youtu.be/..." --key "C major"
    python preprocessor.py --url "https://youtu.be/..." --key "A minor"
    python preprocessor.py --url "https://youtu.be/..." --key "G major" --start 12.5 --end 245.0

Key format: "<note> <major|minor>"
  Notes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B

--start / --end are optional seconds into the YouTube video where the music
actually begins/ends (e.g. to skip applause or outro). If omitted, the
script auto-detects via silence trimming.
"""

import os
import argparse
import tempfile
import numpy as np
import pandas as pd
import librosa
import yt_dlp

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 22050   # Hz — librosa's default; good balance of quality vs size
WINDOW_SECONDS = 5       # how many seconds each audio slice covers
NUM_WINDOWS    = 5       # how many slices to take per song
                         # (1 at start, 3 spread through middle, 1 near end)

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "features.csv")

# 24 key labels in a stable order (index = label integer stored in CSV)
KEY_LABELS = [
    "C major",  "C# major", "D major",  "D# major",
    "E major",  "F major",  "F# major", "G major",
    "G# major", "A major",  "A# major", "B major",
    "C minor",  "C# minor", "D minor",  "D# minor",
    "E minor",  "F minor",  "F# minor", "G minor",
    "G# minor", "A minor",  "A# minor", "B minor",
]

CHROMA_COLS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ── Step 1: Download audio ────────────────────────────────────────────────────

def download_audio(url: str, output_dir: str) -> str:
    """
    Download audio from a YouTube URL to output_dir as a .wav file.
    Returns the path to the downloaded file.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]

    wav_path = os.path.join(output_dir, f"{video_id}.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"Expected .wav at {wav_path} — check that ffmpeg is installed."
        )
    return wav_path


# ── Step 2: Slice audio into windows ─────────────────────────────────────────

def get_window_offsets(
    music_start: float,
    music_end:   float,
    n_windows:   int,
    window_sec:  float,
) -> list[float]:
    """
    Return start-times (in seconds) for n_windows evenly spaced across
    [music_start, music_end], so windows never reach outside that range.

    Example: music 8s->188s, 5 windows, 5-sec window:
        [8.0, 53.5, 99.0, 144.5, 183.0]
    """
    usable_end = max(music_end - window_sec, music_start)
    offsets = np.linspace(music_start, usable_end, n_windows)
    return offsets.tolist()


# ── Step 2b: Detect music boundaries ─────────────────────────────────────────

def detect_music_boundaries(
    y:            np.ndarray,
    sr:           int,
    manual_start: float | None,
    manual_end:   float | None,
) -> tuple[float, float]:
    """
    Return (music_start, music_end) in seconds.

    Priority:
      1. Use manual values directly if provided.
      2. Auto-detect via librosa.effects.trim() for whichever bound is missing.
         trim() removes leading/trailing audio below a dB threshold — handles
         silence well, and does a reasonable job on applause since the threshold
         can be tuned. Not perfect, but a safe fallback for studio recordings.

    Prints what it found so you can sanity-check each song.
    """
    total_duration = librosa.get_duration(y=y, sr=sr)

    if manual_start is not None and manual_end is not None:
        print(f"  Boundaries : manual  [{manual_start:.1f}s -> {manual_end:.1f}s]")
        return manual_start, manual_end

    # top_db=20: trim anything 20 dB below the peak.
    # Raise to 30-40 if applause still bleeds through on a particular recording.
    _, (trim_start_sample, trim_end_sample) = librosa.effects.trim(y, top_db=20)
    auto_start = trim_start_sample / sr
    auto_end   = trim_end_sample   / sr

    music_start = manual_start if manual_start is not None else auto_start
    music_end   = manual_end   if manual_end   is not None else auto_end

    source_start = "manual" if manual_start is not None else f"auto ({auto_start:.1f}s)"
    source_end   = "manual" if manual_end   is not None else f"auto ({auto_end:.1f}s)"
    print(f"  Boundaries : start={music_start:.1f}s [{source_start}]  "
          f"end={music_end:.1f}s [{source_end}]  "
          f"(total file: {total_duration:.1f}s)")

    return music_start, music_end


# ── Step 3: Extract chroma features from a slice ─────────────────────────────

def extract_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a chroma_cqt feature matrix for an audio array,
    then collapse the time axis by taking the mean.

    Returns a (12,) vector of floats in [0, 1].
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)   # shape: (12, T)
    chroma_mean = chroma.mean(axis=1)                  # shape: (12,)

    # Normalize to [0, 1] so values are comparable across songs
    max_val = chroma_mean.max()
    if max_val > 0:
        chroma_mean = chroma_mean / max_val

    return chroma_mean


# ── Step 4: Process a full song -> list of feature rows ──────────────────────

def process_song(
    audio_path:   str,
    manual_start: float | None = None,
    manual_end:   float | None = None,
) -> list[np.ndarray]:
    """
    Load audio, detect music boundaries, slice into NUM_WINDOWS windows,
    and extract chroma from each.

    manual_start / manual_end: seconds into the file where the music actually
    begins/ends (e.g. to skip applause). If None, auto-detected via trim().

    Returns a list of (12,) numpy arrays — one per window.
    """
    print(f"  Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    music_start, music_end = detect_music_boundaries(
        y, sr, manual_start, manual_end
    )
    music_duration = music_end - music_start
    print(f"  Music span : {music_duration:.1f}s — "
          f"extracting {NUM_WINDOWS} windows of {WINDOW_SECONDS}s each")

    offsets = get_window_offsets(music_start, music_end, NUM_WINDOWS, WINDOW_SECONDS)

    features = []
    for i, start in enumerate(offsets):
        start_sample = int(start * sr)
        end_sample   = int((start + WINDOW_SECONDS) * sr)
        window_audio = y[start_sample:end_sample]

        chroma = extract_chroma(window_audio, sr)
        features.append(chroma)
        print(f"  Window {i+1}/{NUM_WINDOWS} @ {start:.1f}s -> chroma extracted")

    return features


# ── Step 5: Save rows to CSV ──────────────────────────────────────────────────

def _next_song_id(csv_path: str) -> int:
    """Return the next available song_id (max existing + 1, or 0 if file is new)."""
    if not os.path.exists(csv_path):
        return 0
    df = pd.read_csv(csv_path, usecols=["song_id"])
    return int(df["song_id"].max()) + 1


def save_to_csv(feature_rows: list[np.ndarray], key_label: str, csv_path: str):
    """
    Append feature rows to the CSV file.
    Creates the file with a header if it doesn't exist yet.

    Each row gets the same song_id so GroupKFold can keep all windows
    of one song together in the same fold — preventing data leakage.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if key_label not in KEY_LABELS:
        raise ValueError(
            f"Unknown key: '{key_label}'\n"
            f"Valid keys: {KEY_LABELS}"
        )
    key_index = KEY_LABELS.index(key_label)
    song_id   = _next_song_id(csv_path)

    rows = []
    for chroma in feature_rows:
        row = {"song_id": song_id}
        row.update({col: round(float(val), 6) for col, val in zip(CHROMA_COLS, chroma)})
        row["key"] = key_index
        rows.append(row)

    new_df = pd.DataFrame(rows, columns=["song_id"] + CHROMA_COLS + ["key"])

    if os.path.exists(csv_path):
        new_df.to_csv(csv_path, mode="a", header=False, index=False)
        print(f"  Appended {len(rows)} rows to {csv_path} (song_id={song_id})")
    else:
        new_df.to_csv(csv_path, index=False)
        print(f"  Created {csv_path} with {len(rows)} rows (song_id={song_id})")


# ── Main entry point ──────────────────────────────────────────────────────────

def preprocess(
    url:          str,
    key_label:    str,
    manual_start: float | None = None,
    manual_end:   float | None = None,
):
    print(f"\n{'='*55}")
    print(f"  Song URL : {url}")
    print(f"  Key      : {key_label}")
    if manual_start is not None or manual_end is not None:
        print(f"  Bounds   : start={manual_start}s  end={manual_end}s")
    print(f"{'='*55}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n[1/3] Downloading audio...")
        audio_path = download_audio(url, tmpdir)

        print("\n[2/3] Extracting chroma features...")
        features = process_song(audio_path, manual_start, manual_end)

        print("\n[3/3] Saving to CSV...")
        save_to_csv(features, key_label, CSV_PATH)

    print("\nDone!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract chroma features from a YouTube song.")
    parser.add_argument("--url",   required=True,  help='YouTube URL')
    parser.add_argument("--key",   required=True,  help='Key, e.g. "C major" or "A minor"')
    parser.add_argument("--start", type=float, default=None,
                        help="Seconds into video where music starts (skip applause/intro)")
    parser.add_argument("--end",   type=float, default=None,
                        help="Seconds into video where music ends (skip outro/applause)")
    args = parser.parse_args()

    preprocess(args.url, args.key, args.start, args.end)