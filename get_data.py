"""
get_data.py

Orchestrates the data generation pipeline:
1. Extract facial landmark coordinates from the videos (src/process_videos.py)
2. Compute fatigue signs from these coordinates (src/signs.py)
3. Display a quick preview and diagnostics on the final DataFrame (missing values, classes)

Usage:
    python get_data.py
"""

import os
import subprocess
import sys

import pandas as pd

from src import signs

COORDINATES_CSV_PATH = "csv/videos_coordinates.csv"
SIGNS_CSV_PATH = "csv/signes.csv"
FRAMES_PER_VIDEO = 875

SIGN_COLUMNS = ["EAR_left", "EAR_right", "EAR_mean", "MAR", "EBR", "PERCLOS", "HOP_gd", "HOP_hb"]

# Mapping from the drowsiness level (KSS scale, extracted from the file name) to a numeric class
CLASS_MAPPING = {
    "1-3": 0,  # Alert
    "6-7": 1,  # Intermediate
    "8-9": 2,  # Drowsy
}


def extract_coordinates():
    """
    Runs src/process_videos.py, which handles resuming on already-processed
    videos itself (via videos/videos_traitees.txt). It can therefore be
    called on every run without risk of reprocessing already-done videos.
    """
    print(f"\n{'=' * 60}")
    print("Extracting coordinates (src/process_videos.py)...")
    print(f"{'=' * 60}\n")

    result = subprocess.run([sys.executable, "src/process_videos.py"])

    if result.returncode != 0:
        print(f"\nError while running src/process_videos.py (exit code {result.returncode})")
        sys.exit(result.returncode)

    if not os.path.exists(COORDINATES_CSV_PATH):
        print(f"\n{COORDINATES_CSV_PATH} not found after extraction, aborting.")
        sys.exit(1)

    print("\nCoordinates extraction complete.")


def build_signs_columns(sign_columns, nb_frames):
    """Builds the list of column names for the signs DataFrame."""
    columns = ["video_name"]
    for frame_idx in range(nb_frames):
        for sign in sign_columns:
            columns.append(f"{sign}_{frame_idx}")
    return columns


def compute_signs_from_coordinates(coordinates_csv_path, sign_columns, nb_frames):
    """
    Computes the fatigue signs for each video from the coordinates CSV,
    processing rows in batches (chunks) to limit memory usage.

    :return: DataFrame of computed signs, with the "class" column added
    """
    coordinates_chunks = pd.read_csv(coordinates_csv_path, chunksize=1000)
    columns = build_signs_columns(sign_columns, nb_frames)

    # Rows are accumulated in a Python list rather than using
    # df.loc[len(df)] = ... on every iteration (much faster, avoids
    # copying the whole DataFrame on every new row).
    rows = []

    for chunk_nb, chunk in enumerate(coordinates_chunks, start=1):
        print(f"Chunk {chunk_nb}...")
        for row_idx in range(chunk.shape[0]):
            video_name = chunk.iloc[row_idx, 0]
            print(f"Processing {row_idx + 1}/{chunk.shape[0]} {video_name}...")

            video_and_coordinates = [video_name]
            for i in range(1, chunk.shape[1], 3):
                point_coordinates = [chunk.iloc[row_idx, i], chunk.iloc[row_idx, i + 1], chunk.iloc[row_idx, i + 2]]
                video_and_coordinates.append(point_coordinates)

            results, _ = signs.calculate_signs(video_and_coordinates)
            rows.append([video_name] + results)

    df = pd.DataFrame(rows, columns=columns)
    df["classe"] = df["video_name"].str[4:7].map(CLASS_MAPPING)
    return df


def display_diagnostics(df, sign_columns):
    """Displays a preview, descriptive statistics and the class distribution of the DataFrame."""
    print(f"\n{'=' * 60}")
    print("DataFrame preview:")
    print(df.head())
    print(f"{'=' * 60}")

    print(f"{'=' * 60}")
    print("Descriptive statistics:")
    print(df.describe())
    print(f"{'=' * 60}")

    print(f"{'=' * 60}")
    print("Missing values per sign:")
    for sign in sign_columns:
        sign_cols = [c for c in df.columns if c.startswith(f"{sign}_")]
        nb_nan = df[sign_cols].isnull().sum().sum()
        print(f"  {sign}: {nb_nan} missing values out of {len(sign_cols) * len(df)} total values")
    print(f"{'=' * 60}")

    print(f"{'=' * 60}")
    print("Class distribution:")
    print(df["classe"].value_counts())
    print(f"{'=' * 60}")


def compute_signs():
    """
    Computes the fatigue signs from csv/videos_coordinates.csv (or loads
    csv/signes.csv if it already exists), displays quick diagnostics, then
    saves the result.
    """
    print(f"\n{'=' * 60}")
    print("Computing fatigue signs...")
    print(f"{'=' * 60}\n")

    if os.path.exists(SIGNS_CSV_PATH):
        print(f"{SIGNS_CSV_PATH} already exists, loading it directly.")
        df = pd.read_csv(SIGNS_CSV_PATH)
    else:
        df = compute_signs_from_coordinates(COORDINATES_CSV_PATH, SIGN_COLUMNS, FRAMES_PER_VIDEO)
        df.to_csv(SIGNS_CSV_PATH, index=False)

    display_diagnostics(df, SIGN_COLUMNS)

    print(f"\nSigns computation complete, file available at: {SIGNS_CSV_PATH}")


def main():
    extract_coordinates()
    compute_signs()
    print(f"\n{'=' * 60}")
    print("get_data.py pipeline completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()