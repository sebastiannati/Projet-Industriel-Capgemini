"""
process_videos.py

For each video in the `videos/` folder, extracts the (x, y, z) coordinates
of a set of facial landmarks of interest (eyes, mouth, head), using the
MediaPipe FaceLandmarker API.

Results are saved to `csv/videos_coordinates.csv`. Videos that have already
been processed (or deemed unreadable/unusable) are tracked in
`videos/videos_traitees.txt` so the script can be re-run without
reprocessing what has already been done.

Usage:
    python src/process_videos.py
"""

import os

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration constants -------------------------------------------------

VIDEOS_FOLDER = "videos"
MODEL_PATH = "src/face_landmarker.task"
COORDINATES_CSV_PATH = "csv/videos_coordinates.csv"
PROCESSED_VIDEOS_PATH = os.path.join(VIDEOS_FOLDER, "videos_traitees.txt")
UNREADABLE_VIDEOS_PATH = os.path.join(VIDEOS_FOLDER, "videos_illisibles.txt")

TARGET_FPS = 25
FRAMES_PER_VIDEO = 875           # 875 frames at 25 fps = 35 seconds of video
NULL_RATIO_THRESHOLD = 0.10      # Above 10% missing values, the video is discarded
MAX_VIDEOS_PER_RUN = 1200        # Batch processing, to avoid losing everything on interruption

# MediaPipe FaceMesh landmarks used to compute the fatigue signs
RIGHT_EYE = [33, 133, 160, 144, 159, 145, 158, 153]
LEFT_EYE = [263, 362, 387, 373, 386, 374, 385, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
HEAD = [10, 152]
POINTS_OF_INTEREST = RIGHT_EYE + LEFT_EYE + MOUTH + HEAD


# --- Resource loading -----------------------------------------------------

def load_face_landmarker(model_path):
    """
    Loads the MediaPipe FaceLandmarker model.

    :param model_path: Path to the model's .task file
    :return: Ready-to-use vision.FaceLandmarker instance
    :raises FileNotFoundError: if the model file is missing
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_path}' is required for extraction. "
            "Download it from the official MediaPipe documentation "
            "(FaceLandmarker task file)."
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def list_videos_to_process(videos_folder):
    """Returns the list of .mp4 video paths found in the folder."""
    files = [f for f in os.listdir(videos_folder) if f.endswith(".mp4")]
    return [os.path.join(videos_folder, f) for f in files]


def load_already_processed_videos(file_path):
    """
    Loads, in a single pass, the set of video names that have already been
    processed, to allow fast (O(1)) lookups in the main loop instead of
    re-reading the file on every iteration.

    :param file_path: Path to the tracking file (videos_traitees.txt)
    :return: set of already processed video names
    """
    if not os.path.exists(file_path):
        # Create the file empty if it doesn't exist yet
        open(file_path, "a").close()
        return set()

    with open(file_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def load_or_create_dataframe(csv_path, columns):
    """Loads the existing coordinates CSV, or creates an empty DataFrame with the right columns."""
    if os.path.exists(csv_path):
        print(f"Existing file found, opening {csv_path}...")
        return pd.read_csv(csv_path)

    print("No existing file found, creating a new DataFrame.")
    return pd.DataFrame(columns=columns)


def build_coordinates_columns(points_of_interest, nb_frames):
    """Builds the list of column names for the coordinates DataFrame."""
    columns = ["video_name"]
    for frame_idx in range(nb_frames):
        for point_id in points_of_interest:
            columns.append(f"x_{point_id}_{frame_idx}")
            columns.append(f"y_{point_id}_{frame_idx}")
            columns.append(f"z_{point_id}_{frame_idx}")
    return columns


# --- Single video processing ---------------------------------------------------------

def extract_video_coordinates(video_path, detector, points_of_interest, target_fps=TARGET_FPS,
                               max_frames=FRAMES_PER_VIDEO):
    """
    Extracts the coordinates of the landmarks of interest for a given video,
    keeping only the frames matching a target fps (normalizes videos with
    different source frame rates).

    :param video_path: Path of the video to process
    :param detector: Already loaded MediaPipe FaceLandmarker detector
    :param points_of_interest: List of landmark ids to extract
    :param target_fps: Target frame rate (default 25 fps)
    :param max_frames: Maximum number of frames to keep per video
    :return: tuple (video_name, values, video_unreadable)
        - video_name (str): video file name (without the folder)
        - values (list): extracted x, y, z coordinates, frame by frame
        - video_unreadable (bool): True if reading the video failed from the very beginning
    """
    video_name = os.path.basename(video_path)
    values = []

    cap = cv2.VideoCapture(video_path.strip())
    
    if not cap.isOpened():
        cap.release()
        return video_name, values, True

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    step = actual_fps / target_fps if actual_fps > 0 else 1

    frame_count = 0
    frames_kept = 0
    next_frame_to_keep = 0.0
    video_unreadable = False

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            # Si on n'a absolument rien pu lire dès la première frame, c'est une vidéo illisible/corrompue
            if frame_count == 0:
                video_unreadable = True
            # Sinon, c'est simplement la fin normale de la vidéo (courte), on s'arrête proprement
            break

        if frames_kept >= max_frames:
            break

        if frame_count >= next_frame_to_keep:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]
                for landmark_id in points_of_interest:
                    point = face_landmarks[landmark_id]
                    values.append(point.x * frame_rgb.shape[1])
                    values.append(point.y * frame_rgb.shape[0])
                    values.append(point.z)

            frames_kept += 1
            next_frame_to_keep += step

        frame_count += 1

    cap.release()
    return video_name, values, video_unreadable

def log_unreadable_video(video_name):
    """Appends a video to the unreadable/discarded videos list (append only, never overwritten)."""
    with open(UNREADABLE_VIDEOS_PATH, "a+") as f:
        f.write(video_name + "\n")


def mark_video_as_processed(video_name):
    """Appends a video to the processed videos list, so it is never reprocessed."""
    with open(PROCESSED_VIDEOS_PATH, "a+") as f:
        f.write(video_name + "\n")


# --- Orchestration -------------------------------------------------------------------

def main():
    video_paths = list_videos_to_process(VIDEOS_FOLDER)
    print(f"Found {len(video_paths)} videos in '{VIDEOS_FOLDER}'.")

    detector = load_face_landmarker(MODEL_PATH)

    already_processed = load_already_processed_videos(PROCESSED_VIDEOS_PATH)
    print(f"{len(already_processed)} videos already processed previously.")

    columns = build_coordinates_columns(POINTS_OF_INTEREST, FRAMES_PER_VIDEO)
    existing_df = load_or_create_dataframe(COORDINATES_CSV_PATH, columns)

    # New rows are accumulated in a Python list rather than running a
    # pd.concat for every video (much faster: avoids copying the whole
    # DataFrame on every iteration).
    new_rows = []
    videos_processed = 0

    for video_path in video_paths:
        video_name = os.path.basename(video_path)

        if video_name in already_processed:
            continue

        videos_processed += 1
        print(f"Processing {videos_processed}/{len(video_paths) - len(already_processed)}: {video_name}")

        video_name, values, video_unreadable = extract_video_coordinates(
            video_path, detector, POINTS_OF_INTEREST
        )

        if video_unreadable:
            print("  -> Unreadable video, discarded.")
            log_unreadable_video(video_name)
            mark_video_as_processed(video_name)
            continue

        new_row = [video_name] + values
        expected_nb_values = len(columns) - 1  # -1 for the "video_name" column
        nb_zeros_to_add = expected_nb_values - len(values)

        # If more than 10% of the values are missing (frames with no face
        # detected, or a shorter-than-expected video), the video is deemed
        # too degraded to be usable and is discarded rather than padded with zeros.
        if nb_zeros_to_add >= NULL_RATIO_THRESHOLD * expected_nb_values:
            print(f"  -> Too many missing values ({nb_zeros_to_add}/{expected_nb_values}), discarded.")
            log_unreadable_video(video_name)
            mark_video_as_processed(video_name)
            continue

        # Pad the missing values at the end of the row with zeros
        new_row += [0] * nb_zeros_to_add
        new_rows.append(new_row)
        mark_video_as_processed(video_name)

        if videos_processed >= MAX_VIDEOS_PER_RUN:
            print(f"Reached the limit of {MAX_VIDEOS_PER_RUN} videos for this run, stopping this batch.")
            break

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=columns)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = existing_df

    final_df.to_csv(COORDINATES_CSV_PATH, index=False)
    cv2.destroyAllWindows()

    print(f"\nDone! {len(new_rows)} new videos added.")
    print(f"File available at: {COORDINATES_CSV_PATH}")


if __name__ == "__main__":
    main()