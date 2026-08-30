"""
signes.py

Computes fatigue signs (EAR, MAR, EBR, PERCLOS, HOP) from the facial
landmark coordinates extracted by `process_videos.py`.

Two modes are provided:
- `calculate_signs`: processes a full video, frame by frame (batch)
- `calculate_signs_live`: processes a single frame in a continuous stream
  (webcam/live), with state kept between calls (ear_history, closed_history, etc.)
"""

import logging

import numpy as np

from src import EAR
from src import MAR
from src import PERCLOS
from src import hop

logger = logging.getLogger(__name__)

# Number of frames corresponding to 1 second of video at 25 fps
FRAMES_PER_SECOND = 25
# Eye-closure EAR threshold for the first second (before the adaptive threshold is available)
INITIAL_EAR_CLOSED_THRESHOLD = 0.2
# Relative drop in mean EAR (compared to the rolling average) considered a blink
EAR_BLINK_DROP_RATIO = 0.9
# Number of frames over which PERCLOS is computed in live mode
LIVE_PERCLOS_WINDOW = 875


def _extract_frame_points(list_points, offset=0):
    """
    Extracts the eyes, mouth and head coordinates from a list of points,
    starting at a given offset.

    :param list_points: Full list of coordinates for the frame
    :param offset: Starting index from which to extract the points
    :return: tuple (right_eye_coord, left_eye_coord, mouth_coord, head_coord)
    """
    right_eye_coord = list_points[offset:offset + 8]
    left_eye_coord = list_points[offset + 8:offset + 16]
    mouth_coord = list_points[offset + 16:offset + 24]
    head_coord = list_points[offset + 24:offset + 26]
    return right_eye_coord, left_eye_coord, mouth_coord, head_coord


def calculate_signs(list_points):
    """
    Computes the fatigue signs for every frame of a video.

    :param list_points: List of coordinates for all frames of the video
        (structure: [video_name, frame_0_points..., frame_1_points..., ...])
    :return: tuple (results, ebr_history)
        - results (list): signs computed for each frame, concatenated
          (EAR_left, EAR_right, EAR_mean, MAR, EBR, PERCLOS, HOP_gd, HOP_hb)
        - ebr_history (list): EBR value frame by frame
    """
    ear_history = []
    frame_count = 0
    results = []
    closed_count = 0
    blink_history = []
    eyes_state = "open"
    ebr_history = []

    # One starting point per frame: each frame occupies 8*3 (eyes+mouth) + 2 (head) + 2 = 28 values
    frame_size = 8 * 3 + 2
    for i in range(1, len(list_points), frame_size):
        right_eye_coord, left_eye_coord, mouth_coord, head_coord = _extract_frame_points(list_points, offset=i)

        ear_right = EAR.eye_aspect_ratio(right_eye_coord)
        ear_left = EAR.eye_aspect_ratio(left_eye_coord)
        mar = MAR.mouth_aspect_ratio(mouth_coord)
        hop_hb, hop_gd = hop.hop(head_coord)
        ear_mean = (ear_right + ear_left) / 2

        # --- Adaptive eye-closure threshold ---
        if frame_count < FRAMES_PER_SECOND:
            # During the first second, no rolling average is available yet
            ear_history.append(ear_mean)
            closed = ear_mean < INITIAL_EAR_CLOSED_THRESHOLD
        else:
            # Afterwards, closure is detected via a relative drop in EAR
            closed = ear_mean < EAR_BLINK_DROP_RATIO * np.mean(np.array(ear_history))
            ear_history.pop(0)
            ear_history.append(ear_mean)
            blink_history.pop(0)

        # --- Blink detection ---
        if closed:
            closed_count += 1
            if eyes_state == "open":
                # Open -> closed transition: count a new blink
                eyes_state = "closed"
                blink_history.append(1)
            else:
                # Already closed on the previous frame: no new blink
                blink_history.append(0)
        else:
            eyes_state = "open"
            blink_history.append(0)

        perclos = PERCLOS.perclos(closed_count, frame_count + 1)
        ebr = np.sum(blink_history)
        ebr_history.append(ebr)

        frame_count += 1

        results.extend([ear_left, ear_right, ear_mean, mar, ebr, perclos, hop_gd, hop_hb])

    return results, ebr_history


def calculate_signs_live(list_points, frame_count, ear_history, closed_history, blink_history,
                          eyes_state, perclos_history, ebr_history):
    """
    Computes the fatigue signs for a single frame, in continuous stream mode
    (live webcam or video). The state (ear_history, closed_history,
    blink_history, eyes_state) must be kept and passed back in on the next call.

    :param list_points: Coordinates of the points for the current frame
    :param frame_count: Index of the current frame (0-indexed)
    :param ear_history: Rolling history of the mean EAR (mutable, updated in place)
    :param closed_history: History of eye-closure states (mutable)
    :param blink_history: History of detected blinks (mutable)
    :param eyes_state: Current eye state ("open" or "closed")
    :param perclos_history: History of PERCLOS values (mutable)
    :param ebr_history: History of EBR values (mutable)
    :return: tuple (results, ear_history, closed_history, blink_history, eyes_state, perclos_history, ebr_history)
    """
    right_eye_coord, left_eye_coord, mouth_coord, head_coord = _extract_frame_points(list_points, offset=0)

    ear_right = EAR.eye_aspect_ratio(right_eye_coord)
    ear_left = EAR.eye_aspect_ratio(left_eye_coord)
    mar = MAR.mouth_aspect_ratio(mouth_coord)
    hop_hb, hop_gd = hop.hop(head_coord)
    ear_mean = (ear_right + ear_left) / 2

    # --- Adaptive eye-closure threshold ---
    if frame_count < FRAMES_PER_SECOND:
        ear_history.append(ear_mean)
        closed = ear_mean < INITIAL_EAR_CLOSED_THRESHOLD

        if closed:
            closed_history.append(1)
            blink_history.append(1)
        else:
            closed_history.append(0)
            blink_history.append(0)

        logger.debug(
            "frame=%s closed=%s closed_history_size=%s", frame_count, closed, len(closed_history)
        )
    else:
        closed = ear_mean < EAR_BLINK_DROP_RATIO * np.mean(np.array(ear_history))
        ear_history.pop(0)
        ear_history.append(ear_mean)
        blink_history.pop(0)

        if closed:
            closed_history.append(1)
            if eyes_state == "open":
                eyes_state = "closed"
                blink_history.append(1)
            else:
                blink_history.append(0)
        else:
            eyes_state = "open"
            blink_history.append(0)
            closed_history.append(0)

    # PERCLOS is computed over a rolling window of the last frames
    closed_count = np.sum(closed_history[-LIVE_PERCLOS_WINDOW:])
    perclos = PERCLOS.perclos(closed_count, LIVE_PERCLOS_WINDOW)
    perclos_history.append(perclos)

    ebr = np.sum(blink_history)
    ebr_history.append(ebr)

    logger.debug("closed_history_size=%s blink_history=%s", len(closed_history), blink_history)

    results = [ear_left, ear_right, ear_mean, mar, ebr, perclos, hop_gd, hop_hb]

    return results, ear_history, closed_history, blink_history, eyes_state, perclos_history, ebr_history