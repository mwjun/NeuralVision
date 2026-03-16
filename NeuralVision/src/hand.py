"""
Hand and finger detection via MediaPipe (free, open source).
Supports both MediaPipe 0.10+ (tasks API) and legacy 0.9 (solutions.hands).
Returns thumb, index finger, middle finger, ring finger, pinky as separate detections.
"""
from __future__ import annotations

import os
from pathlib import Path

# Fingertip landmark indices (MediaPipe 21-landmark hand model)
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

FINGER_LANDMARKS = [
    (THUMB_TIP, "thumb"),
    (INDEX_TIP, "index finger"),
    (MIDDLE_TIP, "middle finger"),
    (RING_TIP, "ring finger"),
    (PINKY_TIP, "pinky"),
]

FINGER_BOX_SIZE = 28

# Cache dir for hand_landmarker.task (MediaPipe 0.10+)
def _hand_model_path() -> Path:
    d = Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))) / "neuralvision"
    d.mkdir(parents=True, exist_ok=True)
    return d / "hand_landmarker.task"


def _download_hand_model() -> Path:
    path = _hand_model_path()
    if path.is_file():
        return path
    try:
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, path)
    except Exception:
        pass
    return path


def _run_hand_detection_v2(frame_rgb, width: int, height: int) -> list:
    """MediaPipe 0.10+ Task API (HandLandmarker). Never raises; returns [] on error."""
    import numpy as np
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python
    except ImportError:
        return []
    try:
        model_path = _download_hand_model()
        if not model_path.is_file():
            return []
    except Exception:
        return []
    try:
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.4,
            running_mode=vision.RunningMode.IMAGE,
        )
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception:
        return []
    try:
        if hasattr(frame_rgb, "shape"):
            img = frame_rgb
        else:
            img = np.asarray(frame_rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            return []
        img = np.asarray(img, dtype=np.uint8)
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=img)
        result = detector.detect(mp_image)
        if not result or not result.hand_landmarks:
            return []
        detections = []
        half = FINGER_BOX_SIZE // 2
        for hand_landmarks in result.hand_landmarks:
            for idx, label in FINGER_LANDMARKS:
                if idx >= len(hand_landmarks):
                    continue
                lm = hand_landmarks[idx]
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                x = max(0, cx - half)
                y = max(0, cy - half)
                box_w = min(FINGER_BOX_SIZE, width - x)
                box_h = min(FINGER_BOX_SIZE, height - y)
                if box_w > 0 and box_h > 0:
                    detections.append((x, y, box_w, box_h, label, 0.92))
        return detections
    except Exception:
        return []


def _run_hand_detection_legacy(frame_rgb, width: int, height: int) -> list:
    """Legacy MediaPipe (<0.10) solutions.hands.Hands. Never raises; returns [] on error."""
    try:
        import numpy as np
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
    except (ImportError, AttributeError):
        return []
    try:
        if hasattr(frame_rgb, "shape"):
            img = frame_rgb
        else:
            img = np.asarray(frame_rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            return []
        results = hands.process(img)
        try:
            hands.close()
        except Exception:
            pass
        if not results.multi_hand_landmarks:
            return []
        detections = []
        half = FINGER_BOX_SIZE // 2
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, label in FINGER_LANDMARKS:
                lm = hand_landmarks.landmark[idx]
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                x = max(0, cx - half)
                y = max(0, cy - half)
                box_w = min(FINGER_BOX_SIZE, width - x)
                box_h = min(FINGER_BOX_SIZE, height - y)
                if box_w > 0 and box_h > 0:
                    detections.append((x, y, box_w, box_h, label, 0.92))
        return detections
    except Exception:
        return []


def run_hand_detection(frame_rgb, width: int, height: int) -> list[tuple[int, int, int, int, str, float]]:
    """
    Run hand/finger detection. Uses MediaPipe 0.10+ Task API if available, else legacy API.
    Returns list of (x, y, w, h, label, score) for each detected fingertip. Never raises; returns [] on error.
    """
    try:
        import numpy as np
        if hasattr(frame_rgb, "shape"):
            img = frame_rgb
        else:
            img = np.asarray(frame_rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            return []
        h, w = img.shape[:2]
        if width <= 0 or height <= 0:
            width, height = w, h
        try:
            from mediapipe.tasks.python import vision  # noqa: F401
            return _run_hand_detection_v2(img, width, height)
        except ImportError:
            return _run_hand_detection_legacy(img, width, height)
    except Exception:
        return []


HAND_CLASS_KEYS = {"thumb", "finger", "fingers", "index", "middle", "ring", "pinky", "hand", "index finger", "middle finger", "ring finger"}


def is_hand_class(name: str) -> bool:
    """True if this class name is a hand/finger request."""
    n = name.strip().lower()
    if n in HAND_CLASS_KEYS:
        return True
    if "finger" in n or "thumb" in n:
        return True
    return False


def wants_hand_detection(class_names: list[str]) -> bool:
    """True if the user asked for any hand/finger class."""
    return any(is_hand_class(c) for c in class_names)


def filter_hand_detections(detections: list, class_names: list[str]) -> list:
    """Keep only hand detections that match what the user asked for."""
    if not detections:
        return []
    requested = {s.strip().lower() for s in class_names if s.strip()}
    if not requested:
        return detections
    out = []
    for (x, y, w, h, label, score) in detections:
        lab_lower = label.lower()
        if lab_lower in requested:
            out.append((x, y, w, h, label, score))
            continue
        if "finger" in requested and lab_lower in ("thumb", "index finger", "middle finger", "ring finger", "pinky"):
            out.append((x, y, w, h, label, score))
            continue
        if "thumb" in requested and lab_lower == "thumb":
            out.append((x, y, w, h, label, score))
    return out
