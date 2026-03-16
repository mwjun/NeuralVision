"""
Large object vocabulary for open-vocabulary detection (1200+ items).
Used when the user enables "Detect 1200+ objects" in Live mode.
"""
from pathlib import Path

# In-memory cache
_LVIS_PRIMARY: list[str] | None = None


def get_lvis_primary_names() -> list[str]:
    """Return 1203 LVIS primary class names + common body parts (nose, mouth, etc.)."""
    global _LVIS_PRIMARY
    if _LVIS_PRIMARY is not None:
        return list(_LVIS_PRIMARY)
    path = Path(__file__).resolve().parent / "data" / "lvis_primary.txt"
    if not path.exists():
        _LVIS_PRIMARY = _fallback_object_list()
        return list(_LVIS_PRIMARY)
    with open(path, encoding="utf-8") as f:
        lvis = [line.strip() for line in f if line.strip()]
    # Add common body parts that aren't in LVIS (fine-grained facial/body features)
    body_parts = [
        "nose", "mouth", "lips", "ear", "ears", "eyebrow", "eyebrows", "chin", "cheek", "cheeks",
        "forehead", "neck", "shoulder", "shoulders", "elbow", "elbows", "knee", "knees",
        "ankle", "ankles", "wrist", "wrists", "finger", "fingers", "thumb", "thumbs",
        "toe", "toes", "hair", "eyelash", "eyelashes", "tongue", "teeth", "tooth",
    ]
    # Combine LVIS + body parts, dedupe
    seen = {s.lower() for s in lvis}
    extended = list(lvis)
    for part in body_parts:
        if part.lower() not in seen:
            seen.add(part.lower())
            extended.append(part)
    _LVIS_PRIMARY = extended
    return list(_LVIS_PRIMARY)


def _fallback_object_list() -> list[str]:
    """Fallback if data file is missing: COCO 80 + common extras (~100)."""
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        "hat", "glasses", "watch", "lamp", "mirror", "pillow", "door", "window", "picture", "poster",
    ]
