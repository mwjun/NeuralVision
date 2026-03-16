"""Basic tests for NeuralVision detection (run from project root: pytest tests/)."""
import sys
from pathlib import Path

# Run from project root so src is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image


def test_imports():
    """Imports should succeed."""
    from src.model import get_model
    from src.realtime_detect import run_detection
    assert get_model is not None
    assert run_detection is not None


def test_run_detection_returns_list():
    """run_detection on a small image should return a list of detections."""
    from src.model import get_model
    from src.realtime_detect import run_detection

    # Small RGB image (320x240)
    pil_image = Image.new("RGB", (320, 240), color=(128, 128, 128))
    processor, model, device = get_model()
    detections = run_detection(processor, model, device, pil_image, confidence_threshold=0.5)

    assert isinstance(detections, list)
    # Each item: (x, y, w, h, label_name, score)
    for d in detections:
        assert len(d) == 6
        x, y, w, h, label, score = d
        assert isinstance(x, int) and isinstance(y, int)
        assert isinstance(w, int) and isinstance(h, int)
        assert isinstance(label, str) and isinstance(score, (int, float))
