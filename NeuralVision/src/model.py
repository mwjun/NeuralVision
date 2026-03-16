"""
Load object detection models (DETR = 91 classes, OWLv2 = any classes via text).
Uses Hugging Face Transformers - free and open source.
"""
# Use certifi's CA bundle for SSL if available (fixes CERTIFICATE_VERIFY_FAILED on macOS)
try:
    import os
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def get_model(device: str | None = None):
    """Load DETR model and processor (91 COCO classes). 100% free."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model = model.to(device)
    model.eval()
    return processor, model, device


def get_owl_model(device: str | None = None):
    """Load OWLv2 model and processor (open-vocabulary: detect any class by name)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = model.to(device)
    model.eval()
    return processor, model, device


def get_yolo_model(size: str = "n"):
    """
    Load YOLOv8 model (80 COCO classes). Fast.
    size: 'n' (nano), 's', 'm', 'l'. Returns (None, model, device_str).
    """
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(f"yolov8{size}.pt")
    return None, model, device


def get_yolo_oiv7_model(size: str = "n"):
    """
    Load YOLOv8 trained on Open Images V7 — 600 classes, same speed as YOLO.
    Detects many more things (animals, objects, etc.) with no lag.
    size: 'n' (nano), 's', 'm', 'l'. Returns (None, model, device_str).
    """
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(f"yolov8{size}-oiv7.pt")
    return None, model, device


def get_yolo_world_model(size: str = "s"):
    """
    Load YOLO-World — open-vocabulary, any classes via set_classes([...]). Still fast.
    size: 's', 'm', 'l'. Call model.set_classes(["person", "dog", ...]) before predict.
    Returns (None, model, device_str).
    """
    try:
        from ultralytics import YOLOWorld
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLOWorld(f"yolov8{size}-worldv2.pt")
    except ImportError:
        # Fallback to YOLO if YOLOWorld not available
        from ultralytics import YOLO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(f"yolov8{size}-worldv2.pt")
    return None, model, device
