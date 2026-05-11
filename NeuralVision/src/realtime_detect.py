"""
Real-time and single-frame object detection (DETR, OWLv2, YOLO, YOLO-OIV7, YOLO-World).
CLI: python -m src.realtime_detect --source 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .model import get_model, get_yolo_model, get_yolo_oiv7_model, get_yolo_world_model


def _batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_detection(
    processor,
    model,
    device: str,
    pil_image: Image.Image,
    confidence_threshold: float = 0.5,
) -> list[tuple[int, int, int, int, str, float]]:
    """DETR: return list of (x, y, w, h, label, score) in pixel coordinates."""
    pil_image = pil_image.convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = _batch_to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs)
    w, h = pil_image.size
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )[0]
    id2label = model.config.id2label
    detections: list[tuple[int, int, int, int, str, float]] = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        x0, y0, x1, y1 = box.tolist()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        lid = int(label_id.item()) if hasattr(label_id, "item") else int(label_id)
        lab = id2label.get(lid, str(lid))
        detections.append((x0, y0, max(0, x1 - x0), max(0, y1 - y0), str(lab), float(score)))
    return detections


def run_detection_owl(
    processor,
    model,
    device: str,
    pil_image: Image.Image,
    *,
    class_names: list[str],
    confidence_threshold: float = 0.5,
) -> list[tuple[int, int, int, int, str, float]]:
    """OWLv2: open-vocabulary detection for given class name strings."""
    if not class_names:
        return []

    pil_image = pil_image.convert("RGB")
    text_queries = [[f"a photo of a {name}" for name in class_names]]
    inputs = processor(text=text_queries, images=pil_image, return_tensors="pt")
    inputs = _batch_to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs)
    w, h = pil_image.size
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )[0]
    detections: list[tuple[int, int, int, int, str, float]] = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        x0, y0, x1, y1 = box.tolist()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        idx = int(label_id.item()) if hasattr(label_id, "item") else int(label_id)
        idx = min(max(idx, 0), len(class_names) - 1)
        lab = class_names[idx]
        detections.append((x0, y0, max(0, x1 - x0), max(0, y1 - y0), lab, float(score)))
    return detections


def run_detection_yolo(
    model,
    device: str,
    bgr: np.ndarray,
    confidence: float = 0.5,
) -> list[tuple[int, int, int, int, str, float]]:
    """YOLOv8 or YOLO-OIV7 on a BGR uint8 image."""
    results = model.predict(bgr, conf=confidence, verbose=False, device=device)
    detections: list[tuple[int, int, int, int, str, float]] = []
    for r in results:
        names = r.names or {}
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x0, y0, x1, y1 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            sc = float(box.conf[0].item())
            ci = int(box.cls[0].item())
            lab = names.get(ci, str(ci))
            detections.append(
                (int(x0), int(y0), int(x1 - x0), int(y1 - y0), lab, sc)
            )
    return detections


def run_detection_yolo_world(
    model,
    device: str,
    bgr: np.ndarray,
    class_names: list[str],
    confidence: float = 0.35,
) -> list[tuple[int, int, int, int, str, float]]:
    """YOLO-World with set_classes; BGR uint8 image."""
    if not class_names:
        return []
    model.set_classes(class_names)
    return run_detection_yolo(model, device, bgr, confidence)


def _draw_detections(bgr: np.ndarray, detections: list[tuple[int, int, int, int, str, float]]) -> np.ndarray:
    out = bgr.copy()
    for (x, y, bw, bh, label, score) in detections:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 100), 2)
        cv2.putText(
            out,
            f"{label} {score:.2f}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 100),
            1,
            cv2.LINE_AA,
        )
    return out


def _parse_world_classes(s: str | None) -> list[str]:
    if not s:
        return ["person", "car", "chair"]
    return [p.strip() for p in s.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralVision real-time detection")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (0, 1, …) or path to image/video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo-oiv7",
        choices=("yolo-oiv7", "yolo-world", "yolo", "detr"),
    )
    parser.add_argument(
        "--yolo-world-classes",
        type=str,
        default="",
        help='Comma-separated classes for yolo-world (default: "person,car,chair")',
    )
    parser.add_argument("--yolo-size", type=str, default="n", choices=("n", "s", "m", "l"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--every-n", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    # normalize yolo-size (strip accidental space)
    yolo_size = args.yolo_size.strip()

    every_n = args.every_n
    if every_n is None:
        every_n = 8 if args.model == "detr" else 4

    max_sz = args.max_size
    if max_sz is None:
        max_sz = 512 if args.model == "detr" else 640

    src = args.source
    if src.isdigit():
        cap_source: int | str = int(src)
        is_cam_or_video = True
    else:
        cap_source = src
        is_cam_or_video = str(src).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

    # Load model
    processor = None
    if args.model == "detr":
        processor, model, device = get_model()
        print("DETR loaded on", device, file=sys.stderr)
    elif args.model == "yolo":
        processor, model, device = get_yolo_model(yolo_size)
        print(f"YOLOv8 ({yolo_size}) on", device, file=sys.stderr)
    elif args.model == "yolo-oiv7":
        processor, model, device = get_yolo_oiv7_model(yolo_size)
        print(f"YOLOv8-OIV7 ({yolo_size}) on", device, file=sys.stderr)
    else:
        processor, model, device = get_yolo_world_model("s" if yolo_size == "n" else yolo_size)
        print(f"YOLO-World ({yolo_size}) on", device, file=sys.stderr)

    world_classes = _parse_world_classes(args.yolo_world_classes or None)

    def run_on_pil_or_bgr(pil: Image.Image | None, bgr: np.ndarray | None) -> list[tuple[int, int, int, int, str, float]]:
        if args.model == "detr" and pil is not None:
            return run_detection(processor, model, device, pil, args.confidence)
        if args.model == "yolo" or args.model == "yolo-oiv7":
            assert bgr is not None
            return run_detection_yolo(model, device, bgr, args.confidence)
        assert bgr is not None
        return run_detection_yolo_world(model, device, bgr, world_classes, args.confidence)

    # Single image file
    if not str(cap_source).isdigit() and not is_cam_or_video:
        path = Path(cap_source)
        if not path.is_file():
            print(f"Not a file: {path}", file=sys.stderr)
            sys.exit(1)
        bgr = cv2.imread(str(path))
        if bgr is None:
            print("Could not read image:", path, file=sys.stderr)
            sys.exit(1)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        max_side = max(bgr.shape[0], bgr.shape[1])
        scale = 1.0
        small_bgr = bgr
        if max_side > max_sz:
            scale = max_sz / max_side
            small_bgr = cv2.resize(
                bgr,
                (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale)),
            )
        small_pil = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
        dets = run_on_pil_or_bgr(small_pil, small_bgr)
        if scale != 1.0:
            inv = 1.0 / scale
            dets = [
                (int(x * inv), int(y * inv), int(w * inv), int(h * inv), lab, s)
                for (x, y, w, h, lab, s) in dets
            ]
        if args.no_show:
            for x, y, w, h, lab, s in dets:
                print(f"{lab}\t{s:.3f}\t{x},{y},{w},{h}")
            return
        vis = _draw_detections(bgr, dets)
        cv2.imshow("NeuralVision", vis)
        print("Press any key to close…", file=sys.stderr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print("Could not open source:", cap_source, file=sys.stderr)
        sys.exit(1)

    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_i += 1
            if frame_i % every_n != 0 and frame_i > 1:
                if not args.no_show:
                    cv2.imshow("NeuralVision", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
                continue

            h, w = frame.shape[:2]
            max_side = max(h, w)
            scale = 1.0
            small = frame
            if max_side > max_sz:
                scale = max_sz / max_side
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            pil_small = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            dets = run_on_pil_or_bgr(pil_small, small)
            if scale != 1.0:
                inv = 1.0 / scale
                dets = [
                    (int(x * inv), int(y * inv), int(bw * inv), int(bh * inv), lab, s)
                    for (x, y, bw, bh, lab, s) in dets
                ]
            vis = _draw_detections(frame, dets)
            if not args.no_show:
                cv2.imshow("NeuralVision", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            elif frame_i % (every_n * 10) == 0:
                print(f"frame {frame_i} dets: {len(dets)}", file=sys.stderr)
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
