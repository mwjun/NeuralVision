"""
NeuralVision GUI: drag-and-drop image detection and live webcam detector.
Uses Tkinter + tkinterdnd2 (free & open source). Run: python -m src.gui
"""
from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .hand import filter_hand_detections, is_hand_class, run_hand_detection, wants_hand_detection
from .model import get_model, get_owl_model, get_yolo_model, get_yolo_oiv7_model, get_yolo_world_model
from .realtime_detect import run_detection, run_detection_owl, run_detection_yolo, run_detection_yolo_world
from .vocabulary import get_lvis_primary_names

# Image extensions we accept
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def _parse_drop_paths(data: str) -> list[Path]:
    """Parse DnD event data into file paths (handles {path} and plain path)."""
    paths = []
    # Some systems use curly braces: {/path/to/file.jpg}
    for part in re.findall(r"\{[^}]+\}|[^\s{}]+", data.strip()):
        part = part.strip("{}").strip()
        if part:
            p = Path(part)
            if p.exists():
                paths.append(p)
    return paths


def _find_first_image(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.suffix.lower() in IMAGE_EXT:
            return p
    return None


DETECTOR_OPTIONS = [
    ("YOLO-OIV7 (600 classes, fast)", "yolo_oiv7"),
    ("YOLO-World (any classes, fast)", "yolo_world"),
    ("YOLO (80 classes, fast)", "yolo"),
    ("OWLv2 (any classes)", "owl"),
    ("DETR (91 classes, transformer)", "detr"),
]
DETECTOR_DISPLAY_TO_KEY = {d: k for d, k in DETECTOR_OPTIONS}
DETECTOR_KEY_TO_DISPLAY = {k: d for d, k in DETECTOR_OPTIONS}


class NeuralVisionGUI:
    def __init__(self, root: tk.Tk, confidence: float = 0.2):
        self.root = root
        self.confidence = confidence
        self.processor = None
        self.model = None
        self.device = None
        self._model_loaded = False
        self._loaded_detector: str | None = None  # "owl" | "yolo" | "detr" — which model is in memory
        self._current_photo = None  # keep reference for Label

        root.title("NeuralVision — Object Detection")
        root.minsize(800, 500)
        root.geometry("1000x620")

        self._build_toolbar()
        self._build_main()
        self._register_drop()

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(toolbar, text="Detector:").pack(side=tk.LEFT, padx=(0, 4))
        self.detector_var = tk.StringVar(value=DETECTOR_OPTIONS[0][0])
        self.detector_combo = ttk.Combobox(
            toolbar, textvariable=self.detector_var, state="readonly", width=28
        )
        self.detector_combo["values"] = [d for d, _ in DETECTOR_OPTIONS]
        self.detector_combo.pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="(change anytime, applies on next run)").pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Button(toolbar, text="Open image…", command=self._on_open).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Live…", command=self._open_live).pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="  or drag an image into the box →").pack(side=tk.LEFT)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Label(toolbar, text="Confidence:").pack(side=tk.LEFT, padx=2)
        self.conf_var = tk.DoubleVar(value=self.confidence)
        spin = ttk.Spinbox(
            toolbar, from_=0.1, to=0.99, increment=0.05,
            textvariable=self.conf_var, width=5
        )
        spin.pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="(lower = more detections)", foreground="gray").pack(side=tk.LEFT)

    def _build_main(self) -> None:
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Left: drop box + image display
        left = ttk.Frame(main)
        main.add(left, weight=2)
        # Dedicated drop zone (visible box with dashed border)
        self.drop_frame = tk.Frame(left, bg="#e8f4ea", highlightbackground="#2d7d4a", highlightthickness=2)
        self.drop_frame.pack(fill=tk.BOTH, expand=True)
        self.drop_frame.configure(highlightbackground="#2d7d4a")
        self._drop_label = tk.Label(
            self.drop_frame,
            text="Drop image here\n\nor click «Open image…» in the toolbar",
            anchor=tk.CENTER,
            font=("Helvetica", 14),
            fg="#333",
            bg="#e8f4ea",
            cursor="hand2"
        )
        self._drop_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self._drop_label.bind("<Button-1>", lambda e: self._on_open())
        # Image result goes in the same frame (we replace drop_label content with photo)
        self.img_label = self._drop_label  # reuse so we can show image in the box

        # Right: classes to detect + results list
        right = ttk.Frame(main)
        main.add(right, weight=1)
        ttk.Label(right, text="Classes to detect (OWLv2 & YOLO-World):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.classes_var = tk.StringVar(
            value="person, dog, cat, car, bottle, cup, strawberry, apple, chair, bowl"
        )
        classes_row = ttk.Frame(right)
        classes_row.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
        classes_entry = ttk.Entry(classes_row, textvariable=self.classes_var, width=22)
        classes_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(classes_row, text="Human only", width=10, command=self._set_human_only).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(right, text="OWLv2 & YOLO-World: type exactly what to detect (e.g. cell phone, poster). No hardcoded list.", font=("Helvetica", 9), foreground="gray").pack(anchor=tk.W)
        ttk.Label(right, text="Tip: lower Confidence = more detections", font=("Helvetica", 9), foreground="gray").pack(anchor=tk.W)
        ttk.Label(right, text="Detections", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(8, 0))
        results_frame = ttk.Frame(right)
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_list = tk.Listbox(results_frame, font=("Menlo", 11), height=20)
        scroll = ttk.Scrollbar(results_frame)
        self.results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_list.config(yscrollcommand=scroll.set)
        scroll.config(command=self.results_list.yview)

    def _register_drop(self) -> None:
        """Register drag-and-drop on the drop box and the whole window."""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            for widget in (self.drop_frame, self._drop_label, self.root):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", self._on_drop)
                except Exception:
                    pass
        except Exception:
            pass  # DnD not available; Open button still works

    def _get_detector_key(self) -> str:
        """Current detector choice from dropdown: 'owl', 'yolo', or 'detr'."""
        return DETECTOR_DISPLAY_TO_KEY.get(self.detector_var.get(), "owl")

    def _ensure_model(self) -> bool:
        """Load the currently selected detector if not already loaded. Switch model in real time."""
        wanted = self._get_detector_key()
        if self._model_loaded and self._loaded_detector == wanted:
            return True
        try:
            self._drop_label.config(text=f"Loading {self.detector_var.get()}…")
            self.root.update()
            if wanted == "owl":
                self.processor, self.model, self.device = get_owl_model()
            elif wanted == "yolo":
                self.processor, self.model, self.device = get_yolo_model("n")
            elif wanted == "yolo_oiv7":
                self.processor, self.model, self.device = get_yolo_oiv7_model("n")
            elif wanted == "yolo_world":
                self.processor, self.model, self.device = get_yolo_world_model("s")
            else:
                self.processor, self.model, self.device = get_model()
            self._loaded_detector = wanted
            self._model_loaded = True
            return True
        except Exception as e:
            messagebox.showerror("Model load failed", str(e))
            self._show_placeholder()
            return False

    def _set_human_only(self) -> None:
        """Preset: detect only people (human detector mode)."""
        self.classes_var.set("person")
        self.conf_var.set(0.35)

    def _open_live(self) -> None:
        """Open the live webcam detector window (type what to detect, e.g. human or strawberry)."""
        _LiveDetectorWindow(self.root, self)

    def _on_open(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", " ".join(f"*{e}" for e in IMAGE_EXT)),
                ("All files", "*.*"),
            ]
        )
        if path:
            self._process_path(Path(path))

    def _on_drop(self, event) -> None:
        data = getattr(event, "data", None) or ""
        if not isinstance(data, str):
            data = str(data)
        paths = _parse_drop_paths(data)
        path = _find_first_image(paths)
        if path:
            self._process_path(path)
        elif paths:
            messagebox.showinfo("Not an image", "Dropped file is not a supported image.")

    def _process_path(self, path: Path) -> None:
        if not self._ensure_model():
            return
        try:
            pil_image = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Open image failed", str(e))
            return

        self._drop_label.config(text="Detecting…")
        self.root.update()

        try:
            conf = float(self.conf_var.get())
        except (TypeError, ValueError):
            conf = 0.5
        try:
            class_names = [s.strip() for s in self.classes_var.get().split(",") if s.strip()]
            other_classes = [c for c in class_names if not is_hand_class(c)]
            hand_wanted = wants_hand_detection(class_names)

            if self._loaded_detector == "owl":
                if not other_classes and not hand_wanted:
                    messagebox.showinfo("No classes", "For OWLv2 enter at least one class (e.g. strawberry, person, thumb).")
                    self._show_placeholder()
                    return
                detections = run_detection_owl(
                    self.processor, self.model, self.device,
                    pil_image, class_names=other_classes,
                    confidence_threshold=conf
                ) if other_classes else []
            elif self._loaded_detector == "yolo_world":
                if not other_classes and not hand_wanted:
                    messagebox.showinfo("Classes needed", "Type at least one class to detect (e.g. cell phone, thumb, cup).")
                    self._show_placeholder()
                    return
                world_conf = max(0.15, conf * 0.5)
                bgr = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
                detections = run_detection_yolo_world(self.model, self.device, bgr, other_classes, world_conf) if other_classes else []
            elif self._loaded_detector in ("yolo", "yolo_oiv7"):
                bgr = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
                detections = run_detection_yolo(self.model, self.device, bgr, conf)
            else:
                detections = run_detection(self.processor, self.model, self.device, pil_image, conf)

            if hand_wanted:
                try:
                    img_arr = np.asarray(pil_image)
                    hand_dets = run_hand_detection(img_arr, pil_image.width, pil_image.height)
                    hand_dets = filter_hand_detections(hand_dets, class_names)
                    detections = list(detections) + hand_dets
                except Exception:
                    pass  # skip hand detections on error
        except Exception as e:
            messagebox.showerror("Detection failed", str(e))
            self._show_placeholder()
            return

        # Draw boxes on a copy
        draw_image = pil_image.copy()
        draw = ImageDraw.Draw(draw_image)
        for (x, y, w, h, label, score) in detections:
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 100), width=3)
            draw.text((x, max(0, y - 16)), f"{label} {score:.2f}", fill=(0, 255, 100))

        # Scale to fit display (max 700px wide)
        w, h = draw_image.size
        max_w = 700
        if w > max_w:
            ratio = max_w / w
            new_w, new_h = max_w, int(h * ratio)
            draw_image = draw_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self._show_image(draw_image)
        self._show_results(detections)

    def _show_placeholder(self) -> None:
        self._current_photo = None
        self._drop_label.config(
            image="", text="Drop image here\n\nor click «Open image…» in the toolbar",
            fg="#333", bg="#e8f4ea"
        )

    def _show_image(self, pil_image: Image.Image) -> None:
        self._current_photo = ImageTk.PhotoImage(pil_image)
        self._drop_label.config(image=self._current_photo, text="", bg="#e8f4ea")

    def _show_results(self, detections: list) -> None:
        self.results_list.delete(0, tk.END)
        if not detections:
            self.results_list.insert(tk.END, "(no detections above threshold)")
            self.results_list.insert(tk.END, "")
            self.results_list.insert(tk.END, "Try lowering Confidence (e.g. 0.3–0.5)")
            self.results_list.insert(tk.END, "in the toolbar and run again.")
        else:
            for (_x, _y, _w, _h, label, score) in detections:
                self.results_list.insert(tk.END, f"  {label}: {score:.2f}")


class _LiveDetectorWindow:
    """Live webcam window: type what to detect (e.g. human, strawberry), only those get boxes."""

    def __init__(self, parent: tk.Tk, main_app: NeuralVisionGUI) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title("NeuralVision — Live detector")
        self.win.minsize(800, 600)
        self.win.geometry("1280x800")
        self.win.resizable(True, True)
        self.main_app = main_app
        self._running = False
        self._thread: threading.Thread | None = None
        self._result_queue: queue.Queue = queue.Queue()
        self._photo_ref: ImageTk.PhotoImage | None = None  # keep reference
        self._live_classes: list[str] = []  # Only what user typed; updated on Enter (no hardcoded list)

        # Top: controls bar
        top = ttk.Frame(self.win, padding=(10, 10, 10, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text="What to detect:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 6))
        self.entry_var = tk.StringVar(value="")
        entry = ttk.Entry(top, textvariable=self.entry_var, width=36, font=("Helvetica", 10))
        entry.pack(side=tk.LEFT, padx=4)
        entry.bind("<Return>", self._on_enter_classes)
        ttk.Label(top, text="(comma-separated; press Enter to apply)", foreground="gray", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        self._use_1200_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Detect 1200+ objects (YOLO-World)", variable=self._use_1200_var, command=self._on_toggle_1200).pack(side=tk.LEFT, padx=8)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Button(top, text="Start", command=self._start, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Stop", command=self._stop, width=8).pack(side=tk.LEFT, padx=2)
        self._status_var = tk.StringVar(value="Stopped")
        ttk.Label(top, textvariable=self._status_var, foreground="gray", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(12, 0))

        # Content: video (left) + detection log (right) so it’s clear and resizable
        content = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        video_container = ttk.Frame(content, padding=(0, 0, 4, 0))
        content.add(video_container, weight=3)
        self._video_inner = tk.Frame(video_container, bg="#1a1a1a", highlightbackground="#444", highlightthickness=1)
        self._video_inner.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(
            self._video_inner,
            text="Click Start to begin live detection.\n\nResize this window to see a larger view.",
            font=("Helvetica", 12),
            fg="#888",
            bg="#1a1a1a",
            compound=tk.CENTER,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.Frame(content)
        content.add(log_frame, weight=1)
        ttk.Label(log_frame, text="Detection log", font=("Helvetica", 10, "bold")).pack(side=tk.TOP, anchor=tk.W)
        log_inner = ttk.Frame(log_frame)
        log_inner.pack(fill=tk.BOTH, expand=True)
        self._log_listbox = tk.Listbox(log_inner, font=("Menlo", 10), height=20, width=32)
        log_scroll = ttk.Scrollbar(log_inner)
        self._log_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_listbox.config(yscrollcommand=log_scroll.set)
        log_scroll.config(command=self._log_listbox.yview)
        ttk.Button(log_frame, text="Clear log", command=self._clear_log).pack(side=tk.TOP, pady=(6, 0))

        self._log_max_lines = 500
        self._log_already_written = False  # only log once per run when first detection happens
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _clear_log(self) -> None:
        self._log_listbox.delete(0, tk.END)

    def _on_toggle_1200(self) -> None:
        """When toggling 1200+ objects, reset log so next detection is logged."""
        self._log_already_written = False
        if self._use_1200_var.get():
            n = len(get_lvis_primary_names())
            self._status_var.set(f"Will detect {n}+ objects (YOLO-World)")
        elif self._live_classes:
            self._status_var.set(f"Detecting: {', '.join(self._live_classes[:5])}{'…' if len(self._live_classes) > 5 else ''}")
        else:
            self._status_var.set("No classes — type e.g. cell phone, poster and press Enter")

    def _on_enter_classes(self, event=None) -> None:
        """On Enter: use exactly what the user typed as the class list. No hardcoded fallback."""
        raw = self.entry_var.get().strip()
        self._live_classes[:] = [s.strip() for s in raw.split(",") if s.strip()]
        self._log_already_written = False  # allow a new log entry when user changes what to detect
        if self._use_1200_var.get():
            self._status_var.set(f"Will detect {len(get_lvis_primary_names())}+ objects (1200+ mode)")
        elif self._live_classes:
            self._status_var.set(f"Detecting: {', '.join(self._live_classes[:5])}{'…' if len(self._live_classes) > 5 else ''}")
        else:
            self._status_var.set("No classes — type e.g. cell phone, poster and press Enter")

    def _on_close(self) -> None:
        self._stop()
        self.win.destroy()

    def _start(self) -> None:
        if self._running:
            return
        # Load or switch detector based on main window dropdown (no Python restart needed)
        try:
            self.video_label.config(text="Loading detector…")
            self.win.update()
            if not self.main_app._ensure_model():
                return
        except Exception as e:
            messagebox.showerror("Model load failed", str(e), parent=self.win)
            return
        self._running = True
        self._log_already_written = False
        if self.main_app._loaded_detector == "yolo_world" and self._use_1200_var.get():
            self._status_var.set(f"Live — detecting {len(get_lvis_primary_names())}+ objects")
        elif self.main_app._loaded_detector == "yolo_world" and not self._live_classes:
            self._status_var.set("Live — type classes (e.g. cell phone, poster), press Enter")
        else:
            self._status_var.set("Live")
        self.video_label.config(text="")
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._pump_queue()

    def _stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._status_var.set("Stopped")
        self.video_label.config(text="Stopped. Click Start to begin again.", image="")

    def _worker(self) -> None:
        # Try multiple camera indices (0, 1, 2) - macOS might use different index
        cap = None
        for idx in [0, 1, 2]:
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    cap = test_cap
                    break
                test_cap.release()
        if cap is None:
            self._result_queue.put(("error", "Could not open webcam.\n\nTroubleshooting:\n• Ensure camera is connected\n• Grant camera permission (macOS: System Settings > Privacy & Security > Camera)\n• Close other apps using the camera\n• Try restarting the app"))
            return
        processor = self.main_app.processor
        model = self.main_app.model
        device = self.main_app.device
        detector = self.main_app._loaded_detector
        conf = 0.3
        try:
            conf = float(self.main_app.conf_var.get())
        except (TypeError, ValueError):
            pass
        max_side = 640  # inference at 640 for speed; we draw on full-res frame for crisp display
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                if max(h, w) > max_side:
                    scale = max_side / max(h, w)
                    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    small = frame
                    scale = 1.0
                class_names = list(self._live_classes)
                other_classes = [c for c in class_names if not is_hand_class(c)]
                hand_wanted = wants_hand_detection(class_names)
                try:
                    if detector == "owl":
                        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                        detections = run_detection_owl(
                            processor, model, device, pil,
                            class_names=other_classes, confidence_threshold=conf
                        ) if other_classes else []
                    elif detector == "yolo_world":
                        if self._use_1200_var.get():
                            world_classes = get_lvis_primary_names()
                        else:
                            world_classes = other_classes if other_classes else []
                        world_conf = max(0.15, conf * 0.5)
                        detections = run_detection_yolo_world(model, device, small, world_classes, world_conf) if world_classes else []
                    elif detector in ("yolo", "yolo_oiv7"):
                        detections = run_detection_yolo(model, device, small, conf)
                    else:
                        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                        detections = run_detection(processor, model, device, pil, conf)
                except Exception:
                    detections = []
                if scale != 1.0:
                    inv = 1.0 / scale
                    detections = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv), lab, s) for x, y, w, h, lab, s in detections]
                if hand_wanted:
                    try:
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        hand_dets = run_hand_detection(rgb, small.shape[1], small.shape[0])
                        hand_dets = filter_hand_detections(hand_dets, class_names)
                        if scale != 1.0:
                            inv = 1.0 / scale
                            hand_dets = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv), lab, s) for x, y, w, h, lab, s in hand_dets]
                        detections = list(detections) + hand_dets
                    except Exception:
                        pass  # keep webcam running; skip hand detections this frame
                draw_frame = frame.copy()
                # Scale font/line with frame size so labels stay readable when window is large
                font_scale = max(0.5, min(1.2, 0.0008 * max(draw_frame.shape[:2])))
                thickness = max(1, int(round(1.5 * font_scale)))
                for (x, y, bw, bh, label, score) in detections:
                    cv2.rectangle(draw_frame, (x, y), (x + bw, y + bh), (0, 255, 100), max(2, thickness))
                    cv2.putText(draw_frame, f"{label} {score:.2f}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 100), thickness)
                rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                pil_out = Image.fromarray(rgb)
                # Keep display crisp: cap at 1280 so we don't downscale too much (was 640)
                pw, ph = pil_out.size
                if max(pw, ph) > 1280:
                    r = 1280 / max(pw, ph)
                    pil_out = pil_out.resize((int(pw * r), int(ph * r)), Image.Resampling.LANCZOS)
                self._result_queue.put(("frame", pil_out, detections))
        finally:
            cap.release()

    def _pump_queue(self) -> None:
        # Classes are updated only when user presses Enter (_on_enter_classes), not every frame
        try:
            while True:
                msg = self._result_queue.get_nowait()
                if msg[0] == "error":
                    messagebox.showerror("Webcam", msg[1], parent=self.win)
                    self._stop()
                    return
                if msg[0] == "frame":
                    pil_img = msg[1]
                    detections = msg[2] if len(msg) > 2 else []
                    if detections and not self._log_already_written:
                        self._log_already_written = True
                        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        parts = [f"{label} {score:.2f}" for (_, _, _, _, label, score) in detections]
                        line = f"[{ts}] " + ", ".join(parts)
                        self._log_listbox.insert(tk.END, line)
                        n = self._log_listbox.size()
                        if n > self._log_max_lines:
                            self._log_listbox.delete(0, n - self._log_max_lines - 1)
                        self._log_listbox.see(tk.END)
                    # Scale to fit current label size so the video fills the resizable window
                    lw = max(320, self.video_label.winfo_width())
                    lh = max(240, self.video_label.winfo_height())
                    pw, ph = pil_img.size
                    if lw > 1 and lh > 1 and pw > 0 and ph > 0:
                        r = min(lw / pw, lh / ph)
                        if abs(r - 1.0) > 0.01:
                            new_w, new_h = max(1, int(pw * r)), max(1, int(ph * r))
                            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    self._photo_ref = ImageTk.PhotoImage(pil_img)
                    self.video_label.config(image=self._photo_ref, text="")
        except queue.Empty:
            pass
        if self._running:
            self.win.after(80, self._pump_queue)


def main() -> None:
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    app = NeuralVisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
