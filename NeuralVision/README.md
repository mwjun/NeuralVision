# NeuralVision

**What it does:** You give it a photo (or webcam feed). The app finds **everyday objects** in the image—people, cars, dogs, chairs, bottles, laptops, etc.—draws boxes around them, and tells you what each one is and how confident it is (e.g. “person 0.95”, “cup 0.87”). No cloud or paid APIs; everything runs on your machine.

**Purpose:** See what’s in an image at a glance, count or find specific objects, or use it as a starting point for building something that reacts to what the camera sees (e.g. “person in frame”, “car detected”).

**What it detects:** In the **GUI** we use **OWLv2** (open-vocabulary): you type the class names you care about (e.g. `strawberry, person, bowl`), so you can detect **any objects**, not just a fixed list. The **CLI** (webcam/video) still uses DETR with 91 COCO classes. Both models are free and run locally.

**All tools are free and open source** — no paid APIs or cloud services required.

## Stack (free / open source only)

| Component        | Tool                          | License   |
|-----------------|-------------------------------|-----------|
| Framework       | PyTorch                       | BSD       |
| Models          | Hugging Face (DETR, OWLv2)       | Apache 2.0 |
| Vision / I/O    | OpenCV, Pillow                | Apache 2.0 / HPND |
| Dataset (train) | COCO, Open Images (optional)  | CC / CC-BY |

## Setup

- **Python**: 3.10 or newer (tested with 3.13).
- From the project root (the folder containing `src/` and `requirements.txt`):

```bash
cd NeuralVision
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **First run**: The DETR model is downloaded from Hugging Face on first use (one-time, ~160MB).

## Usage

Run from the project root.

### GUI (drag-and-drop)

Open the graphical app, then drag and drop an image onto the window, or click **Open image…**. Detections appear in the panel on the right; the image is shown with bounding boxes.

```bash
python -m src.gui
```

- **Detector (dropdown):** **YOLO-OIV7** (600 classes, fast, default), **YOLO-World** (any classes you type, fast), **YOLO** (80 classes), **OWLv2** (any classes), **DETR** (91 classes). Use **YOLO-OIV7** to detect many more things with no lag. Change anytime; applies on next Open image or Live → Start.
- **Classes to detect:** For OWLv2 and YOLO-World—comma-separated names. You can also type **thumb**, **finger** (or **index finger**, **middle finger**, **ring finger**, **pinky**) to detect fingers via MediaPipe; the thumb is labeled separately from other fingers.
- **Confidence** (0.1–0.99): detection threshold. **Note:** YOLO-World works better with lower confidence (0.2–0.4); the GUI auto-adjusts this.
- Supported image formats: JPEG, PNG, BMP, WebP, GIF.

**Live webcam detector:** Click **Live…** in the toolbar. Choose the detector in the main window first (or change it, then Stop and Start again in the Live window). Type what to detect in the input (e.g. `human`, `strawberry`, `cup`); the webcam feed shows boxes only for those objects. Change the text anytime while it’s running. Start/Stop to turn the camera on or off.

**Webcam troubleshooting:** If you see "Could not open webcam":
- **macOS**: Grant camera permission in System Settings → Privacy & Security → Camera → enable for Terminal/Python
- Close other apps using the camera (Zoom, FaceTime, etc.)
- Ensure a camera is connected

### Command-line

**Real-time detection (webcam):**

```bash
python -m src.realtime_detect --source 0
```

**Detection on image:** (use a real file path, e.g. `./photo.jpg`)

```bash
python -m src.realtime_detect --source path/to/your_image.jpg
```

**Detection on video file:**

```bash
python -m src.realtime_detect --source path/to/video.mp4
```

**Options:**

- `--model yolo-oiv7` (default), `yolo-world`, `yolo`, or `detr` — **yolo-oiv7** = 600 classes, usually **most accurate** for everyday objects (phone vs book, etc.). **yolo-world** = any classes; use `--yolo-world-classes` for better accuracy.
- `--yolo-world-classes "a,b,c"` — For **yolo-world** only. Comma-separated list of classes (e.g. `person,cell phone,laptop,book,poster,cup`). **Fewer, specific classes = more accurate**; a long list can cause confusion (e.g. phone labeled as book).
- `--yolo-size n|s|m|l` — `n` = fastest, `s`/`m`/`l` = more accurate. For yolo-world use `s` or `m` for better labels.
- `--confidence 0.5` — detection threshold. Lower = more objects.
- `--every-n N` — run every N frames (default: 4 for YOLO variants, 8 for DETR).
- `--max-size N` — max input size (default: 640 YOLO, 512 DETR).
- `--no-show` — disable the display window.

**Closing the app:** Press **Q**, click the **Quit** button (bottom-right), or close the window (X).

**More accurate labels (phone vs book, poster vs book, etc.):** Use **YOLO-OIV7** (default) — 600 fixed classes, trained to distinguish e.g. cell phone, book, poster. If you use **YOLO-World**, give it a **short, specific** class list via `--yolo-world-classes "person,cell phone,laptop,book,poster,cup"` or in the GUI; too many classes makes it confuse similar objects.

**Less lag:** Defaults are tuned for YOLO (every 4 frames, 640px). If still laggy: `--every-n 6` or `--max-size 416`. With DETR, use `--every-n 10 --max-size 416`.

**Fine-tuning (improve recognition on your own objects):** To get better on specific items (e.g. a certain product or animal), you can fine-tune YOLOv8 or DETR on your own labeled data. See the "Training / fine-tuning" section at the bottom of this README. Both models are free to train with PyTorch and public datasets (COCO, etc.) or your own.

**Troubleshooting:** If the display window does not appear (e.g. with `opencv-python-headless` on some setups), use `--no-show` or install `opencv-python` instead for GUI support. For webcam, ensure a camera is connected and not in use by another app.

## Testing

**Manual (quick check)**

1. **CLI on an image** — use any image file you have:
   ```bash
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   python -m src.realtime_detect --source /path/to/your/image.jpg --no-show
   ```
   You should see “Loading DETR…” then “Model on cpu” (or “cuda”), then the process exits after one frame. (With default `--model yolo-oiv7` you'll see "Loading YOLOv8-OIV7n…".)

2. **GUI** — open the app and open or drag-and-drop an image:
   ```bash
   python -m src.gui
   ```
   Load an image; the left panel should show it with green boxes and the right panel should list detections.

**Automated (optional)**

From the project root, install pytest and run the test suite (first run may download the DETR model):

```bash
pip install pytest
pytest tests/ -v
```

- `tests/test_detection.py` checks that imports work and that `run_detection` returns a list of detections for a small test image.

## Project structure

```
NeuralVision/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── gui.py               # Drag-and-drop GUI for image detection
│   ├── realtime_detect.py   # Real-time object detection (CLI)
│   └── model.py             # Transformer model loading (DETR)
├── tests/
│   ├── __init__.py
│   └── test_detection.py    # Pytest: imports and run_detection
└── outputs/                 # Saved images/videos (optional)
```

## Training / fine-tuning (optional)

To train or fine-tune on your own data (e.g. to approach 98.7% on a specific benchmark):

- Use PyTorch + Hugging Face `Trainer` with a DETR (or other transformer) checkpoint.
- Datasets: COCO (free), Open Images, or your own labeled data.
- Run on your own machine or free-tier GPU (e.g. Google Colab free tier, Kaggle notebooks).

## License

This project uses only open-source dependencies; see each package’s license. Project code: MIT.
