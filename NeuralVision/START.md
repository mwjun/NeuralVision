# Starting NeuralVision

You must be in **this folder** — the one that contains `src/`, `requirements.txt`, and `.venv`.  
(On this machine that is usually `NeuralVision/NeuralVision` inside the repo, not the outer `NeuralVision` folder.)

## Easiest: run the launcher script

On macOS, double‑clicking a `.sh` file usually does nothing useful; use Terminal:

```bash
cd /path/to/NeuralVision/NeuralVision
chmod +x start_gui.sh   # once
./start_gui.sh
```

This always uses `.venv/bin/python`, so you avoid the wrong‑Python errors below.

## One-time setup

```bash
cd /path/to/NeuralVision/NeuralVision
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(If you already have `.venv`, skip creating it and only run `pip install` when `requirements.txt` changes.)

## Start the graphical app (recommended)

Always use the **venv interpreter** (not system `python3` — system Python does not have OpenCV, PyTorch, etc.):

```bash
cd /path/to/NeuralVision/NeuralVision
.venv/bin/python -m src.gui
```

**Alternative** — activate the venv first, *then* `python` / `python3` points at the venv:

```bash
cd /path/to/NeuralVision/NeuralVision
source .venv/bin/activate
python -m src.gui
```

In the app: **Open image…** or drag an image onto the window. Use **Live…** for webcam (macOS: grant Camera access to Terminal or your IDE if prompted).

## Start from the command line

**Webcam:**

```bash
.venv/bin/python -m src.realtime_detect --source 0
```

**Single image:**

```bash
.venv/bin/python -m src.realtime_detect --source path/to/image.jpg
```

**Video file:**

```bash
.venv/bin/python -m src.realtime_detect --source path/to/video.mp4
```

## Troubleshooting

| Issue | What to try |
|--------|----------------|
| `No module named 'src'` | You are in the wrong directory. `cd` into the folder that contains `src/` (see top of this file), or run `./start_gui.sh` from that folder. |
| `No module named 'cv2'` (or `torch`, `PIL`, …) | You used system Python. Use `.venv/bin/python -m src.gui` or `source .venv/bin/activate` first — never plain `python3 -m src.gui` without the venv. |
| `python: command not found` | Use `.venv/bin/python` (full path) or `python3` only **after** `source .venv/bin/activate`. |
| Webcam does not open | Close other apps using the camera; on macOS check **System Settings → Privacy & Security → Camera**. |
| More options and models | See **README.md** in this folder. |
