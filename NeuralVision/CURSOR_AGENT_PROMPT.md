# Prompt for Cursor Agent — NeuralVision

**Copy everything below the line into a new Cursor chat when you have the NeuralVision project open as the workspace. The agent should work in this project folder only.**

---

I need you to get the **NeuralVision** project running on my machine.

## What this project is
- **NeuralVision**: Real-time object detection and scene understanding using transformer-based models (DETR).
- All dependencies must remain **free and open source** (no paid APIs or cloud services).
- Stack: Python, PyTorch, Hugging Face Transformers (DETR), OpenCV, Pillow.

## Project location and structure
- **Workspace**: This repo is the NeuralVision project root (the folder containing `src/`, `requirements.txt`, and this file).
- **Entry point for detection**: `src/realtime_detect.py` — run with `python -m src.realtime_detect --source 0` for webcam, or `--source path/to/image.jpg` for an image.
- **Model loading**: `src/model.py` loads the DETR model from Hugging Face (`facebook/detr-resnet-50`).

## Your tasks
1. **Environment**: Ensure a Python virtual environment is used (e.g. `.venv` in the project root). Create it if missing.
2. **Dependencies**: Install from `requirements.txt`. Resolve any version or platform issues (e.g. Windows vs macOS, Python 3.10+).
3. **Run the app**: Run real-time object detection:
   - Webcam: `python -m src.realtime_detect --source 0`
   - Image: `python -m src.realtime_detect --source path/to/image.jpg`
4. **Fix any errors**: If the program crashes, fails to import, or the model fails to load, fix the code or dependencies so that:
   - Imports succeed.
   - The DETR model and processor load from Hugging Face (first run may download the model).
   - Detection runs on at least one source (webcam or image) and draws bounding boxes with labels.
5. **Document**: If you change setup steps or add new requirements, update `README.md` so I can run it again later.

## Constraints
- Do not introduce paid or proprietary APIs or services.
- Keep the project runnable with `python -m src.realtime_detect` from the project root after `pip install -r requirements.txt`.

Please start by checking the environment and `requirements.txt`, then install, run, and fix any issues until detection works.
