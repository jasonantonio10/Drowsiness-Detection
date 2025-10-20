# Drowsiness Eye Detection

Real-time driver drowsiness detection using eye landmark tracking and the Eye Aspect Ratio (EAR). The app monitors blink frequency and eye openness from a live webcam feed and triggers an audible/visual alert when drowsiness is detected.

## Features
- Real-time eye landmark tracking from webcam
- EAR-based drowsiness detection with adjustable thresholds
- Blink detection and consecutive-frame logic to reduce false alarms
- On-screen overlays: FPS, EAR, status, and guidance
- Audible alert when drowsiness is detected
- Modular design to swap between MediaPipe or dlib facial landmarks

## Tech Stack
- Python 3.8+
- OpenCV (`cv2`) for video capture and drawing
- One of: MediaPipe Face Mesh or dlib facial landmarks
- NumPy for vector math
- Optional: `playsound`/`simpleaudio`/`winsound` for alarm

## How It Works (EAR)
The Eye Aspect Ratio (EAR) is computed from six eye landmarks. When eyes are open, EAR is high; when eyes close, EAR drops. If the EAR remains below a threshold for a sequence of frames, the system considers the user drowsy.

Basic formula per eye (p1..p6 are eye landmarks):

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

Typical thresholds: `EAR_THRESHOLD ≈ 0.20–0.30`, `CONSEC_FRAMES ≈ 15–48` depending on FPS and camera.

## Project Structure (example)
Adjust to match your repo:

```
.
├─ src/
│  ├─ main.py                # entry point
│  ├─ detector.py            # EAR, blink logic
│  ├─ landmarks_mediapipe.py # MediaPipe backend (optional)
│  ├─ landmarks_dlib.py      # dlib backend (optional)
│  └─ utils.py               # helpers (drawing, smoothing)
├─ models/                   # shape predictor or assets (if any)
├─ requirements.txt
└─ README
```

## Setup
1) Create a virtual environment (Windows PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:
- If you have `requirements.txt`:
```
pip install -r requirements.txt
```
- Otherwise, start with:
```
pip install opencv-python numpy mediapipe playsound
```
If you use dlib instead of MediaPipe, install it per your platform:
```
pip install dlib
```
(Note: dlib often requires CMake and a C++ compiler. MediaPipe is simpler on most systems.)

## Usage
Run the main entry point (adjust path/name if different):
```
python src/main.py --backend mediapipe --camera 0 --ear-threshold 0.25 --consec-frames 30 --alarm on
```

Common flags (may vary by your implementation):
- `--backend {mediapipe|dlib}`: landmark detection backend
- `--camera <index>`: webcam index (default 0)
- `--ear-threshold <float>`: EAR threshold for closed eyes
- `--consec-frames <int>`: frames below threshold before alert
- `--alarm {on|off}`: enable/disable audible alert

Hotkeys (suggested):
- `q` to quit
- `a` toggle alarm
- `h` toggle on-screen help

## Configuration Tips
- Calibrate `--ear-threshold` per person/camera. Start at 0.25 and tune ±0.03.
- Calibrate `--consec-frames` based on FPS. Higher FPS usually needs more frames.
- If using an external camera, try `--camera 1` (or higher) if 0 fails.
- Consider smoothing EAR with a small moving average to reduce jitter.

## Performance
- Prefer MediaPipe for simpler installs; dlib is CPU-heavy without AVX.
- Reduce frame size (e.g., 640×480) for better FPS.
- Draw fewer overlays or update text less frequently to save cycles.

## Troubleshooting
- Camera not found: ensure no other app uses the camera; try another index.
- Low FPS: reduce resolution, switch to MediaPipe, or disable heavy overlays.
- dlib install errors: install CMake and a C++ toolchain, or switch to MediaPipe.
- No audio: verify speakers, try `winsound` on Windows or `simpleaudio`.
- EAR unstable: ensure good lighting; re-tune threshold; try smoothing.

## Contributing
Issues and PRs are welcome. Please describe environment (OS, Python, backend) and include logs/screenshots.

## License
[Choose a license and replace this section, e.g., MIT]

## Acknowledgments
- EAR method from Tereza Soukupová and Jan Čech, “Real-Time Eye Blink Detection Using Facial Landmarks”.
- MediaPipe Face Mesh and dlib facial landmarks for robust tracking.

