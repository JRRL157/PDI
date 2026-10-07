# PDI - Digital Puppetry & Facial Reenactment

A real-time facial expression transfer and digital puppetry project using **OpenCV** and **MediaPipe FaceMesh**. The system tracks facial landmarks on a "Master" face and transfers their expressions onto a "Puppet" face using homography mapping and piecewise affine triangle warping.

---

## Prerequisites

- **Python Version**: **Python 3.10** (Python 3.10 to 3.12 are supported).  
  > [!IMPORTANT]
  > Python 3.13+ is **not supported** because MediaPipe's legacy solutions API (`mp.solutions.face_mesh`) was deprecated and removed in newer versions.
- **Cameras**: Two video capture sources (e.g., two webcams, or one webcam + a virtual camera such as OBS / DroidCam / video stream).

---

## Quickstart Guide

Follow these steps to set up and run the project from scratch:

### 1. Clone the Repository

```bash
git clone https://github.com/jrrl/PDI.git
cd PDI
```

### 2. Set Up a Virtual Environment

Ensure you are using **Python 3.10** (or 3.11/3.12):

#### Using standard `venv`:
```bash
# On Linux / macOS
python3.10 -m venv .venv
source .venv/bin/activate

# On Windows
py -3.10 -m venv .venv
.venv\Scripts\activate
```

#### Using `uv` (Fast alternative):
```bash
# Tip: Use --seed if you want standard 'pip' installed inside the venv
uv venv --seed --python 3.10 .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the pinned dependencies from `requirements.txt`:

```bash
# If using standard pip:
pip install -r requirements.txt

# Or if using uv:
uv pip install -r requirements.txt
```

### 4. Run the Application

```bash
# Default (640x480 resolution, master=0, puppet=2)
python src/main.py

# High performance mode (lower resolution for faster FPS):
python src/main.py --width 480 --height 360

# Lightweight mode:
python src/main.py --width 320 --height 240

# Custom camera indices:
python src/main.py --master 0 --puppet 2 --width 640 --height 480
```

---

## Usage & Controls

- **Master Window**: Displays the video feed controlling the expressions (`--master`, default `0`).
- **Puppet Window**: Displays the target face receiving the expressions (`--puppet`, default `2`).
- **Result Window**: Real-time synthesized output with the puppet face animated by the master.
- **Quit**: Press the **`q`** key while focused on any OpenCV window to close the application.

---

## Camera & Resolution Configuration

### Resolution Tuning
Piecewise affine warping across 72 facial triangles is computationally intensive. If you experience latency or low FPS:
- **Default (`640x480`)**: Balanced quality and speed.
- **Fast (`480x360`)**: Recommended for real-time responsiveness on CPUs.
- **Ultra-fast (`320x240`)**: Maximum frame rate.

### Camera Indices on Linux
On Linux, physical USB and built-in webcams often expose two device nodes (a video stream node and a metadata node):
- `0`: Built-in laptop webcam video stream (`/dev/video0`)
- `1`: Built-in webcam metadata node (`/dev/video1`, not a video stream!)
- `2`: External webcam video stream (`/dev/video2`)

By default, `src/main.py` uses `--master 0` and `--puppet 2` (falling back to `1` if `2` is unavailable).

If you only have one physical webcam or want to test with a pre-recorded video:
1. You can route your phone or a secondary video stream through **OBS Virtual Camera** or **DroidCam**.
2. Or pass pre-recorded videos by adapting `VideoCapture` in `src/main.py`.

---

## How It Works

1. **Facial Landmark Detection**: MediaPipe FaceMesh tracks 468+ facial keypoints on both video streams simultaneously.
2. **Homography Estimation**: A planar homography matrix is computed using landmark correspondences to map the master face geometry into the puppet's coordinate space.
3. **Delaunay Triangulation & Warping**: Face meshes are divided into triangles (`face_triangles`). For each triangle, an affine transformation matrix is calculated (`cv2.getAffineTransform`) to warp pixels from the puppet to the transformed master expression (`cv2.warpAffine`).
4. **Alpha Blending & Masking**: The warped regions are masked and blended back into the result canvas in real-time.

---

## Project Structure

```text
├── src/
│   └── main.py              # Main execution script
├── assets/                  # Sample faces and reference graphics
├── references/              # Project documentation and triangulation diagrams
├── model/                   # Super-resolution models (FSRCNN)
├── .python-version          # Python version lock for version managers (3.10)
├── pyproject.toml           # Project metadata and version constraints
├── requirements.txt         # Pinned dependencies
└── README.md                # Project documentation
```