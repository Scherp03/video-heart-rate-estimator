# VIDEO HEART RATE ESTIMATOR
A real-time heart rate estimator from webcam.

Project developed by @Scherp03 and @nicocecchin for the 2025 Signal, Image and Video course.

## Project Outline

This project is a real-time implementation of remote photoplethysmography (rPPG). The baseline concept was inspired by [webcam-pulse-detector](https://github.com/thearn/webcam-pulse-detector).
The project was successfully tested on Ubuntu, MacOS and Windows. The higher the camera quality, the greater the precision of the estimator.

### Methodology

1.  **Face Detection (ROI)**
    We utilize **[MediaPipe](https://ai.google.dev/edge/mediapipe/) Face Mesh** to detect facial landmarks. To minimize noise, we dynamically define the Region of Interest (ROI) by masking out the eyes, mouth, and eyebrows, focusing on skin areas with consistent blood flow visibility.

2.  **Signal Extraction (POS)**
    We implement the **Plane-Orthogonal-to-Skin (POS)** algorithm [^1] to extract the blood volume pulse signal from the ROI. This method projects, after performing spatial averaging on the ROI, the RGB color channels into a plane orthogonal to the skin tone, effectively separating the pulse signal from motion artifacts and illumination variations.

3.  **Frequency Analysis**
    A **Fast Fourier Transform (FFT)** is applied to the extracted temporal signal to identify the dominant frequency, which corresponds to the heart rate (BPM).

[^1]: Wang, W., et al. "Algorithmic Principles of Remote PPG." IEEE Transactions on Biomedical Engineering (2016).

---

## Prerequisites & Setup

To run this project, you need **Python 3.11**. Follow the steps below for your operating system.

### 1. Install Python 3.11

**Linux (Ubuntu/Debian)**
If `python3.11 --version` returns an error, use the DeadSnakes PPA:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

**macOS**
Install using [Homebrew](https://brew.sh/):
```bash
brew install python@3.11
```

**Windows**
Download and install Python 3.11 from [python.org](https://www.python.org/downloads/).

---

### 2. Create a Virtual Environment

Navigate to the project root and create a virtual environment.

**Linux / macOS**
```bash
python3.11 -m venv .venv
```

**Windows**
```powershell
python -m venv .venv
```

---

### 3. Activate the Environment

You need to activate the environment every time you start a new terminal session.

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate
```

---

### 4. Install Dependencies

Once the environment is active (you should see `(venv)` in your prompt), install the required libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Run the project

**Make sure you have a working webcam or an external camera connected**

From the project root, run the following command:

**Linux / macOS**
```bash
python3 main.py
```

**Windows**
```bash
python main.py
```
