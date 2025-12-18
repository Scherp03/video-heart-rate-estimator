# VIDEO HEART RATE ESTIMATOR
A real-time heart rate estimator from webcam.

Project developed by @Scherp03 and @nicocecchin for the 2025 Signal, Image and Video course.

## Prerequisites & Setup

To run this project, you need Python 3.11. If you are on Ubuntu and don't have it installed, follow the steps below to set up the environment correctly.
1. Install Python 3.11

If python3.11 --version returns an error, install it using the DeadSnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

2. Create a Virtual Environment

Navigate to the project root and create a virtual environment specifically for this project:

```bash
python3.11 -m venv .venv
```

3. Activate the Environment

You need to activate the environment every time you start a new terminal session:

Linux/macOS:

```bash
source .venv/bin/activate
```

4. Install Dependencies

Once the environment is active (you should see (.venv) in your prompt), install the required libraries:
Bash

```bash
pip install --upgrade pip
pip install -r requirements.txt
```