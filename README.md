# ForensIQ — Image Forensics System

A web-based image forensics system that detects photo manipulation and AI-generated content using SIFT, SURF, AKAZE, ORB algorithms combined with CLIP-based deep learning.

## Features
- Copy-move forgery detection (SIFT, SURF, AKAZE, ORB, VB)
- Error Level Analysis (ELA)
- AI-generated image detection (CLIP + statistical signals)
- Heatmap visualization of manipulated regions
- Screenshot / digital content detection

## Requirements
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

## Usage
```bash
python app.py
```
Then open `http://localhost:5050` in your browser.

## CLIP Model Setup
The CLIP model (~374MB) must be downloaded separately:

1. Open the [Google Colab notebook](https://colab.research.google.com/drive/1yyEnbJWWXP3CgrDudvUNYt0ilYQK1nNc?usp=sharing)
2. Run all cells — `clip_processor.zip` will be downloaded automatically
3. Extract `clip_processor.zip` into the project root folder

Without CLIP, the system falls back to statistical analysis mode.

## Project Structure
```
ForensIQ/
├── app.py                 # Flask server
├── forensics_engine.py    # Core algorithms
├── ai_detector.py         # AI detection engine
├── main.html              # Web interface
├── requirements.txt
└── clip_processor/        # CLIP model (download via Colab)
```

## Tech Stack
Python · Flask · OpenCV · PyTorch · CLIP · HTML / CSS / JS