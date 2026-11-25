# ForensIQ — Image Forensics System

A web-based image forensics system that detects photo manipulation and AI-generated content using SIFT, SURF, AKAZE, ORB algorithms combined with CLIP-based deep learning.

## Features
- Copy-move forgery detection (SIFT, SURF, AKAZE, ORB, VB)
- Error Level Analysis (ELA)
- AI-generated image detection — **three independent channels, fully offline**
  (CLIP zero-shot + a supervised Swin Transformer detector + 5 statistical signals)
- **Medium classification** — photograph / artwork / 3D render / screenshot, so that
  "not a photograph" is never mistaken for "AI generated"
- **Person counting** — Faster R-CNN (COCO) body detection + OpenCV YuNet faces
- **Age estimation** — a single integer per face, with a margin derived from
  the model's own probability spread
- **Metadata / AI tag inspection** — EXIF, PNG tEXt (Stable Diffusion parameters),
  XMP (IPTC `trainedAlgorithmicMedia`), C2PA / Content Credentials, camera-trace evidence
- Heatmap visualization of manipulated regions
- Screenshot / digital content detection

## API
| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/analyze` | POST | Full forensics report (image + selected algorithms) |
| `/api/metadata` | POST | Metadata / AI tag inspection only (no pixel analysis) |
| `/api/persons` | POST | Person counting only (body + face detection) |
| `/api/medium` | POST | Content-type classification (photo / artwork / render / screenshot) |
| `/api/ages` | POST | Age estimation for every detected face |
| `/api/algorithms` | GET | Available algorithms |
| `/api/models` | GET | AI detection channels and how many models are loaded |

## Tests

### White box (unit) tests
60 tests, ~3 seconds, **no model files required** — they exercise the decision
logic only, so CLIP / SwinV2 / Faster R-CNN are never loaded:
```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

| File | Tests | Covers |
|---|---|---|
| `tests/test_fsm.py` | 8 | FSM transition validation |
| `tests/test_metadata_inspector.py` | 12 | IPTC / C2PA source-type parsing |
| `tests/test_ai_karar.py` | 20 | Metadata boost, edit evidence, verdict thresholds |
| `tests/test_kisi_ve_yas.py` | 20 | Box de-duplication, counting rule, age expected value |

Most of them are regression tests for bugs that actually happened — for example
`test_kompozit_tamamen_ai_sayilmaz` guards the substring bug where
`compositeWithTrainedAlgorithmicMedia` was read as `trainedAlgorithmicMedia`.

### Test fixtures
Metadata detection fixtures (images with known AI tags) are generated locally:
```bash
python tests/make_fixtures.py
```
They are written to `tests/fixtures/` and are test-only files, not product output.

### Test document
Acceptance test cases and measured results: `docs/STD-Test-Dokumani.md`.

## Requirements
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface_hub
```

No API key and no internet connection are needed at runtime — every detection
channel runs locally.

## Usage
```bash
python app.py
```
Then open `http://localhost:5050` in your browser.

**First start takes 20-60 seconds.** Importing PyTorch + transformers alone costs
~15 s (longer on a cold file cache), before any model file is read. The server
prints a `[1/4] … [4/4]` progress trace and a `Hazır (N sn)` line when it is
ready — if the console looks idle it is still loading, not hung. Do not press
Ctrl+C: interrupting mid-import produces a long `KeyboardInterrupt` traceback
through `sympy`/`torch._dynamo` that looks like a crash but is not one.

If port 5050 is already taken by an earlier run, the second process starts but
never serves. Find and stop the old one first:
```powershell
netstat -ano | findstr "127.0.0.1:5050"
taskkill /PID <pid> /F
```

## CLIP Model Setup
The CLIP model (~374MB) must be downloaded separately:

1. Open the [Google Colab notebook](https://colab.research.google.com/drive/1yyEnbJWWXP3CgrDudvUNYt0ilYQK1nNc?usp=sharing)
2. Run all cells — `clip_processor.zip` will be downloaded automatically
3. Extract `clip_processor.zip` into the project root folder

Without CLIP, the system falls back to statistical analysis mode.

## Model Downloads

Model files are not stored in the repository (~490 MB in total). Download them
once; afterwards everything runs offline:

```bash
python tools/download_models.py                      # everything that is missing
python tools/download_models.py --list               # components and variants
python tools/download_models.py --component person   # just one component
```

| Component | What it is | Size | Used by |
|---|---|---|---|
| `ai` | Supervised AI-image classifier (SwinV2) | ~745 MB | US-3 |
| `person` | torchvision Faster R-CNN, COCO weights | ~160 MB | US-5 |
| `face` | OpenCV YuNet face detector (ONNX) | ~230 KB | US-5, US-6 |
| `age` | Age classifier (ViT, FairFace) | ~327 MB | US-6 |

Every component is optional: if a model is missing, that feature degrades
gracefully and the rest of the analysis still runs.

### Second AI detector (User Story-3)

Default: [`haywoodsloan/ai-image-detector-deploy`](https://huggingface.co/haywoodsloan/ai-image-detector-deploy)
(SwinV2, Apache-2.0). Other variants
are available via `--component ai --variant swin|vit` (see `--list`).

Combination weights: `35% CLIP + 65% supervised model` for the semantic score,
then `82% semantic + 18% statistical`. A confirmed AI tag in the file metadata
raises the final score by a bounded amount (capped at 95%).

A **medium classification** stage (CLIP zero-shot) runs alongside: if the image is
confidently a screenshot, no AI verdict is issued; for artwork and 3D renders the
verdict stands but the UI states that the distinction is less reliable there.
Measured on our 16-image test set, this pipeline scores 12/14 (sensitivity 5/5,
specificity 7/9) — see `gelistirme/05-OLCUM-SONUCLARI.md`.

### Person counting (User Story-5)

Body boxes come from Faster R-CNN (COCO `person` class, score ≥ 0.70); face boxes
come from YuNet (score ≥ 0.60). Any face that does not fall inside a body box is
counted as an additional person whose body is hidden or cropped. Both models were
trained on real photographs, so drawings, anime and game screenshots may be
undercounted — the UI states this explicitly.

## Project Structure
```
ForensIQ/
├── app.py                   # Flask server
├── forensics_engine.py      # Core algorithms
├── ai_detector.py           # AI detection engine
├── deep_detector.py         # Second deep model loader (US-3)
├── medium_classifier.py     # Photo / artwork / render / screenshot (CLIP zero-shot)
├── person_detector.py       # Person counting: body + face (US-5)
├── age_estimator.py         # Age estimation per face (US-6)
├── metadata_inspector.py    # Metadata / AI tag inspection (US-4)
├── main.html                # Web interface
├── tools/download_models.py # One-time model downloads (US-3, US-5, US-6)
├── tests/make_fixtures.py   # Test fixtures for metadata detection
├── gelistirme/              # Development log: analysis, decisions, roadmap
├── requirements.txt
├── clip_processor/          # CLIP model (download via Colab)
└── models/                  # Downloaded models (AI detector, person, face)
```

## Tech Stack
Python · Flask · OpenCV · PyTorch · CLIP · HTML / CSS / JS