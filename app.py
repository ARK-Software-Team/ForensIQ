"""
@file app.py
@brief Image Forgery Detection System - Main Flask Application
@author ImageForensics Team
@version 1.0.0
@date 2026

@mainpage Image Forgery Detection System

@section intro_sec Introduction
This system detects image manipulation using multiple computer vision algorithms:
SIFT, SURF, AKAZE, ORB, and VGG-based feature matching.

@section install_sec Installation
Run: pip install -r requirements.txt

@section usage_sec Usage
python app.py
"""

import logging
import sys
import time

# Windows konsolu varsayılan olarak cp1252/cp857 kullanır. Bu iki soruna yol açar:
#   1. Türkçe karakter basmak UnicodeEncodeError ile programı düşürebilir,
#   2. Düşürmezse bile "Görüntü" yerine "G├╢r├╝nt├╝" gibi bozuk çıktı verir.
# Hem konsolun kod sayfasını hem de Python akışlarını UTF-8'e alıyoruz.
# İkisi birlikte yapılmazsa bozukluk sürer.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Açılış bildirimi
# ---------------------------------------------------------------------------
# PyTorch + transformers import zinciri tek başına ~15-20 saniye sürüyor
# (ilk açılışta, dosya önbelleği soğukken daha da uzun). Bu süre boyunca hiçbir
# çıktı olmazsa program donmuş gibi görünür ve kullanıcı Ctrl+C'ye basar.
# Bu yüzden her aşama başlamadan ÖNCE duyurulur.
_BOOT_T0 = time.perf_counter()

print("=" * 62, flush=True)
print("  ForensIQ — Görüntü Adli Analiz Sistemi", flush=True)
print("=" * 62, flush=True)
print("  Modeller yükleniyor. İlk açılış 20-60 saniye sürebilir;", flush=True)
print("  program donmuş değildir — lütfen Ctrl+C'ye basmayın.", flush=True)
print("-" * 62, flush=True)

import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from forensics_engine import ForensicsEngine

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
## Tekrarlanan istemci hata mesajları (SonarQube: yinelenen metin sabiti).
HATA_DOSYA_YOK = "No file provided"
HATA_GORUNTU_OKUNAMADI = "Cannot read image file"

app = Flask(__name__)
# CORS: only allow requests from localhost (development)
CORS(app, origins=["http://localhost:5050", "http://127.0.0.1:5050"])

engine = ForensicsEngine()

_BOOT_SECONDS = time.perf_counter() - _BOOT_T0
print("-" * 62, flush=True)
print(f"  Hazır ({_BOOT_SECONDS:.1f} sn). Tarayıcıda açın: "
      f"http://localhost:5050", flush=True)
print("  Durdurmak için: Ctrl+C", flush=True)
print("=" * 62, flush=True)

# ---------------------------------------------------------------------------
# FSM States (used for effort/state tracking)
# ---------------------------------------------------------------------------
# States: IDLE -> UPLOAD -> PREPROCESS -> ANALYZE -> REPORT -> IDLE
FSM_STATE = "IDLE"
FSM_TRANSITIONS = {
    "IDLE": ["UPLOAD"],
    "UPLOAD": ["PREPROCESS"],
    "PREPROCESS": ["ANALYZE"],
    "ANALYZE": ["REPORT"],
    "REPORT": ["IDLE"]
}


def fsm_transition(current: str, next_state: str) -> str:
    """
    @brief Perform FSM state transition with validation.
    @param current Current FSM state string.
    @param next_state Desired next FSM state string.
    @return New state string if transition is valid.
    @throws ValueError if the transition is not allowed.
    """
    if next_state not in FSM_TRANSITIONS.get(current, []):
        raise ValueError(f"Invalid FSM transition: {current} -> {next_state}")
    logger.info("FSM: %s -> %s", current, next_state)
    return next_state


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """@brief Serve the main HTML interface."""
    return send_from_directory(".", "main.html")


@app.route("/api/health", methods=["GET"])
def health():
    """
    @brief Health check endpoint.
    @return JSON with status 'ok'.
    """
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    @brief Main analysis endpoint. Accepts an image and runs all forensics algorithms.

    @details
    FSM Flow: IDLE -> UPLOAD -> PREPROCESS -> ANALYZE -> REPORT -> IDLE

    Expects multipart/form-data with:
    - file: image file (jpg, png, bmp, tiff, webp)
    - algorithms: comma-separated list (sift,surf,akaze,orb)

    @return JSON with full forensics report including:
            - tampered (bool)
            - confidence (float 0-100)
            - ai_generated (bool)
            - ai_confidence (float 0-100, metadata artışı uygulanmış nihai skor)
            - ai_confidence_base (float, metadata artışı öncesi piksel analizi skoru)
            - ai_metadata_boost (dict|None, K-12: uygulanan sınırlı artışın dökümü)
            - ai_model_scores (list, US-3: her AI algoritmasının kendi skoru)
            - ai_model_count (int, yüklü öğrenen model sayısı)
            - algorithm_results (dict)
            - metadata_report (dict, US-4: dosya yapısında AI etiketi taraması)
            - medium_report (dict: içerik türü — fotoğraf/tablo/3B/ekran görüntüsü)
            - person_count (int, US-5: görüntüdeki kişi sayısı)
            - person_report (dict, US-5: gövde/yüz kutuları ve sayım dökümü)
            - age_report (dict, US-6: her yüz için tam sayı yaş tahmini)
            - persons_b64 (str, base64 PNG: kişi kutuları çizilmiş görüntü)
            - heatmap_b64 (str, base64 PNG)
            - annotated_b64 (str, base64 PNG)
            - ela_b64 (str, base64 PNG)
    """
    global FSM_STATE
    FSM_STATE = "IDLE"

    # --- UPLOAD state ---
    try:
        FSM_STATE = fsm_transition(FSM_STATE, "UPLOAD")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if "file" not in request.files:
        return jsonify({"error": HATA_DOSYA_YOK}), 400

    file = request.files["file"]
    algorithms_param = request.form.get("algorithms", "sift,surf,akaze,orb")
    selected_algos = [a.strip().lower() for a in algorithms_param.split(",")]

    try:
        img_bytes = file.read()
        pil_raw = Image.open(BytesIO(img_bytes))
        image_format = getattr(pil_raw, "format", "") or ""
        is_gif = image_format.upper() == "GIF"
        is_animated_gif = is_gif and getattr(pil_raw, "n_frames", 1) > 1
        n_frames = getattr(pil_raw, "n_frames", 1) if is_gif else 1
        if is_gif:
            pil_raw.seek(0)
        pil_img = pil_raw.convert("RGB")
    except Exception:
        logger.exception("Image load failed")
        return jsonify({"error": HATA_GORUNTU_OKUNAMADI}), 400

    # --- PREPROCESS state ---
    FSM_STATE = fsm_transition(FSM_STATE, "PREPROCESS")
    img_np = np.array(pil_img)

    # --- ANALYZE state ---
    FSM_STATE = fsm_transition(FSM_STATE, "ANALYZE")
    try:
        result = engine.run_full_analysis(img_np, img_bytes, selected_algos)
        if is_gif:
            result["file_format"] = "GIF"
            result["gif_animated"] = is_animated_gif
            result["gif_frames"] = n_frames
    except Exception:
        logger.exception("Analysis failed")
        return jsonify({"error": "Analysis error"}), 500

    # --- REPORT state ---
    FSM_STATE = fsm_transition(FSM_STATE, "REPORT")
    FSM_STATE = fsm_transition(FSM_STATE, "IDLE")

    return jsonify(result)


@app.route("/api/metadata", methods=["POST"])
def metadata_only():
    """
    @brief Yalnızca dosya yapısı (metadata) incelemesi yapan hafif endpoint — US-4.

    @details
    Piksel analizi çalıştırmaz; sadece EXIF / XMP / PNG-text / C2PA bloklarını
    tarar. Hızlı kontrol ve otomatik testler için kullanılır.

    Expects multipart/form-data with:
    - file: image file

    @return JSON metadata raporu (verdict, confidence, findings, camera_evidence).
    """
    if "file" not in request.files:
        return jsonify({"error": HATA_DOSYA_YOK}), 400

    try:
        img_bytes = request.files["file"].read()
    except Exception:
        logger.exception("Metadata: dosya okunamadı")
        return jsonify({"error": HATA_GORUNTU_OKUNAMADI}), 400

    if not img_bytes:
        return jsonify({"error": "Empty file"}), 400

    try:
        report = engine.inspect_metadata(img_bytes)
    except Exception:
        logger.exception("Metadata incelemesi başarısız")
        return jsonify({"error": "Metadata error"}), 500

    return jsonify(report)


def _istekten_goruntu_oku(islem_adi: str):
    """
    @brief İstekten görüntüyü okur ve doğrular (üç hafif endpoint ortak kullanır).

    @param islem_adi Log'da görünecek işlem adı.
    @return `(img_np, hata_yaniti)` — başarıdaysa `(dizi, None)`,
            hatada `(None, (jsonify(...), kod))`.

    @note Bu yardımcı, `/api/persons`, `/api/medium` ve `/api/ages` uçlarındaki
          aynı doğrulama bloğunun tekrarını önler (kod tekrarı giderildi).
    """
    if "file" not in request.files:
        return None, (jsonify({"error": HATA_DOSYA_YOK}), 400)

    try:
        img_bytes = request.files["file"].read()
        if not img_bytes:
            return None, (jsonify({"error": "Empty file"}), 400)
        img_np = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
        return img_np, None
    except Exception:
        logger.exception("%s: dosya okunamadı", islem_adi)
        return None, (jsonify({"error": HATA_GORUNTU_OKUNAMADI}), 400)


@app.route("/api/persons", methods=["POST"])
def persons_only():
    """
    @brief Yalnızca kişi sayımı yapan hafif endpoint — US-5.

    @details
    Sahtecilik/AI analizi çalıştırmaz; sadece gövde ve yüz tespiti yapar.
    Hızlı kontrol ve otomatik testler için kullanılır.

    Expects multipart/form-data with:
    - file: image file

    @return JSON kişi raporu (person_count, body_count, face_count, persons, faces).
    """
    img_np, hata = _istekten_goruntu_oku("Kişi tespiti")
    if hata is not None:
        return hata

    try:
        report = engine.detect_persons(img_np)
    except Exception:
        logger.exception("Kişi tespiti başarısız")
        return jsonify({"error": "Person detection error"}), 500

    return jsonify(report)


@app.route("/api/ages", methods=["POST"])
def ages_only():
    """
    @brief Yalnızca yaş tahmini yapan hafif endpoint — US-6.

    @details
    Yüzleri tespit edip her biri için tam sayı yaş tahmini üretir.
    Sahtecilik / AI analizi çalıştırmaz.

    Expects multipart/form-data with:
    - file: image file

    @return JSON yaş raporu (count, ages[], summary, model).
    """
    img_np, hata = _istekten_goruntu_oku("Yaş tahmini")
    if hata is not None:
        return hata

    try:
        return jsonify(engine.estimate_ages(img_np))
    except Exception:
        logger.exception("Yaş tahmini başarısız")
        return jsonify({"error": "Age estimation error"}), 500


@app.route("/api/medium", methods=["POST"])
def medium_only():
    """
    @brief Yalnızca içerik türü (medium) sınıflandırması yapan hafif endpoint.

    @details
    Görselin fotoğraf mı, tablo/çizim mi, 3B render mı yoksa ekran görüntüsü mü
    olduğunu CLIP zero-shot ile belirler. AI kararının bağlamını verir:
    "fotoğraf değil" ile "AI üretimi" aynı şey değildir.

    Expects multipart/form-data with:
    - file: image file

    @return JSON medium raporu (medium, confidence, scores, caution).
    """
    img_np, hata = _istekten_goruntu_oku("Medium")
    if hata is not None:
        return hata

    try:
        return jsonify(engine.classify_medium(img_np))
    except Exception:
        logger.exception("Medium sınıflandırması başarısız")
        return jsonify({"error": "Medium classification error"}), 500


@app.route("/api/algorithms", methods=["GET"])
def list_algorithms():
    """
    @brief Returns available algorithms and their descriptions.
    @return JSON list of algorithm objects.
    """
    algos = [
        {"id": "sift",  "name": "SIFT",  "full": "Scale-Invariant Feature Transform",
         "description": "Kopyala-yapıştır manipülasyonlarını tespit eder."},
        {"id": "surf",  "name": "SURF",  "full": "Speeded-Up Robust Features (BRISK)",
         "description": "Patent kısıtlaması nedeniyle BRISK ile ikame edildi. Benzer doğruluk, daha hızlı."},
        {"id": "akaze", "name": "AKAZE", "full": "Accelerated KAZE",
         "description": "Doğrusal olmayan difüzyon filtresi tabanlı."},
        {"id": "orb",   "name": "ORB",   "full": "Oriented FAST and Rotated BRIEF",
         "description": "Hızlı ve açık kaynaklı."},
    ]
    return jsonify(algos)


@app.route("/api/models", methods=["GET"])
def list_models():
    """
    @brief AI tespit kanallarını ve yüklü model sayısını döndürür — US-3.

    @details
    Hangi algoritmaların gerçekten yüklü olduğunu şeffaf biçimde gösterir
    (CLIP, eğitimli derin model, istatistiksel sinyaller, metadata taraması).
    Harici API kanalı yoktur; sistem tamamen çevrimdışı çalışır.

    @return JSON: `{"model_count": int, "channels": [...]}`
    """
    try:
        return jsonify(engine.ai_model_info())
    except Exception:
        logger.exception("Model bilgisi alınamadı")
        return jsonify({"error": "Model info error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Image Forensics Server on http://localhost:5050")
    # host="127.0.0.1" — only local access, not all interfaces
    app.run(host="127.0.0.1", port=5050, debug=False)