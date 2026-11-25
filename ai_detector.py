"""
@file ai_detector.py
@brief AI-Generated Image Detector — Tamamen çevrimdışı, çok algoritmalı sistem
@details
  Algoritma 1: CLIP zero-shot (ViT-B/32) — anlamsal, eğitim gerektirmez
  Algoritma 2: DeepAIDetector (Swin/ViT, denetimli eğitilmiş) — bkz. deep_detector.py
  Algoritma 3: İstatistiksel analiz (5 sinyal) — her zaman aktif

  Üç kanal birbirinden bağımsız çalışır; sonuçlar ağırlıklı olarak birleştirilir.
  Hiçbiri internet veya API anahtarı gerektirmez (User Story-3).

@note v4.0.0 ile Anthropic Vision API katmanı **kaldırıldı**: proje tamamen
      ücretsiz ve çevrimdışı çalışmalı, harici ücretli servise bağımlı olmamalı.
@author ImageForensics Team
@version 4.0.0
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from PIL import Image

# İkinci derin model (denetimli eğitilmiş AI dedektörü) — opsiyonel
try:
    from deep_detector import DeepAIDetector
    _DEEP_DETECTOR_AVAILABLE = True
except ImportError:
    _DEEP_DETECTOR_AVAILABLE = False

logger = logging.getLogger(__name__)

_BASE_DIR      = Path(__file__).parent
CLIP_WEIGHTS   = _BASE_DIR / "clip_weights.pth"
CLIP_PROCESSOR = _BASE_DIR / "clip_processor"

# Semantik kanalların kendi içindeki ağırlığı (ikisi birden yüklüyse).
#
# Derin model daha ağır basar (%35 CLIP / %65 SwinV2). Gerekçe
# (ölçüm: images/testing, 14 görsel — ekran görüntüleri hariç, 23.11.2026):
#   - CLIP tek başına            :  9/14
#   - SwinV2 tek başına          : 11/14
#   - CLIP %50 + SwinV2 %50      : 11/14 (grup selfie'si %49,6 ile sınırda)
#   - CLIP %35 + SwinV2 %65      : 11/14 (aynı doğruluk, daha geniş pay)
#   + medium aşaması (ekran kapısı) ile → **12/14**
#
# Not: Önceki ikinci model (Organika/sdxl-detector, Swin v1) tek başına 8/14
# veriyordu ve tablolara %99,9 AI diyordu; 23.11.2026'da SwinV2 ile değiştirildi.
#
# @warning 14 görsellük bir set doğrulanmış doğruluk oranı değildir; ağırlık
#          daha geniş bir test setiyle yeniden ölçülmelidir.
W_DEEP_VS_CLIP = 0.65

# Semantik kanal ↔ istatistiksel kanal ağırlığı
W_SEMANTIC = 0.82
W_STAT     = 0.18

_clip_model     = None
_clip_processor = None
_clip_loaded    = False


def _score_kurtosis_dark(kurt: float) -> float:
    """@brief Score DCT kurtosis for dark scenes."""
    if kurt < 3:
        return 80.0
    if kurt < 10:
        return 55.0
    if kurt < 25:
        return 30.0
    if kurt < 60:
        return 8.0
    return 0.0


def _score_kurtosis_normal(kurt: float) -> float:
    """@brief Score DCT kurtosis for normal brightness scenes."""
    if kurt < 5:
        return 85.0
    if kurt < 18:
        return 65.0
    if kurt < 45:
        return 42.0
    if kurt < 100:
        return 12.0
    return 0.0


def _score_dnr(dnr: float, jpeg: bool) -> float:
    """@brief Score noise/detail ratio."""
    if dnr < 0.10:
        return 70.0
    if dnr < 0.18:
        return 50.0
    if dnr < 0.28:
        return 15.0 if jpeg else 35.0
    return 0.0


def _score_channel_corr(corr: float, bypass: bool) -> float:
    """@brief Score color channel correlation."""
    if bypass:
        return 10.0 if corr > 0.95 else 0.0
    if corr > 0.92:
        return 50.0
    if corr > 0.85:
        return 30.0
    if corr > 0.75:
        return 15.0
    return 0.0


def _load_clip():
    """@brief Load CLIP model from local files (one-time)."""
    global _clip_model, _clip_processor, _clip_loaded

    if _clip_loaded is not False:
        return _clip_loaded is True

    if not CLIP_PROCESSOR.exists():
        logger.info("CLIP files not found — statistical mode active")
        _clip_loaded = None
        return False

    model_safe     = CLIP_PROCESSOR / "model.safetensors"
    model_bin      = CLIP_PROCESSOR / "pytorch_model.bin"
    has_full_model = model_safe.exists() or model_bin.exists()
    has_weights    = CLIP_WEIGHTS.exists()

    if not has_full_model and not has_weights:
        logger.info("CLIP weights not found — statistical mode active")
        _clip_loaded = None
        return False

    try:
        # Bu import zinciri (torch -> torch._dynamo -> sympy -> transformers)
        # tek başına 15 saniyeyi bulabiliyor ve bu sürede hiçbir çıktı olmuyor.
        # Kullanıcı donma sanmasın diye başlamadan önce duyuruyoruz.
        logger.info("[1/4] PyTorch ve transformers yükleniyor "
                    "(en uzun adım, ~15 sn)...")
        import torch
        from transformers import CLIPModel, CLIPProcessor as CP

        logger.info("[2/4] CLIP yükleniyor: %s", CLIP_PROCESSOR)
        proc = CP.from_pretrained(str(CLIP_PROCESSOR))

        if has_full_model:
            logger.info("Loading full CLIP model from clip_processor/")
            model = CLIPModel.from_pretrained(str(CLIP_PROCESSOR))
        else:
            logger.info("Loading CLIP weights: %s", CLIP_WEIGHTS)
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            # weights_only=True is safer than default pickle loading
            state      = torch.load(str(CLIP_WEIGHTS), map_location="cpu",
                                    weights_only=True)
            model_keys = set(model.state_dict().keys())
            filtered   = {k: v for k, v in state.items() if k in model_keys}
            model.load_state_dict(filtered, strict=False)

        model.eval()
        _clip_processor = proc
        _clip_model     = model
        _clip_loaded    = True
        logger.info("CLIP model loaded successfully")
        return True

    except Exception as e:
        logger.warning("CLIP load failed: %s", e)
        _clip_loaded = None
        return False


class AIDetector:
    """
    @class AIDetector
    @brief Çok algoritmalı, tamamen çevrimdışı AI görsel dedektörü (User Story-3).

    @details
    Üç bağımsız kanal:
      1. **CLIP zero-shot** (ViT-B/32) — metin-görsel benzerliği, eğitim yok
      2. **DeepAIDetector** — bu göreve özel denetimli eğitilmiş sınıflandırıcı
      3. **İstatistiksel analiz** (5 sinyal) — her zaman aktif

    Birleştirme: iki semantik model varsa aralarında
    `%65 CLIP + %35 derin model`, sonra `%82 semantik + %18 istatistiksel`.
    Tek semantik model varsa doğrudan o kullanılır; hiçbiri yoksa yalnız
    istatistiksel skor döner (model dosyaları eksikse sistem çökmez).

    @note Harici API / internet bağımlılığı yoktur.
    """

    def __init__(self):
        """@brief CLIP ve derin modeli yükle; ikisi de opsiyoneldir."""
        self._clip_ready = _load_clip()

        if _DEEP_DETECTOR_AVAILABLE:
            self._deep = DeepAIDetector()
        else:
            self._deep = None
            logger.info("deep_detector.py yok — ikinci derin model devre dışı")

        active = []
        if self._clip_ready:
            active.append("CLIP")
        if self._deep is not None and self._deep.available:
            active.append(self._deep.display_name)
        active.append("İstatistiksel")
        logger.info("AIDetector v4.0 aktif kanallar: %s", " + ".join(active))

    @property
    def model_count(self) -> int:
        """@brief Yüklü öğrenen (eğitimli) model sayısı — US-3 kanıtı."""
        n = 1 if self._clip_ready else 0
        if self._deep is not None and self._deep.available:
            n += 1
        return n

    def detect(self, img_np: np.ndarray, img_bytes: bytes = b"") -> dict:
        """
        @brief Görselin AI üretimi olup olmadığını tespit eder.
        @param img_np    RGB uint8 numpy dizisi.
        @param img_bytes Ham dosya baytları (geriye dönük uyumluluk; kullanılmıyor).
        @return          generated, confidence, method, model_scores, signals içeren dict.
        """
        gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        stat_res = self._statistical_analysis(img_np, gray)

        clip_res = self._clip_inference(img_np) if self._clip_ready else None

        deep_res = None
        if self._deep is not None and self._deep.available:
            deep_res = self._deep.predict(img_np)

        return self._combine(stat_res, clip_res, deep_res)

    # ------------------------------------------------------------------
    # Algoritma 1: CLIP Zero-Shot
    # ------------------------------------------------------------------

    def _clip_inference(self, img_np: np.ndarray):
        """
        @brief CLIP zero-shot AI/real classification.
        @param img_np  RGB image array.
        @return        Dict with confidence and generated, or None on error.
        """
        try:
            import torch

            pil = Image.fromarray(img_np)
            real_texts = [
                "a real photograph taken with a camera",
                "a genuine photo with natural camera noise and real lighting",
                "a photo taken by a person with a smartphone or DSLR",
            ]
            ai_texts = [
                "an AI generated image created by a neural network",
                "a synthetic image made by Midjourney or Stable Diffusion",
                "a digitally generated photorealistic image that is not real",
            ]
            all_texts = real_texts + ai_texts

            inputs = _clip_processor(
                text=all_texts, images=pil,
                return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                out   = _clip_model(**inputs)
                probs = out.logits_per_image.softmax(dim=1).squeeze().tolist()

            real_p = sum(probs[:len(real_texts)])
            ai_p   = sum(probs[len(real_texts):])
            total  = real_p + ai_p + 1e-9
            ai_pct = (ai_p / total) * 100.0

            logger.info("CLIP: AI=%.1f%% Real=%.1f%%", ai_pct, real_p/total*100)
            return {
                "ai_generated": ai_pct >= 52.0,
                "confidence":   round(ai_pct, 2),
                "reason":       f"CLIP: AI={ai_pct:.1f}% / Real={real_p/total*100:.1f}%",
            }
        except Exception as e:
            logger.warning("CLIP inference error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Algoritma 3: İstatistiksel analiz
    # ------------------------------------------------------------------

    def _statistical_analysis(self, img_np: np.ndarray,
                               gray: np.ndarray) -> dict:
        """@brief 5-signal JPEG/dark-tolerant statistical AI analysis."""
        signals = {
            "freq":    self._freq_analysis(gray),
            "noise":   self._noise_analysis(gray),
            "color":   self._color_analysis(img_np),
            "edge":    self._edge_analysis(gray),
            "texture": self._texture_analysis(gray),
        }
        weights = {
            "freq": 0.40, "noise": 0.30, "color": 0.10,
            "edge": 0.12, "texture": 0.08,
        }
        raw = sum(signals[k]["score"] * weights[k] for k in signals)

        import re
        bm         = re.search(r'bright=([\d.]+)',
                               signals["freq"].get("note", ""))
        brightness = float(bm.group(1)) if bm else 100.0
        n_note     = signals["noise"].get("note", "")
        jpeg       = "jpeg=True" in n_note
        lm         = re.search(r'lap_s=([\d.]+)', n_note)
        lap_s      = float(lm.group(1)) if lm else 99.0

        if brightness < 80.0 and not jpeg and lap_s < 12.0:
            raw *= 0.72
        elif brightness < 100.0 and not jpeg and lap_s < 12.0:
            raw *= 0.88

        return {
            "confidence": round(self._sigmoid(raw, 47.0, 9.0), 2),
            "raw":        round(raw, 2),
            "signals":    {k: round(v["score"], 1) for k, v in signals.items()},
            "notes":      {k: v.get("note", "")    for k, v in signals.items()},
        }

    def _freq_analysis(self, gray: np.ndarray) -> dict:
        """@brief DCT kurtosis based frequency analysis (brightness-adaptive)."""
        try:
            crop = cv2.resize(gray, (256, 256)).astype(np.float32)
            dct  = cv2.dct(crop / 255.0)

            dct_abs = np.abs(dct).sum() + 1e-9
            mid  = float(np.abs(dct[32:96, 32:96]).sum()) / dct_abs
            high = float(np.abs(dct[96:,   96: ]).sum()) / dct_abs

            from scipy.stats import kurtosis as sk
            kurt       = float(sk(dct[4:64, 4:64].flatten(), fisher=True))
            brightness = float(gray.mean())
            dark       = brightness < 85.0

            if not np.isfinite(kurt):
                score = 25.0
            elif dark:
                score = _score_kurtosis_dark(kurt)
            else:
                score = _score_kurtosis_normal(kurt)

            high_mid_ratio = high / (mid + 1e-9)
            if high_mid_ratio > 1.5:
                score += 12.0

            return {
                "score": min(100.0, score),
                "note":  f"kurt={kurt:.1f} bright={brightness:.0f} dark={dark}",
            }
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    def _noise_analysis(self, gray: np.ndarray) -> dict:
        """@brief Noise/detail ratio — JPEG tolerant."""
        try:
            med    = cv2.medianBlur(gray, 3)
            noise  = gray.astype(np.float32) - med.astype(np.float32)
            n_std  = float(noise.std())
            lap_s  = float(cv2.Laplacian(gray, cv2.CV_64F).std())
            dnr    = n_std / (lap_s + 1e-6)
            jpeg   = n_std > 1.5 and lap_s > 15.0

            h, w  = gray.shape
            step  = max(32, min(h, w) // 8)
            rs    = [
                float(noise[y:y+step, x:x+step].std())
                for y in range(0, h - step, step)
                for x in range(0, w - step, step)
            ]
            rcv = float(np.std(rs) / (np.mean(rs) + 1e-6)) if rs else 1.0

            score = _score_dnr(dnr, jpeg)

            if n_std < 1.5 and not jpeg:
                score += 20.0
            elif n_std < 2.5 and not jpeg:
                score += 10.0

            if rcv < 0.20:
                score += 18.0
            elif rcv < 0.40:
                score += 6.0

            return {
                "score": min(100.0, score),
                "note":  f"dnr={dnr:.3f} n={n_std:.2f} lap_s={lap_s:.1f} jpeg={jpeg}",
            }
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    def _color_analysis(self, img_np: np.ndarray) -> dict:
        """@brief Color channel correlation — single-color scene bypass."""
        try:
            h, w = img_np.shape[:2]
            rng  = np.random.default_rng(seed=42)
            idx  = rng.choice(h * w, min(8000, h * w), replace=False)

            r = img_np[:, :, 0].flatten()[idx].astype(np.float64)
            g = img_np[:, :, 1].flatten()[idx].astype(np.float64)
            b = img_np[:, :, 2].flatten()[idx].astype(np.float64)

            corr = (
                abs(float(np.corrcoef(r, g)[0, 1]))
                + abs(float(np.corrcoef(r, b)[0, 1]))
                + abs(float(np.corrcoef(g, b)[0, 1]))
            ) / 3.0

            rm, gm, bm = (float(img_np[:, :, c].mean()) for c in range(3))
            dom        = max(rm, gm, bm) / (rm + gm + bm + 1e-6)
            bypass     = dom > 0.50 or (rm + gm + bm) / 3 < 40.0

            def hist_smoothness(channel_idx: int) -> float:
                hist = cv2.calcHist(
                    [img_np], [channel_idx], None, [256], [0, 256]
                ).flatten()
                hist = hist / (hist.sum() + 1e-9)
                return float(np.mean(np.abs(np.diff(hist))))

            hs = (hist_smoothness(0) + hist_smoothness(1)) / 2.0

            score = _score_channel_corr(corr, bypass)
            if hs < 0.0015:
                score += 25.0
            elif hs < 0.003:
                score += 10.0

            return {
                "score": min(100.0, score),
                "note":  f"corr={corr:.3f} bypass={bypass}",
            }
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    def _edge_analysis(self, gray: np.ndarray) -> dict:
        """@brief Edge quality analysis."""
        try:
            edge_lo = float(cv2.Canny(gray, 50,  150).mean())
            edge_hi = float(cv2.Canny(gray, 100, 200).mean())
            ratio   = edge_hi / (edge_lo + 1e-6)

            gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2).flatten()
            mcv = float(mag.std() / (mag.mean() + 1e-6))

            if ratio > 0.75:
                score = 35.0
            elif ratio < 0.20:
                score = 30.0
            else:
                score = 10.0

            if mcv < 1.5:
                score += 35.0
            elif mcv < 2.5:
                score += 15.0

            return {
                "score": min(100.0, score),
                "note":  f"ratio={ratio:.2f} mag_cv={mcv:.2f}",
            }
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    def _texture_analysis(self, gray: np.ndarray) -> dict:
        """@brief Texture uniformity — JPEG artifact bypass."""
        try:
            dh  = np.abs(
                gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)
            )
            dv  = np.abs(
                gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16)
            )
            dhm = float(dh.mean())
            dvm = float(dv.mean())
            iso  = 1.0 - abs(dhm - dvm) / (dhm + dvm + 1e-6)
            sm   = 1.0 / (1.0 + (float(dh.std()) + float(dv.std())) / 2.0)
            jiso = iso > 0.98 and dhm > 3 and dvm > 3

            score = 0.0
            if iso > 0.97 and not jiso:
                score = 28.0
            elif iso > 0.95 and not jiso:
                score = 15.0
            elif iso > 0.90:
                score = 8.0

            if sm > 0.05:
                score += 30.0
            elif sm > 0.02:
                score += 12.0

            return {
                "score": min(100.0, score),
                "note":  f"iso={iso:.3f} jpeg_iso={jiso}",
            }
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    # ------------------------------------------------------------------
    # Combine & Helpers
    # ------------------------------------------------------------------

    def _combine(self, stat: dict, clip_res, deep_res) -> dict:
        """
        @brief İki semantik modeli ve istatistiksel kanalı birleştirir.

        @param stat      İstatistiksel analiz sonucu (zorunlu).
        @param clip_res  CLIP zero-shot sonucu veya None.
        @param deep_res  DeepAIDetector sonucu veya None.
        @return          Birleşik sonuç dict'i (model_scores dâhil).

        @details Ağırlıklar: iki model varsa `%65 CLIP + %35 derin model`,
                 ardından `%82 semantik + %18 istatistiksel`. Tek model varsa
                 o modelin skoru semantik skordur. Hiç model yoksa yalnızca
                 istatistiksel skor döner.
        """
        model_scores = []

        if clip_res is not None:
            model_scores.append({
                "key":          "clip",
                "name":         "CLIP Zero-Shot",
                "architecture": "ViT-B/32 (zero-shot)",
                "score":        round(float(clip_res["confidence"]), 2),
                "note":         clip_res.get("reason", ""),
            })

        if deep_res is not None:
            model_scores.append({
                "key":          "deep",
                "name":         deep_res.get("name", "Derin Model"),
                "architecture": deep_res.get("architecture", ""),
                "score":        round(float(deep_res["confidence"]), 2),
                "note":         deep_res.get("reason", ""),
            })

        # --- Semantik skor ---
        if clip_res is not None and deep_res is not None:
            semantic_conf = (
                clip_res["confidence"] * (1.0 - W_DEEP_VS_CLIP)
                + deep_res["confidence"] * W_DEEP_VS_CLIP
            )
            source = f"CLIP + {deep_res.get('name', 'Derin Model')}"
            reason = (
                f"CLIP %{clip_res['confidence']:.1f} · "
                f"{deep_res.get('name', 'Derin model')} %{deep_res['confidence']:.1f}"
            )
        elif clip_res is not None:
            semantic_conf = float(clip_res["confidence"])
            source        = "CLIP Zero-Shot"
            reason        = clip_res.get("reason", "")
        elif deep_res is not None:
            semantic_conf = float(deep_res["confidence"])
            source        = deep_res.get("name", "Derin Model")
            reason        = deep_res.get("reason", "")
        else:
            semantic_conf = None
            source        = "İstatistiksel Analiz"
            reason        = self._build_reason(stat)

        # --- İstatistiksel kanalla birleştir ---
        if semantic_conf is not None:
            combined = semantic_conf * W_SEMANTIC + stat["confidence"] * W_STAT
        else:
            combined = stat["confidence"]

        model_scores.append({
            "key":          "statistical",
            "name":         "İstatistiksel Analiz",
            "architecture": "5 sinyal (DCT / gürültü / renk / kenar / doku)",
            "score":        round(float(stat["confidence"]), 2),
            "note":         self._build_reason(stat),
        })

        return {
            "generated":     combined > 50.0,
            "confidence":    round(combined, 2),
            "method":        source,
            "reason":        reason,
            "stat_conf":     stat["confidence"],
            "semantic_conf": round(semantic_conf, 2) if semantic_conf is not None else None,
            "model_scores":  model_scores,
            "model_count":   len(model_scores) - 1,  # istatistiksel kanal hariç
            "signals":       stat.get("signals", {}),
            "signal_notes":  stat.get("notes", {}),
        }

    def _build_reason(self, stat: dict) -> str:
        """@brief Build human-readable reason from statistical signals."""
        s      = stat.get("signals", {})
        keys   = [k for k in ("freq", "noise", "color", "edge", "texture")
                  if s.get(k, 0) > 60]
        labels = {
            "freq":    "artificial frequency spectrum",
            "noise":   "uniform noise",
            "color":   "artificial color channels",
            "edge":    "artificial edge quality",
            "texture": "isotropic texture",
        }
        return ", ".join(labels[k] for k in keys) if keys else "no clear AI signature"

    @staticmethod
    def _sigmoid(x: float, c: float = 50.0, s: float = 10.0) -> float:
        """@brief Sigmoid calibration function."""
        return float(100.0 / (1.0 + np.exp(-(x - c) / s)))