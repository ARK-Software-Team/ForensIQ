"""
@file ai_detector.py
@brief AI-Generated Image Detector — Three-Layer Hybrid System
@details
  Layer 1: CLIP Local Model (clip_weights.pth) — 85-90% accuracy
  Layer 2: Anthropic Vision API (ANTHROPIC_API_KEY) — 90%+ accuracy
  Layer 3: Statistical Analysis (always active) — fallback

  Priority: CLIP > Anthropic API > Statistical
@author ImageForensics Team
@version 3.1.0
"""

import os
import cv2
import base64
import logging
import numpy as np
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_BASE_DIR      = Path(__file__).parent
CLIP_WEIGHTS   = _BASE_DIR / "clip_weights.pth"
CLIP_PROCESSOR = _BASE_DIR / "clip_processor"

ANTHROPIC_API   = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-opus-4-5"

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
        import torch
        from transformers import CLIPModel, CLIPProcessor as CP

        logger.info("Loading CLIP processor: %s", CLIP_PROCESSOR)
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
    @brief Three-layer hybrid AI image detector.

    @details
    Priority order:
      1. CLIP local model — no internet needed
      2. Anthropic Vision API — requires ANTHROPIC_API_KEY env var
      3. Statistical analysis — always active (fallback)

    When semantic source available: 82% semantic + 18% statistical
    """

    def __init__(self):
        self._api_key       = os.environ.get("ANTHROPIC_API_KEY", "")
        self._api_available = bool(self._api_key)
        self._clip_ready    = _load_clip()

        if self._clip_ready:
            logger.info("AIDetector v3.1: CLIP local model active")
        elif self._api_available:
            logger.info("AIDetector v3.1: Anthropic API active")
        else:
            logger.info("AIDetector v3.1: Statistical mode only")

    def detect(self, img_np: np.ndarray, img_bytes: bytes) -> dict:
        """
        @brief Detect whether image is AI-generated.
        @param img_np    RGB uint8 numpy array.
        @param img_bytes Raw image bytes.
        @return          Dict with generated, confidence, method, signals.
        """
        gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        stat_res = self._statistical_analysis(img_np, gray)

        if self._clip_ready:
            clip_res = self._clip_inference(img_np)
            if clip_res is not None:
                return self._combine(stat_res, clip_res, source="CLIP Local Model")

        if self._api_available:
            api_res = self._anthropic_vision(img_bytes)
            if api_res is not None:
                return self._combine(stat_res, api_res, source="Anthropic Vision API")

        return self._combine(stat_res, None, source="Statistical Analysis")

    # ------------------------------------------------------------------
    # Layer 1: CLIP Zero-Shot
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
    # Layer 2: Anthropic Vision API
    # ------------------------------------------------------------------

    def _anthropic_vision(self, img_bytes: bytes):
        """
        @brief Anthropic Claude Vision AI detection.
        @param img_bytes Raw image bytes.
        @return          Result dict or None.
        """
        try:
            pil   = Image.open(BytesIO(img_bytes)).convert("RGB")
            ratio = min(768 / pil.width, 768 / pil.height, 1.0)
            if ratio < 1.0:
                new_w = int(pil.width * ratio)
                new_h = int(pil.height * ratio)
                pil   = pil.resize((new_w, new_h), Image.LANCZOS)
            buf = BytesIO()
            pil.save(buf, "JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            prompt = (
                "Inspect this image. Return ONLY JSON:\n"
                '{"ai_generated": true/false, "confidence": 0-100, '
                '"reason": "max 80 chars"}\n'
                "AI signs: perfect smooth skin, weird hands, physics errors, "
                "fantasy scene. Only JSON."
            )
            resp = requests.post(
                ANTHROPIC_API,
                headers={
                    "x-api-key":          self._api_key,
                    "anthropic-version":  "2023-06-01",
                    "content-type":       "application/json",
                },
                json={
                    "model":      ANTHROPIC_MODEL,
                    "max_tokens": 120,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image",
                             "source": {"type":       "base64",
                                        "media_type": "image/jpeg",
                                        "data":       img_b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                },
                timeout=20,
            )
            if resp.status_code != 200:
                return None
            import json
            raw = resp.json()["content"][0]["text"].strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed = json.loads(raw)
            return {
                "ai_generated": bool(parsed.get("ai_generated", False)),
                "confidence":   float(parsed.get("confidence", 50)),
                "reason":       str(parsed.get("reason", "")),
            }
        except Exception as e:
            logger.warning("Anthropic API error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Layer 3: Statistical Analysis
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

            score = 35.0 if ratio > 0.75 else (30.0 if ratio < 0.20 else 10.0)
            score += 35.0 if mcv < 1.5 else (15.0 if mcv < 2.5 else 0.0)

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

            score += 30.0 if sm > 0.05 else (12.0 if sm > 0.02 else 0.0)

            return {
                "score": min(100.0, score),
                "note":  f"iso={iso:.3f} jpeg_iso={jiso}",
            }
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    # ------------------------------------------------------------------
    # Combine & Helpers
    # ------------------------------------------------------------------

    def _combine(self, stat: dict, semantic, source: str) -> dict:
        """
        @brief Combine semantic (CLIP/API) and statistical results (82/18).
        @param stat      Statistical analysis result dict.
        @param semantic  Semantic result dict or None.
        @param source    Source label string.
        @return          Final combined result dict.
        """
        if semantic is not None:
            combined  = semantic["confidence"] * 0.82 + stat["confidence"] * 0.18
            generated = combined > 50.0
            reason    = semantic.get("reason", "")
        else:
            combined  = stat["confidence"]
            generated = combined > 50.0
            reason    = self._build_reason(stat)

        return {
            "generated":    generated,
            "confidence":   round(combined, 2),
            "method":       source,
            "reason":       reason,
            "stat_conf":    stat["confidence"],
            "api_conf":     semantic["confidence"] if semantic is not None else None,
            "signals":      stat.get("signals", {}),
            "signal_notes": stat.get("notes", {}),
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
