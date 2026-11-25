"""
@file medium_classifier.py
@brief Görselin *türünü* (medium) belirler: fotoğraf mı, tablo mu, render mı?

@details
  **Neden gerekli?**
  AI dedektörleri aslında "bu bir kamera fotoğrafı mı?" sorusunu yanıtlıyor.
  Bir tablo için doğru cevap "hayır"dır — ama sistem bu "hayır"ı doğrudan
  "AI üretimi" diye yorumlarsa **kategori hatası** yapar. 23 Kasım 2026
  ölçümünde Las Meninas (1656) tablosuna %65 AI skoru verilmesi bu hatanın
  somut sonucuydu (bkz. `gelistirme/05-OLCUM-SONUCLARI.md`).

  Bu modül *medium* eksenini *köken* ekseninden ayırır:

  | Eksen | Değerler |
  |---|---|
  | **Medium** (bu modül) | fotoğraf · tablo/çizim · 3B render · ekran görüntüsü |
  | **Köken** (ai_detector) | kamera · yapay zekâ |

  Sınıflandırma CLIP zero-shot ile yapılır; **ek model indirmez**,
  `ai_detector` içinde zaten yüklü olan CLIP örneğini kullanır.

  **Kullanımı (ölçüme dayalı):**
  - `ekran görüntüsü` yüksek güvenle tespit edilirse AI kararı **geçersiz**
    sayılır (oyun arayüzleri ve anime kareleri istatistiksel olarak AI'ya benzer).
  - `tablo/çizim` ve `3B render` için karar **engellenmez**, yalnızca arayüzde
    "bu bir fotoğraf değil, ayrım daha az güvenilir" uyarısı gösterilir.
    Gerekçe: engelleme denendi, gerçek tabloları kurtarırken AI üretimi
    illüstrasyonları da kaybettiriyordu (net kazanç sıfır).

@note CLIP yüklü değilse modül `available=False` döner ve sistem eskisi gibi
      çalışır. Hiçbir koşulda istisna fırlatmaz (Ek Madde-1).

@author ImageForensics Team
@version 1.0.0
"""

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

## Medium sınıfları ve her biri için CLIP metin tanımları.
MEDIUM_PROMPTS = {
    "photograph": [
        "a photograph taken with a camera",
        "a real photo of a person taken with a smartphone",
        "a documentary photograph of a real scene",
    ],
    "artwork": [
        "an oil painting on canvas",
        "a hand-drawn illustration or cartoon",
        "a classical painting by an old master",
        "a flat vector illustration with solid colors",
    ],
    "render": [
        "a 3D rendered image from a video game",
        "computer generated 3D graphics of a character",
    ],
    "screenshot": [
        "a screenshot of a computer user interface",
        "a screenshot with menus, buttons and text overlay",
    ],
}

## Kullanıcıya gösterilecek Türkçe etiketler.
MEDIUM_LABELS = {
    "photograph": "Fotoğraf",
    "artwork":    "Tablo / Çizim",
    "render":     "3B Render / Oyun",
    "screenshot": "Ekran Görüntüsü",
    "unknown":    "Belirlenemedi",
}

## Fotoğraf dışı sayılan sınıflar.
NON_PHOTOGRAPHIC = ("artwork", "render", "screenshot")

## Bir medium kararının "güvenli" sayılması için gereken en düşük skor (%).
MEDIUM_CONFIDENT_THRESHOLD = 70.0


class MediumClassifier:
    """
    @class MediumClassifier
    @brief CLIP zero-shot ile görselin türünü belirler.

    @details
    Kullanım:
    @code
        mc = MediumClassifier()
        if mc.available:
            rapor = mc.classify(img_np)
            if rapor["blocks_ai_verdict"]:
                ...  # ekran görüntüsü: AI kararı geçersiz
    @endcode
    """

    VERSION = "1.0.0"

    def __init__(self):
        """@brief CLIP'in yüklü olup olmadığını kontrol eder (yeni model indirmez)."""
        self._texts  = []
        self._owners = []
        for medium, prompts in MEDIUM_PROMPTS.items():
            self._texts  += prompts
            self._owners += [medium] * len(prompts)

        try:
            import ai_detector
            self._ai_detector_module = ai_detector
            self.available = bool(ai_detector._load_clip())
        except Exception as exc:
            logger.warning("MediumClassifier: CLIP erişilemedi: %s", exc)
            self._ai_detector_module = None
            self.available = False

        if self.available:
            logger.info("MediumClassifier v%s aktif (CLIP zero-shot)", self.VERSION)
        else:
            logger.info("MediumClassifier: CLIP yok, medium tespiti devre dışı")

    def classify(self, img_np: np.ndarray) -> dict:
        """
        @brief Görselin türünü belirler.

        @param img_np RGB uint8 görüntü dizisi.
        @return       Medium raporu (aşağıdaki alanlarla).

        @details Dönen rapor:
        - `available`          : sınıflandırma yapılabildi mi
        - `medium`             : photograph / artwork / render / screenshot / unknown
        - `medium_label`       : Türkçe etiket
        - `confidence`         : seçilen sınıfın skoru (%)
        - `scores`             : tüm sınıfların skorları
        - `photographic`       : fotoğraf mı
        - `confident`          : skor eşiği aştı mı
        - `blocks_ai_verdict`  : AI kararı geçersiz sayılmalı mı (ekran görüntüsü)
        - `caution`            : arayüzde gösterilecek uyarı veya ""
        """
        if not self.available:
            return self._empty_report()

        try:
            import torch

            module = self._ai_detector_module
            model  = module._clip_model
            proc   = module._clip_processor
            if model is None or proc is None:
                return self._empty_report()

            pil    = Image.fromarray(img_np).convert("RGB")
            inputs = proc(text=self._texts, images=pil,
                          return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                probs = model(**inputs).logits_per_image.softmax(dim=1)
                probs = probs.squeeze().tolist()

            scores = {
                medium: round(
                    sum(p for p, o in zip(probs, self._owners) if o == medium) * 100.0,
                    2
                )
                for medium in MEDIUM_PROMPTS
            }
            medium     = max(scores, key=scores.get)
            confidence = scores[medium]
            confident  = confidence >= MEDIUM_CONFIDENT_THRESHOLD

            logger.info("Medium: %s (%%%.1f)", medium, confidence)

            return {
                "available":         True,
                "classifier_version": self.VERSION,
                "medium":            medium,
                "medium_label":      MEDIUM_LABELS[medium],
                "confidence":        confidence,
                "scores":            scores,
                "photographic":      medium == "photograph",
                "confident":         confident,
                "blocks_ai_verdict": medium == "screenshot" and confident,
                "caution":           self._caution_for(medium, confident),
                "threshold":         MEDIUM_CONFIDENT_THRESHOLD,
            }

        except Exception as exc:
            logger.warning("Medium sınıflandırma hatası: %s", exc)
            return self._empty_report()

    @staticmethod
    def _caution_for(medium: str, confident: bool) -> str:
        """
        @brief Fotoğraf dışı içerik için gösterilecek uyarı metnini üretir.
        @param medium    Belirlenen medium sınıfı.
        @param confident Karar eşiği aşıldı mı.
        @return          Uyarı metni; gerekmiyorsa boş string.
        """
        if medium == "photograph" or not confident:
            return ""
        if medium == "screenshot":
            return ("Ekran görüntüsü tespit edildi. Arayüz grafikleri istatistiksel "
                    "olarak AI görsellerine benzediği için AI kararı verilmedi.")
        if medium == "artwork":
            return ("Bu içerik bir fotoğraf değil (tablo / çizim). İnsan yapımı sanat "
                    "ile AI üretimi sanatın ayrımı fotoğraflara göre daha az "
                    "güvenilirdir; skoru bu çerçevede değerlendirin.")
        return ("Bu içerik bir fotoğraf değil (3B render / oyun görüntüsü). "
                "AI ayrımı fotoğraflara göre daha az güvenilirdir.")

    @staticmethod
    def _empty_report() -> dict:
        """
        @brief Sınıflandırma yapılamadığında dönen güvenli rapor.
        @return `available=False` olan, kararı etkilemeyen rapor.
        """
        return {
            "available":         False,
            "classifier_version": MediumClassifier.VERSION,
            "medium":            "unknown",
            "medium_label":      MEDIUM_LABELS["unknown"],
            "confidence":        0.0,
            "scores":            {},
            "photographic":      True,
            "confident":         False,
            "blocks_ai_verdict": False,
            "caution":           "",
            "threshold":         MEDIUM_CONFIDENT_THRESHOLD,
        }
