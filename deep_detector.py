"""
@file deep_detector.py
@brief İkinci derin öğrenme modeli: denetimli eğitilmiş AI görsel dedektörü (User Story-3).

@details
  CLIP zero-shot kanalı "eğitimsiz" bir benzerlik ölçümü yapar. Bu modül,
  doğrudan "AI üretimi / gerçek" ikili sınıflandırması için **etiketli veriyle
  eğitilmiş** ayrı bir modeli ikinci bağımsız algoritma olarak çalıştırır.

  Modül model-bağımsızdır: `models/ai_image_detector/` klasöründeki herhangi bir
  HuggingFace `*ForImageClassification` modelini (Swin, SwinV2, ViT, SigLIP,
  ConvNeXt, EfficientNet...) yükleyebilir. Model dosyaları repoya konmaz;
  `tools/download_models.py` ile bir kez indirilir
  (mevcut `clip_processor/` kalıbının aynısı).

  **Çevrimdışı çalışır:** indirme sonrası internet veya API anahtarı gerekmez.

@note Model klasörü yoksa veya yükleme başarısız olursa modül sessizce devre
      dışı kalır (`available == False`) ve analiz CLIP + istatistiksel kanalla
      sürer. Hiçbir koşulda istisna fırlatmaz (Ek Madde-1: çökme = -10 puan).

@author ImageForensics Team
@version 1.0.0
"""

import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_BASE_DIR  = Path(__file__).parent
MODELS_DIR = _BASE_DIR / "models"

## Varsayılan model klasörü (indirme script'i buraya yazar).
DEFAULT_MODEL_DIR = MODELS_DIR / "ai_image_detector"

## Sınıf etiketlerinde "AI üretimi" anlamına gelen anahtar kelimeler.
AI_LABEL_TOKENS = (
    "artificial", "ai", "ai_generated", "aigenerated", "fake", "synthetic",
    "generated", "gan", "midjourney", "stable", "diffusion", "sahte", "yapay",
)

## Sınıf etiketlerinde "gerçek fotoğraf" anlamına gelen anahtar kelimeler.
REAL_LABEL_TOKENS = (
    "human", "real", "hum", "authentic", "camera", "photo", "photograph",
    "natural", "nature", "gercek", "gerçek",
)

## Mimari kodu → insan okunur ad.
ARCH_LABELS = {
    "swin":       "Swin Transformer",
    "swinv2":     "Swin Transformer V2",
    "vit":        "Vision Transformer (ViT)",
    "siglip":     "SigLIP",
    "convnext":   "ConvNeXt (CNN)",
    "convnextv2": "ConvNeXt V2 (CNN)",
    "resnet":     "ResNet (CNN)",
    "efficientnet": "EfficientNet (CNN)",
    "deit":       "DeiT",
    "beit":       "BEiT",
}


def _normalize(label: str) -> str:
    """
    @brief Etiket metnini karşılaştırma için sadeleştirir.
    @param label Ham sınıf etiketi.
    @return      Küçük harfli, ayraçları temizlenmiş etiket.
    """
    return "".join(ch for ch in label.lower() if ch.isalnum())


def _classify_label(label: str) -> str:
    """
    @brief Bir sınıf etiketinin AI mı, gerçek mi anlamına geldiğini belirler.
    @param label Model config'indeki `id2label` değeri (örn. "artificial").
    @return      "ai", "real" veya "unknown".
    """
    norm = _normalize(label)
    for token in AI_LABEL_TOKENS:
        if norm == token or norm.startswith(token):
            return "ai"
    for token in REAL_LABEL_TOKENS:
        if norm == token or norm.startswith(token):
            return "real"
    return "unknown"


class DeepAIDetector:
    """
    @class DeepAIDetector
    @brief Yerel klasörden yüklenen, eğitimli AI görsel sınıflandırıcısı.

    @details
    Kullanım:
    @code
        det = DeepAIDetector()
        if det.available:
            res = det.predict(img_np)   # {"confidence": 93.2, ...} veya None
    @endcode

    `available` False ise `predict()` daima None döner; çağıran taraf bu durumu
    normal kabul etmelidir (model dosyaları opsiyoneldir).
    """

    VERSION = "1.0.0"

    def __init__(self, model_dir: Path = None):
        """
        @brief Modeli diskten yükler; başarısızlık hâlinde devre dışı kalır.
        @param model_dir Model klasörü; None ise `models/ai_image_detector/`.
        """
        self.model_dir    = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.available    = False
        self.display_name = "Derin AI Dedektörü"
        self.architecture = ""
        self.source       = ""
        self.license      = ""

        self._model     = None
        self._processor = None
        self._ai_ids    = []
        self._real_ids  = []

        self._load()

    # ------------------------------------------------------------------
    # Yükleme
    # ------------------------------------------------------------------

    def _read_card(self) -> dict:
        """
        @brief İndirme script'inin yazdığı `model_card.json` dosyasını okur.
        @return Kart sözlüğü; dosya yoksa boş sözlük.
        """
        card_path = self.model_dir / "model_card.json"
        if not card_path.exists():
            return {}
        try:
            with open(card_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("model_card.json okunamadı: %s", exc)
            return {}

    def _load(self) -> None:
        """
        @brief Model ve ön-işlemciyi yükler, etiket eşlemesini kurar.
        @note Tüm hatalar yutulur; yalnızca log'a yazılır.
        """
        if not self.model_dir.exists():
            logger.info(
                "İkinci derin model bulunamadı (%s) — CLIP + istatistiksel mod. "
                "İndirmek için: python tools/download_models.py --component ai",
                self.model_dir,
            )
            return

        try:
            import torch  # noqa: F401  (yalnızca varlık kontrolü)
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            logger.info("[3/4] İkinci AI modeli yükleniyor (~745 MB)...")
            processor = AutoImageProcessor.from_pretrained(str(self.model_dir))
            model     = AutoModelForImageClassification.from_pretrained(
                str(self.model_dir)
            )
            model.eval()

            id2label = getattr(model.config, "id2label", {}) or {}
            for idx, label in id2label.items():
                kind = _classify_label(str(label))
                if kind == "ai":
                    self._ai_ids.append(int(idx))
                elif kind == "real":
                    self._real_ids.append(int(idx))

            if not self._ai_ids:
                logger.warning(
                    "Model etiketlerinde AI sınıfı tanınamadı (%s) — devre dışı",
                    list(id2label.values()),
                )
                return

            card              = self._read_card()
            model_type        = str(getattr(model.config, "model_type", "")).lower()
            self.architecture = card.get("architecture") or ARCH_LABELS.get(
                model_type, model_type.upper() or "Bilinmeyen mimari"
            )
            self.display_name = card.get("display_name") or self.architecture
            self.source       = card.get("source", "")
            self.license      = card.get("license", "")

            self._processor = processor
            self._model     = model
            self.available  = True

            logger.info(
                "DeepAIDetector yüklendi: %s (%s), AI sınıf id=%s",
                self.display_name, self.architecture, self._ai_ids,
            )

        except Exception as exc:
            logger.warning("DeepAIDetector yüklenemedi: %s", exc)
            self.available = False

    # ------------------------------------------------------------------
    # Çıkarım
    # ------------------------------------------------------------------

    def predict(self, img_np: np.ndarray):
        """
        @brief Görsel için AI üretimi olasılığını hesaplar.
        @param img_np RGB uint8 numpy dizisi.
        @return       `{"confidence", "ai_generated", "name", "architecture",
                      "reason", "probabilities"}` veya hata/devre dışı hâlinde None.
        """
        if not self.available:
            return None

        try:
            import torch

            pil    = Image.fromarray(img_np).convert("RGB")
            inputs = self._processor(images=pil, return_tensors="pt")

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1).squeeze(0).tolist()

            if isinstance(probs, float):      # tek sınıflı model koruması
                probs = [probs]

            ai_p   = sum(probs[i] for i in self._ai_ids   if i < len(probs))
            real_p = sum(probs[i] for i in self._real_ids if i < len(probs))

            # Etiketlerin bir kısmı tanınmadıysa yalnız tanınanlar normalize edilir
            total  = ai_p + real_p
            ai_pct = (ai_p / total * 100.0) if total > 1e-9 else ai_p * 100.0

            id2label = getattr(self._model.config, "id2label", {}) or {}
            named    = {
                str(id2label.get(i, i)): round(p * 100.0, 2)
                for i, p in enumerate(probs)
            }

            logger.info("DeepAIDetector: AI=%.1f%% (%s)", ai_pct, self.display_name)

            return {
                "confidence":    round(float(ai_pct), 2),
                "ai_generated":  ai_pct >= 50.0,
                "name":          self.display_name,
                "architecture":  self.architecture,
                "reason":        f"Eğitimli sınıflandırıcı: AI=%{ai_pct:.1f}",
                "probabilities": named,
            }

        except Exception as exc:
            logger.warning("DeepAIDetector çıkarım hatası: %s", exc)
            return None

    def info(self) -> dict:
        """
        @brief Modelin arayüzde/raporda gösterilecek künyesi.
        @return available, name, architecture, source, license alanları.
        """
        return {
            "available":    self.available,
            "name":         self.display_name,
            "architecture": self.architecture,
            "source":       self.source,
            "license":      self.license,
            "model_dir":    str(self.model_dir),
        }
