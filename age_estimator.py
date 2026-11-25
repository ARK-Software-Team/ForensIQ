"""
@file age_estimator.py
@brief Görüntüdeki kişilerin yaşlarının tahmini (User Story-6).

@details
  Yaş modelleri yaşı **aralık sınıfları** olarak tahmin eder (0-2, 3-9, 10-19…).
  Bu modül, sınıf olasılıklarından **beklenen değeri** hesaplayarak tek bir
  tam sayı üretir (Karar K-7):

  @f[ \\hat{a} = \\sum_i p_i \\cdot orta_i @f]

  Örnek: `20-29` %60 + `30-39` %40 → `0.6 \\cdot 24.5 + 0.4 \\cdot 34.5 = 28.5`
  → **"29 yaş"**. Tek bir aralığı ("20-29") göstermek yerine bu yöntem,
  komşu sınıflar arasındaki kararsızlığı da hesaba katar.

  Yanına daima bir **sapma payı** yazılır. Pay sabit değildir: modelin kendi
  olasılık dağılımının standart sapmasından hesaplanır, en az `MIN_MARGIN`
  olmak üzere. Dağılım tek bir sınıfta yoğunsa pay küçük, iki sınıfa
  yayılmışsa büyük olur.

  Girdi olarak **yüz kutuları** kullanılır; bunları `person_detector.py`
  (YuNet) üretir. Yüz bulunamazsa yaş tahmini yapılmaz.

@note Model `models/age_estimator/` klasöründen yüklenir, repoda tutulmaz.
      Klasör yoksa modül sessizce devre dışı kalır (`available == False`)
      ve analiz yaş olmadan sürer. İstisna fırlatmaz (Ek Madde-1).

@warning Yaş tahmini doğası gereği **yaklaşıktır**. Işık, poz, makyaj, sakal,
         çözünürlük ve etnik köken tahmini belirgin biçimde kaydırabilir.
         Çıktı adli bir delil değil, bir tahmindir; arayüzde böyle sunulur.

@author ImageForensics Team
@version 1.0.0
"""

import json
import logging
import re
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_BASE_DIR  = Path(__file__).parent
MODELS_DIR = _BASE_DIR / "models"

## Varsayılan yaş modeli klasörü.
DEFAULT_MODEL_DIR = MODELS_DIR / "age_estimator"

## Yüz kutusunun her yönde bu oranda genişletilmesi (model bağlam ister).
FACE_CROP_MARGIN = 0.35

## Gösterilecek en küçük sapma payı (yıl).
MIN_MARGIN = 3

## Gösterilecek en büyük sapma payı (yıl).
MAX_MARGIN = 12

## Açık uçlu üst sınıf ("70+", "80 +") için varsayılan üst yaş.
OPEN_ENDED_TOP = 85

## Beklenen değer hesabına katılacak komşu sınıf sayısı (en olası sınıfın her
## iki yanında). Tüm sınıflar üzerinden hesap yapmak uzak sınıfların küçük
## olasılıklarının tahmini sürüklemesine yol açıyordu: bir bebeğin yüzünde
## en olası sınıf `0-2` iken tahmin 7 yaşa çıkıyordu. Komşulukla sınırlayınca
## aynı yüz 3 yaş veriyor ve kendi test setimizde ortalama hata 3,5 → 2,7 yıla
## düşüyor. Komşuluk 1'de kalır: amaç bitişik sınıflar arasındaki kararsızlığı
## yumuşatmak, uzak sınıfları karıştırmak değil.
NEIGHBOUR_SPAN = 1


def parse_age_label(label: str):
    """
    @brief Model etiketini sayısal yaş aralığına çevirir.

    @param label Ham etiket ("0-2", "20-29", "more than 70", "age 41-55", "80 +").
    @return      `(alt, ust)` yaş ikilisi veya tanınmazsa None.

    @details Farklı modeller farklı etiket biçimi kullanır; bu yüzden
             metinden sayılar çekilir ve açık uçlu ifadeler
             (`more than`, `+`, `over`) üst sınıf olarak yorumlanır.
    """
    if label is None:
        return None
    text = str(label).lower().strip()
    sayilar = [int(n) for n in re.findall(r"\d+", text)]

    acik_uclu = any(k in text for k in ("more than", "+", "over", "above", "ustu"))

    if not sayilar:
        return None
    if len(sayilar) >= 2 and not acik_uclu:
        return min(sayilar), max(sayilar)
    if acik_uclu:
        return sayilar[0], OPEN_ENDED_TOP
    # Tek sayı: dar bir aralık varsay
    return sayilar[0], sayilar[0]


class AgeEstimator:
    """
    @class AgeEstimator
    @brief Yüz kutularından yaş tahmini üretir (US-6).

    @details
    Kullanım:
    @code
        est = AgeEstimator()
        if est.available:
            sonuclar = est.estimate(img_np, yuz_kutulari)
            # [{"age": 29, "margin": 4, "range": "20-29", ...}, ...]
    @endcode

    `available` False ise `estimate()` boş liste döner.
    """

    VERSION = "1.0.0"

    def __init__(self, model_dir: Path = None):
        """
        @brief Yaş modelini diskten yükler; başarısızlıkta devre dışı kalır.
        @param model_dir Model klasörü; None ise `models/age_estimator/`.
        """
        self.model_dir    = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.available    = False
        self.display_name = "Yaş Tahmini"
        self.architecture = ""
        self.source       = ""
        self.license      = ""

        self._model     = None
        self._processor = None
        self._buckets   = []   # [(idx, alt, ust, orta, etiket), ...]

        self._load()

    # ------------------------------------------------------------------
    # Yükleme
    # ------------------------------------------------------------------

    def _read_card(self) -> dict:
        """
        @brief İndirme script'inin yazdığı `model_card.json` künyesini okur.
        @return Künye sözlüğü; dosya yoksa boş sözlük.
        """
        yol = self.model_dir / "model_card.json"
        if not yol.exists():
            return {}
        try:
            with open(yol, "r", encoding="utf-8") as fh:
                veri = json.load(fh)
            return veri if isinstance(veri, dict) else {}
        except Exception as exc:
            logger.warning("Yaş modeli künyesi okunamadı: %s", exc)
            return {}

    def _load(self) -> None:
        """
        @brief Model, ön-işlemci ve yaş sınıflarını hazırlar.
        @note Tüm hatalar yutulur; yalnızca log'a yazılır.
        """
        if not self.model_dir.exists():
            logger.info(
                "Yaş modeli bulunamadı (%s) — yaş tahmini devre dışı. "
                "İndirmek için: python tools/download_models.py --component age",
                self.model_dir,
            )
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            logger.info("Yaş tahmini modeli yükleniyor...")
            processor = AutoImageProcessor.from_pretrained(str(self.model_dir))
            model     = AutoModelForImageClassification.from_pretrained(
                str(self.model_dir)
            )
            model.eval()

            id2label = getattr(model.config, "id2label", {}) or {}
            kovalar  = []
            for idx, etiket in id2label.items():
                aralik = parse_age_label(etiket)
                if aralik is None:
                    continue
                alt, ust = aralik
                kovalar.append((int(idx), alt, ust, (alt + ust) / 2.0, str(etiket)))

            if len(kovalar) < 2:
                logger.warning("Yaş sınıfları çözümlenemedi (%s) — devre dışı",
                               list(id2label.values()))
                return

            kovalar.sort(key=lambda k: k[3])

            kart = self._read_card()
            self.architecture = kart.get("architecture") or str(
                getattr(model.config, "model_type", "")).upper()
            self.display_name = kart.get("display_name") or "Yaş Tahmini"
            self.source       = kart.get("source", "")
            self.license      = kart.get("license", "")

            self._processor = processor
            self._model     = model
            self._buckets   = kovalar
            self.available  = True

            logger.info("AgeEstimator yüklendi: %s (%s), %d yaş sınıfı",
                        self.display_name, self.architecture, len(kovalar))

        except Exception as exc:
            logger.warning("Yaş modeli yüklenemedi: %s", exc)
            self.available = False

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_face(img_np: np.ndarray, box, margin: float = FACE_CROP_MARGIN):
        """
        @brief Yüz kutusunu paylı biçimde kırpar.
        @param img_np RGB görüntü.
        @param box    `[x1, y1, x2, y2]` yüz kutusu.
        @param margin Her yöne eklenecek oran.
        @return       Kırpılmış RGB dizi veya kutu geçersizse None.
        @note Yaş modelleri saç/çene hattı gibi bağlamı kullandığı için
              kutu olduğu gibi değil, genişletilerek verilir.
        """
        h, w = img_np.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            return None

        px, py = bw * margin, bh * margin
        cx1 = max(0, int(round(x1 - px)))
        cy1 = max(0, int(round(y1 - py)))
        cx2 = min(w, int(round(x2 + px)))
        cy2 = min(h, int(round(y2 + py)))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None
        return img_np[cy1:cy2, cx1:cx2]

    def _expected_age(self, probs) -> tuple:
        """
        @brief Sınıf olasılıklarından beklenen yaşı ve sapma payını hesaplar.

        @param probs Sınıf olasılıkları listesi (model sırasında).
        @return      `(yas, pay, en_olasi_etiket, dagilim)`

        @details Beklenen değer `Σ p_i · orta_i`, sapma payı ise aynı
                 dağılımın standart sapmasıdır (en az `MIN_MARGIN`).
                 Böylece model kararsızken pay kendiliğinden büyür.
        """
        gecerli = [(i, orta, etiket) for (i, _a, _u, orta, etiket) in self._buckets
                   if i < len(probs)]
        if not gecerli:
            return None, MIN_MARGIN, "", {}

        dagilim = {etiket: round(float(probs[i]) * 100.0, 1)
                   for i, _o, etiket in gecerli}

        # --- Sapma payı: TÜM dağılımın standart sapması ---
        # Bilerek tam dağılımdan hesaplanır; model kararsızsa pay büyük olmalı.
        tam_toplam = sum(float(probs[i]) for i, _o, _e in gecerli)
        if tam_toplam <= 1e-9:
            return None, MIN_MARGIN, "", {}
        tam_ort = sum(float(probs[i]) * o for i, o, _e in gecerli) / tam_toplam
        tam_kare = sum(float(probs[i]) * (o ** 2) for i, o, _e in gecerli) / tam_toplam
        sapma = max(0.0, tam_kare - tam_ort ** 2) ** 0.5

        # --- Nokta tahmini: yalnızca en olası sınıf ve komşuları ---
        en_sira = max(range(len(gecerli)), key=lambda k: probs[gecerli[k][0]])
        bas = max(0, en_sira - NEIGHBOUR_SPAN)
        son = min(len(gecerli), en_sira + NEIGHBOUR_SPAN + 1)
        yerel = gecerli[bas:son]

        yerel_toplam = sum(float(probs[i]) for i, _o, _e in yerel)
        if yerel_toplam <= 1e-9:
            return None, MIN_MARGIN, "", dagilim
        beklenen = sum(float(probs[i]) * o for i, o, _e in yerel) / yerel_toplam

        # Payı yaşla orantılı kırp: 3 yaşındaki birine "±12" demek anlamsız.
        pay = max(MIN_MARGIN, min(MAX_MARGIN, sapma))
        pay = min(pay, max(float(MIN_MARGIN), beklenen * 0.5))
        pay = int(round(pay))

        en_olasi = gecerli[en_sira][2]
        return int(round(beklenen)), pay, en_olasi, dagilim

    def estimate(self, img_np: np.ndarray, face_boxes: list) -> list:
        """
        @brief Verilen yüz kutuları için yaş tahmini üretir (US-6).

        @param img_np     RGB uint8 görüntü.
        @param face_boxes `[{"box": [x1,y1,x2,y2], ...}, ...]` yüz listesi.
        @return           Her yüz için tahmin listesi; devre dışıysa boş liste.

        @details Her öğe:
        - `face_index`   : kaçıncı yüz (Y1, Y2… ile aynı sıra)
        - `age`          : tam sayı yaş tahmini
        - `margin`       : sapma payı (yıl)
        - `display`      : "29 yaş (±4)"
        - `likely_range` : modelin en olası sınıfı
        - `distribution` : tüm sınıfların yüzdeleri
        """
        if not self.available or not face_boxes:
            return []

        try:
            import torch

            kirpiklar = []
            indisler  = []
            for i, yuz in enumerate(face_boxes):
                kirpik = self._crop_face(img_np, yuz.get("box", [0, 0, 0, 0]))
                if kirpik is None:
                    continue
                kirpiklar.append(Image.fromarray(kirpik).convert("RGB"))
                indisler.append(i)

            if not kirpiklar:
                return []

            girdi = self._processor(images=kirpiklar, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**girdi).logits
                tumu   = torch.softmax(logits, dim=-1).tolist()

            sonuclar = []
            for sira, probs in zip(indisler, tumu):
                yas, pay, en_olasi, dagilim = self._expected_age(probs)
                if yas is None:
                    continue
                sonuclar.append({
                    "face_index":   sira,
                    "age":          yas,
                    "margin":       pay,
                    "display":      f"{yas} yaş (±{pay})",
                    "likely_range": en_olasi,
                    "distribution": dagilim,
                })

            logger.info("Yaş tahmini: %s",
                        ", ".join(s["display"] for s in sonuclar) or "yok")
            return sonuclar

        except Exception as exc:
            logger.warning("Yaş tahmini başarısız: %s", exc)
            return []

    def info(self) -> dict:
        """
        @brief Modelin künyesi (arayüz ve `GET /api/models` için).
        @return available, name, architecture, source, license, buckets.
        """
        return {
            "available":    self.available,
            "name":         self.display_name,
            "architecture": self.architecture,
            "source":       self.source,
            "license":      self.license,
            "buckets":      [k[4] for k in self._buckets],
            "model_dir":    str(self.model_dir),
        }
