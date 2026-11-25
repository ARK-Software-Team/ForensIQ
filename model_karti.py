"""
@file model_karti.py
@brief Model künyesi (`model_card.json`) okuma ve metin temizleme yardımcıları.

@details
  `tools/download_models.py`, indirdiği her modelin yanına bir
  `model_card.json` dosyası yazar: mimari adı, görünen ad, kaynak ve lisans.
  Bu değerler hem **günlüğe** yazılır hem de `/api/models` ucuyla
  **arayüzde gösterilir**.

  Dosya depo dışından (Hugging Face) indirildiği için içeriği güvenilmez
  kabul edilir. Satır sonu veya denetim karakteri taşıyan bir değer:
  - günlükte **sahte satır** oluşturabilir (log injection, CWE-117),
  - arayüzde düzeni bozabilir.

  Bu modül iki şeyi tek yerde toplar:
  1. Künye dosyasını güvenli okuma (`kunye_oku`),
  2. Künyeden gelen metinleri temizleme (`guvenli_metin`).

  `deep_detector.py` ve `age_estimator.py` aynı işi ayrı ayrı yapıyordu;
  burada birleştirildi.

@note SonarQube `pythonsecurity:S5145` (Logging should not be vulnerable to
      injection attacks) bulgusu bu modülle kapatılır.
@author ImageForensics Team
@version 1.0.0
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

## Künye dosyasının adı (indirme script'i bu adla yazar).
KUNYE_DOSYASI = "model_card.json"

## Künyeden gelen bir metnin en fazla uzunluğu.
MAKS_UZUNLUK = 120

## Yazdırılabilir ASCII dışındaki her şey (satır sonu, sekme, denetim
## karakterleri) tek boşluğa indirgenir. Türkçe harfler korunur.
_DENETIM_DESENI = re.compile(r"[\x00-\x1f\x7f-\x9f]+")

## Birden çok boşluk tek boşluğa iner.
_BOSLUK_DESENI = re.compile(r"\s{2,}")


def guvenli_metin(deger, varsayilan: str = "") -> str:
    """
    @brief Künyeden gelen bir değeri günlüğe ve arayüze yazılabilir hâle getirir.

    @param deger      Künyeden okunan ham değer (her tür olabilir).
    @param varsayilan Değer boş veya kullanılamazsa dönecek metin.
    @return           Satır sonu ve denetim karakteri içermeyen, kırpılmış metin.

    @details Satır sonları günlükte sahte kayıt üretebileceği için (CWE-117)
             boşluğa çevrilir; uzunluk `MAKS_UZUNLUK` ile sınırlanır.
    """
    if deger is None:
        return varsayilan
    if not isinstance(deger, str):
        deger = str(deger)
    temiz = _DENETIM_DESENI.sub(" ", deger)
    temiz = _BOSLUK_DESENI.sub(" ", temiz).strip()
    if not temiz:
        return varsayilan
    return temiz[:MAKS_UZUNLUK]


def kunye_oku(model_dir: Path, aciklama: str = "Model künyesi") -> dict:
    """
    @brief `model_card.json` dosyasını okur ve **tüm metin alanlarını temizler**.

    @param model_dir Model klasörü.
    @param aciklama  Hata günlüğünde kullanılacak ad.
    @return          Künye sözlüğü; dosya yoksa veya bozuksa boş sözlük.

    @details Dönen sözlükteki her metin değeri `guvenli_metin()` ile
             geçirildiği için çağıran tarafın ayrıca temizlik yapması
             gerekmez.
    """
    yol = Path(model_dir) / KUNYE_DOSYASI
    if not yol.exists():
        return {}
    try:
        with open(yol, "r", encoding="utf-8") as fh:
            veri = json.load(fh)
    except Exception as exc:
        logger.warning("%s okunamadı: %s", aciklama, guvenli_metin(exc))
        return {}

    if not isinstance(veri, dict):
        return {}

    return {k: (guvenli_metin(v) if isinstance(v, str) else v)
            for k, v in veri.items()}
