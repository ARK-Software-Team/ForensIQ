"""
@file tests/test_ai_karar.py
@brief White box test: AI kararı, metadata artışı ve düzenleme kanıtı.

@details Sınanan kurallar:
  - **K-12** metadata artışı: kesin AI etiketi skoru sınırlı ölçüde yükseltir,
    skor asla düşürülmez, üst sınır %95'tir.
  - **K-30** "AI ile düzenlenmiş" bulgusu AI *üretim* skorunu yükseltmez;
    düzenleme (US-2) kanalına gider.
  - **K-22** medium ekseni: ekran görüntüsünde AI kararı verilmez, tablo/çizimde
    yalnızca uyarı gösterilir.

@note `ForensicsEngine` normalde kurucusunda ~700 MB model yükler. Test
      yalnızca saf karar metotlarını sınadığı için nesne `object.__new__` ile
      modelsiz oluşturulur.
"""

import pytest

from forensics_engine import (
    METADATA_BOOST_CAP,
    METADATA_BOOST_RULES,
    ForensicsEngine,
)


@pytest.fixture(scope="module")
def motor():
    """@brief Model yüklemeden `ForensicsEngine` örneği (saf metotlar için)."""
    return object.__new__(ForensicsEngine)


def _ai_sonuc(skor):
    """@brief Test için basit bir AI sonuç sözlüğü üretir."""
    return {
        "generated": skor > 50.0,
        "confidence": skor,
        "verdict_label": "test",
        "signal_detail": [],
        "model_scores": [],
        "model_count": 2,
    }


def _meta(verdict, **ek):
    """@brief Test için basit bir metadata raporu üretir."""
    rapor = {"verdict": verdict, "confidence": 95.0, "generators": [],
             "verdict_label": verdict, "c2pa": {}}
    rapor.update(ek)
    return rapor


# ---------------------------------------------------------------------------
# TC-06: Metadata artışı (K-12)
# ---------------------------------------------------------------------------

def test_kesin_ai_etiketi_skoru_yukseltir(motor):
    """Üretici imzası bulunan görselin skoru taban sınırına çekilmeli."""
    sonuc, artis = motor._apply_metadata_boost(
        _ai_sonuc(20.0), _meta("ai_signature", generators=["Midjourney"]))

    assert artis is not None and artis["applied"] is True
    assert sonuc["confidence"] >= 75.0, "ai_signature için taban %75"
    assert sonuc["generated"] is True
    assert "Midjourney" in artis["generators"]


def test_artis_ust_sinirini_asamaz(motor):
    """
    Metadata başka dosyadan kopyalanabildiği için skor %95'i geçmemeli —
    sistem asla "%100 kesin" dememeli.
    """
    sonuc, artis = motor._apply_metadata_boost(
        _ai_sonuc(94.0), _meta("ai_signature", generators=["DALL-E"]))
    assert sonuc["confidence"] <= METADATA_BOOST_CAP
    assert artis["cap"] == METADATA_BOOST_CAP


def test_skor_asla_dusurulmez(motor):
    """
    Kamera EXIF'i veya temiz metadata, yüksek bir AI skorunu AŞAĞI çekmemeli:
    metadata'nın temiz olması görüntünün gerçek olduğunu kanıtlamaz.
    """
    for verdict in ("camera_evidence", "stripped", "no_trace"):
        sonuc, artis = motor._apply_metadata_boost(_ai_sonuc(88.0), _meta(verdict))
        assert sonuc["confidence"] == 88.0, f"{verdict} skoru değiştirdi"
        assert artis is None


def test_bozuk_metadata_raporu_cokmez(motor):
    """Beklenmedik girdi analizi düşürmemeli (Ek Madde-1)."""
    for bozuk in (None, "metin", 42, {}):
        sonuc, artis = motor._apply_metadata_boost(_ai_sonuc(30.0), bozuk)
        assert sonuc["confidence"] == 30.0
        assert artis is None


# ---------------------------------------------------------------------------
# TC-07: "AI ile düzenlenmiş" üretim skorunu yükseltmez (K-30 regresyonu)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["c2pa_ai_composite", "c2pa_ai_enhanced"])
def test_duzenleme_beyani_ai_skorunu_yukseltmez(motor, verdict):
    """
    K-30 REGRESYONU — AI ile iyileştirilmiş/düzenlenmiş gerçek bir fotoğrafın
    içeriği gerçektir; AI *üretim* skoru yükseltilmemeli.
    """
    assert verdict not in METADATA_BOOST_RULES, (
        f"{verdict} artış tablosuna geri eklenmiş — K-30 kararı bozuldu"
    )
    sonuc, artis = motor._apply_metadata_boost(_ai_sonuc(24.5), _meta(verdict))
    assert sonuc["confidence"] == 24.5
    assert artis is None


@pytest.mark.parametrize("verdict,beklenen", [
    ("c2pa_ai_composite", True),
    ("c2pa_ai_enhanced",  True),
    ("c2pa_ai",           False),   # tamamen AI üretimi: düzenleme kanıtı değil
    ("stripped",          False),
])
def test_duzenleme_kaniti_dogru_toplanir(verdict, beklenen):
    """Kompozit ve iyileştirme beyanları düzenleme kanıtı sayılmalı."""
    kanit = ForensicsEngine._collect_edit_evidence(_meta(verdict))
    assert kanit["present"] is beklenen


def test_duzenleme_kaniti_yazilim_ve_eylemi_tasir():
    """Kanıt, künyedeki yazılım ve eylem bilgisini kullanıcıya taşımalı."""
    kanit = ForensicsEngine._collect_edit_evidence(_meta(
        "c2pa_ai_composite",
        c2pa={"software": ["Adobe Photoshop", "Adobe Firefly"],
              "actions": ["edited"]}))
    assert kanit["present"] is True
    assert "Adobe Photoshop" in kanit["detail"]
    assert "edited" in kanit["detail"]
    assert kanit["source"].startswith("C2PA")


def test_duzenleme_kaniti_bozuk_girdide_cokmez():
    """Geçersiz girdi boş kanıt döndürmeli, istisna değil."""
    for bozuk in (None, "metin", 7, []):
        kanit = ForensicsEngine._collect_edit_evidence(bozuk)
        assert kanit["present"] is False


# ---------------------------------------------------------------------------
# TC-08: Karar etiketi eşikleri
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("skor,beklenen", [
    (95.0, "AI ÜRETİMİ — KESİN"),
    (80.0, "AI ÜRETİMİ — KESİN"),
    (70.0, "AI ÜRETİMİ — YÜKSEK"),
    (55.0, "AI ÜRETİMİ — ZAYIF"),
    (40.0, "GERÇEK — ZAYIF"),
    (25.0, "GERÇEK — YÜKSEK"),
    (5.0,  "GERÇEK — KESİN"),
])
def test_verdict_etiketi_esikleri(skor, beklenen):
    """Skor aralıkları doğru etikete çevrilmeli."""
    assert ForensicsEngine._verdict_label_for(skor) == beklenen


def test_elli_esigi_ai_tarafinda():
    """
    Eşik davranışı net olmalı: %50 üstü AI, altı gerçek.
    (Animasyonlu GIF hatasında bu eşik belirleyiciydi.)
    """
    assert "AI" in ForensicsEngine._verdict_label_for(50.1)
    assert "GERÇEK" in ForensicsEngine._verdict_label_for(49.9)
