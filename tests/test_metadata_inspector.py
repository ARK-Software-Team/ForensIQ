"""
@file tests/test_metadata_inspector.py
@brief White box test: metadata / C2PA karar mantığı (US-4).

@details Bu dosyadaki testlerin çoğu **gerçek bir hatanın regresyon testidir**.
         25 Kasım 2026'da şu hata bulundu: IPTC `DigitalSourceType` değerleri
         sırayla aranıp ilk eşleşmede duruluyordu. `trainedalgorithmicmedia`
         metni `compositeWithTrainedAlgorithmicMedia` **içinde de geçtiği için**
         AI ile düzenlenmiş gerçek bir fotoğraf "tamamen AI üretimi" sayılıyordu.
         (bkz. karar K-29, `gelistirme/02-KARAR-GUNLUGU.md`)

@note Model dosyası gerektirmez; yalnızca saf çözümleme mantığı sınanır.
"""

import pytest

from metadata_inspector import (
    IPTC_SOURCE_TYPES,
    MetadataInspector,
    match_source_type,
)


# ---------------------------------------------------------------------------
# TC-04: IPTC kaynak türü ayrıştırma (K-29 regresyonu)
# ---------------------------------------------------------------------------

IPTC_TABAN = "http://cv.iptc.org/newscodes/digitalsourcetype/"


@pytest.mark.parametrize("deger,beklenen_tur,ai_uretimi", [
    ("trainedalgorithmicmedia",             "generated",   True),
    ("compositewithtrainedalgorithmicmedia", "composite",  True),
    ("algorithmicallyenhanced",             "enhanced",    False),
    ("algorithmicmedia",                    "algorithmic", False),
    ("digitalcapture",                      "capture",     False),
])
def test_kaynak_turu_dogru_siniflanir(deger, beklenen_tur, ai_uretimi):
    """Her IPTC değeri kendi türüne eşlenmeli."""
    sonuc = match_source_type(IPTC_TABAN + deger)
    assert sonuc is not None, f"{deger} tanınmadı"
    assert sonuc[1] == beklenen_tur
    assert sonuc[3] is ai_uretimi


def test_kompozit_tamamen_ai_sayilmaz():
    """
    K-29 REGRESYONU — bu testin kırılması, düzelttiğimiz hatanın geri
    geldiği anlamına gelir.

    `compositeWithTrainedAlgorithmicMedia` metni `trainedAlgorithmicMedia`
    alt dizesini içerir. Kısa olan önce aranırsa kompozit bir görüntü
    "tamamen AI üretimi" sayılır.
    """
    sonuc = match_source_type(IPTC_TABAN + "compositewithtrainedalgorithmicmedia")
    assert sonuc[1] == "composite"
    assert sonuc[1] != "generated", (
        "Kompozit beyan 'tamamen AI üretimi' olarak okunuyor — K-29 hatası geri geldi"
    )


def test_kaynak_turu_listesi_uzundan_kisaya_sirali():
    """
    Eşleştirmenin doğru çalışması listenin sırasına bağlı. Daha kısa bir
    değer, kendisini içeren daha uzun bir değerden ÖNCE gelmemeli.
    """
    anahtarlar = [k for k, _, _, _ in IPTC_SOURCE_TYPES]
    for i, kisa in enumerate(anahtarlar):
        for uzun in anahtarlar[i + 1:]:
            assert kisa not in uzun, (
                f"'{kisa}' listede '{uzun}'dan önce geliyor ama onun alt dizesi; "
                f"sıra bozuk"
            )


def test_tanimsiz_deger_none_doner():
    """Bilinmeyen bir kaynak türü sessizce None dönmeli, hata değil."""
    assert match_source_type("http://example.com/birseyler") is None
    assert match_source_type("") is None
    assert match_source_type(None) is None


# ---------------------------------------------------------------------------
# TC-05: Sentetik dosyalarla uçtan uca metadata kararı
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inspector():
    """@brief Durumsuz inceleyici; model gerektirmez."""
    return MetadataInspector()


def test_bos_dosya_cokmeden_rapor_doner(inspector):
    """Geçersiz girdi analizi düşürmemeli (Ek Madde-1)."""
    rapor = inspector.inspect(b"")
    assert isinstance(rapor, dict)
    assert rapor["verdict"] in ("error", "stripped", "no_trace")
    assert rapor["ai_tag_found"] is False


def test_bozuk_dosya_cokmeden_rapor_doner(inspector):
    """Rastgele baytlar da istisna fırlatmamalı."""
    rapor = inspector.inspect(b"bu bir goruntu dosyasi degil" * 50)
    assert isinstance(rapor, dict)
    assert "verdict" in rapor


def test_sd_parametre_tanima():
    """
    Stable Diffusion üretim parametreleri en az iki belirteçle tanınmalı;
    tek belirteç yetmemeli (yanlış pozitifi önler).
    """
    tanir = MetadataInspector._looks_like_sd_params
    assert tanir("Steps: 30, Sampler: Euler a, CFG scale: 7, Seed: 12345")
    assert tanir("negative prompt: blurry, steps: 20")
    assert not tanir("Bu fotoğrafta bir adım var")
    assert not tanir("")


def test_kontrol_karakterleri_temizlenir():
    """
    Metadata üçüncü taraf verisidir: kontrol karakterleri temizlenmeli ve
    değer kırpılmalı (arayüze XSS/bozuk çıktı gitmesin).
    """
    from metadata_inspector import _clean
    kirli = "Midjourney\x00\x07 v6\n\n   sürüm"
    temiz = _clean(kirli)
    assert "\x00" not in temiz, "NUL baytı temizlenmeli"
    assert "\x07" not in temiz, "Denetim karakteri temizlenmeli"
    assert "  " not in temiz, "Ardışık boşluklar sadeleştirilmeli"

    uzun = _clean("A" * 500, max_len=50)
    assert len(uzun) <= 50, "Kırpma işareti dâhil max_len aşılmamalı"
    assert uzun.endswith("..."), "Kırpıldığı belli olmalı"
