"""
@file tests/test_kisi_ve_yas.py
@brief White box test: kişi sayımı (US-5) ve yaş tahmini (US-6) mantığı.

@details Sınanan kurallar:
  - **K-19** mükerrer kutu elemesi: aynı kişiye ikinci kutu ve iki kişiyi
    kapsayan "grup kutusu" elenmeli; uzaktaki küçük bir kişi SİLİNMEMELİ.
  - **K-17** sayım kuralı: hiçbir gövdeye düşmeyen yüz ek kişi sayılır.
  - **K-32** yaş tahmini: beklenen değer yalnızca en olası sınıf ve komşuları
    üzerinden hesaplanır (uzak sınıflar tahmini sürüklememeli).
  - **K-33** sapma payı dinamik ve yaşla orantılı kırpılır.

@note Model dosyası gerektirmez; saf geometri ve olasılık mantığı sınanır.
"""

import pytest

from person_detector import (
    DEDUP_IOU_THRESHOLD,
    FACE_IN_BODY_RATIO,
    PersonDetector,
    _box_area,
    _iou,
    _overlap_ratio,
)
from age_estimator import MIN_MARGIN, AgeEstimator, parse_age_label


# ---------------------------------------------------------------------------
# TC-09: Kutu geometrisi
# ---------------------------------------------------------------------------

def test_kutu_alani_ve_kesisim():
    """Temel geometri doğru hesaplanmalı."""
    assert _box_area([0, 0, 10, 10]) == 100
    assert _box_area([10, 10, 0, 0]) == 0, "Ters kutu negatif alan vermemeli"

    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)


def test_icerilme_orani_yonlu():
    """
    `_overlap_ratio(iç, dış)` yönlüdür: küçük kutunun ne kadarının büyük
    kutuda kaldığını verir. Sayım kuralı buna dayanır.
    """
    kucuk = [10, 10, 20, 20]
    buyuk = [0, 0, 100, 100]
    assert _overlap_ratio(kucuk, buyuk) == pytest.approx(1.0)
    assert _overlap_ratio(buyuk, kucuk) < 0.05
    assert _overlap_ratio([0, 0, 0, 0], buyuk) == 0.0, "Sıfır alan 0 dönmeli"


# ---------------------------------------------------------------------------
# TC-10: Mükerrer kutu elemesi (K-19 regresyonu)
# ---------------------------------------------------------------------------

def test_govde_ici_parca_kutu_elenir():
    """
    K-19 REGRESYONU — 1 kişilik stüdyo portresinde tam gövde kutusunun içinde
    ikinci bir üst-beden kutusu çıkıyor ve 2 kişi sayılıyordu.
    """
    kutular = [
        {"box": [91, 180, 425, 682], "score": 0.999},   # tam gövde
        {"box": [143, 178, 363, 548], "score": 0.772},  # aynı kişinin üst bedeni
    ]
    kalan = PersonDetector._deduplicate(kutular)
    assert len(kalan) == 1, "Aynı kişiye ait iki kutu birleştirilmeli"
    assert kalan[0]["score"] == 0.999, "Yüksek skorlu kutu korunmalı"


def test_iki_kisiyi_kapsayan_grup_kutusu_elenir():
    """
    K-19 REGRESYONU — yan yana iki kişiyi birden kapsayan düşük skorlu kutu
    sayımı şişiriyordu.
    """
    kutular = [
        {"box": [40, 136, 210, 775], "score": 0.997},   # kişi A
        {"box": [170, 118, 352, 760], "score": 0.998},  # kişi B
        {"box": [93, 108, 294, 768], "score": 0.781},   # ikisini kapsayan kutu
    ]
    kalan = PersonDetector._deduplicate(kutular)
    assert len(kalan) == 2, "Grup kutusu elenmeli, iki kişi kalmalı"
    assert all(k["score"] > 0.9 for k in kalan)


def test_uzaktaki_kucuk_kisi_silinmez():
    """
    Elemenin TERS yönde hata yapmaması kritik: büyük bir kişinin kutusu içinde
    kalan uzaktaki küçük bir kişi silinmemeli. Alan oranı koşulu bunun içindir.
    """
    kutular = [
        {"box": [0, 0, 400, 900], "score": 0.99},      # öndeki büyük kişi
        {"box": [150, 100, 190, 190], "score": 0.85},  # arkadaki küçük kişi
    ]
    kalan = PersonDetector._deduplicate(kutular)
    assert len(kalan) == 2, "Uzaktaki küçük kişi yanlışlıkla elendi"


def test_ayri_kisiler_korunur():
    """Üst üste binmeyen kutular hiç elenmemeli."""
    kutular = [{"box": [0, 0, 50, 100], "score": 0.9},
               {"box": [200, 0, 250, 100], "score": 0.9},
               {"box": [400, 0, 450, 100], "score": 0.9}]
    assert len(PersonDetector._deduplicate(kutular)) == 3


def test_eleme_esikleri_makul_aralikta():
    """Eşikler ölçümle seçildi; kazara değiştirilirse test uyarmalı."""
    assert 0.40 <= DEDUP_IOU_THRESHOLD <= 0.50
    assert 0.50 <= FACE_IN_BODY_RATIO <= 0.70


# ---------------------------------------------------------------------------
# TC-11: Sayım kuralı — gövdeye düşmeyen yüz ek kişidir (K-17)
# ---------------------------------------------------------------------------

def test_yuz_govdeye_eslesir_veya_ek_kisi_olur():
    """
    Bir yüzün alanının `FACE_IN_BODY_RATIO` kadarı bir gövdenin içindeyse o
    gövdeye aittir; değilse gövdesi görünmeyen ayrı bir kişidir.
    """
    govde = [100, 100, 300, 700]
    icerideki_yuz = [150, 130, 230, 220]
    disaridaki_yuz = [500, 130, 580, 220]

    assert _overlap_ratio(icerideki_yuz, govde) >= FACE_IN_BODY_RATIO
    assert _overlap_ratio(disaridaki_yuz, govde) < FACE_IN_BODY_RATIO


# ---------------------------------------------------------------------------
# TC-12: Yaş etiketi çözümleme
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("etiket,alt,ust", [
    ("0-2",            0,  2),
    ("20-29",         20, 29),
    ("age 41-55",     41, 55),
    ("more than 70",  70, None),   # açık uçlu
    ("age 80 +",      80, None),
])
def test_yas_etiketi_cozumlenir(etiket, alt, ust):
    """Farklı modellerin etiket biçimleri aynı sayısal aralığa çevrilmeli."""
    sonuc = parse_age_label(etiket)
    assert sonuc is not None, f"{etiket} çözümlenemedi"
    assert sonuc[0] == alt
    if ust is not None:
        assert sonuc[1] == ust
    else:
        assert sonuc[1] > alt, "Açık uçlu sınıf üst sınır almalı"


def test_tanimsiz_yas_etiketi_none():
    """Sayı içermeyen etiket çözümlenememeli."""
    assert parse_age_label("bilinmiyor") is None
    assert parse_age_label(None) is None


# ---------------------------------------------------------------------------
# TC-13: Beklenen değer komşulukla sınırlı (K-32 regresyonu)
# ---------------------------------------------------------------------------

def _yas_tahminci(kovalar):
    """@brief Model yüklemeden, verilen sınıflarla bir tahminci kurar."""
    est = object.__new__(AgeEstimator)
    est._buckets = kovalar
    return est


# (indis, alt, ust, orta, etiket) — gerçek ViT modelinin sınıfları
VIT_KOVALARI = [
    (0, 0, 2, 1.0, "0-2"),
    (1, 3, 9, 6.0, "3-9"),
    (2, 10, 19, 14.5, "10-19"),
    (3, 20, 29, 24.5, "20-29"),
    (4, 30, 39, 34.5, "30-39"),
    (5, 40, 49, 44.5, "40-49"),
    (6, 50, 59, 54.5, "50-59"),
    (7, 60, 69, 64.5, "60-69"),
    (8, 70, 85, 77.5, "more than 70"),
]


def test_uzak_siniflar_bebek_tahminini_surukleyemez():
    """
    K-32 REGRESYONU — bebeğin yüzünde en olası sınıf `0-2` olmasına rağmen
    tüm sınıflar hesaba katılınca sonuç 7 yaşa çıkıyordu.
    """
    est = _yas_tahminci(VIT_KOVALARI)
    # En olası sınıf 0-2, ama kuyrukta dağınık olasılıklar var
    probs = [0.55, 0.20, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
    yas, pay, en_olasi, dagilim = est._expected_age(probs)

    assert en_olasi == "0-2"
    assert yas <= 5, f"Uzak sınıflar tahmini sürükledi: {yas}"
    assert len(dagilim) == len(VIT_KOVALARI), "Dağılımın tamamı raporlanmalı"


def test_komsu_siniflar_arasinda_yumusatma_yapilir():
    """
    Komşuluk kuralı bitişik sınıflar arasındaki kararsızlığı yine de hesaba
    katmalı: %60 `20-29` + %40 `30-39` -> ~28-29.
    """
    est = _yas_tahminci(VIT_KOVALARI)
    probs = [0, 0, 0, 0.6, 0.4, 0, 0, 0, 0]
    yas, _, en_olasi, _ = est._expected_age(probs)
    assert en_olasi == "20-29"
    assert 27 <= yas <= 30, f"Komşu sınıf yumuşatması çalışmadı: {yas}"


# ---------------------------------------------------------------------------
# TC-14: Sapma payı dinamik (K-33)
# ---------------------------------------------------------------------------

def test_kararli_dagilim_dar_pay_verir():
    """Model tek sınıfta yoğunsa pay alt sınıra yakın olmalı."""
    est = _yas_tahminci(VIT_KOVALARI)
    probs = [0, 0, 0, 0, 0, 0, 0.97, 0.03, 0]
    _, pay, _, _ = est._expected_age(probs)
    assert pay == MIN_MARGIN, f"Kararlı dağılımda pay büyük: {pay}"


def test_kararsiz_dagilim_genis_pay_verir():
    """
    Olasılık geniş bir aralığa yayılmışsa pay büyümeli.

    @note Tahmin yetişkin bir yaşta olmalı; çocuk yaşlarında yaş oranlı kırpma
          (bir sonraki test) baskın gelir ve bu kuralı gölgeler.
    """
    est = _yas_tahminci(VIT_KOVALARI)
    #            0-2   3-9  10-19 20-29 30-39 40-49 50-59 60-69  70+
    probs = [0.12, 0.03, 0.03, 0.08, 0.10, 0.26, 0.10, 0.13, 0.15]
    yas, pay, en_olasi, _ = est._expected_age(probs)

    assert en_olasi == "40-49"
    assert yas >= 30, "Yetişkin yaş beklenirdi"
    assert pay > MIN_MARGIN, f"Kararsız dağılımda pay büyümeli, ±{pay} geldi"


def test_pay_yasla_orantili_kirpilir():
    """3 yaşındaki birine '±12' demek anlamsız; pay yaşla sınırlanmalı."""
    est = _yas_tahminci(VIT_KOVALARI)
    probs = [0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1]
    yas, pay, _, _ = est._expected_age(probs)
    assert pay <= max(MIN_MARGIN, yas * 0.5) + 0.5, (
        f"{yas} yaş için pay ±{pay} çok geniş"
    )


def test_bos_olasilik_cokmez():
    """Geçersiz olasılık vektörü istisna fırlatmamalı."""
    est = _yas_tahminci(VIT_KOVALARI)
    yas, pay, _, _ = est._expected_age([0.0] * 9)
    assert yas is None
    assert pay == MIN_MARGIN
