"""
@file tests/test_fsm.py
@brief White box test: sonlu durum makinesi (FSM) geçiş doğrulaması.

@details `app.fsm_transition()` analiz akışının durum makinesini uygular:
         `IDLE -> UPLOAD -> PREPROCESS -> ANALYZE -> REPORT -> IDLE`.
         Geçersiz bir geçiş istendiğinde `ValueError` fırlatmalıdır; bu,
         API'nin 400 döndürmesini sağlayan mekanizmadır.

@note `app` modülü içe aktarılırken Flask uygulaması kurulur ve modeller
      yüklenir. Testlerin hızlı kalması için bu dosya **yalnızca** FSM
      fonksiyonunu kendi başına yeniden kurar; böylece model yüklenmez.
      Fonksiyonun ve geçiş tablosunun kaynaktan okunarak alınması, kopyanın
      gerçek koddan sapmasını engeller.
"""

import re
from pathlib import Path

import pytest

PROJE_KOKU = Path(__file__).resolve().parent.parent


def _fsm_yukle():
    """
    @brief `app.py` içinden FSM tablosunu ve fonksiyonunu modelsiz çıkarır.
    @return `(fsm_transition, FSM_TRANSITIONS)` ikilisi.
    @details Kaynaktan yalnızca ilgili bölüm alınır ve ayrı bir ad alanında
             çalıştırılır. Böylece test, gerçek kodu sınar ama Flask ve
             modeller yüklenmez.
    """
    kaynak = (PROJE_KOKU / "app.py").read_text(encoding="utf-8")

    tablo = re.search(r"FSM_TRANSITIONS = \{.*?\n\}", kaynak, re.S)
    fonk  = re.search(r"def fsm_transition\(.*?(?=\n@app\.|\ndef |\Z)", kaynak, re.S)
    assert tablo and fonk, "app.py içinde FSM tanımı bulunamadı"

    # Fonksiyon log yazıyor; ad alanına sessiz bir logger koyuyoruz.
    import logging
    ad_alani = {"logger": logging.getLogger("test_fsm")}
    exec(tablo.group(0) + "\n\n" + fonk.group(0), ad_alani)
    return ad_alani["fsm_transition"], ad_alani["FSM_TRANSITIONS"]


fsm_transition, FSM_TRANSITIONS = _fsm_yukle()


# ---------------------------------------------------------------------------
# TC-01: Geçerli tam tur
# ---------------------------------------------------------------------------

def test_tam_analiz_turu_gecerli():
    """Tam analiz akışı baştan sona geçerli olmalı."""
    durum = "IDLE"
    for hedef in ("UPLOAD", "PREPROCESS", "ANALYZE", "REPORT", "IDLE"):
        durum = fsm_transition(durum, hedef)
        assert durum == hedef

    assert durum == "IDLE", "Tur sonunda başlangıç durumuna dönülmeli"


# ---------------------------------------------------------------------------
# TC-02: Adım atlama reddedilmeli
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mevcut,hedef", [
    ("IDLE", "ANALYZE"),      # yükleme atlandı
    ("IDLE", "REPORT"),       # her şey atlandı
    ("UPLOAD", "REPORT"),     # ön işleme ve analiz atlandı
    ("PREPROCESS", "IDLE"),   # analiz edilmeden kapanış
    ("ANALYZE", "UPLOAD"),    # geriye dönüş
])
def test_gecersiz_gecis_hata_firlatir(mevcut, hedef):
    """Tanımsız her geçiş ValueError ile reddedilmeli."""
    with pytest.raises(ValueError):
        fsm_transition(mevcut, hedef)


# ---------------------------------------------------------------------------
# TC-03: Geçiş tablosunun bütünlüğü
# ---------------------------------------------------------------------------

def test_tablodaki_tum_durumlar_ulasilabilir():
    """
    Tabloda tanımlı her hedef durum, yine tabloda bir anahtar olmalı.
    Aksi hâlde makine çıkmaz bir duruma düşer.
    """
    for kaynak, hedefler in FSM_TRANSITIONS.items():
        for hedef in hedefler:
            assert hedef in FSM_TRANSITIONS, (
                f"{kaynak} -> {hedef} geçişi var ama {hedef} tabloda tanımlı değil"
            )


def test_bilinmeyen_durumdan_gecis_reddedilir():
    """Tabloda olmayan bir durumdan çıkış denenirse hata verilmeli."""
    with pytest.raises(ValueError):
        fsm_transition("YOK_BOYLE_BIR_DURUM", "IDLE")
