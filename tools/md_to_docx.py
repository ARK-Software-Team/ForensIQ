"""
@file tools/md_to_docx.py
@brief Teslim belgelerini Markdown'dan Word (.docx) biçimine çevirir.

@details
  Teslim edilecek belgeler `.docx` olmalıdır; kaynak metinler ise sürüm
  kontrolünde okunabilir kalsın diye Markdown tutulur. Bu araç ikisini
  bağlar: kaynak `.md`, çıktı `.docx`.

  Desteklenen Markdown öğeleri:
  - Başlıklar (`#` … `######`)
  - Paragraflar, kalın (`**...**`), eğik (`*...*`), satır içi kod (`` `...` ``)
  - Madde listeleri (`-`, `*`) ve numaralı listeler
  - Tablolar (`| ... |` ve hizalama satırı)
  - Kod blokları (``` ile çevrili)
  - Yatay çizgi (`---`)
  - Alıntı (`>`)

  Kullanım:
  @code
      python tools/md_to_docx.py                      # docs/ içindeki hepsi
      python tools/md_to_docx.py docs/STD-Test-Dokumani.md
  @endcode

@note Pandoc gibi harici bir araca ihtiyaç duymaz; yalnızca `python-docx`
      kullanır. Böylece belge üretimi projenin kendi ortamında tekrarlanabilir.
@author ImageForensics Team
@version 1.0.0
"""

import re
import sys
from pathlib import Path

# Windows konsolunda Türkçe karakter sorunu yaşanmasın
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = _BASE_DIR / "docs"

## Varsayılan olarak çevrilecek teslim belgeleri.
VARSAYILAN_BELGELER = [
    "Yazilim-Sartnamesi-ve-Sozlesmesi.md",
    "STD-Test-Dokumani.md",
    "TaskBoard-Scrum-Panosu.md",
]

## Gövde yazı tipi ve kod yazı tipi.
GOVDE_FONT = "Calibri"
KOD_FONT = "Consolas"


def _satir_ici_bicimle(paragraf, metin: str) -> None:
    """
    @brief Satır içi biçimlendirmeyi (kalın / eğik / kod) paragrafa uygular.

    @param paragraf Hedef `docx` paragrafı.
    @param metin    Markdown metni.

    @details `**kalın**`, `*eğik*` ve `` `kod` `` parçaları ayrı "run"lar
             hâlinde eklenir; geri kalan düz metin olarak yazılır.
    """
    desen = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")
    for parca in desen.split(metin):
        if not parca:
            continue
        if parca.startswith("**") and parca.endswith("**"):
            paragraf.add_run(parca[2:-2]).bold = True
        elif parca.startswith("`") and parca.endswith("`"):
            run = paragraf.add_run(parca[1:-1])
            run.font.name = KOD_FONT
        elif parca.startswith("*") and parca.endswith("*") and len(parca) > 2:
            paragraf.add_run(parca[1:-1]).italic = True
        else:
            paragraf.add_run(parca)


def _tablo_mu(satir: str) -> bool:
    """
    @brief Satırın Markdown tablo satırı olup olmadığını söyler.
    @param satir Tek satır metin.
    @return      Tablo satırıysa True.
    """
    s = satir.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def _hucreler(satir: str) -> list:
    """
    @brief Tablo satırını hücrelere ayırır.
    @param satir `| a | b |` biçiminde satır.
    @return      Hücre metinleri listesi.
    """
    return [h.strip() for h in satir.strip().strip("|").split("|")]


def _hizalama_satiri_mi(satir: str) -> bool:
    """
    @brief Satırın tablo hizalama satırı (`|---|---|`) olup olmadığını söyler.
    @param satir Tek satır metin.
    @return      Hizalama satırıysa True.
    """
    return bool(re.fullmatch(r"\|[\s:|-]+\|", satir.strip()))


def _tablo_ekle(belge, satirlar: list) -> None:
    """
    @brief Markdown tablosunu Word tablosuna çevirir.
    @param belge    Hedef `Document`.
    @param satirlar Tabloya ait Markdown satırları.
    """
    veri = [_hucreler(s) for s in satirlar if not _hizalama_satiri_mi(s)]
    if not veri:
        return
    sutun = max(len(r) for r in veri)

    tablo = belge.add_table(rows=len(veri), cols=sutun)
    tablo.style = "Table Grid"

    for i, satir in enumerate(veri):
        for j in range(sutun):
            hucre = tablo.cell(i, j)
            hucre.text = ""
            p = hucre.paragraphs[0]
            _satir_ici_bicimle(p, satir[j] if j < len(satir) else "")
            if i == 0:
                for run in p.runs:
                    run.bold = True


def md_to_docx(md_yolu: Path, docx_yolu: Path = None) -> Path:
    """
    @brief Bir Markdown dosyasını Word belgesine çevirir.

    @param md_yolu   Kaynak `.md` dosyası.
    @param docx_yolu Hedef `.docx`; None ise aynı ad ve klasör kullanılır.
    @return          Üretilen dosyanın yolu.
    """
    from docx import Document
    from docx.shared import Pt, RGBColor

    hedef = docx_yolu or md_yolu.with_suffix(".docx")
    belge = Document()

    # Gövde yazı tipi
    stil = belge.styles["Normal"]
    stil.font.name = GOVDE_FONT
    stil.font.size = Pt(10.5)

    satirlar = md_yolu.read_text(encoding="utf-8").split("\n")
    i = 0
    kod_icinde = False
    kod_tamponu = []

    while i < len(satirlar):
        satir = satirlar[i]
        kirpik = satir.strip()

        # --- Kod bloğu ---
        if kirpik.startswith("```"):
            if kod_icinde:
                p = belge.add_paragraph()
                run = p.add_run("\n".join(kod_tamponu))
                run.font.name = KOD_FONT
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
                kod_tamponu = []
            kod_icinde = not kod_icinde
            i += 1
            continue
        if kod_icinde:
            kod_tamponu.append(satir)
            i += 1
            continue

        # --- Boş satır ---
        if not kirpik:
            i += 1
            continue

        # --- Yatay çizgi ---
        if re.fullmatch(r"-{3,}|={3,}|\*{3,}", kirpik):
            belge.add_paragraph("_" * 60)
            i += 1
            continue

        # --- Başlık ---
        baslik = re.match(r"^(#{1,6})\s+(.*)$", kirpik)
        if baslik:
            seviye = len(baslik.group(1))
            metin = baslik.group(2).replace("**", "").strip()
            belge.add_heading(metin, level=min(seviye, 4))
            i += 1
            continue

        # --- Tablo ---
        if _tablo_mu(satir):
            blok = []
            while i < len(satirlar) and _tablo_mu(satirlar[i]):
                blok.append(satirlar[i])
                i += 1
            _tablo_ekle(belge, blok)
            belge.add_paragraph()
            continue

        # --- Alıntı ---
        if kirpik.startswith(">"):
            p = belge.add_paragraph(style="Intense Quote")
            _satir_ici_bicimle(p, kirpik.lstrip("> ").strip())
            i += 1
            continue

        # --- Madde listesi ---
        madde = re.match(r"^\s*[-*]\s+(.*)$", satir)
        if madde:
            p = belge.add_paragraph(style="List Bullet")
            _satir_ici_bicimle(p, madde.group(1))
            i += 1
            continue

        # --- Numaralı liste ---
        numarali = re.match(r"^\s*\d+[.)]\s+(.*)$", satir)
        if numarali:
            p = belge.add_paragraph(style="List Number")
            _satir_ici_bicimle(p, numarali.group(1))
            i += 1
            continue

        # --- Normal paragraf ---
        p = belge.add_paragraph()
        _satir_ici_bicimle(p, kirpik)
        i += 1

    belge.save(hedef)
    return hedef


def main() -> int:
    """
    @brief Komut satırı girişi.
    @return Çıkış kodu (0 = başarılı).
    """
    if len(sys.argv) > 1:
        kaynaklar = [Path(a) for a in sys.argv[1:]]
    else:
        kaynaklar = [DOCS_DIR / ad for ad in VARSAYILAN_BELGELER]

    basarili = 0
    for kaynak in kaynaklar:
        if not kaynak.exists():
            print(f"  atlandı (yok): {kaynak}")
            continue
        try:
            hedef = md_to_docx(kaynak)
            boyut = hedef.stat().st_size / 1024
            print(f"  {kaynak.name}  ->  {hedef.name}  ({boyut:.0f} KB)")
            basarili += 1
        except Exception as exc:
            print(f"  HATA: {kaynak.name} -> {exc}", file=sys.stderr)

    print(f"\n{basarili}/{len(kaynaklar)} belge üretildi.")
    return 0 if basarili == len(kaynaklar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
