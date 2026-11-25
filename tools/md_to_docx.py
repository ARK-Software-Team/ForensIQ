"""
@file tools/md_to_docx.py
@brief Teslim belgelerini Markdown'dan Word (.docx) biçimine çevirir.

@details
  Teslim edilecek belgeler `.docx` olmalıdır; kaynak metinler ise sürüm
  kontrolünde okunabilir kalsın diye Markdown tutulur. Bu araç ikisini
  bağlar: kaynak `.md`, çıktı `.docx`.

  **Stil profilleri.** Hocanın paylaştığı örnek belgelerin görsel biçimi
  (yazı tipi, punto, satır aralığı, kalınlık, renk, tablo çizgisi, kenar
  boşluğu) ölçülerek iki profile dönüştürülmüştür:

  | Profil | Örnek kaynak | Gövde |
  |---|---|---|
  | `std`      | `docs/örnek dosyalar/STD-Ornek.pdf` + `Test_Case_Ornegi…docx` | Arial 12 pt |
  | `sozlesme` | `docs/örnek dosyalar/sözlesme_ornegi.doc`                      | Times New Roman 12 pt, iki yana yaslı |

  Profil değerleri tahmin edilmedi; `PyMuPDF` ve `python-docx` ile örnek
  dosyalardan **okundu** (ayrıntı: `gelistirme/02-KARAR-GUNLUGU.md`, K-46).

  Desteklenen Markdown öğeleri:
  - Başlıklar (`#` … `######`)
  - Paragraflar, kalın (`**...**`), eğik (`*...*`), satır içi kod (`` `...` ``)
  - Madde listeleri (`-`, `*`) ve numaralı listeler
  - Tablolar (`| ... |` ve hizalama satırı); başlık satırı boşsa
    (`| | |`) başlık biçimi uygulanmaz — etiket/değer tablosu sayılır
  - Kod blokları (``` ile çevrili)
  - Yatay çizgi (`---`) → paragraf alt kenarlığı
  - Alıntı (`>`)

  Kullanım:
  @code
      python tools/md_to_docx.py                      # üç teslim belgesi
      python tools/md_to_docx.py "docs/v1 gereksinimleri/STD-Test-Dokumani.md"
      python tools/md_to_docx.py --stil sozlesme kaynak.md
  @endcode

@note Pandoc gibi harici bir araca ihtiyaç duymaz; yalnızca `python-docx`
      kullanır. Böylece belge üretimi projenin kendi ortamında tekrarlanabilir.
@author ImageForensics Team
@version 2.0.0
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

## Markdown kaynaklarının bulunduğu klasör (kullanıcı buraya taşıdı).
KAYNAK_DIR = DOCS_DIR / "v1 gereksinimleri"

## Üretilen `.docx` dosyalarının yazılacağı klasör.
CIKTI_DIR = DOCS_DIR

## Varsayılan olarak çevrilecek teslim belgeleri ve kullanacakları stil profili.
VARSAYILAN_BELGELER = {
    "Yazilim-Sartnamesi-ve-Sozlesmesi.md": "sozlesme",
    "STD-Test-Dokumani.md": "std",
    "TaskBoard-Scrum-Panosu.md": "std",
}

## Belgeye özel kenar boşluğu (sol, sağ, üst, alt — cm).
##
## Profildeki değeri ezer. Buradaki değerler **kullanıcının Word'de kendi
## ayarladığı** boşluklardır; belgeler yeniden üretildiğinde kaybolmasın
## diye kayda geçirildi (25.11.2026).
BELGE_KENAR_BOSLUKLARI = {
    "STD-Test-Dokumani.md": (1.27, 1.27, 1.27, 1.27),
    "Yazilim-Sartnamesi-ve-Sozlesmesi.md": (2.0, 2.0, 2.0, 2.0),
    "TaskBoard-Scrum-Panosu.md": (2.5, 2.5, 2.5, 2.5),
}

## Markdown içinde kullanılan yönergeler.
##   <!-- pagebreak -->  : sayfa sonu
##   <!-- cover -->      : sonraki boş satıra kadar olan blok kapak sayfası olur
##   <!-- center -->     : sonraki satır ortalanır
YONERGE_SAYFA_SONU = "<!-- pagebreak -->"
YONERGE_KAPAK = "<!-- cover -->"
YONERGE_ORTA = "<!-- center -->"


# ===========================================================================
# STİL PROFİLLERİ — değerler örnek belgelerden ÖLÇÜLDÜ
# ===========================================================================

## @brief `docs/örnek dosyalar/STD-Ornek.pdf` dosyasından ölçülen profil.
##
## Ölçülen değerler:
##  - sayfa A4 (595 × 842 pt), sol kenar 71 pt = 2,5 cm
##  - gövde Arial (PDF'te Helvetica) 12 pt, siyah
##  - satır aralığı ~13,8 pt → tek
##  - H1 "1. INTRODUCTION" → Arial **kalın 15 pt**
##  - H2 "1.1 Overview"    → Arial **kalın eğik 12 pt**
##  - kapak başlığı 16 pt kalın ortalı, diğer kapak satırları 14 pt kalın ortalı
##  - kapak başlığı sayfanın tepesinden ~211 pt aşağıda
##  - test case tabloları: Table Grid, **gölge yok, başlık satırı yok**,
##    etiket sütunu toplam genişliğin ~%22'si, 11 pt
PROFIL_STD = {
    "ad": "std",
    "govde_font": "Arial",
    "govde_punto": 12.0,
    "govde_hiza": "left",
    "satir_araligi": 1.0,
    "paragraf_sonrasi": 6.0,
    "kenar_cm": (2.5, 2.5, 2.5, 2.5),  # sol, sağ, üst, alt
    "metin_rengi": (0x00, 0x00, 0x00),
    "baslikler": {
        1: {"punto": 15.0, "kalin": True, "egik": False, "once": 18.0, "sonra": 4.0},
        2: {"punto": 12.0, "kalin": True, "egik": True, "once": 16.0, "sonra": 6.0},
        3: {"punto": 12.0, "kalin": True, "egik": False, "once": 12.0, "sonra": 4.0},
        4: {"punto": 12.0, "kalin": True, "egik": False, "once": 10.0, "sonra": 4.0},
    },
    "kapak": {
        "baslik_punto": 16.0,
        "diger_punto": 14.0,
        "ust_bosluk": 200.0,
        "satir_sonrasi": 10.0,
    },
    "tablo": {
        "punto": 11.0,
        "basligi_golgele": False,
        "basligi_kalinlastir": True,
        "kaynak_kalini_koru": True,
        "ilk_sutun_orani": 0.22,
        "hucre_sonrasi": 2.0,
    },
    "kod_font": "Consolas",
    "kod_punto": 9.5,
}

## @brief `docs/örnek dosyalar/sözlesme_ornegi.doc` dosyasından ölçülen profil.
##
## Ölçülen değerler:
##  - sayfa A4, sol/sağ kenar 2,5 cm, üst 2,13 cm
##  - gövde **Times New Roman 12 pt**, siyah, **iki yana yaslı** (JUSTIFY)
##  - satır aralığı 14 pt → tek; paragraflar arası ~28 pt → bir satır boşluk
##  - belge başlığı Times New Roman **kalın 16 pt**
##  - madde başlıkları ("1. TARAFLAR", "3.5.1. …") gövdeyle **aynı punto**,
##    yalnızca kalın — büyütülmez
##  - alt maddeler "2.1. Etiket : açıklama" biçiminde; etiket kalın, gerisi düz
PROFIL_SOZLESME = {
    "ad": "sozlesme",
    "govde_font": "Times New Roman",
    "govde_punto": 12.0,
    "govde_hiza": "justify",
    "satir_araligi": 1.0,
    "paragraf_sonrasi": 12.0,
    "kenar_cm": (2.5, 2.5, 2.13, 2.5),
    "metin_rengi": (0x00, 0x00, 0x00),
    "baslikler": {
        1: {"punto": 12.0, "kalin": True, "egik": False, "once": 18.0, "sonra": 6.0},
        2: {"punto": 12.0, "kalin": True, "egik": False, "once": 12.0, "sonra": 6.0},
        3: {"punto": 12.0, "kalin": True, "egik": False, "once": 10.0, "sonra": 4.0},
        4: {"punto": 12.0, "kalin": True, "egik": False, "once": 8.0, "sonra": 4.0},
    },
    "kapak": {
        "baslik_punto": 16.0,
        "diger_punto": 14.0,
        "ust_bosluk": 200.0,
        "satir_sonrasi": 10.0,
    },
    "tablo": {
        "punto": 11.0,
        "basligi_golgele": False,
        "basligi_kalinlastir": True,
        "kaynak_kalini_koru": True,
        "ilk_sutun_orani": 0.26,
        "hucre_sonrasi": 2.0,
    },
    "kod_font": "Consolas",
    "kod_punto": 9.5,
}

## Profil adı → profil sözlüğü.
STIL_PROFILLERI = {
    "std": PROFIL_STD,
    "sozlesme": PROFIL_SOZLESME,
}


# ===========================================================================
# Düşük seviyeli yardımcılar
# ===========================================================================

def _font_ata(font, ad: str) -> None:
    """
    @brief Yazı tipini ASCII, hAnsi, eastAsia ve cs alanlarının tamamına yazar.

    @param font `docx` font nesnesi (run.font veya style.font).
    @param ad   Yazı tipi adı.

    @details `python-docx` yalnızca `w:ascii` alanını yazar; Türkçe
             karakterler `hAnsi` alanından çözümlendiği için Word o
             karakterleri farklı bir yazı tipiyle gösterebilir. Bu yüzden
             dört alan birlikte ayarlanır.
    """
    from docx.oxml.ns import qn
    font.name = ad
    rpr = font.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        from docx.oxml import OxmlElement
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for alan in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(alan), ad)


def _hiza(ad: str):
    """
    @brief Profildeki hizalama adını `docx` sabitine çevirir.
    @param ad "left", "center", "right" veya "justify".
    @return   `WD_ALIGN_PARAGRAPH` değeri.
    """
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    return {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[ad]


def _alt_kenarlik(paragraf, renk: str = "808080", kalinlik: int = 6) -> None:
    """
    @brief Paragrafa alt kenarlık (yatay çizgi) ekler.
    @param paragraf Hedef paragraf.
    @param renk     6 haneli onaltılık renk.
    @param kalinlik Kenarlık kalınlığı (1/8 pt birimi).
    """
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    ppr = paragraf._p.get_or_add_pPr()
    kenarlik = OxmlElement("w:pBdr")
    alt = OxmlElement("w:bottom")
    alt.set(qn("w:val"), "single")
    alt.set(qn("w:sz"), str(kalinlik))
    alt.set(qn("w:space"), "1")
    alt.set(qn("w:color"), renk)
    kenarlik.append(alt)
    ppr.append(kenarlik)


def _hucre_golgele(hucre, renk: str = "D9E2F3") -> None:
    """
    @brief Tablo hücresine arka plan rengi verir (başlık satırı için).
    @param hucre Hedef hücre.
    @param renk  6 haneli onaltılık renk.
    """
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    golge = OxmlElement("w:shd")
    golge.set(qn("w:val"), "clear")
    golge.set(qn("w:fill"), renk)
    hucre._tc.get_or_add_tcPr().append(golge)


def _satir_bolunmesin(satir) -> None:
    """
    @brief Tablo satırının sayfa sonunda ikiye bölünmesini engeller.
    @param satir Hedef tablo satırı.
    """
    from docx.oxml import OxmlElement
    satir._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def _tablo_sabit_duzen(tablo) -> None:
    """
    @brief Tabloyu sabit düzene alır; böylece verilen sütun genişlikleri korunur.
    @param tablo Hedef tablo.
    """
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    tblpr = tablo._tbl.tblPr
    duzen = OxmlElement("w:tblLayout")
    duzen.set(qn("w:type"), "fixed")
    tblpr.append(duzen)


# ===========================================================================
# Belge iskeleti
# ===========================================================================

class DocxYazici:
    """
    @brief Markdown satırlarını, verilen stil profiline göre Word belgesine yazar.

    @details Profil; yazı tipi, punto, satır aralığı, başlık ağırlıkları, kapak
             düzeni ve tablo görünümünü belirler. Böylece aynı Markdown
             kaynağından farklı örnek belgelere uyan çıktılar üretilebilir.
    """

    def __init__(self, profil: dict):
        """
        @brief Boş bir belge açar ve profili uygular.
        @param profil `STIL_PROFILLERI` içindeki sözlüklerden biri.
        """
        from docx import Document
        self.profil = profil
        self.belge = Document()
        ## Sayfa sonu yönergesi görüldüğünde hemen boş bir paragraf eklemek
        ## yerine bayrak kaldırılır; sonrasında gelen ilk blok "önce sayfa
        ## sonu" özelliğiyle yazılır. Aksi hâlde, sayfa zaten dolmuşsa
        ## **tamamen boş bir sayfa** oluşuyordu.
        self._bekleyen_sayfa_sonu = False
        ## Tablodan sonra eklenen ayırıcı paragraf; hemen ardından sayfa
        ## sonu gelirse bu paragraf tek başına yeni bir sayfaya düşüp
        ## **boş sayfa** oluşturuyordu, bu yüzden geri alınabilir tutulur.
        self._bosluk_paragrafi = None
        self._sayfa_duzeni()
        self._stilleri_kur()

    def sayfa_sonu(self) -> None:
        """@brief Sonraki bloğun yeni sayfadan başlamasını işaretler."""
        self._bekleyen_sayfa_sonu = True

    def _sayfa_sonu_uygula(self, paragraf) -> None:
        """
        @brief Bekleyen sayfa sonunu verilen paragrafa uygular.
        @param paragraf Yeni sayfadan başlaması istenen paragraf.
        """
        if self._bekleyen_sayfa_sonu:
            # Sayfa sonundan hemen önceki ayırıcı paragraf gereksizdir;
            # bırakılırsa tek başına boş bir sayfa oluşturur.
            if self._bosluk_paragrafi is not None:
                oge = self._bosluk_paragrafi._p
                oge.getparent().remove(oge)
            paragraf.paragraph_format.page_break_before = True
            self._bekleyen_sayfa_sonu = False
        self._bosluk_paragrafi = None

    # --- kurulum -----------------------------------------------------------

    def _sayfa_duzeni(self) -> None:
        """
        @brief Sayfa boyutunu ve kenar boşluklarını profilden uygular.

        @details `python-docx`'in hazır şablonu **Letter** (612 × 792 pt)
                 boyutundadır; hocanın örnek belgelerinin tamamı **A4**
                 (595 × 842 pt). Boyut burada açıkça ayarlanır.
        """
        from docx.shared import Cm, Mm
        sol, sag, ust, alt = self.profil["kenar_cm"]
        for bolum in self.belge.sections:
            bolum.page_width = Mm(210)
            bolum.page_height = Mm(297)
            bolum.left_margin = Cm(sol)
            bolum.right_margin = Cm(sag)
            bolum.top_margin = Cm(ust)
            bolum.bottom_margin = Cm(alt)

    def _stilleri_kur(self) -> None:
        """
        @brief `Normal`, `Heading 1..4`, liste ve alıntı stillerini profile göre
               yeniden tanımlar.

        @details Word'ün hazır başlık stilleri Calibri Light ve mavi renktedir;
                 örnek belgelerde başlıklar gövde yazı tipiyle ve siyahtır.
                 Stiller doğrudan değiştirildiği için anahat (outline) düzeyi
                 korunur, yalnızca görünüm değişir.
        """
        from docx.shared import Pt, RGBColor
        p = self.profil
        renk = RGBColor(*p["metin_rengi"])

        # --- Normal ---
        normal = self.belge.styles["Normal"]
        _font_ata(normal.font, p["govde_font"])
        normal.font.size = Pt(p["govde_punto"])
        normal.font.color.rgb = renk
        npf = normal.paragraph_format
        npf.line_spacing = p["satir_araligi"]
        npf.space_before = Pt(0)
        npf.space_after = Pt(p["paragraf_sonrasi"])
        npf.alignment = _hiza(p["govde_hiza"])

        # --- Başlıklar ---
        for seviye, bicim in p["baslikler"].items():
            try:
                stil = self.belge.styles[f"Heading {seviye}"]
            except KeyError:
                continue
            _font_ata(stil.font, p["govde_font"])
            stil.font.size = Pt(bicim["punto"])
            stil.font.bold = bicim["kalin"]
            stil.font.italic = bicim["egik"]
            stil.font.color.rgb = renk
            spf = stil.paragraph_format
            spf.space_before = Pt(bicim["once"])
            spf.space_after = Pt(bicim["sonra"])
            spf.line_spacing = p["satir_araligi"]
            spf.keep_with_next = True
            spf.alignment = _hiza("left")

        # --- Listeler: paragraf boşluğu daraltılır ---
        for ad in ("List Bullet", "List Number"):
            try:
                stil = self.belge.styles[ad]
            except KeyError:
                continue
            _font_ata(stil.font, p["govde_font"])
            stil.font.size = Pt(p["govde_punto"])
            stil.font.color.rgb = renk
            lpf = stil.paragraph_format
            lpf.line_spacing = p["satir_araligi"]
            lpf.space_before = Pt(0)
            lpf.space_after = Pt(2)

        # --- Alıntı: hazır stil mavi/eğiktir, gövdeye çekilir ---
        try:
            alinti = self.belge.styles["Intense Quote"]
            _font_ata(alinti.font, p["govde_font"])
            alinti.font.size = Pt(p["govde_punto"])
            alinti.font.italic = True
            alinti.font.bold = False
            alinti.font.color.rgb = renk
            qpf = alinti.paragraph_format
            qpf.line_spacing = p["satir_araligi"]
            qpf.space_after = Pt(p["paragraf_sonrasi"])
            # Word'ün hazır "Intense Quote" stili mavi bir alt çizgi çizer;
            # örnek belgelerde böyle bir çizgi yok, kaldırılır.
            from docx.oxml.ns import qn
            ppr = alinti.element.get_or_add_pPr()
            for kenarlik in ppr.findall(qn("w:pBdr")):
                ppr.remove(kenarlik)
        except KeyError:
            pass

    # --- satır içi biçim ---------------------------------------------------

    def satir_ici(self, paragraf, metin: str, punto: float = None,
                  kalin: bool = None, egik: bool = None) -> None:
        """
        @brief Satır içi biçimlendirmeyi (kalın / eğik / kod) paragrafa uygular.

        @param paragraf Hedef `docx` paragrafı.
        @param metin    Markdown metni.
        @param punto    Zorlanacak punto; None ise stilden gelir.
        @param kalin    Tüm parçalara zorlanacak kalınlık; None ise Markdown'a bakılır.
        @param egik     Tüm parçalara zorlanacak eğiklik.

        @details `**kalın**`, `*eğik*` ve `` `kod` `` parçaları ayrı "run"lar
                 hâlinde eklenir; geri kalan düz metin olarak yazılır.
        """
        from docx.shared import Pt
        p = self.profil
        desen = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")

        def _uygula(run, kod: bool = False):
            if kod:
                _font_ata(run.font, p["kod_font"])
                run.font.size = Pt(punto - 1.0 if punto else p["govde_punto"] - 1.0)
            else:
                _font_ata(run.font, p["govde_font"])
                if punto is not None:
                    run.font.size = Pt(punto)
            if kalin is not None:
                run.font.bold = kalin
            if egik is not None:
                run.font.italic = egik

        for parca in desen.split(metin):
            if not parca:
                continue
            if parca.startswith("**") and parca.endswith("**"):
                run = paragraf.add_run(parca[2:-2])
                _uygula(run)
                if kalin is None:
                    run.font.bold = True
            elif parca.startswith("`") and parca.endswith("`"):
                run = paragraf.add_run(parca[1:-1])
                _uygula(run, kod=True)
            elif parca.startswith("*") and parca.endswith("*") and len(parca) > 2:
                run = paragraf.add_run(parca[1:-1])
                _uygula(run)
                if egik is None:
                    run.font.italic = True
            else:
                run = paragraf.add_run(parca)
                _uygula(run)

    # --- bloklar -----------------------------------------------------------

    def kapak(self, satirlar: list) -> None:
        """
        @brief Kapak sayfasını profildeki düzene göre yazar.

        @param satirlar Kapak bloğundaki Markdown satırları.

        @details Örnek STD belgesinde kapak başlığı sayfanın tepesinden
                 ~200 pt aşağıda, 16 pt kalın ve ortalıdır; diğer kapak
                 satırları 14 pt kalın ve ortalıdır.
        """
        from docx.shared import Pt
        k = self.profil["kapak"]
        for sira, metin in enumerate(satirlar):
            paragraf = self.belge.add_paragraph()
            paragraf.alignment = _hiza("center")
            pf = paragraf.paragraph_format
            pf.space_before = Pt(k["ust_bosluk"] if sira == 0 else 0)
            pf.space_after = Pt(k["satir_sonrasi"])
            pf.line_spacing = 1.0
            baslik = re.match(r"^(#{1,6})\s+(.*)$", metin)
            if baslik:
                punto = k["baslik_punto"] if len(baslik.group(1)) == 1 else k["diger_punto"]
                self.satir_ici(paragraf, baslik.group(2).replace("**", ""),
                               punto=punto, kalin=True)
            else:
                self.satir_ici(paragraf, metin, punto=k["diger_punto"], kalin=True)
            if sira == 0:
                self._sayfa_sonu_uygula(paragraf)
        self.sayfa_sonu()

    def baslik(self, seviye: int, metin: str) -> None:
        """
        @brief Başlık paragrafı ekler.
        @param seviye 1–6 arası Markdown başlık düzeyi.
        @param metin  Başlık metni.
        """
        from docx.shared import Pt
        sinir = min(seviye, max(self.profil["baslikler"]))
        bicim = self.profil["baslikler"][sinir]
        paragraf = self.belge.add_paragraph(style=f"Heading {sinir}")
        self._sayfa_sonu_uygula(paragraf)
        self.satir_ici(paragraf, metin.replace("**", ""),
                       punto=bicim["punto"], kalin=bicim["kalin"], egik=bicim["egik"])

    def paragraf(self, metin: str) -> None:
        """
        @brief Normal paragraf ekler.
        @param metin Markdown metni.
        """
        paragraf = self.belge.add_paragraph()
        self._sayfa_sonu_uygula(paragraf)
        self.satir_ici(paragraf, metin)

    def liste(self, metin: str, numarali: bool = False) -> None:
        """
        @brief Madde veya numaralı liste öğesi ekler.
        @param metin    Öğe metni.
        @param numarali True ise numaralı liste stili kullanılır.
        """
        stil = "List Number" if numarali else "List Bullet"
        paragraf = self.belge.add_paragraph(style=stil)
        self._sayfa_sonu_uygula(paragraf)
        paragraf.paragraph_format.alignment = _hiza("left")
        self.satir_ici(paragraf, metin)

    def alinti(self, metin: str) -> None:
        """
        @brief Alıntı paragrafı ekler.
        @param metin Alıntı metni.
        """
        try:
            paragraf = self.belge.add_paragraph(style="Intense Quote")
        except KeyError:
            paragraf = self.belge.add_paragraph()
        self._sayfa_sonu_uygula(paragraf)
        self.satir_ici(paragraf, metin)

    def kod_blogu(self, satirlar: list) -> None:
        """
        @brief Kod bloğunu tek paragrafta, sabit genişlikli yazı tipiyle ekler.
        @param satirlar Kod satırları.
        """
        from docx.shared import Pt, RGBColor
        p = self.profil
        paragraf = self.belge.add_paragraph()
        self._sayfa_sonu_uygula(paragraf)
        pf = paragraf.paragraph_format
        pf.alignment = _hiza("left")
        pf.line_spacing = 1.0
        pf.space_after = Pt(p["paragraf_sonrasi"])
        run = paragraf.add_run("\n".join(satirlar))
        _font_ata(run.font, p["kod_font"])
        run.font.size = Pt(p["kod_punto"])
        run.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)

    def yatay_cizgi(self) -> None:
        """@brief Yatay ayraç (paragraf alt kenarlığı) ekler."""
        from docx.shared import Pt
        paragraf = self.belge.add_paragraph()
        self._sayfa_sonu_uygula(paragraf)
        paragraf.paragraph_format.space_before = Pt(2)
        paragraf.paragraph_format.space_after = Pt(8)
        _alt_kenarlik(paragraf)

    def tablo(self, satirlar: list) -> None:
        """
        @brief Markdown tablosunu Word tablosuna çevirir.

        @param satirlar Tabloya ait Markdown satırları.

        @details Örnek belgelerdeki görünüm uygulanır: `Table Grid` çizgileri,
                 hücre içi tek satır aralığı, profil yazı tipi ve puntosu.
                 Markdown başlık satırı boşsa (`| | |`) tablo bir
                 **etiket/değer** tablosu sayılır; başlık satırı hiç
                 yazılmaz ve ilk sütun daraltılır — hocanın test case
                 örneğindeki düzen budur.
        """
        from docx.shared import Pt, Emu

        p = self.profil
        t = p["tablo"]
        veri = [_hucreler(s) for s in satirlar if not _hizalama_satiri_mi(s)]
        if not veri:
            return

        # Başlık satırı tamamen boşsa etiket/değer tablosudur.
        basliksiz = not any(h.strip() for h in veri[0])
        if basliksiz:
            veri = veri[1:]
            if not veri:
                return

        # Bekleyen sayfa sonu, tablodan önce sıfır yükseklikli bir
        # paragrafa uygulanır; tablolara doğrudan uygulanamaz.
        if self._bekleyen_sayfa_sonu:
            onculu = self.belge.add_paragraph()
            self._sayfa_sonu_uygula(onculu)
            opf = onculu.paragraph_format
            opf.space_before = Pt(0)
            opf.space_after = Pt(0)
            opf.line_spacing = 1.0
            onculu.add_run().font.size = Pt(1)

        sutun = max(len(r) for r in veri)
        tablo = self.belge.add_table(rows=len(veri), cols=sutun)
        tablo.style = "Table Grid"
        tablo.autofit = False
        _tablo_sabit_duzen(tablo)

        # Kullanılabilir genişlik = sayfa - kenar boşlukları
        bolum = self.belge.sections[0]
        kullanilabilir = bolum.page_width - bolum.left_margin - bolum.right_margin
        oranlar = _sutun_oranlari(veri, sutun, t["ilk_sutun_orani"])
        genislikler = [Emu(int(kullanilabilir * o)) for o in oranlar]

        for i, satir in enumerate(veri):
            for j in range(sutun):
                hucre = tablo.cell(i, j)
                hucre.width = genislikler[j]
                hucre.text = ""
                paragraf = hucre.paragraphs[0]
                pf = paragraf.paragraph_format
                pf.line_spacing = 1.0
                pf.space_before = Pt(0)
                pf.space_after = Pt(t["hucre_sonrasi"])
                pf.alignment = _hiza("left")
                self.satir_ici(paragraf, satir[j] if j < len(satir) else "",
                               punto=t["punto"])
                if i == 0 and not basliksiz:
                    if t["basligi_kalinlastir"]:
                        for run in paragraf.runs:
                            run.font.bold = True
                    if t["basligi_golgele"]:
                        _hucre_golgele(hucre)

        # Etiket/değer tablolarında satırlar sayfa sonunda ikiye
        # bölünmesin; başlıklı tablolarda başlık her sayfada yinelensin.
        if basliksiz:
            for satir_nes in tablo.rows:
                _satir_bolunmesin(satir_nes)
        else:
            tablo.rows[0]._tr.get_or_add_trPr().append(_tekrar_eden_baslik())

    def tablo_sonrasi_bosluk(self) -> None:
        """
        @brief Tablodan sonra küçük bir ayırıcı paragraf ekler.

        @details Tam boy boş paragraf, sayfa sonlarında gereksiz taşmaya
                 yol açıyordu; bu yüzden yüksekliği 6 pt'a indirildi.
        """
        from docx.shared import Pt
        paragraf = self.belge.add_paragraph()
        pf = paragraf.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        paragraf.add_run().font.size = Pt(6)
        self._bosluk_paragrafi = paragraf

    def kaydet(self, hedef: Path) -> Path:
        """
        @brief Belgeyi diske yazar.
        @param hedef Çıktı yolu.
        @return      Yazılan yol.
        """
        hedef.parent.mkdir(parents=True, exist_ok=True)
        self.belge.save(str(hedef))
        return hedef


def _sutun_oranlari(veri: list, sutun: int, ilk_oran: float) -> list:
    """
    @brief Tablo sütun genişliklerini **içerik uzunluğuna göre** dağıtır.

    @param veri      Satır listesi (her satır hücre metinleri listesi).
    @param sutun     Sütun sayısı.
    @param ilk_oran  İki sütunlu etiket/değer tablolarında ilk sütunun oranı.
    @return          Toplamı 1,0 olan oran listesi.

    @details Eşit dağıtım, "#" veya "No" gibi dar sütunları gereksiz
             genişletip metni dar sütunlara sıkıştırıyordu. Burada her
             sütunun en uzun hücresi esas alınır; ardından hiçbir sütun
             %10'un altına düşmeyecek ve %55'i aşmayacak biçimde sınırlanır.
             Alt sınır %8 iken "QG-1" gibi kısa etiketler bile satır
             kaydırmasına uğruyordu.
    """
    if sutun == 2:
        return [ilk_oran, 1.0 - ilk_oran]

    uzunluklar = []
    for j in range(sutun):
        en_uzun = 0
        for satir in veri:
            if j < len(satir):
                en_uzun = max(en_uzun, len(satir[j]))
        uzunluklar.append(max(en_uzun, 1))

    toplam = float(sum(uzunluklar))
    oranlar = [u / toplam for u in uzunluklar]
    oranlar = [min(max(o, 0.10), 0.55) for o in oranlar]
    toplam = sum(oranlar)
    return [o / toplam for o in oranlar]


def _tekrar_eden_baslik():
    """
    @brief Tablo başlık satırının her sayfada yinelenmesini sağlayan öğeyi üretir.
    @return `w:tblHeader` XML öğesi.
    """
    from docx.oxml import OxmlElement
    return OxmlElement("w:tblHeader")


# ===========================================================================
# Markdown ayrıştırma yardımcıları
# ===========================================================================

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


def _blok_baslangici_mi(satir: str) -> bool:
    """
    @brief Satırın yeni bir Markdown bloğu başlatıp başlatmadığını söyler.

    @param satir Tek satır metin.
    @return      Başlık, tablo, liste, alıntı, kod çiti, yönerge, yatay çizgi
                 veya boş satırsa True.

    @details Markdown'da bir paragraf **boş satırda** biter; arada kalan
             satır kaydırmaları aynı paragrafın parçasıdır. Bu yardımcı,
             paragrafın nerede bittiğini belirlemek için kullanılır.
    """
    k = satir.strip()
    if not k:
        return True
    if k.startswith("<!--") or k.startswith("```") or k.startswith(">"):
        return True
    if re.match(r"^#{1,6}\s+", k):
        return True
    if re.fullmatch(r"-{3,}|={3,}|\*{3,}", k):
        return True
    if _tablo_mu(satir):
        return True
    if re.match(r"^\s*[-*]\s+", satir) or re.match(r"^\s*\d+[.)]\s+", satir):
        return True
    return False


## Künye satırı deseni: satırın tamamı `**Etiket:**` ile başlar.
KUNYE_DESENI = re.compile(r"^\*\*[^*]{1,45}:\*\*")


def _kunye_satiri_mi(satir: str) -> bool:
    """
    @brief Satırın `**Etiket:** değer` biçiminde bir künye satırı olup
           olmadığını söyler.
    @param satir Tek satır metin.
    @return      Desene uyuyorsa True.
    """
    return bool(KUNYE_DESENI.match(satir.strip()))


def _kunye_blogu_mu(satirlar: list, i: int) -> bool:
    """
    @brief Satırın, **ardışık** künye satırlarından oluşan bir bloğun
           parçası olup olmadığını söyler.

    @param satirlar Tüm Markdown satırları.
    @param i        İncelenen satırın sırası.
    @return         Üstünde veya altında başka bir künye satırı varsa True.

    @details "Proje: … / Araç: … / Tarih: …" gibi arka arkaya gelen künye
             satırları **ayrı paragraflar** olmalıdır. Buna karşılık
             sözleşmedeki "**2.1. ForensIQ:** Dijital görüntüler üzerinde…"
             maddesi tek başınadır ve devam satırlarıyla **birleşmelidir**.
             Ayrım, komşu satırın da künye olup olmamasına bakılarak yapılır.
    """
    if not _kunye_satiri_mi(satirlar[i]):
        return False
    onceki = i - 1 >= 0 and _kunye_satiri_mi(satirlar[i - 1])
    sonraki = i + 1 < len(satirlar) and _kunye_satiri_mi(satirlar[i + 1])
    return onceki or sonraki


def _blogu_birlestir(satirlar: list, i: int) -> tuple:
    """
    @brief Bir paragrafa ait ardışık satırları tek metinde birleştirir.

    @param satirlar Tüm Markdown satırları.
    @param i        Paragrafın ilk satırının sırası.
    @return         (birleştirilmiş metin, bir sonraki satırın sırası)
    """
    if _kunye_blogu_mu(satirlar, i):
        return satirlar[i].strip(), i + 1
    parcalar = [satirlar[i].strip()]
    j = i + 1
    while (j < len(satirlar) and not _blok_baslangici_mi(satirlar[j])
           and not _kunye_blogu_mu(satirlar, j)):
        parcalar.append(satirlar[j].strip())
        j += 1
    return " ".join(parcalar), j


# ===========================================================================
# Ana dönüştürücü
# ===========================================================================

def md_to_docx(md_yolu: Path, docx_yolu: Path = None, stil: str = "std",
               kenar_cm: tuple = None) -> Path:
    """
    @brief Bir Markdown dosyasını, verilen stil profiliyle Word belgesine çevirir.

    @param md_yolu   Kaynak `.md` dosyası.
    @param docx_yolu Hedef `.docx`; None ise `docs/` altına aynı adla yazılır.
    @param stil      `STIL_PROFILLERI` anahtarı ("std" veya "sozlesme").
    @param kenar_cm  (sol, sağ, üst, alt) cm; verilirse profildekini ezer.
                     Verilmezse `BELGE_KENAR_BOSLUKLARI` içindeki kayıt
                     kullanılır.
    @return          Üretilen dosyanın yolu.
    """
    if stil not in STIL_PROFILLERI:
        raise ValueError(f"bilinmeyen stil profili: {stil}")

    profil = STIL_PROFILLERI[stil]
    ozel = kenar_cm or BELGE_KENAR_BOSLUKLARI.get(md_yolu.name)
    if ozel:
        profil = dict(profil)
        profil["kenar_cm"] = tuple(ozel)

    yazici = DocxYazici(profil)
    hedef = docx_yolu or (CIKTI_DIR / (md_yolu.stem + ".docx"))

    satirlar = md_yolu.read_text(encoding="utf-8").split("\n")
    i = 0
    kod_icinde = False
    kod_tamponu = []

    while i < len(satirlar):
        satir = satirlar[i]
        kirpik = satir.strip()

        # --- Yönergeler ---
        if kirpik == YONERGE_SAYFA_SONU:
            yazici.sayfa_sonu()
            i += 1
            continue

        if kirpik == YONERGE_KAPAK:
            i += 1
            blok = []
            while i < len(satirlar) and satirlar[i].strip():
                blok.append(satirlar[i].strip())
                i += 1
            yazici.kapak(blok)
            continue

        if kirpik == YONERGE_ORTA:
            i += 1
            if i < len(satirlar):
                paragraf = yazici.belge.add_paragraph()
                yazici._sayfa_sonu_uygula(paragraf)
                paragraf.alignment = _hiza("center")
                yazici.satir_ici(paragraf, satirlar[i].strip())
                i += 1
            continue

        # --- Kod bloğu ---
        if kirpik.startswith("```"):
            if kod_icinde:
                yazici.kod_blogu(kod_tamponu)
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
            yazici.yatay_cizgi()
            i += 1
            continue

        # --- Başlık ---
        baslik = re.match(r"^(#{1,6})\s+(.*)$", kirpik)
        if baslik:
            yazici.baslik(len(baslik.group(1)), baslik.group(2).strip())
            i += 1
            continue

        # --- Tablo ---
        if _tablo_mu(satir):
            blok = []
            while i < len(satirlar) and _tablo_mu(satirlar[i]):
                blok.append(satirlar[i])
                i += 1
            yazici.tablo(blok)
            yazici.tablo_sonrasi_bosluk()
            continue

        # --- Alıntı ---
        if kirpik.startswith(">"):
            blok = []
            while i < len(satirlar) and satirlar[i].strip().startswith(">"):
                blok.append(satirlar[i].strip().lstrip("> ").strip())
                i += 1
            yazici.alinti(" ".join(blok))
            continue

        # --- Madde listesi ---
        madde = re.match(r"^\s*[-*]\s+(.*)$", satir)
        if madde:
            metin = madde.group(1).strip()
            i += 1
            while i < len(satirlar) and not _blok_baslangici_mi(satirlar[i]):
                metin += " " + satirlar[i].strip()
                i += 1
            yazici.liste(metin)
            continue

        # --- Numaralı liste ---
        numarali = re.match(r"^\s*\d+[.)]\s+(.*)$", satir)
        if numarali:
            metin = numarali.group(1).strip()
            i += 1
            while i < len(satirlar) and not _blok_baslangici_mi(satirlar[i]):
                metin += " " + satirlar[i].strip()
                i += 1
            yazici.liste(metin, numarali=True)
            continue

        # --- Normal paragraf (satır kaydırmaları birleştirilir) ---
        metin, i = _blogu_birlestir(satirlar, i)
        yazici.paragraf(metin)

    return yazici.kaydet(hedef)


def main() -> int:
    """
    @brief Komut satırı girişi.

    @details Argüman verilmezse `VARSAYILAN_BELGELER` sözlüğündeki üç teslim
             belgesi, her birine atanmış stil profiliyle üretilir.
             `--stil <ad>` ile profil elle seçilebilir.
    @return Çıkış kodu (0 = başarılı).
    """
    argv = sys.argv[1:]
    zorlanan_stil = None
    if "--stil" in argv:
        k = argv.index("--stil")
        try:
            zorlanan_stil = argv[k + 1]
        except IndexError:
            print("--stil için profil adı gerekli", file=sys.stderr)
            return 2
        del argv[k:k + 2]

    if argv:
        isler = [(Path(a), zorlanan_stil or "std") for a in argv]
    else:
        isler = [(KAYNAK_DIR / ad, zorlanan_stil or stil)
                 for ad, stil in VARSAYILAN_BELGELER.items()]

    basarili = 0
    for kaynak, stil in isler:
        if not kaynak.exists():
            print(f"  atlandı (yok): {kaynak}")
            continue
        try:
            hedef = md_to_docx(kaynak, stil=stil)
            boyut = hedef.stat().st_size / 1024
            print(f"  {kaynak.name}  ->  {hedef.name}  [{stil}]  ({boyut:.0f} KB)")
            basarili += 1
        except Exception as exc:
            print(f"  HATA: {kaynak.name} -> {exc}", file=sys.stderr)

    print(f"\n{basarili}/{len(isler)} belge üretildi.")
    return 0 if basarili == len(isler) else 1


if __name__ == "__main__":
    raise SystemExit(main())
