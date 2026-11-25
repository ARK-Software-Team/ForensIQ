"""
@file metadata_inspector.py
@brief Dosya yapisindan (metadata) AI etiketi tespiti - User Story-4.
@details
  Goruntunun ikili icerigini ve metadata bloklarini tarayarak yapay zeka
  uretimine isaret eden etiketleri arar:

    1. EXIF         : Software / Make / Model / ImageDescription / UserComment ...
    2. PNG tEXt     : Stable Diffusion ve ComfyUI "parameters", "prompt", "workflow"
    3. XMP paketi   : xmp:CreatorTool, Iptc4xmpExt:DigitalSourceType, GenAI alanlari
    4. C2PA / JUMBF : Content Credentials imzasi (trainedAlgorithmicMedia)
    5. GIF / WebP   : comment uzantilari
    6. Kamera izi   : ISO, diyafram, lens, GPS, DateTimeOriginal (gercek foto gostergesi)

  Harici bagimlilik yoktur: yalnizca Pillow ve standart kutuphane kullanilir.

@note Metadata silinebilir. "Iz yok" sonucu goruntunun gercek oldugunu KANITLAMAZ;
      bu yuzden rapor ayri bir sinyal olarak sunulur, tek basina karar vermez.
@author ImageForensics Team
@version 1.0.0
"""

import re
import logging
from io import BytesIO

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imza sozlukleri
# ---------------------------------------------------------------------------

## AI uretici imzalari: regex -> insan okunur urun adi
AI_GENERATOR_PATTERNS = [
    (r"midjourney", "Midjourney"),
    (r"dall[\s_\-]?e|openai", "DALL-E / OpenAI"),
    (r"stable[\s_\-]?diffusion|sd[\-_]?xl|sdxl|stability\.?ai",
     "Stable Diffusion / SDXL"),
    (r"automatic1111|a1111|comfyui|invokeai|forge[\s_\-]?ui|dreamstudio",
     "Stable Diffusion arayuzu (A1111/ComfyUI/InvokeAI)"),
    (r"adobe[\s_\-]?firefly|firefly", "Adobe Firefly"),
    (r"leonardo\.?ai", "Leonardo.Ai"),
    (r"playground[\s_\-]?ai", "Playground AI"),
    (r"nightcafe|artbreeder|craiyon|novelai|getimg|clipdrop|recraft",
     "AI goruntu servisi"),
    (r"black[\s_\-]?forest[\s_\-]?labs|flux\.?1|flux[\s_\-]?dev|flux[\s_\-]?pro",
     "FLUX (Black Forest Labs)"),
    (r"ideogram", "Ideogram"),
    (r"imagen|nano[\s_\-]?banana|google[\s_\-]?ai|gemini",
     "Google Imagen / Gemini"),
    (r"grok|xai[\s_\-]?image|aurora", "xAI Grok"),
    (r"bing[\s_\-]?image[\s_\-]?creator|designer\.microsoft",
     "Bing Image Creator"),
    (r"runway|pika[\s_\-]?labs|luma[\s_\-]?ai|kling|hunyuan|seedream|qwen[\s_\-]?image",
     "Uretici video/goruntu modeli"),
    (r"generative[\s_\-]?ai|gen[\s_\-]?ai|ai[\s_\-]?generated|aigc|synthetic[\s_\-]?media",
     "Genel AI uretim etiketi"),
]

## Duzenleme yazilimi imzalari (AI degil, "editlenmis" sinyali)
EDITOR_PATTERNS = [
    (r"adobe[\s_\-]?photoshop|photoshop", "Adobe Photoshop"),
    (r"adobe[\s_\-]?lightroom|lightroom", "Adobe Lightroom"),
    (r"gimp", "GIMP"),
    (r"paint\.net|paintshop", "Paint.NET / PaintShop"),
    (r"affinity[\s_\-]?photo", "Affinity Photo"),
    (r"canva", "Canva"),
    (r"snapseed|picsart|facetune|lensa", "Mobil retus uygulamasi"),
    (r"pixelmator|luminar|capture[\s_\-]?one", "Profesyonel retus yazilimi"),
]

## IPTC/C2PA "dijital kaynak tipi" degerleri
## IPTC DigitalSourceType degerleri -> (tur, etiket, AI uretimi mi)
##
## DIKKAT - sira onemli: "compositewithtrainedalgorithmicmedia" metninin
## ICINDE "trainedalgorithmicmedia" ve "algorithmicmedia" de gecer. Kisa olani
## once ararsak kompozit bir goruntuyu "tamamen AI uretimi" sayariz. Bu yuzden
## liste en UZUN/en OZEL degerden kisaya dogru siralidir ve ilk eslesme kazanir.
##
## Anlamlari (IPTC NewsCodes):
##   trainedAlgorithmicMedia            : tamamen AI ile uretilmis
##   compositeWithTrainedAlgorithmicMedia: gercek icerik + AI uretimi parcalar
##   algorithmicallyEnhanced            : gercek icerik, AI ile iyilestirilmis
##                                        (cozunurluk artirma, gurultu temizleme)
##   algorithmicMedia                   : algoritmayla uretilmis ama AI degil (3B render)
##   digitalCapture                     : kamerayla cekilmis
IPTC_SOURCE_TYPES = [
    ("compositewithtrainedalgorithmicmedia", "composite",
     "Kismen AI uretimi (kompozit)", True),
    ("trainedalgorithmicmedia", "generated",
     "Tamamen AI uretimi", True),
    ("algorithmicallyenhanced", "enhanced",
     "AI ile iyilestirilmis (uretilmemis)", False),
    ("algorithmicmedia", "algorithmic",
     "Algoritmayla uretilmis (AI degil)", False),
    ("digitalcapture", "capture",
     "Kamerayla cekilmis", False),
]

## Geriye donuk uyumluluk: yalnizca "AI ici var" anlamina gelen degerler.
IPTC_AI_SOURCE_TYPES = [k for k, _, _, _ in IPTC_SOURCE_TYPES if _ in ("composite", "generated")]


def match_source_type(text: str):
    """
    @brief Metinde gecen IPTC DigitalSourceType degerini bulur.
    @param text Kucuk harfe cevrilmis metin (XMP, C2PA baytlari vb.).
    @return     `(anahtar, tur, etiket, ai_uretimi_mi)` veya None.
    @note En ozel (en uzun) degerden basladigi icin kompozit/iyilestirilmis
          beyanlar "tamamen AI uretimi" ile karistirilmaz.
    """
    if not text:
        return None
    for key, kind, label, is_generated in IPTC_SOURCE_TYPES:
        if key in text:
            return key, kind, label, is_generated
    return None

## Stable Diffusion / ComfyUI tarzi uretim parametresi barindiran anahtarlar
SD_PARAM_KEYS = {
    "parameters", "prompt", "negative_prompt", "workflow", "sd-metadata",
    "dream", "comment", "generation_data", "aiparams", "invokeai_metadata",
}

## Gercek fotograf gostergesi EXIF alanlari
CAMERA_FIELDS = [
    "Make", "Model", "LensModel", "LensMake", "ISOSpeedRatings",
    "PhotographicSensitivity", "FNumber", "ExposureTime", "FocalLength",
    "DateTimeOriginal", "ShutterSpeedValue", "ApertureValue", "Flash",
    "WhiteBalance", "MeteringMode", "ExifImageWidth",
]

## Raporda gosterilecek EXIF alanlari
INTERESTING_EXIF = [
    "Software", "Make", "Model", "ImageDescription", "UserComment",
    "Artist", "Copyright", "XPComment", "XPTitle", "XPSubject",
    "DateTime", "DateTimeOriginal", "HostComputer", "ProcessingSoftware",
]

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------

def _clean(value, max_len: int = 180) -> str:
    """
    @brief Metadata degerini guvenli, kisa ve okunur metne cevirir.
    @param value   Ham metadata degeri (str / bytes / sayi).
    @param max_len Maksimum karakter sayisi.
    @return Temizlenmis metin.
    @note Metadata ucuncu taraf kontrolunde oldugu icin kontrol karakterleri
          temizlenir ve uzunluk sinirlanir.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)
    text = str(value)
    text = text.replace("\x00", "")
    text = _CTRL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        # Kirpma isareti dahil toplam uzunluk max_len'i asmamali.
        text = text[: max(0, max_len - 3)] + "..."
    return text


def _match_patterns(text: str, patterns: list) -> list:
    """
    @brief Metin icinde imza listesini arar.
    @param text     Aranacak metin (kucuk harfe cevrilir).
    @param patterns (regex, etiket) ciftleri.
    @return Bulunan etiketlerin tekrarsiz listesi.
    """
    if not text:
        return []
    low = text.lower()
    found = []
    for pattern, label in patterns:
        if re.search(pattern, low) and label not in found:
            found.append(label)
    return found


# ---------------------------------------------------------------------------
# MetadataInspector
# ---------------------------------------------------------------------------

class MetadataInspector:
    """
    @class MetadataInspector
    @brief Dosya yapisindaki AI etiketlerini inceleyen tek sorumluluklu sinif.
    @details Durumsuzdur (stateless); inspect() her cagri icin bagimsiz calisir.
             ForensicsEngine tarafindan tek bir ornek olarak tutulur.
    """

    VERSION = "1.0.0"

    # ------------------------------------------------------------------
    # Ana giris noktasi
    # ------------------------------------------------------------------

    def inspect(self, img_bytes: bytes) -> dict:
        """
        @brief Goruntunun metadata yapisini AI etiketleri icin inceler.
        @param img_bytes Dosyanin ham baytlari.
        @return Rapor sozlugu: verdict, confidence, findings, camera_evidence, ...
        """
        report = {
            "available":         True,
            "inspector_version": self.VERSION,
            "file_format":       "?",
            "file_size_kb":      round(len(img_bytes) / 1024.0, 1),
            "ai_tag_found":      False,
            "verdict":           "no_trace",
            "verdict_label":     "",
            "confidence":        0.0,
            "summary":           "",
            "findings":          [],
            "generators":        [],
            "editors":           [],
            "camera_evidence":   {"present": False, "count": 0, "fields": {}},
            "metadata_stripped": False,
            "c2pa":              {"present": False, "ai_assertion": False,
                                  "markers": [], "source_type": None,
                                  "source_kind": None, "source_label": None,
                                  "actions": [], "software": []},
            "blocks":            {"exif": False, "xmp": False, "png_text": False,
                                  "iptc": False, "comment": False},
        }

        try:
            pil = Image.open(BytesIO(img_bytes))
            report["file_format"] = (getattr(pil, "format", "") or "?").upper()
        except Exception as e:
            logger.warning("Metadata: goruntu acilamadi: %s", e)
            report["available"]     = False
            report["verdict"]       = "error"
            report["verdict_label"] = "Metadata okunamadi"
            report["summary"]       = "Dosya yapisi cozumlenemedi."
            return report

        # --- Blok bazli toplama ---
        exif_fields  = self._read_exif(pil, report)
        png_texts    = self._read_png_text(pil, report)
        xmp_text     = self._read_xmp(img_bytes, pil, report)
        comment_text = self._read_comment(pil, report)
        self._read_c2pa(img_bytes, report)

        # --- Imza taramasi ---
        self._scan_exif(exif_fields, report)
        self._scan_png_text(png_texts, report)
        self._scan_xmp(xmp_text, report)
        self._scan_comment(comment_text, report)
        self._collect_camera_evidence(exif_fields, report)

        # --- Karar ---
        self._finalize(report)
        return report

    # ------------------------------------------------------------------
    # Blok okuyucular
    # ------------------------------------------------------------------

    def _read_exif(self, pil: Image.Image, report: dict) -> dict:
        """
        @brief EXIF (IFD0 + Exif IFD + GPS) alanlarini isimli sozluge cevirir.
        @param pil    Acilmis PIL goruntusu.
        @param report Guncellenecek rapor.
        @return Alan adi -> deger sozlugu.
        """
        fields = {}
        try:
            exif = pil.getexif()
            if not exif:
                return fields

            for tag_id, value in exif.items():
                name = ExifTags.TAGS.get(tag_id, "Tag%s" % tag_id)
                fields[name] = value

            for ifd_name, ifd_id in (("exif", 0x8769), ("gps", 0x8825)):
                try:
                    sub = exif.get_ifd(ifd_id)
                except Exception:
                    sub = None
                if not sub:
                    continue
                table  = ExifTags.GPSTAGS if ifd_name == "gps" else ExifTags.TAGS
                prefix = "GPS" if ifd_name == "gps" else ""
                for tag_id, value in sub.items():
                    name = table.get(tag_id, "Tag%s" % tag_id)
                    fields.setdefault(prefix + name, value)

            if fields:
                report["blocks"]["exif"] = True
        except Exception as e:
            logger.debug("EXIF okuma hatasi: %s", e)
        return fields

    def _read_png_text(self, pil: Image.Image, report: dict) -> dict:
        """
        @brief PNG tEXt / iTXt chunklarini okur (Stable Diffusion parametreleri burada durur).
        @param pil    Acilmis PIL goruntusu.
        @param report Guncellenecek rapor.
        @return Anahtar -> deger sozlugu.
        """
        texts = {}
        try:
            raw = dict(getattr(pil, "text", {}) or {})
            for key, value in (pil.info or {}).items():
                if isinstance(value, str) and key not in ("transparency",):
                    raw.setdefault(key, value)
            for key, value in raw.items():
                if key.lower() in ("xml:com.adobe.xmp",):   # XMP ayri islenir
                    continue
                texts[key] = value
            if texts:
                report["blocks"]["png_text"] = True
        except Exception as e:
            logger.debug("PNG text okuma hatasi: %s", e)
        return texts

    def _read_xmp(self, img_bytes: bytes, pil: Image.Image, report: dict) -> str:
        """
        @brief XMP paketini (Adobe/IPTC metadata) cikarir.
        @param img_bytes Ham dosya baytlari.
        @param pil       Acilmis PIL goruntusu.
        @param report    Guncellenecek rapor.
        @return XMP XML metni ya da bos metin.
        """
        xmp = ""
        try:
            info     = pil.info or {}
            info_xmp = info.get("XML:com.adobe.xmp") or info.get("xmp")
            if isinstance(info_xmp, bytes):
                info_xmp = info_xmp.decode("utf-8", errors="ignore")
            if isinstance(info_xmp, str) and info_xmp.strip():
                xmp = info_xmp
            else:
                start = img_bytes.find(b"<x:xmpmeta")
                if start == -1:
                    start = img_bytes.find(b"<?xpacket")
                if start != -1:
                    end = img_bytes.find(b"</x:xmpmeta>", start)
                    if end == -1:
                        end = img_bytes.find(b"<?xpacket end", start)
                    if end != -1:
                        xmp = img_bytes[start:end + 12].decode("utf-8", errors="ignore")
            if xmp:
                report["blocks"]["xmp"] = True
                if re.search(r"iptc4xmp|photoshop:|dc:creator", xmp, re.I):
                    report["blocks"]["iptc"] = True
        except Exception as e:
            logger.debug("XMP okuma hatasi: %s", e)
        return xmp

    def _read_comment(self, pil: Image.Image, report: dict) -> str:
        """
        @brief GIF/JPEG/WebP comment uzantisini okur.
        @param pil    Acilmis PIL goruntusu.
        @param report Guncellenecek rapor.
        @return Yorum metni.
        """
        try:
            comment = (pil.info or {}).get("comment", "")
            if isinstance(comment, bytes):
                comment = comment.decode("utf-8", errors="ignore")
            comment = comment or ""
            if comment.strip():
                report["blocks"]["comment"] = True
            return comment
        except Exception:
            return ""

    def _read_c2pa(self, img_bytes: bytes, report: dict) -> None:
        """
        @brief C2PA / Content Credentials (JUMBF) imzasini arar.
        @param img_bytes Ham dosya baytlari.
        @param report    Guncellenecek rapor (c2pa alani doldurulur).
        @details C2PA manifesti JUMBF kutusunda tasinir. Burada kriptografik imza
                 dogrulamasi YAPILMAZ; yalnizca manifest varligi ve AI beyani aranir.
        """
        try:
            low     = img_bytes.lower()
            markers = []
            for marker in (b"c2pa", b"jumbf", b"jumd", b"contentcredentials",
                           b"content credentials"):
                if marker in low:
                    markers.append(marker.decode("ascii", errors="ignore"))

            if not markers:
                return

            report["c2pa"]["present"] = True
            report["c2pa"]["markers"] = markers

            # Hangi DigitalSourceType beyan edilmis? (kompozit/iyilestirilmis
            # ile tamamen AI uretimi ayri seylerdir)
            text = low.decode("latin-1", errors="ignore")
            eslesme = match_source_type(text)

            ai_assertion = bool(eslesme and eslesme[3])
            report["c2pa"]["ai_assertion"] = ai_assertion
            report["c2pa"]["source_type"]  = eslesme[0] if eslesme else None
            report["c2pa"]["source_kind"]  = eslesme[1] if eslesme else None
            report["c2pa"]["source_label"] = eslesme[2] if eslesme else None

            # Manifestteki duzenleme eylemi ve kullanilan yazilim: asil hikaye
            # cogu zaman burada ("Adobe Photoshop ile duzenlendi, Firefly kullanildi")
            self._collect_c2pa_actions(img_bytes, report)

            if eslesme:
                kind = eslesme[1]
                severity = "high" if eslesme[3] else "medium"
                if kind == "generated":
                    note = ("Content Credentials manifesti goruntunun tamamen AI ile "
                            "uretildigini beyan ediyor.")
                elif kind == "composite":
                    note = ("Manifest, gercek icerigin AI uretimi parcalarla "
                            "birlestirildigini beyan ediyor - goruntunun TAMAMI "
                            "AI uretimi degil.")
                elif kind == "enhanced":
                    note = ("Manifest, goruntunun AI ile IYILESTIRILDIGINI beyan "
                            "ediyor (cozunurluk/gurultu). Icerik AI uretimi degil.")
                    severity = "medium"
                elif kind == "algorithmic":
                    note = ("Manifest algoritmayla uretim beyan ediyor (3B render "
                            "gibi) - yapay zeka degil.")
                    severity = "info"
                else:
                    note = "Manifest goruntunun kamerayla cekildigini beyan ediyor."
                    severity = "good"

                report["findings"].append({
                    "category": "c2pa",
                    "severity": severity,
                    "key":      "C2PA - digitalSourceType",
                    "value":    eslesme[2],
                    "note":     note,
                })
            else:
                report["findings"].append({
                    "category": "c2pa",
                    "severity": "info",
                    "key":      "C2PA manifesti",
                    "value":    "mevcut (AI beyani yok)",
                    "note":     "Icerik kimlik bilgisi tasiyor; AI beyani bulunamadi. "
                                "Imza kriptografik olarak dogrulanmadi.",
                })
        except Exception as e:
            logger.debug("C2PA tarama hatasi: %s", e)

    # ------------------------------------------------------------------
    # Imza tarayicilar
    # ------------------------------------------------------------------

    def _scan_exif(self, fields: dict, report: dict) -> None:
        """
        @brief EXIF alanlarinda AI / duzenleme yazilimi imzalarini arar.
        @param fields EXIF alan sozlugu.
        @param report Guncellenecek rapor.
        """
        for name in INTERESTING_EXIF:
            if name not in fields:
                continue
            value = _clean(fields[name])
            if not value:
                continue

            generators = _match_patterns(value, AI_GENERATOR_PATTERNS)
            editors    = _match_patterns(value, EDITOR_PATTERNS)

            if generators:
                self._add_generators(report, generators, "EXIF - " + name, value,
                                     "Dosya yapisinda AI uretici imzasi bulundu.")
            elif editors:
                self._add_editors(report, editors, "EXIF - " + name, value)
            elif name in ("Software", "ProcessingSoftware", "HostComputer"):
                report["findings"].append({
                    "category": "exif",
                    "severity": "info",
                    "key":      "EXIF - " + name,
                    "value":    value,
                    "note":     "Bilinen AI/retus imzasi degil.",
                })

    def _scan_png_text(self, texts: dict, report: dict) -> None:
        """
        @brief PNG metin chunklarinda uretim parametrelerini ve imzalari arar.
        @param texts  PNG tEXt/iTXt sozlugu.
        @param report Guncellenecek rapor.
        """
        for key, raw_value in texts.items():
            value      = _clean(raw_value, max_len=240)
            key_low    = key.lower()
            generators = _match_patterns(key + " " + value, AI_GENERATOR_PATTERNS)
            editors    = _match_patterns(key + " " + value, EDITOR_PATTERNS)

            if generators:
                self._add_generators(report, generators, "PNG - " + key, value,
                                     "PNG metin blogunda AI uretici imzasi bulundu.")
                continue

            if key_low in SD_PARAM_KEYS and self._looks_like_sd_params(value):
                report["ai_tag_found"] = True
                report["findings"].append({
                    "category": "png_text",
                    "severity": "high",
                    "key":      "PNG - " + key,
                    "value":    value,
                    "note":     "Difuzyon modeli uretim parametreleri "
                                "(prompt/seed/sampler/steps) gomulu.",
                })
                continue

            if editors:
                self._add_editors(report, editors, "PNG - " + key, value)
            elif value:
                report["findings"].append({
                    "category": "png_text",
                    "severity": "info",
                    "key":      "PNG - " + key,
                    "value":    value,
                    "note":     "Metin blogu mevcut, AI imzasi tespit edilmedi.",
                })

    @staticmethod
    def _collect_c2pa_actions(img_bytes: bytes, report: dict) -> None:
        """
        @brief C2PA manifestindeki eylemleri ve kullanilan yazilimi cikarir.

        @param img_bytes Ham dosya baytlari.
        @param report    Guncellenecek rapor (`c2pa.actions` ve `c2pa.software`).

        @details Manifest CBOR kodludur; burada tam cozumleme yapilmaz, yalnizca
                 `c2pa.<eylem>` adlari ve `softwareAgent` / `claim_generator`
                 degerleri metin olarak taranir. Bu, "AI ile uretildi" ile
                 "AI araciyla duzenlendi" ayrimini kullaniciya gosterebilmek
                 icin yeterlidir.
        @note Kriptografik imza dogrulanmaz.
        """
        import re as _re

        try:
            text = img_bytes.decode("latin-1", errors="ignore")

            # c2pa.created / c2pa.edited / c2pa.placed ...
            eylemler = []
            for m in _re.finditer(r"c2pa\.(created|edited|placed|filtered|"
                                  r"color_adjustments|drawing|resized|cropped|"
                                  r"converted|opened|published)", text):
                ad = m.group(1)
                if ad not in eylemler:
                    eylemler.append(ad)
            report["c2pa"]["actions"] = eylemler[:6]

            # softwareAgent ve claim_generator degerleri
            # Kullanilan yazilimi cikar. Iki kaynak var:
            #   1. claim_generator_info CBOR haritasi: "dname" + uzunluk bayti + ad
            #      (CBOR'da 0x60+n kisa metin demektir, uzunluk bayttan okunur)
            #   2. actions icindeki softwareAgent degeri
            # Manifest ic etiketlerini ("jumbf manifest", "c2pa.*") disarida birakiyoruz.
            COP = ("jumbf", "manifest", "dname", "version", "c2pa", "urn:")

            def temiz_mi(deger: str) -> bool:
                dusuk = deger.lower()
                if len(deger) < 4 or len(deger) > 40:
                    return False
                return not any(c in dusuk for c in COP)

            yazilimlar = []

            for m in _re.finditer(r"dname(.)", text):
                uzunluk = ord(m.group(1)) - 0x60
                if 2 <= uzunluk <= 30:
                    ad = _clean(text[m.end():m.end() + uzunluk], 40)
                    if temiz_mi(ad) and ad not in yazilimlar:
                        yazilimlar.append(ad)

            for m in _re.finditer(r"softwareAgent.(.{3,40}?)(?:qdigital|odigital|[^ -~])",
                                  text):
                ad = _clean(m.group(1), 40).split("/")[0].strip(" .-_")
                if temiz_mi(ad) and ad not in yazilimlar:
                    yazilimlar.append(ad)

            report["c2pa"]["software"] = yazilimlar[:4]

            if yazilimlar:
                report["findings"].append({
                    "category": "c2pa",
                    "severity": "info",
                    "key":      "C2PA - kullanilan yazilim",
                    "value":    ", ".join(yazilimlar[:3]),
                    "note":     ("Manifestte kayitli duzenleme yazilimi" +
                                 (" ve eylemler: " + ", ".join(eylemler[:4])
                                  if eylemler else "")),
                })
        except Exception as exc:
            logger.debug("C2PA eylem taramasi basarisiz: %s", exc)

    def _scan_xmp(self, xmp: str, report: dict) -> None:
        """
        @brief XMP paketinde CreatorTool, DigitalSourceType ve GenAI alanlarini arar.
        @param xmp    XMP XML metni.
        @param report Guncellenecek rapor.
        """
        if not xmp:
            return

        low = xmp.lower()

        for source_type in IPTC_AI_SOURCE_TYPES:
            if source_type in low:
                report["ai_tag_found"] = True
                report["findings"].append({
                    "category": "xmp",
                    "severity": "high",
                    "key":      "XMP - Iptc4xmpExt:DigitalSourceType",
                    "value":    source_type,
                    "note":     "IPTC standardina gore icerik egitilmis bir algoritma "
                                "tarafindan uretilmis.",
                })
                break

        creator = (re.search(r"<xmp:CreatorTool>(.*?)</xmp:CreatorTool>", xmp, re.I | re.S)
                   or re.search(r'xmp:CreatorTool="(.*?)"', xmp, re.I | re.S))
        if creator:
            value      = _clean(creator.group(1))
            generators = _match_patterns(value, AI_GENERATOR_PATTERNS)
            editors    = _match_patterns(value, EDITOR_PATTERNS)
            if generators:
                self._add_generators(report, generators, "XMP - xmp:CreatorTool", value,
                                     "XMP paketinde AI uretici imzasi bulundu.")
            elif editors:
                self._add_editors(report, editors, "XMP - xmp:CreatorTool", value)
            elif value:
                report["findings"].append({
                    "category": "xmp",
                    "severity": "info",
                    "key":      "XMP - xmp:CreatorTool",
                    "value":    value,
                    "note":     "Uretici arac bilgisi; bilinen AI imzasi degil.",
                })

        generators = _match_patterns(low, AI_GENERATOR_PATTERNS)
        if generators:
            self._add_generators(report, generators, "XMP - paket govdesi",
                                 ", ".join(generators),
                                 "XMP paketinin govdesinde AI uretici imzasi gecti.")

    def _scan_comment(self, comment: str, report: dict) -> None:
        """
        @brief GIF/JPEG yorum alanini tarar.
        @param comment Yorum metni.
        @param report  Guncellenecek rapor.
        """
        value = _clean(comment, max_len=240)
        if not value:
            return
        generators = _match_patterns(value, AI_GENERATOR_PATTERNS)
        if generators:
            self._add_generators(report, generators, "Dosya yorumu (comment)", value,
                                 "Yorum alaninda AI uretici imzasi bulundu.")
        else:
            report["findings"].append({
                "category": "comment",
                "severity": "info",
                "key":      "Dosya yorumu (comment)",
                "value":    value,
                "note":     "Yorum alani mevcut, AI imzasi tespit edilmedi.",
            })

    def _collect_camera_evidence(self, fields: dict, report: dict) -> None:
        """
        @brief Gercek fotograf gostergesi olan kamera alanlarini toplar.
        @param fields EXIF alan sozlugu.
        @param report Guncellenecek rapor.
        """
        present = {}
        for name in CAMERA_FIELDS:
            if name in fields:
                value = _clean(fields[name], max_len=60)
                if value:
                    present[name] = value

        gps_keys = [k for k in fields if k.startswith("GPS")]
        if gps_keys:
            present["GPS"] = "%d alan" % len(gps_keys)

        report["camera_evidence"] = {
            "present": len(present) >= 3,
            "count":   len(present),
            "fields":  present,
        }

        if len(present) >= 3:
            report["findings"].append({
                "category": "camera",
                "severity": "good",
                "key":      "Kamera izi (EXIF)",
                "value":    ", ".join(list(present.keys())[:6]),
                "note":     "%d kamera alani mevcut - gercek cekim lehine sinyal "
                            "(EXIF taklit edilebilir, kesin kanit degildir)." % len(present),
            })

    # ------------------------------------------------------------------
    # Yardimci kayit metotlari
    # ------------------------------------------------------------------

    def _add_generators(self, report: dict, generators: list, key: str,
                        value: str, note: str) -> None:
        """
        @brief AI uretici bulgusunu rapora ekler ve bayragi kaldirir.
        @param report     Guncellenecek rapor.
        @param generators Bulunan uretici etiketleri.
        @param key        Bulgunun kaynagi (alan adi).
        @param value      Temizlenmis alan degeri.
        @param note       Aciklama metni.
        """
        report["ai_tag_found"] = True
        for g in generators:
            if g not in report["generators"]:
                report["generators"].append(g)
        report["findings"].append({
            "category": "generator",
            "severity": "high",
            "key":      key,
            "value":    value,
            "note":     "%s (%s)" % (note, ", ".join(generators)),
        })

    def _add_editors(self, report: dict, editors: list, key: str, value: str) -> None:
        """
        @brief Duzenleme yazilimi bulgusunu rapora ekler.
        @param report  Guncellenecek rapor.
        @param editors Bulunan yazilim etiketleri.
        @param key     Bulgunun kaynagi (alan adi).
        @param value   Temizlenmis alan degeri.
        """
        for e in editors:
            if e not in report["editors"]:
                report["editors"].append(e)
        report["findings"].append({
            "category": "editor",
            "severity": "medium",
            "key":      key,
            "value":    value,
            "note":     "Duzenleme yazilimi izi (%s) - goruntu islenmis olabilir, "
                        "AI uretimi anlamina gelmez." % ", ".join(editors),
        })

    @staticmethod
    def _looks_like_sd_params(value: str) -> bool:
        """
        @brief Metnin difuzyon modeli parametre blogu olup olmadigini sezer.
        @param value Chunk degeri.
        @return Parametre blogu gorunumundeyse True.
        """
        low = value.lower()
        hits = sum(1 for token in ("steps:", "sampler", "cfg scale", "cfg_scale",
                                   "seed:", "model hash", "denoising",
                                   "negative prompt", "scheduler", "class_type")
                   if token in low)
        return hits >= 2

    # ------------------------------------------------------------------
    # Karar
    # ------------------------------------------------------------------

    @staticmethod
    def _c2pa_summary(report: dict, taban: str) -> str:
        """
        @brief C2PA ozetine kullanilan yazilim ve eylem bilgisini ekler.
        @param report Rapor sozlugu.
        @param taban  Temel aciklama cumlesi.
        @return       Yazilim/eylem bilgisi eklenmis ozet.
        """
        parcalar = [taban]
        yazilim = report["c2pa"].get("software") or []
        eylem   = report["c2pa"].get("actions") or []
        if yazilim:
            parcalar.append("Kayitli yazilim: %s." % ", ".join(yazilim[:3]))
        if eylem:
            parcalar.append("Eylemler: %s." % ", ".join(eylem[:4]))
        return " ".join(parcalar)

    def _finalize(self, report: dict) -> None:
        """
        @brief Bulgulari tek bir karara ve guven skoruna indirger.
        @param report Tamamlanacak rapor.
        @details Skor yalnizca METADATA kaynakli guveni ifade eder; piksel
                 analizinden (CLIP / istatistiksel sinyaller) bagimsizdir.
        """
        has_block = any(report["blocks"].values()) or report["c2pa"]["present"]
        report["metadata_stripped"] = not has_block

        severities = set(f["severity"] for f in report["findings"])

        if report["generators"]:
            report["verdict"]       = "ai_signature"
            report["confidence"]    = 97.0
            report["verdict_label"] = ("AI etiketi bulundu - dosya yapisi uretici "
                                       "imzasi tasiyor")
            report["summary"]       = ("Metadata icinde AI uretici imzasi tespit "
                                       "edildi: %s." % ", ".join(report["generators"]))
        elif report["c2pa"]["source_kind"] == "generated":
            report["verdict"]       = "c2pa_ai"
            report["confidence"]    = 95.0
            report["ai_tag_found"]  = True
            report["verdict_label"] = "C2PA manifesti AI uretimi beyan ediyor"
            report["summary"]       = ("Content Credentials manifesti "
                                       "digitalSourceType = trainedAlgorithmicMedia "
                                       "beyani iceriyor: goruntu tamamen AI uretimi.")
        elif report["c2pa"]["source_kind"] == "composite":
            # Gercek icerik + AI uretimi parcalar. "Tamamen AI uretimi" DEGIL.
            report["verdict"]       = "c2pa_ai_composite"
            report["confidence"]    = 70.0
            report["ai_tag_found"]  = True
            report["verdict_label"] = "C2PA: kismen AI uretimi (kompozit)"
            report["summary"]       = self._c2pa_summary(
                report,
                "Manifest, gercek icerigin AI uretimi parcalarla birlestirildigini "
                "beyan ediyor (compositeWithTrainedAlgorithmicMedia). Goruntunun "
                "tamami AI uretimi degildir.")
        elif report["c2pa"]["source_kind"] == "enhanced":
            # Yalnizca iyilestirme (upscale/denoise). Icerik gercek.
            report["verdict"]       = "c2pa_ai_enhanced"
            report["confidence"]    = 0.0
            report["verdict_label"] = "C2PA: AI ile iyilestirilmis (uretilmemis)"
            report["summary"]       = self._c2pa_summary(
                report,
                "Manifest, goruntunun AI ile iyilestirildigini beyan ediyor "
                "(algorithmicallyEnhanced) - cozunurluk artirma veya gurultu "
                "temizleme gibi. Icerik AI uretimi degildir.")
        elif report["ai_tag_found"]:
            report["verdict"]       = "ai_suspect"
            report["confidence"]    = 85.0
            report["verdict_label"] = "AI uretim parametreleri gomulu"
            report["summary"]       = ("Dosyada difuzyon modeli uretim parametreleri "
                                       "(prompt/seed/sampler) bulundu.")
        elif report["metadata_stripped"]:
            report["verdict"]       = "stripped"
            report["confidence"]    = 0.0
            report["verdict_label"] = "Metadata yok / temizlenmis"
            report["summary"]       = ("Dosyada hic metadata blogu yok. Bu, AI uretimi "
                                       "oldugunu da gercek oldugunu da gostermez - "
                                       "sosyal medya yuklemeleri metadatayi siler.")
            report["findings"].append({
                "category": "stripped",
                "severity": "info",
                "key":      "Metadata blogu",
                "value":    "bulunamadi",
                "note":     "EXIF / XMP / PNG-text / C2PA bloklarinin hicbiri yok.",
            })
        elif report["camera_evidence"]["present"]:
            report["verdict"]       = "camera_evidence"
            report["confidence"]    = 0.0
            report["verdict_label"] = "AI etiketi yok - kamera izi mevcut"
            report["summary"]       = ("AI imzasi bulunamadi; %d kamera EXIF alani "
                                       "gercek cekim lehine sinyal veriyor."
                                       % report["camera_evidence"]["count"])
        else:
            report["verdict"]       = "no_trace"
            report["confidence"]    = 0.0
            report["verdict_label"] = "AI etiketi bulunamadi"
            report["summary"]       = ("Metadata mevcut ancak AI uretici imzasi yok. "
                                       "Etiketler silinebildigi icin bu tek basina "
                                       "goruntunun gercek oldugunu kanitlamaz.")

        if "medium" in severities and report["verdict"] in ("no_trace", "camera_evidence"):
            report["summary"] += (" Not: duzenleme yazilimi izi var (%s)."
                                  % ", ".join(report["editors"]))

        order = {"high": 0, "medium": 1, "good": 2, "info": 3}
        report["findings"].sort(key=lambda f: order.get(f["severity"], 9))
