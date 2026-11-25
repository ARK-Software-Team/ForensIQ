"""
@file tests/make_fixtures.py
@brief Metadata (US-4) testleri icin sentetik ornek dosyalar uretir.
@details
  images/testing altindaki gercek ornek goruntulerin metadata bloklari
  (indirme / ekran goruntusu alma sirasinda) temizlenmis durumdadir. Bu yuzden
  AI etiketi tespitini dogrulamak icin etiketleri BILINEN kontrollu dosyalar
  uretilir:

    1. exif_midjourney.jpg  : EXIF Software = "Midjourney v6"
    2. png_sd_params.png    : PNG tEXt "parameters" = SD uretim parametreleri
    3. xmp_c2pa_ai.jpg      : XMP + sentetik JUMBF/C2PA blogu (trainedAlgorithmicMedia)
    4. camera_photo.jpg     : Kamera EXIF alanlari (gercek cekim senaryosu)
    5. photoshop_edited.jpg : EXIF Software = "Adobe Photoshop 25.0"
    6. gif_comment_ai.gif   : GIF comment uzantisinda uretici imzasi
    7. clean_no_meta.jpg    : Hic metadata yok (temizlenmis dosya senaryosu)

@note Bu dosyalar YALNIZCA test amaclidir; gercek bir C2PA manifesti degildir
      (kriptografik imza icermez). Uretilen goruntuler duz renk/gurultu desenidir.
@warning Uretilen dosyalar gercek bir urunun cikti dosyasi gibi sunulmamalidir.

Kullanim:
    python tests/make_fixtures.py
"""

import os
import struct
from io import BytesIO

import numpy as np
from PIL import Image, PngImagePlugin

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# EXIF etiket kimlikleri
TAG_MAKE             = 0x010F
TAG_MODEL            = 0x0110
TAG_SOFTWARE         = 0x0131
TAG_DATETIME         = 0x0132
TAG_IMAGE_DESC       = 0x010E
TAG_ARTIST           = 0x013B
TAG_EXIF_IFD         = 0x8769
TAG_EXPOSURE_TIME    = 0x829A
TAG_FNUMBER          = 0x829D
TAG_ISO              = 0x8827
TAG_DATETIME_ORIG    = 0x9003
TAG_FOCAL_LENGTH     = 0x920A
TAG_LENS_MODEL       = 0xA434

XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def _base_image(seed: int = 7, size=(320, 240)) -> Image.Image:
    """
    @brief Testler icin deterministik, anlamsiz bir goruntu uretir.
    @param seed Rastgelelik tohumu (tekrarlanabilirlik icin).
    @param size (genislik, yukseklik).
    @return RGB PIL goruntusu.
    """
    rng   = np.random.default_rng(seed)
    noise = rng.integers(60, 190, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(noise, "RGB")


def _insert_jpeg_segment(jpeg_bytes: bytes, marker: bytes, payload: bytes) -> bytes:
    """
    @brief JPEG akisina SOI hemen sonrasina yeni bir APP segmenti ekler.
    @param jpeg_bytes Kaynak JPEG baytlari.
    @param marker     Iki baytlik segment isaretcisi (ornek: b"\\xff\\xe1").
    @param payload    Segment govdesi (uzunluk alani haric).
    @return Yeni JPEG baytlari.
    @details Segment yapisi: marker + uzunluk(2 bayt, big-endian, kendisi dahil) + govde.
    """
    if not jpeg_bytes.startswith(b"\xff\xd8"):
        raise ValueError("Gecerli bir JPEG baslangici (SOI) bulunamadi")
    length  = struct.pack(">H", len(payload) + 2)
    segment = marker + length + payload
    return jpeg_bytes[:2] + segment + jpeg_bytes[2:]


def make_exif_midjourney(path: str) -> None:
    """@brief EXIF Software alaninda AI uretici imzasi tasiyan JPEG uretir."""
    img  = _base_image(1)
    exif = Image.Exif()
    exif[TAG_SOFTWARE]   = "Midjourney v6"
    exif[TAG_IMAGE_DESC] = "an astronaut riding a horse, cinematic lighting"
    exif[TAG_ARTIST]     = "AI generated"
    img.save(path, "JPEG", quality=92, exif=exif)


def make_png_sd_params(path: str) -> None:
    """@brief Stable Diffusion uretim parametrelerini tEXt chunkta tasiyan PNG uretir."""
    img  = _base_image(2)
    meta = PngImagePlugin.PngInfo()
    meta.add_text(
        "parameters",
        "a cat wearing sunglasses, highly detailed\n"
        "Negative prompt: blurry, low quality\n"
        "Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7.5, "
        "Seed: 1234567890, Size: 512x512, Model hash: 6ce0161689",
    )
    img.save(path, "PNG", pnginfo=meta)


def make_xmp_c2pa_ai(path: str) -> None:
    """
    @brief XMP + sentetik C2PA/JUMBF blogu tasiyan JPEG uretir.
    @note Gercek bir C2PA manifesti DEGILDIR; yalnizca tespit yolunu test eder.
    """
    buf = BytesIO()
    _base_image(3).save(buf, "JPEG", quality=92)
    data = buf.getvalue()

    xmp = (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
        'xmp:CreatorTool="Adobe Firefly" '
        'Iptc4xmpExt:DigitalSourceType='
        '"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
        "</rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    ).encode("utf-8")

    # Sentetik JUMBF kutusu (APP11): C2PA manifest varligini taklit eder
    jumbf = (b"JP\x00\x00\x00\x01" + b"jumb" + b"jumd" + b"c2pa" +
             b'{"claim_generator":"c2pa-test","assertions":'
             b'[{"label":"c2pa.actions","data":{"digitalSourceType":'
             b'"trainedAlgorithmicMedia"}}]}')

    data = _insert_jpeg_segment(data, b"\xff\xe1", XMP_HEADER + xmp)
    data = _insert_jpeg_segment(data, b"\xff\xeb", jumbf)

    with open(path, "wb") as fh:
        fh.write(data)


def make_camera_photo(path: str) -> None:
    """@brief Kamera EXIF alanlari tasiyan (gercek cekim senaryosu) JPEG uretir."""
    img  = _base_image(4)
    exif = Image.Exif()
    exif[TAG_MAKE]     = "Canon"
    exif[TAG_MODEL]    = "Canon EOS 90D"
    exif[TAG_DATETIME] = "2026:05:14 18:42:11"

    sub = exif.get_ifd(TAG_EXIF_IFD)
    sub[TAG_DATETIME_ORIG] = "2026:05:14 18:42:11"
    sub[TAG_ISO]           = 200
    sub[TAG_FNUMBER]       = (28, 10)
    sub[TAG_EXPOSURE_TIME] = (1, 250)
    sub[TAG_FOCAL_LENGTH]  = (50, 1)
    sub[TAG_LENS_MODEL]    = "EF50mm f/1.8 STM"

    img.save(path, "JPEG", quality=92, exif=exif)


def make_photoshop_edited(path: str) -> None:
    """@brief EXIF Software alaninda retus yazilimi imzasi tasiyan JPEG uretir."""
    img  = _base_image(5)
    exif = Image.Exif()
    exif[TAG_SOFTWARE] = "Adobe Photoshop 25.0 (Windows)"
    exif[TAG_MAKE]     = "NIKON CORPORATION"
    exif[TAG_MODEL]    = "NIKON D750"
    img.save(path, "JPEG", quality=92, exif=exif)


def make_gif_comment_ai(path: str) -> None:
    """@brief GIF comment uzantisinda AI uretici imzasi tasiyan dosya uretir."""
    img = _base_image(6, size=(160, 120))
    img.save(path, "GIF", comment=b"Created with Stable Diffusion WebUI (AUTOMATIC1111)")


def make_clean_no_meta(path: str) -> None:
    """@brief Hicbir metadata blogu icermeyen JPEG uretir."""
    _base_image(8).save(path, "JPEG", quality=92)


## Uretilecek dosyalar: (dosya adi, uretici fonksiyon, beklenen verdict)
FIXTURES = [
    ("exif_midjourney.jpg",  make_exif_midjourney,  "ai_signature"),
    ("png_sd_params.png",    make_png_sd_params,    "ai_suspect"),
    ("xmp_c2pa_ai.jpg",      make_xmp_c2pa_ai,      "ai_signature"),
    ("camera_photo.jpg",     make_camera_photo,     "camera_evidence"),
    ("photoshop_edited.jpg", make_photoshop_edited, "no_trace"),
    ("gif_comment_ai.gif",   make_gif_comment_ai,   "ai_signature"),
    ("clean_no_meta.jpg",    make_clean_no_meta,    "stripped"),
]


def build_all(target_dir: str = FIXTURE_DIR) -> list:
    """
    @brief Tum ornek dosyalari uretir.
    @param target_dir Cikti klasoru.
    @return (yol, beklenen_verdict) ciftlerinin listesi.
    """
    os.makedirs(target_dir, exist_ok=True)
    produced = []
    for name, builder, expected in FIXTURES:
        path = os.path.join(target_dir, name)
        builder(path)
        produced.append((path, expected))
    return produced


if __name__ == "__main__":
    for path, expected in build_all():
        size_kb = os.path.getsize(path) / 1024.0
        print("uretildi: %-24s %7.1f KB   beklenen verdict: %s"
              % (os.path.basename(path), size_kb, expected))
