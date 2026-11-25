"""
@file tools/download_models.py
@brief ForensIQ'nun kullandığı tüm model dosyalarını `models/` altına indirir.

@details
  Model dosyaları repoya konmaz (toplam ~905 MB); bir kez indirilir ve
  sonrasında sistem tamamen çevrimdışı çalışır. `clip_processor/` kalıbının
  aynısıdır.

  Bileşenler:
  | anahtar  | ne indirir                                  | boyut  | kullanan |
  |----------|---------------------------------------------|--------|----------|
  | `ai`     | AI görsel dedektörü (SwinV2 sınıflandırıcı) | ~745 MB | US-3 |
  | `person` | torchvision Faster R-CNN COCO ağırlıkları     | ~160 MB | US-5 |
  | `face`   | OpenCV YuNet yüz dedektörü (ONNX)             | ~230 KB | US-5, US-6 |
  | `age`    | Yas tahmini siniflandiricisi (ViT)            | ~327 MB | US-6 |

  Kullanım:
  @code
      python tools/download_models.py                      # eksik olan her şeyi indir
      python tools/download_models.py --list               # seçenekleri göster
      python tools/download_models.py --component person   # yalnızca kişi dedektörü
      python tools/download_models.py --component ai --variant swinv2
      python tools/download_models.py --force              # var olanları yeniden indir
  @endcode

@note İndirme başarısız olursa sistem çalışmaya devam eder; yalnızca ilgili
      özellik devre dışı kalır (Ek Madde-1: çökme riski sıfır).
@author ImageForensics Team
@version 2.0.0
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Windows konsolu varsayılan olarak cp1252 kullanır ve Türkçe karakterlerde
# UnicodeEncodeError verir. Çıktıyı UTF-8'e sabitliyoruz.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_BASE_DIR  = Path(__file__).resolve().parent.parent
MODELS_DIR = _BASE_DIR / "models"

# ---------------------------------------------------------------------------
# US-3: AI görsel dedektörü (HuggingFace sınıflandırıcı)
# ---------------------------------------------------------------------------

AI_TARGET_DIR = MODELS_DIR / "ai_image_detector"

## HuggingFace'ten indirilecek dosya kalıpları (eğitim artıkları hariç).
AI_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "pytorch_model.bin",
]

AI_VARIANTS = {
    "swin": {
        "repo_id":      "Organika/sdxl-detector",
        "display_name": "Swin AI Dedektörü",
        "architecture": "Swin Transformer (denetimli eğitilmiş)",
        "license":      "CC BY-NC 3.0 (ticari olmayan kullanım)",
        "size_mb":      331,
        "note":         "SDXL ile ince ayarlı. ⚠️ Kendi test setimizde tablolara "
                        "%99,9 AI verdi (8/14); 23.11.2026'da varsayılanlıktan çıkarıldı.",
        "default":      False,
    },
    "swinv2": {
        "repo_id":      "haywoodsloan/ai-image-detector-deploy",
        "display_name": "SwinV2 AI Dedektörü",
        "architecture": "Swin Transformer V2 (denetimli eğitilmiş)",
        "license":      "Apache-2.0",
        "size_mb":      745,
        "note":         "Kendi test setimizde en iyi sonuç (11/14 tek başına). "
                        "Daha büyük, CPU'da ~0,8 sn.",
        "default":      True,
    },
    "vit": {
        "repo_id":      "dima806/ai_vs_real_image_detection",
        "display_name": "ViT AI Dedektörü",
        "architecture": "Vision Transformer (denetimli eğitilmiş)",
        "license":      "Apache-2.0",
        "size_mb":      327,
        "note":         "CLIP ile aynı mimari aileden — algoritma çeşitliliği daha az.",
        "default":      False,
    },
}

# ---------------------------------------------------------------------------
# US-5: Kişi (gövde) dedektörü — torchvision COCO ağırlıkları
# ---------------------------------------------------------------------------

PERSON_TARGET_DIR = MODELS_DIR / "person_detector"

PERSON_VARIANTS = {
    "resnet50": {
        "url":      "https://download.pytorch.org/models/"
                    "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "filename": "fasterrcnn_resnet50_fpn_coco.pth",
        "display_name": "Faster R-CNN ResNet-50 FPN (COCO)",
        "license":  "BSD-3-Clause (torchvision)",
        "size_mb":  160,
        "note":     "COCO box mAP 37.0 — CPU'da görsel başına ~2 saniye.",
        "default":  True,
    },
    "mobilenet": {
        "url":      "https://download.pytorch.org/models/"
                    "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
        "filename": "fasterrcnn_mobilenet_v3_large_fpn_coco.pth",
        "display_name": "Faster R-CNN MobileNetV3-Large FPN (COCO)",
        "license":  "BSD-3-Clause (torchvision)",
        "size_mb":  74,
        "note":     "COCO box mAP 32.8 — ~5 kat hızlı, yavaş makineler için.",
        "default":  False,
    },
}

# ---------------------------------------------------------------------------
# US-5 / US-6: Yüz dedektörü — OpenCV YuNet
# ---------------------------------------------------------------------------

FACE_TARGET_DIR = MODELS_DIR / "face_detector"

FACE_VARIANTS = {
    "yunet": {
        # Dosya opencv_zoo deposunda Git LFS ile tutulur; media.githubusercontent.com
        # gerçek içeriği verir (raw.githubusercontent.com yalnızca LFS pointer'ı döner).
        "url":      "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                    "main/models/face_detection_yunet/"
                    "face_detection_yunet_2023mar.onnx",
        "filename": "face_detection_yunet_2023mar.onnx",
        "display_name": "YuNet yüz dedektörü (2023mar)",
        "license":  "Apache-2.0 (OpenCV Zoo)",
        "size_mb":  0.23,
        "note":     "Çok küçük ve hızlı; yaş tahmini (US-6) için yüz kutularını üretir.",
        "default":  True,
        "min_bytes": 100_000,   # LFS pointer'ı (~130 bayt) indiyse hata ver
    },
}

# ---------------------------------------------------------------------------
# US-6: Yas tahmini
# ---------------------------------------------------------------------------

AGE_TARGET_DIR = MODELS_DIR / "age_estimator"

AGE_VARIANTS = {
    "vit": {
        "repo_id":      "nateraw/vit-age-classifier",
        "display_name": "ViT Yas Tahmini",
        "architecture": "Vision Transformer (FairFace ile egitilmis)",
        "license":      "Belirtilmemis (model kartinda lisans yok)",
        "size_mb":      327,
        "note":         "Kendi test setimizde ortalama hata ~4,5 yil. "
                        "9 sinif (0-2 dahil), bebeklerde de calisiyor. "
                        "UYARI: lisans beyan edilmemis.",
        "default":      True,
    },
    "siglip": {
        "repo_id":      "prithivMLmods/facial-age-detection",
        "display_name": "SigLIP Yas Tahmini",
        "architecture": "SigLIP (denetimli egitilmis)",
        "license":      "Apache-2.0",
        "size_mb":      354,
        "note":         "Apache-2.0 alternatifi. Kendi setimizde ortalama hata "
                        "~6,8 yil; en alt sinifi 1-10 oldugu icin bebek yasini "
                        "ayirt edemiyor.",
        "default":      False,
    },
}

COMPONENTS = {
    "ai":     ("US-3 — AI görsel dedektörü",      AI_VARIANTS,     AI_TARGET_DIR),
    "person": ("US-5 — Kişi (gövde) dedektörü",   PERSON_VARIANTS, PERSON_TARGET_DIR),
    "face":   ("US-5/US-6 — Yüz dedektörü",       FACE_VARIANTS,   FACE_TARGET_DIR),
    "age":    ("US-6 — Yaş tahmini",              AGE_VARIANTS,    AGE_TARGET_DIR),
}


def default_variant(variants: dict) -> str:
    """
    @brief Bir bileşenin varsayılan sürüm anahtarını döndürür.
    @param variants Sürüm sözlüğü.
    @return         Anahtar (örn. "swin").
    """
    for key, spec in variants.items():
        if spec.get("default"):
            return key
    return next(iter(variants))


def print_catalog() -> None:
    """@brief Tüm bileşenleri ve sürümlerini okunur biçimde yazdırır."""
    for comp, (title, variants, target) in COMPONENTS.items():
        print(f"\n=== {comp}  ({title}) ===")
        print(f"    hedef: {target}")
        for key, spec in variants.items():
            mark = "  (varsayılan)" if spec.get("default") else ""
            print(f"\n  --variant {key}{mark}")
            print(f"     kaynak : {spec.get('repo_id') or spec.get('url')}")
            print(f"     boyut  : ~{spec['size_mb']} MB")
            print(f"     lisans : {spec['license']}")
            print(f"     not    : {spec['note']}")
    print()


def _write_card(target: Path, spec: dict, variant: str) -> None:
    """
    @brief Model klasörüne künye dosyası (`model_card.json`) yazar.
    @param target  Hedef klasör.
    @param spec    Sürüm tanımı.
    @param variant Sürüm anahtarı.
    """
    card = {
        "display_name": spec.get("display_name", variant),
        "architecture": spec.get("architecture", ""),
        "source":       spec.get("repo_id") or spec.get("url", ""),
        "license":      spec.get("license", ""),
        "catalog_key":  variant,
    }
    with open(target / "model_card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, ensure_ascii=False, indent=2)


def _download_file(spec: dict, target: Path, force: bool) -> bool:
    """
    @brief Tek dosyalık bir modeli indirir (torchvision ağırlığı, ONNX...).
    @param spec   Sürüm tanımı (`url`, `filename`, opsiyonel `min_bytes`).
    @param target Hedef klasör.
    @param force  Dosya varsa bile yeniden indir.
    @return       Başarılıysa True.
    """
    target.mkdir(parents=True, exist_ok=True)
    dest = target / spec["filename"]

    if dest.exists() and not force:
        print(f"   zaten var: {dest.name} ({dest.stat().st_size / 1048576:.1f} MB)")
        return True

    print(f"   indiriliyor: {spec['filename']}  (~{spec['size_mb']} MB)")
    try:
        urllib.request.urlretrieve(spec["url"], dest)
    except Exception as exc:
        print(f"   HATA: indirme başarısız — {exc}", file=sys.stderr)
        return False

    size = dest.stat().st_size
    min_bytes = spec.get("min_bytes")
    if min_bytes and size < min_bytes:
        # Git LFS pointer'ı indi: dosya içeriği değil, sadece referans
        dest.unlink(missing_ok=True)
        print(f"   HATA: dosya beklenenden küçük ({size} bayt) — "
              f"Git LFS pointer'ı inmiş olabilir.", file=sys.stderr)
        return False

    print(f"   tamam: {dest.name} ({size / 1048576:.1f} MB)")
    return True


def _download_hf(spec: dict, target: Path, variant: str, force: bool) -> bool:
    """
    @brief HuggingFace deposundan model dosyalarını indirir.
    @param spec    Sürüm tanımı (`repo_id`).
    @param target  Hedef klasör.
    @param variant Sürüm anahtarı.
    @param force   Klasör doluysa bile yeniden indir.
    @return        Başarılıysa True.
    """
    weights = list(target.glob("*.safetensors")) + list(target.glob("*.bin"))
    if weights and not force:
        print(f"   zaten var: {weights[0].name} "
              f"({weights[0].stat().st_size / 1048576:.1f} MB)")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("   HATA: huggingface_hub kurulu değil → pip install huggingface_hub",
              file=sys.stderr)
        return False

    target.mkdir(parents=True, exist_ok=True)
    print(f"   indiriliyor: {spec['repo_id']}  (~{spec['size_mb']} MB)")
    try:
        snapshot_download(
            repo_id=spec["repo_id"],
            local_dir=str(target),
            allow_patterns=AI_ALLOW_PATTERNS,
        )
    except Exception as exc:
        print(f"   HATA: indirme başarısız — {exc}", file=sys.stderr)
        return False

    _write_card(target, spec, variant)
    print(f"   tamam: {target}")
    return True


def download_component(comp: str, variant: str = None, force: bool = False) -> bool:
    """
    @brief Tek bir bileşeni indirir.
    @param comp    "ai", "person" veya "face".
    @param variant Sürüm anahtarı; None ise varsayılan.
    @param force   Var olan dosyaları yeniden indir.
    @return        Başarılıysa True.
    """
    title, variants, target = COMPONENTS[comp]
    key  = variant or default_variant(variants)
    if key not in variants:
        print(f"HATA: '{comp}' için geçersiz sürüm: {key}", file=sys.stderr)
        return False
    spec = variants[key]

    print(f"\n[{comp}] {title} — {key}")
    if comp in ("ai", "age"):
        ok = _download_hf(spec, target, key, force)
    else:
        ok = _download_file(spec, target, force)
        if ok:
            _write_card(target, spec, key)

    if ok:
        print(f"   lisans: {spec['license']}")
    return ok


def main() -> int:
    """
    @brief Komut satırı girişi.
    @return Çıkış kodu (0 = tüm istenen bileşenler hazır).
    """
    parser = argparse.ArgumentParser(
        description="ForensIQ model dosyalarını indirir (US-3, US-5, US-6)."
    )
    parser.add_argument("--component", choices=sorted(COMPONENTS), default=None,
                        help="Yalnızca bu bileşeni indir (varsayılan: hepsi)")
    parser.add_argument("--variant", default=None,
                        help="Bileşenin sürümü (--list ile seçenekleri görün)")
    parser.add_argument("--list", action="store_true",
                        help="Bileşenleri ve sürümleri listele, çık")
    parser.add_argument("--force", action="store_true",
                        help="Dosya varsa bile yeniden indir")
    args = parser.parse_args()

    if args.list:
        print_catalog()
        return 0

    if args.variant and not args.component:
        print("HATA: --variant yalnızca --component ile birlikte kullanılır.",
              file=sys.stderr)
        return 2

    comps = [args.component] if args.component else list(COMPONENTS)
    results = {c: download_component(c, args.variant, args.force) for c in comps}

    print("\n" + "=" * 52)
    for comp, ok in results.items():
        print(f"  {comp:<8} {'TAMAM' if ok else 'BAŞARISIZ'}")
    print("=" * 52)

    if all(results.values()):
        print("Sunucuyu yeniden başlatın: python app.py")
        return 0

    print("Eksik bileşen var; sistem bu özellikler olmadan çalışmaya devam eder.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
