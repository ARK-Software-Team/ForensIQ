"""
@file person_detector.py
@brief Görüntüdeki kişi sayısının tespiti (User Story-5).

@details
  İki bağımsız dedektör birlikte çalışır:

  1. **Gövde dedektörü** — torchvision Faster R-CNN (COCO ile eğitilmiş).
     COCO'daki `person` sınıfı (etiket 1) kutuları sayılır.
  2. **Yüz dedektörü** — OpenCV **YuNet** (ONNX, ~230 KB).
     Hem gövdesi görünmeyen kişileri yakalamak hem de User Story-6'daki
     yaş tahmini için yüz kutularını sağlamak üzere kullanılır.

  **Sayım kuralı:** gövde kutuları temel alınır; hiçbir gövde kutusunun içine
  düşmeyen her yüz, gövdesi görünmeyen ek bir kişi sayılır. Böylece hem
  kalabalıkta birbirini kapatan kişiler hem de yalnız yüzü görünen (yakın
  çekim, kadrajdan taşmış) kişiler sayıma girer.

  Model dosyaları repoda tutulmaz; `tools/download_models.py` ile bir kez
  indirilir ve sonrasında sistem tamamen çevrimdışı çalışır.

@note Modeller yoksa veya yükleme başarısız olursa modül sessizce devre dışı
      kalır (`available == False`) ve analiz kişi sayımı olmadan sürer.
      Hiçbir koşulda istisna fırlatmaz (Ek Madde-1: çökme = -10 puan).

@warning COCO ve YuNet **gerçek fotoğraflarla** eğitilmiştir. Çizim, anime ve
         oyun ekran görüntülerinde kişi sayısı eksik çıkabilir; arayüzde bu
         uyarı kullanıcıya gösterilir.

@author ImageForensics Team
@version 1.0.0
"""

import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE_DIR  = Path(__file__).parent
MODELS_DIR = _BASE_DIR / "models"

## Gövde dedektörü ağırlıklarının bulunduğu klasör.
PERSON_MODEL_DIR = MODELS_DIR / "person_detector"

## YuNet yüz dedektörü ONNX dosyasının bulunduğu klasör.
FACE_MODEL_DIR = MODELS_DIR / "face_detector"

## COCO veri setinde "person" sınıfının etiket numarası.
COCO_PERSON_LABEL = 1

## Bir gövde kutusunun kişi sayılması için gereken en düşük güven skoru.
PERSON_SCORE_THRESHOLD = 0.70

## Kutunun görüntü alanına oranı bunun altındaysa gürültü sayılır (binde 1.5).
MIN_BOX_AREA_RATIO = 0.0015

## YuNet yüz tespiti için en düşük güven skoru.
FACE_SCORE_THRESHOLD = 0.60

## Bir yüzün "şu gövdenin yüzü" sayılması için gereken örtüşme oranı.
FACE_IN_BODY_RATIO = 0.60

# --- Çift kutu temizliği (mükerrer sayım önleme) -----------------------------
#
# Faster R-CNN kendi içinde IoU 0.5 ile NMS uygular; bu eşik iki hatayı geçiriyor:
#   1. Aynı kişiye hem tam gövde hem gövdenin bir parçası için kutu çizilmesi
#      (örn. 1 kişilik fotoğrafta gövde + üst beden = 2 kişi sayılıyordu),
#   2. Yan yana duran iki kişiyi birden kapsayan "grup kutusu".
# Her ikisi de kişi sayısını yukarı şişiriyor.

## Ek NMS eşiği: iki kutunun IoU'su bu değerin üstündeyse düşük skorlu atılır.
DEDUP_IOU_THRESHOLD = 0.45

## İçerilme eşiği: küçük kutunun bu kadarı büyük kutunun içindeyse aday eleme.
DEDUP_CONTAIN_RATIO = 0.85

## İçerilme elemesi yalnızca kutu, içinde kaldığı kutunun bu oranından büyükse
## uygulanır. Böylece büyük bir kişinin kutusu içinde kalan **uzaktaki küçük**
## bir kişi yanlışlıkla silinmez.
DEDUP_MIN_AREA_RATIO = 0.30

## Desteklenen gövde dedektörü mimarileri: dosya adı → torchvision kurucu adı.
BODY_MODEL_FILES = {
    "fasterrcnn_resnet50_fpn_coco.pth": (
        "fasterrcnn_resnet50_fpn", "Faster R-CNN ResNet-50 FPN (COCO)"
    ),
    "fasterrcnn_mobilenet_v3_large_fpn_coco.pth": (
        "fasterrcnn_mobilenet_v3_large_fpn",
        "Faster R-CNN MobileNetV3-Large FPN (COCO)"
    ),
}


def _box_area(box) -> float:
    """
    @brief Kutunun alanını hesaplar.
    @param box `[x1, y1, x2, y2]` biçiminde kutu.
    @return    Piksel cinsinden alan (negatifse 0).
    """
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(box_a, box_b) -> float:
    """
    @brief İki kutunun kesişim/birleşim (IoU) oranını hesaplar.
    @param box_a Birinci kutu `[x1, y1, x2, y2]`.
    @param box_b İkinci kutu.
    @return      0.0 - 1.0 arası IoU değeri.
    """
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(box_a) + _box_area(box_b) - inter
    return (inter / union) if union > 0 else 0.0


def _overlap_ratio(inner, outer) -> float:
    """
    @brief `inner` kutusunun ne kadarının `outer` kutusu içinde kaldığını verir.
    @param inner İçeride olduğu varsayılan kutu (örn. yüz).
    @param outer Dıştaki kutu (örn. gövde).
    @return      0.0 - 1.0 arası oran; `inner` alanı sıfırsa 0.0.
    """
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area  = _box_area(inner)
    return (inter / area) if area > 0 else 0.0


class PersonDetector:
    """
    @class PersonDetector
    @brief Görüntüdeki kişileri sayar ve yüz kutularını çıkarır (US-5).

    @details
    Kullanım:
    @code
        det = PersonDetector()
        if det.available:
            report = det.detect(img_np)
            print(report["person_count"])
    @endcode

    `available` False ise `detect()` `available=False` olan bir rapor döndürür;
    çağıran taraf bunu normal kabul etmelidir (model dosyaları opsiyoneldir).
    """

    VERSION = "1.0.0"

    def __init__(self, person_model_dir: Path = None,
                 face_model_dir: Path = None):
        """
        @brief Gövde ve yüz dedektörlerini diskten yükler.
        @param person_model_dir Gövde modeli klasörü (None → `models/person_detector`).
        @param face_model_dir   YuNet klasörü (None → `models/face_detector`).
        """
        self.person_model_dir = Path(person_model_dir or PERSON_MODEL_DIR)
        self.face_model_dir   = Path(face_model_dir or FACE_MODEL_DIR)

        self._body_model   = None
        self._face_model   = None
        self.body_name     = ""
        self.face_name     = ""

        self._load_body_model()
        self._load_face_model()

    @property
    def available(self) -> bool:
        """@brief En az bir dedektör yüklendiyse True."""
        return self._body_model is not None or self._face_model is not None

    # ------------------------------------------------------------------
    # Model yükleme
    # ------------------------------------------------------------------

    def _pick_body_weights(self):
        """
        @brief Klasördeki desteklenen ilk ağırlık dosyasını seçer.
        @return `(dosya_yolu, kurucu_adı, görünen_ad)` veya None.
        """
        if not self.person_model_dir.exists():
            return None
        for filename, (builder, display) in BODY_MODEL_FILES.items():
            path = self.person_model_dir / filename
            if path.exists():
                return path, builder, display
        return None

    def _load_body_model(self) -> None:
        """
        @brief torchvision Faster R-CNN modelini yerel ağırlıkla yükler.
        @note Ağ bağlantısı kullanılmaz; ağırlık dosyası yoksa sessizce atlanır.
        """
        picked = self._pick_body_weights()
        if picked is None:
            logger.info(
                "Gövde dedektörü ağırlığı bulunamadı (%s) — kişi sayımı için "
                "indirin: python tools/download_models.py --component person",
                self.person_model_dir,
            )
            return

        weights_path, builder_name, display = picked
        try:
            import torch
            from torchvision.models import detection as tv_detection

            logger.info("[4/4] Kişi tespiti modeli yükleniyor (~160 MB)...")
            builder = getattr(tv_detection, builder_name)
            # weights=None: internetten indirme yok, ağırlık diskten gelir
            model   = builder(weights=None, weights_backbone=None, num_classes=91)
            state   = torch.load(str(weights_path), map_location="cpu",
                                 weights_only=True)
            model.load_state_dict(state)
            model.eval()

            self._body_model = model
            self.body_name   = display
            logger.info("Gövde dedektörü yüklendi: %s", display)

        except Exception as exc:
            logger.warning("Gövde dedektörü yüklenemedi: %s", exc)
            self._body_model = None

    def _load_face_model(self) -> None:
        """
        @brief OpenCV YuNet yüz dedektörünü yükler.
        @note ONNX dosyası yoksa veya OpenCV sürümü desteklemiyorsa atlanır.
        """
        if not self.face_model_dir.exists():
            logger.info("YuNet klasörü yok (%s) — yüz tespiti devre dışı",
                        self.face_model_dir)
            return

        onnx_files = sorted(self.face_model_dir.glob("*.onnx"))
        if not onnx_files:
            logger.info("YuNet ONNX dosyası bulunamadı (%s)", self.face_model_dir)
            return

        try:
            import cv2

            if not hasattr(cv2, "FaceDetectorYN"):
                logger.warning("OpenCV sürümü FaceDetectorYN desteklemiyor")
                return

            onnx = onnx_files[0]
            self._face_model = cv2.FaceDetectorYN.create(
                str(onnx), "", (320, 320),
                FACE_SCORE_THRESHOLD, 0.3, 5000
            )
            self.face_name = f"YuNet ({onnx.stem})"
            logger.info("Yüz dedektörü yüklendi: %s", self.face_name)

        except Exception as exc:
            logger.warning("YuNet yüklenemedi: %s", exc)
            self._face_model = None

    # ------------------------------------------------------------------
    # Tespit
    # ------------------------------------------------------------------

    def _detect_bodies(self, img_np: np.ndarray) -> list:
        """
        @brief Faster R-CNN ile kişi (gövde) kutularını bulur.
        @param img_np RGB uint8 görüntü.
        @return       `[{"box": [x1,y1,x2,y2], "score": float}, ...]`
        """
        if self._body_model is None:
            return []

        try:
            import torch
            import torchvision.transforms.functional as TF

            tensor = TF.to_tensor(img_np)
            with torch.no_grad():
                output = self._body_model([tensor])[0]

            boxes  = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            img_area = float(img_np.shape[0] * img_np.shape[1])
            results  = []
            for box, label, score in zip(boxes, labels, scores):
                if int(label) != COCO_PERSON_LABEL:
                    continue
                if float(score) < PERSON_SCORE_THRESHOLD:
                    continue
                if _box_area(box) / img_area < MIN_BOX_AREA_RATIO:
                    continue
                results.append({
                    "box":   [float(v) for v in box],
                    "score": round(float(score), 3),
                })
            return self._deduplicate(results)

        except Exception as exc:
            logger.warning("Gövde tespiti başarısız: %s", exc)
            return []

    @staticmethod
    def _deduplicate(persons: list) -> list:
        """
        @brief Aynı kişiye ait mükerrer kutuları eler.

        @param persons Ham gövde kutusu listesi.
        @return        Temizlenmiş liste (yüksek skorlu kutular korunur).

        @details Kutular skora göre azalan sıralanır; bir kutu, kendisinden
        yüksek skorlu bir kutuyla
        - IoU'su `DEDUP_IOU_THRESHOLD`'u aşıyorsa (aynı kişiye ikinci kutu veya
          iki kişiyi kapsayan "grup kutusu"), **veya**
        - alanının `DEDUP_CONTAIN_RATIO` kadarı o kutunun içindeyse ve kendisi
          o kutunun `DEDUP_MIN_AREA_RATIO`'sundan büyükse (gövde içi parça kutu)
        atılır.

        @note Alan oranı koşulu bilinçlidir: büyük bir kişinin kutusu içinde
              kalan **uzaktaki küçük** bir kişi silinmemelidir.
        """
        ordered = sorted(persons, key=lambda p: -p["score"])
        kept = []
        for candidate in ordered:
            duplicate = False
            for existing in kept:
                if _iou(candidate["box"], existing["box"]) >= DEDUP_IOU_THRESHOLD:
                    duplicate = True
                    break
                inside = _overlap_ratio(candidate["box"], existing["box"])
                ratio  = _box_area(candidate["box"]) / max(
                    _box_area(existing["box"]), 1e-9
                )
                if inside >= DEDUP_CONTAIN_RATIO and ratio >= DEDUP_MIN_AREA_RATIO:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)

        removed = len(ordered) - len(kept)
        if removed:
            logger.info("Mükerrer kutu elendi: %d", removed)
        return kept

    def _detect_faces(self, img_np: np.ndarray) -> list:
        """
        @brief YuNet ile yüz kutularını bulur.
        @param img_np RGB uint8 görüntü.
        @return       `[{"box": [x1,y1,x2,y2], "score": float}, ...]`
        @note US-6 (yaş tahmini) bu kutuları kullanacak.
        """
        if self._face_model is None:
            return []

        try:
            import cv2

            bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            self._face_model.setInputSize((w, h))
            _, faces = self._face_model.detect(bgr)

            if faces is None:
                return []

            results = []
            for face in faces:
                x, y, fw, fh = (float(v) for v in face[:4])
                score = float(face[-1])
                if score < FACE_SCORE_THRESHOLD:
                    continue
                results.append({
                    "box":   [x, y, x + fw, y + fh],
                    "score": round(score, 3),
                })
            return results

        except Exception as exc:
            logger.warning("Yüz tespiti başarısız: %s", exc)
            return []

    def detect(self, img_np: np.ndarray) -> dict:
        """
        @brief Görüntüdeki kişi sayısını tespit eder (US-5).

        @param img_np RGB uint8 görüntü dizisi.
        @return       Kişi sayısı raporu (aşağıdaki alanlarla).

        @details Dönen rapor:
        - `available`      : dedektörler yüklü mü
        - `person_count`   : nihai kişi sayısı
        - `body_count`     : gövdesi tespit edilen kişi sayısı
        - `face_count`     : tespit edilen yüz sayısı
        - `extra_from_face`: yalnız yüzüyle sayılan (gövdesi görünmeyen) kişi sayısı
        - `persons`        : `[{box, score, has_face, face_box}]`
        - `faces`          : `[{box, score, matched_person}]`
        - `summary`        : kullanıcıya gösterilecek Türkçe özet
        - `models`         : kullanılan modellerin künyesi
        """
        if not self.available:
            return self._unavailable_report(
                "Kişi tespiti modelleri yüklü değil."
            )

        try:
            bodies = self._detect_bodies(img_np)
            faces  = self._detect_faces(img_np)

            # Her yüzü, içine düştüğü gövdeyle eşleştir
            for face in faces:
                face["matched_person"] = None
            for idx, person in enumerate(bodies):
                person["has_face"] = False
                person["face_box"] = None
                for face in faces:
                    if face["matched_person"] is not None:
                        continue
                    if _overlap_ratio(face["box"], person["box"]) >= FACE_IN_BODY_RATIO:
                        face["matched_person"] = idx
                        person["has_face"]     = True
                        person["face_box"]     = face["box"]
                        break

            # Hiçbir gövdeye eşleşmeyen yüzler = gövdesi görünmeyen kişiler
            extra = [f for f in faces if f["matched_person"] is None]
            person_count = len(bodies) + len(extra)

            return {
                "available":       True,
                "detector_version": self.VERSION,
                "person_count":    person_count,
                "body_count":      len(bodies),
                "face_count":      len(faces),
                "extra_from_face": len(extra),
                "persons":         bodies,
                "faces":           faces,
                "thresholds": {
                    "person_score": PERSON_SCORE_THRESHOLD,
                    "face_score":   FACE_SCORE_THRESHOLD,
                },
                "models": {
                    "body": self.body_name or "yüklü değil",
                    "face": self.face_name or "yüklü değil",
                },
                "summary": self._build_summary(
                    person_count, len(bodies), len(faces), len(extra)
                ),
            }

        except Exception:
            logger.exception("Kişi tespiti başarısız")
            return self._unavailable_report("Kişi tespiti sırasında hata oluştu.")

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(total: int, bodies: int, faces: int, extra: int) -> str:
        """
        @brief Sayım sonucunu açıklayan Türkçe özet üretir.
        @param total  Nihai kişi sayısı.
        @param bodies Gövde kutusu sayısı.
        @param faces  Yüz kutusu sayısı.
        @param extra  Yalnız yüzle sayılan kişi sayısı.
        @return       Tek cümlelik açıklama.
        """
        if total == 0:
            return ("Görüntüde kişi tespit edilmedi. Not: modeller gerçek "
                    "fotoğraflarla eğitildiği için çizim, anime ve oyun "
                    "görüntülerinde kişiler atlanabilir.")

        parts = [f"{total} kişi tespit edildi"]
        if bodies:
            parts.append(f"{bodies} gövde")
        if faces:
            parts.append(f"{faces} yüz")
        if extra:
            parts.append(f"{extra} kişi yalnız yüzünden sayıldı "
                         f"(gövdesi görünmüyor veya kadraj dışında)")
        return " · ".join(parts) + "."

    def _unavailable_report(self, reason: str) -> dict:
        """
        @brief Dedektör çalışmadığında dönen güvenli rapor.
        @param reason Kullanıcıya gösterilecek sebep.
        @return       `available=False` olan boş rapor.
        """
        return {
            "available":        False,
            "detector_version": self.VERSION,
            "person_count":     0,
            "body_count":       0,
            "face_count":       0,
            "extra_from_face":  0,
            "persons":          [],
            "faces":            [],
            "thresholds":       {},
            "models": {
                "body": self.body_name or "yüklü değil",
                "face": self.face_name or "yüklü değil",
            },
            "summary": reason,
        }

    def info(self) -> dict:
        """
        @brief Yüklü modellerin künyesi (arayüz / `GET /api/models` için).
        @return available, body, face alanları.
        """
        return {
            "available": self.available,
            "body":      self.body_name or "yüklü değil",
            "face":      self.face_name or "yüklü değil",
            "model_dirs": {
                "person": str(self.person_model_dir),
                "face":   str(self.face_model_dir),
            },
        }
