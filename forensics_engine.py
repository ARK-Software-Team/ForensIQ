"""
@file forensics_engine.py
@brief Core image forensics analysis engine.
@details Implements SIFT, SURF, AKAZE, ORB algorithms for copy-move
         forgery detection, plus ELA-based tamper localization and a
         multi-signal AI-generation detector using 8 independent forensic signals.
@author ImageForensics Team
@version 1.2.0
"""

import cv2
import numpy as np
import logging
import base64
import scipy.fft as sfft
import scipy.stats as sstats
from skimage.feature import local_binary_pattern
from io import BytesIO
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

# AIDetector: CLIP + eğitimli derin model + istatistiksel sinyaller (çevrimdışı)
try:
    from ai_detector import AIDetector
    _AI_DETECTOR_AVAILABLE = True
except ImportError:
    _AI_DETECTOR_AVAILABLE = False

# MetadataInspector: dosya yapısından AI etiketi tespiti (User Story-4)
try:
    from metadata_inspector import MetadataInspector
    _METADATA_INSPECTOR_AVAILABLE = True
except ImportError:
    _METADATA_INSPECTOR_AVAILABLE = False

# PersonDetector: görüntüdeki kişi sayısının tespiti (User Story-5)
try:
    from person_detector import PersonDetector
    _PERSON_DETECTOR_AVAILABLE = True
except ImportError:
    _PERSON_DETECTOR_AVAILABLE = False

# AgeEstimator: yüzlerden yaş tahmini (User Story-6)
try:
    from age_estimator import AgeEstimator
    _AGE_ESTIMATOR_AVAILABLE = True
except ImportError:
    _AGE_ESTIMATOR_AVAILABLE = False

# MediumClassifier: görselin türü (fotoğraf / tablo / render / ekran görüntüsü)
try:
    from medium_classifier import MediumClassifier
    _MEDIUM_CLASSIFIER_AVAILABLE = True
except ImportError:
    _MEDIUM_CLASSIFIER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Metadata → AI skoru sınırlı artış kuralları (Karar K-12)
# ============================================================================
#
# Dosya yapısında bulunan AI etiketi, piksel analizinden bağımsız güçlü bir
# kanıttır; ancak metadata kopyalanabilir/uydurulabilir. Bu yüzden:
#   - skor **asla düşürülmez** (kamera EXIF'i "gerçek" kanıtı sayılmaz),
#   - artış sınırlıdır ve `METADATA_BOOST_CAP` değerini geçemez (asla %100 olmaz),
#   - taban skor raporda `ai_confidence_base` olarak görünmeye devam eder.

## verdict → (eklenecek puan, uygulanacak alt sınır)
##
## Not: "AI kullanılmış" ile "AI ile üretilmiş" aynı şey değildir.
## C2PA/IPTC üç ayrı durumu ayırır ve her biri farklı ağırlık hak eder:
##   c2pa_ai            → tamamen AI üretimi            → tam artış
##   c2pa_ai_composite  → gerçek içerik + AI parçalar   → sınırlı artış
##   c2pa_ai_enhanced   → yalnızca iyileştirme (upscale)→ artış YOK
## Son durum bilinçli olarak bu tabloda yoktur: AI ile çözünürlüğü artırılmış
## gerçek bir fotoğraf "AI üretimi" sayılmaz.
METADATA_BOOST_RULES = {
    "ai_signature":      (25.0, 75.0),  # doğrudan üretici imzası (Midjourney, SD...)
    "c2pa_ai":           (25.0, 75.0),  # C2PA: tamamen AI üretimi
    "ai_suspect":        (12.0, 60.0),  # difüzyon üretim parametreleri gömülü
}

# `c2pa_ai_composite` ve `c2pa_ai_enhanced` bilerek bu tabloda YOK.
# Gerekçe: "AI ile düzenlenmiş gerçek bir fotoğraf" sorusunun doğru cevabı
# AI *üretim* skorunu yükseltmek değildir — görüntünün içeriği gerçektir.
# Bu beyanlar DÜZENLEME kanıtıdır ve `_collect_edit_evidence()` üzerinden
# sahtecilik/edit kanalına (US-2) gider.

## Düzenleme kanıtı sayılan metadata kararları → (etiket, ağırlık)
EDIT_EVIDENCE_VERDICTS = {
    "c2pa_ai_composite": "Künyeye göre gerçek içeriğe AI üretimi parçalar eklenmiş",
    "c2pa_ai_enhanced":  "Künyeye göre görüntü AI ile iyileştirilmiş "
                         "(çözünürlük / gürültü)",
}

## Künyede tanınan düzenleme eylemleri.
EDIT_EVIDENCE_ACTIONS = ("edited", "filtered", "placed", "cropped", "resized")

## Düzenleme kanıtında raporlanan en fazla yazılım / eylem sayısı.
EDIT_MAX_SOFTWARE = 3
EDIT_MAX_ACTIONS = 4

## Model yüklenmemişken arayüze ve raporlara yazılan metin.
MODEL_YUKLU_DEGIL = "yüklü değil"

## Metadata artışının çıkabileceği en yüksek skor.
METADATA_BOOST_CAP = 95.0


# ============================================================================
# Animasyonlu görseller (GIF / WEBP) — çok kareli AI analizi
# ============================================================================
#
# US-1 notu olarak duran sınırlama buydu: animasyonlu dosyalarda yalnızca ilk
# kare analiz ediliyordu. 25.11.2026 ölçümü bunun gerçek bir hataya yol açtığını
# gösterdi: AI üretimi 36 karelik bir videoda kare skorları %14 ile %71 arasında
# salınıyor; ilk kare %45 aldığı için dosya "gerçek" sayılıyordu.
#
# Karşılaştırma (aynı ölçüm):
#   kararlı AI GIF   : kareler %87-88, standart sapma 0,5
#   kararsız AI GIF  : kareler %14-71, standart sapma 22,6
#
# Bu yüzden birden çok kare örnekleniyor ve **en yüksek K karenin ortalaması**
# alınıyor. Gerekçe: bir üreticinin parmak izi hareket bulanıklığı ve sıkıştırma
# yüzünden bazı karelerde kaybolur, net karelerde görünür. Ortalama bu kanıtı
# seyreltir; maksimum ise tek bir karenin hatasına açıktır.
#
# @warning Bu eşik, elimizde **gerçek kamera çekimi** bir animasyon olmadığı
#          için yanlış pozitif tarafından kalibre EDİLEMEDİ (bkz. S-22).

## Animasyonlu dosyada en çok kaç kare örneklenir.
ANIMATION_MAX_FRAMES = 6

## Skoru belirlerken ortalaması alınan en yüksek kare sayısı.
ANIMATION_TOP_K = 3


# ============================================================================
# Kişi / yüz kutusu renkleri (RGB — görüntü RGB uzayında çizilir)
# ============================================================================

## Gövde kutusu rengi (yeşil) — arayüzdeki --safe ile aynı aile.
PERSON_BOX_COLOR = (120, 240, 130)

## Yüz kutusu rengi (turkuaz) — arayüzdeki --accent ile aynı aile.
FACE_BOX_COLOR = (80, 215, 255)


# ============================================================================
# ForensicsEngine
# ============================================================================

class ForensicsEngine:
    """
    @class ForensicsEngine
    @brief Orchestrates all image forgery detection algorithms.

    @details
    State machine (FSM) is managed externally by app.py.
    This class is stateless between calls.

    Algorithms:
    - SIFT  : Scale-Invariant Feature Transform (copy-move)
    - SURF  : Speeded-Up Robust Features        (copy-move)
    - AKAZE : Accelerated KAZE                  (copy-move)
    - ORB   : Oriented FAST + Rotated BRIEF     (copy-move)
    - ELA   : Error Level Analysis              (compression artifacts)
    - AI    : AIDetector — CLIP zero-shot + eğitimli derin model + 5 istatistiksel sinyal
    - META  : MetadataInspector — dosya yapısında AI etiketi taraması (US-4)
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self):
        """@brief Initialize engine, matchers and AIDetector."""
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # AIDetector: api_detector.py import edildiyse kullan
        if _AI_DETECTOR_AVAILABLE:
            self._ai_detector = AIDetector()
            logger.info("ForensicsEngine v1.3: AIDetector (hibrit mod) aktif")
        else:
            self._ai_detector = None
            logger.info("ForensicsEngine v1.3: AIDetector yok, 8-sinyal motoru kullanılıyor")

        # MetadataInspector: tek örnek olarak tutulur (durumsuz, US-4)
        if _METADATA_INSPECTOR_AVAILABLE:
            self._metadata_inspector = MetadataInspector()
            logger.info("ForensicsEngine: MetadataInspector v%s aktif",
                        MetadataInspector.VERSION)
        else:
            self._metadata_inspector = None
            logger.warning("ForensicsEngine: MetadataInspector yok, metadata analizi atlanacak")

        # PersonDetector: kişi sayımı (US-5), yüz kutuları US-6'ya zemin hazırlar
        if _PERSON_DETECTOR_AVAILABLE:
            self._person_detector = PersonDetector()
            if self._person_detector.available:
                logger.info("ForensicsEngine: PersonDetector v%s aktif (%s)",
                            PersonDetector.VERSION,
                            self._person_detector.body_name or "yalnız yüz")
            else:
                logger.info("ForensicsEngine: PersonDetector modelleri yok, "
                            "kişi sayımı atlanacak")
        else:
            self._person_detector = None
            logger.warning("ForensicsEngine: person_detector.py yok, kişi sayımı atlanacak")

        # AgeEstimator: yüz kutularından yaş tahmini (US-6)
        if _AGE_ESTIMATOR_AVAILABLE:
            self._age_estimator = AgeEstimator()
        else:
            self._age_estimator = None
            logger.warning("ForensicsEngine: age_estimator.py yok, yaş tahmini atlanacak")

        # MediumClassifier: medium ekseni (fotoğraf/tablo/render/ekran) — US-3 iyileştirmesi
        if _MEDIUM_CLASSIFIER_AVAILABLE:
            self._medium_classifier = MediumClassifier()
        else:
            self._medium_classifier = None
            logger.warning("ForensicsEngine: medium_classifier.py yok, "
                           "yalnızca istatistiksel ekran görüntüsü tespiti kullanılacak")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_analysis(self, img_np: np.ndarray, img_bytes: bytes,
                          algorithms: list) -> dict:
        """
        @brief Run all selected algorithms and aggregate results.

        @param img_np    Image as NumPy array (H x W x 3, RGB).
        @param img_bytes Raw image bytes (for ELA compression analysis).
        @param algorithms List of algorithm IDs to run.
        @return          Full results dict ready for JSON serialization.
        """
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        algo_results = {}
        heatmap = np.zeros((h, w), dtype=np.float32)

        # --- Run each keypoint algorithm ---
        if "sift" in algorithms:
            r = self._run_keypoint_detector("sift", gray, img_np)
            algo_results["sift"] = r
            heatmap = self._accumulate_heatmap(heatmap, r["regions"])

        if "surf" in algorithms:
            r = self._run_keypoint_detector("surf", gray, img_np)
            algo_results["surf"] = r
            heatmap = self._accumulate_heatmap(heatmap, r["regions"])

        if "akaze" in algorithms:
            r = self._run_keypoint_detector("akaze", gray, img_np)
            algo_results["akaze"] = r
            heatmap = self._accumulate_heatmap(heatmap, r["regions"])

        if "orb" in algorithms:
            r = self._run_keypoint_detector("orb", gray, img_np)
            algo_results["orb"] = r
            heatmap = self._accumulate_heatmap(heatmap, r["regions"])

        # --- ELA ---
        ela_result, ela_b64 = self._run_ela(img_bytes)
        ela_heatmap = ela_result["heatmap"]
        if ela_heatmap is not None:
            heatmap += ela_heatmap * 0.5

        # --- Aggregate tampering confidence ---
        scores = [v["confidence"] for v in algo_results.values()]
        ela_conf = ela_result["confidence"]
        scores.append(ela_conf)

        aggregate_confidence = float(np.mean(scores)) if scores else 0.0
        tampered = aggregate_confidence > 35.0

        # --- Ekran görüntüsü tespiti ---
        # Ekran görüntüleri (screenshot, UI, diagram) istatistiksel olarak
        # AI görsellerine benzer: düz renkler, sıfır gürültü, yüksek renk korelasyonu.
        # Bu tür içerikler için AI tespitini bypass ederiz.
        stat_screenshot = self._detect_screenshot(img_np, gray)

        # --- Medium ekseni: fotoğraf / tablo / render / ekran görüntüsü ---
        # AI dedektörleri aslında "bu bir kamera fotoğrafı mı?" sorusunu yanıtlar.
        # Medium'u ayrı ölçmek, "fotoğraf değil" ile "AI üretimi"nin karıştırılmasını
        # önler (Las Meninas tablosuna %65 AI verilmesi bu hatanın sonucuydu).
        medium_report = self._run_medium_classification(img_np)

        # Ekran görüntüsü kapısı.
        #
        # CLIP medium sınıflandırması **güvenli** bir karar verdiyse yetkili odur;
        # istatistiksel dedektör yalnızca yedektir. Gerekçe (23.11.2026 ölçümü):
        # istatistiksel dedektör ChatGPT üretimi iki görselde yanlış tetiklenip
        # %74-80'lik AI skorunu sıfırlıyordu; CLIP ikisine de doğru biçimde
        # "3B render" dedi. Medium güvenilir değilse eski davranış korunur.
        if medium_report.get("available") and medium_report.get("confident"):
            is_screenshot     = medium_report.get("blocks_ai_verdict", False)
            screenshot_source = "CLIP medium sınıflandırması"
        else:
            is_screenshot     = stat_screenshot
            screenshot_source = "istatistiksel sinyaller"

        # --- AI detection: AIDetector (hibrit) veya 8-sinyal yedek ---
        frame_analysis = {"animated": False}

        if is_screenshot:
            # Ekran görüntüsü / arayüz grafiği: AI tespitini devre dışı bırak
            ai_result = {
                "generated":     False,
                "confidence":    0.0,
                "verdict_label": "Ekran Görüntüsü / Dijital İçerik — AI Tespiti Geçersiz",
                "signal_detail": [{"name": "Ekran Görüntüsü Tespiti",
                                   "score": 100.0,
                                   "label": f"Kaynak: {screenshot_source}. Arayüz grafikleri "
                                            f"istatistiksel olarak AI görsellerine benzer."}],
            }
        elif self._ai_detector is not None:
            # Animasyonlu dosyada tek kare yeterli değil: kareler arasında skor
            # ciddi biçimde salınabiliyor (bkz. ANIMATION_* açıklaması).
            kareler   = self._sample_animation_frames(img_bytes)
            kare_rapor = self._analyze_frames(kareler)

            if kare_rapor is not None:
                raw_ai = dict(kare_rapor["best_raw"])
                raw_ai["confidence"] = kare_rapor["aggregate"]
                raw_ai["generated"]  = kare_rapor["aggregate"] > 50.0
                frame_analysis = {
                    "animated":        True,
                    "frames_analyzed": len(kare_rapor["scores"]),
                    "scores":          kare_rapor["scores"],
                    "mean":            kare_rapor["mean"],
                    "std":             kare_rapor["std"],
                    "flagged":         kare_rapor["flagged"],
                    "top_k":           kare_rapor["top_k"],
                    "method":          f"en yüksek {kare_rapor['top_k']} karenin ortalaması",
                }
            else:
                raw_ai = self._ai_detector.detect(img_np, img_bytes)
                frame_analysis = {"animated": False}

            # AIDetector sonucunu ortak formata çevir
            ai_result = {
                "generated":     raw_ai["generated"],
                "confidence":    raw_ai["confidence"],
                "verdict_label": self._ai_verdict_label(raw_ai),
                "signal_detail": self._ai_signal_detail(raw_ai),
                "model_scores":  raw_ai.get("model_scores", []),
                "model_count":   raw_ai.get("model_count", 0),
            }
        else:
            ai_result = self._detect_ai_generated_multisignal(img_np, gray)

        # --- Metadata / AI tag incelemesi (US-4) ---
        # Piksel analizinden bağımsız, ayrı bir kanıt kanalı: dosya yapısında
        # üretici imzası (EXIF/XMP/PNG-text/C2PA) aranır.
        metadata_report = self._run_metadata_inspection(img_bytes)

        # --- Künye beyanlı düzenleme kanıtı (US-2) ---
        # C2PA manifesti "düzenlendi" diyorsa bu, piksel analizinden bağımsız
        # doğrudan bir kanıttır; sahtecilik kararını bu da belirler.
        edit_evidence = self._collect_edit_evidence(metadata_report)
        if edit_evidence["present"]:
            tampered = True

        # --- Metadata bulgusuyla AI skorunu sınırlı ölçüde yükselt (K-12) ---
        ai_base_conf = round(float(ai_result["confidence"]), 2)
        ai_result, metadata_boost = self._apply_metadata_boost(
            ai_result, metadata_report
        )

        # --- Kişi sayısı tespiti (US-5) ---
        person_report = self._run_person_detection(img_np)

        # --- Yaş tahmini (US-6) ---
        # Girdi, US-5'in ürettiği yüz kutularıdır; ayrı bir yüz araması yapılmaz.
        age_report = self._run_age_estimation(img_np, person_report)

        persons_b64 = self._render_person_overlay(img_np, person_report, age_report)

        # --- Normalize and render heatmap ---
        heatmap_b64, annotated_b64 = self._render_outputs(img_np, heatmap)

        return {
            "metadata_report": metadata_report,
            "edit_evidence": edit_evidence,
            "medium_report": medium_report,
            "person_report": person_report,
            "person_count": person_report.get("person_count", 0),
            "age_report": age_report,
            "persons_b64": persons_b64,
            "tampered": tampered,
            "confidence": round(aggregate_confidence, 2),
            "ai_generated": ai_result["generated"],
            "ai_confidence": round(ai_result["confidence"], 2),
            "ai_confidence_base": ai_base_conf,
            "ai_metadata_boost": metadata_boost,
            "ai_frame_analysis": frame_analysis,
            "ai_model_scores": ai_result.get("model_scores", []),
            "ai_model_count": ai_result.get("model_count", 0),
            "ai_signal_detail": ai_result["signal_detail"],
            "ai_verdict_label": ai_result["verdict_label"],
            "algorithm_results": algo_results,
            "ela_confidence": round(ela_conf, 2),
            "heatmap_b64": heatmap_b64,
            "annotated_b64": annotated_b64,
            "ela_b64": ela_b64,
            "image_size": {"width": int(w), "height": int(h)},
        }

    # ------------------------------------------------------------------
    # Metadata / AI tag incelemesi (User Story-4)
    # ------------------------------------------------------------------

    def ai_model_info(self) -> dict:
        """
        @brief AI tespitinde kullanılan algoritma kanallarının künyesi (US-3).

        @return `{"model_count", "channels": [...]}` — her kanal için ad, mimari,
                tür (öğrenen/kural tabanlı) ve yüklü olup olmadığı.
        @note Harici API kanalı yoktur; sistem tamamen çevrimdışıdır.
        """
        channels = []
        model_count = 0

        if self._ai_detector is not None:
            clip_ready = bool(getattr(self._ai_detector, "_clip_ready", False))
            channels.append({
                "name":         "CLIP Zero-Shot",
                "architecture": "ViT-B/32 (zero-shot, eğitim gerektirmez)",
                "type":         "öğrenen model",
                "loaded":       clip_ready,
            })
            if clip_ready:
                model_count += 1

            deep = getattr(self._ai_detector, "_deep", None)
            if deep is not None:
                info = deep.info()
                channels.append({
                    "name":         info["name"],
                    "architecture": info["architecture"] or "model indirilmedi",
                    "type":         "öğrenen model (denetimli eğitilmiş)",
                    "loaded":       info["available"],
                    "source":       info["source"],
                    "license":      info["license"],
                })
                if info["available"]:
                    model_count += 1

        channels.append({
            "name":         "İstatistiksel Analiz",
            "architecture": "5 sinyal (DCT kurtosis / gürültü / renk / kenar / doku)",
            "type":         "kural tabanlı",
            "loaded":       True,
        })
        channels.append({
            "name":         "Metadata / AI Tag",
            "architecture": "EXIF · XMP · PNG-text · C2PA taraması",
            "type":         "kanıt tabanlı",
            "loaded":       self._metadata_inspector is not None,
        })
        age_info = (self._age_estimator.info()
                    if self._age_estimator is not None else None)
        channels.append({
            "name":         "Yaş Tahmini",
            "architecture": (age_info or {}).get("architecture") or MODEL_YUKLU_DEGIL,
            "type":         "yaş tahmini (US-6)",
            "loaded":       bool(age_info and age_info["available"]),
        })
        channels.append({
            "name":         "Medium Sınıflandırma",
            "architecture": "CLIP zero-shot (fotoğraf / tablo / 3B / ekran)",
            "type":         "bağlam kanalı",
            "loaded":       bool(self._medium_classifier
                                 and self._medium_classifier.available),
        })

        # Kişi sayımı kanalları (US-5) — AI tespiti değil, ayrı bir yetenek
        person_info = (self._person_detector.info()
                       if self._person_detector is not None else None)
        channels.append({
            "name":         "Kişi Sayımı (gövde)",
            "architecture": (person_info or {}).get("body", MODEL_YUKLU_DEGIL),
            "type":         "nesne tespiti (US-5)",
            "loaded":       bool(person_info and person_info["available"]
                                 and person_info["body"] != MODEL_YUKLU_DEGIL),
        })
        channels.append({
            "name":         "Yüz Tespiti",
            "architecture": (person_info or {}).get("face", MODEL_YUKLU_DEGIL),
            "type":         "yüz tespiti (US-5 / US-6)",
            "loaded":       bool(person_info and person_info["available"]
                                 and person_info["face"] != MODEL_YUKLU_DEGIL),
        })

        return {"model_count": model_count, "channels": channels}

    @staticmethod
    def _collect_edit_evidence(metadata_report: dict) -> dict:
        """
        @brief Dosya künyesinin beyan ettiği düzenleme kanıtlarını toplar (US-2).

        @param metadata_report MetadataInspector raporu.
        @return `{present, label, detail, software, actions, source}`

        @details C2PA manifesti "bu görüntü düzenlendi" diyorsa bu, piksel
        analizinden **bağımsız ve doğrudan** bir kanıttır: dosyanın kendisi
        beyan eder. Üç kaynak kabul edilir:
          - `c2pa_ai_composite` : gerçek içerik + AI üretimi parçalar
          - `c2pa_ai_enhanced`  : AI ile iyileştirme (upscale / gürültü)
          - manifestteki `edited` / `filtered` / `placed` eylemleri

        @note Bu kanıt AI **üretim** skorunu yükseltmez; görüntünün içeriği
              gerçek olabilir. Yalnızca "düzenlenmiş" bilgisini taşır.
        """
        bos = {"present": False, "label": "", "detail": "",
               "software": [], "actions": [], "source": ""}
        if not isinstance(metadata_report, dict):
            return bos

        verdict = metadata_report.get("verdict", "")
        c2pa    = metadata_report.get("c2pa", {}) or {}
        actions = [a for a in (c2pa.get("actions") or [])
                   if a in EDIT_EVIDENCE_ACTIONS][:EDIT_MAX_ACTIONS]

        label = EDIT_EVIDENCE_VERDICTS.get(verdict)
        if label is None and actions:
            label = "Künyeye göre görüntü düzenlenmiş"
        if label is None:
            return bos

        # Künye bilgisi güvenilmeyen bir dosyadan gelir; liste olduğu
        # doğrulanmadan dilimlenirse bozuk bir manifest TypeError üretir.
        # Dilimleme `actions` gibi doğrudan liste kavrama içinde yapılır:
        # böylece listenin türü tek ifadeden okunur ve statik çözümleyici
        # (SonarQube S6466) dilimi hatalı biçimde "IndexError" saymaz.
        ham_yazilim = c2pa.get("software")
        if not isinstance(ham_yazilim, (list, tuple)):
            ham_yazilim = ()
        software = [str(s).strip() for s in ham_yazilim if str(s).strip()][:EDIT_MAX_SOFTWARE]

        detay = []
        if software:
            detay.append("Kayıtlı yazılım: %s." % ", ".join(software))
        if actions:
            detay.append("Eylemler: %s." % ", ".join(actions))

        return {
            "present":  True,
            "label":    label,
            "detail":   " ".join(detay),
            "software": software,
            "actions":  actions,
            "source":   "C2PA / Content Credentials",
        }

    def estimate_ages(self, img_np: np.ndarray) -> dict:
        """
        @brief Yalnızca yaş tahmini yapan genel API (US-6).
        @param img_np RGB uint8 görüntü dizisi.
        @return       Yaş raporu (yüz tespitini kendisi çalıştırır).
        """
        person_report = self._run_person_detection(img_np)
        return self._run_age_estimation(img_np, person_report)

    def _run_age_estimation(self, img_np: np.ndarray,
                            person_report: dict) -> dict:
        """
        @brief Tespit edilen yüzler için yaş tahmini üretir (US-6).

        @param img_np        RGB uint8 görüntü.
        @param person_report `_run_person_detection` raporu (yüz kutuları burada).
        @return              Yaş raporu.

        @details Dönen rapor:
        - `available`  : yaş modeli yüklü ve çalıştı mı
        - `count`      : yaş tahmini yapılan yüz sayısı
        - `ages`       : `[{face_index, age, margin, display, likely_range, ...}]`
        - `summary`    : kullanıcıya gösterilecek Türkçe özet
        - `model`      : kullanılan modelin künyesi

        @note Yüz yoksa veya model yüklü değilse boş rapor döner; analiz düşmez
              (Ek Madde-1).
        """
        bos = {
            "available": False, "count": 0, "ages": [],
            "summary": "Yaş tahmini yapılmadı.",
            "model": {},
        }

        if self._age_estimator is None or not self._age_estimator.available:
            bos["summary"] = ("Yaş tahmini modeli yüklü değil "
                              "(python tools/download_models.py --component age).")
            return bos

        yuzler = (person_report or {}).get("faces", []) or []
        if not yuzler:
            bos["available"] = True
            bos["model"]     = self._age_estimator.info()
            bos["summary"]   = ("Yüz tespit edilmediği için yaş tahmini "
                                "yapılamadı.")
            return bos

        try:
            sonuclar = self._age_estimator.estimate(img_np, yuzler)
        except Exception:
            logger.exception("Yaş tahmini başarısız")
            bos["summary"] = "Yaş tahmini sırasında hata oluştu."
            return bos

        return {
            "available": True,
            "count":     len(sonuclar),
            "ages":      sonuclar,
            "summary":   self._build_age_summary(sonuclar),
            "model":     self._age_estimator.info(),
        }

    @staticmethod
    def _build_age_summary(sonuclar: list) -> str:
        """
        @brief Yaş sonuçlarını tek cümlelik özete çevirir.
        @param sonuclar `AgeEstimator.estimate()` çıktısı.
        @return         Kullanıcıya gösterilecek özet.
        """
        if not sonuclar:
            return "Yaş tahmini üretilemedi."

        yaslar = [s["age"] for s in sonuclar]
        if len(yaslar) == 1:
            return (f"Tahmini yaş: {sonuclar[0]['display']}. "
                    f"Yaş tahmini yaklaşıktır; ışık, poz ve makyaj sonucu kaydırabilir.")

        return (f"{len(yaslar)} kişi için yaş tahmin edildi "
                f"(en genç {min(yaslar)}, en yaşlı {max(yaslar)}). "
                f"Yaş tahmini yaklaşıktır; ışık, poz ve makyaj sonucu kaydırabilir.")

    @staticmethod
    def _sample_animation_frames(img_bytes: bytes) -> list:
        """
        @brief Animasyonlu bir dosyadan eşit aralıklı kareler örnekler.

        @param img_bytes Ham dosya baytları.
        @return          RGB kare listesi; dosya animasyonlu değilse boş liste.

        @note İlk kare daima örneğe dâhildir. Hata hâlinde boş liste döner;
              analiz tek kareyle sürer (Ek Madde-1).
        """
        try:
            pil = Image.open(BytesIO(img_bytes))
            toplam = int(getattr(pil, "n_frames", 1))
            if toplam <= 1:
                return []

            adim = max(1, toplam // ANIMATION_MAX_FRAMES)
            indisler = list(range(0, toplam, adim))[:ANIMATION_MAX_FRAMES]

            kareler = []
            for i in indisler:
                pil.seek(i)
                kareler.append(np.array(pil.convert("RGB")))
            return kareler

        except Exception as exc:
            logger.warning("Animasyon kareleri okunamadı: %s", exc)
            return []

    def _analyze_frames(self, frames: list) -> dict | None:
        """
        @brief Birden çok karede AI tespiti çalıştırıp sonucu birleştirir.

        @param frames RGB kare listesi (en az 2 kare).
        @return       `{"scores", "aggregate", "best_raw", "std", "flagged"}`
                      veya analiz yapılamazsa None.

        @details Skor, en yüksek `ANIMATION_TOP_K` karenin ortalamasıdır.
                 Ayrıntılı panel (model skorları, sinyaller) **en yüksek
                 kareden** alınır; böylece başlık ile detay tutarlı olur.
        """
        if self._ai_detector is None or len(frames) < 2:
            return None

        ham = []
        for kare in frames:
            try:
                ham.append(self._ai_detector.detect(kare))
            except Exception:
                logger.exception("Kare analizi başarısız")

        if len(ham) < 2:
            return None

        skorlar = [float(h["confidence"]) for h in ham]
        en_iyi  = sorted(range(len(skorlar)), key=lambda i: -skorlar[i])
        ust_k   = [skorlar[i] for i in en_iyi[:ANIMATION_TOP_K]]
        birlesik = sum(ust_k) / len(ust_k)

        ortalama = sum(skorlar) / len(skorlar)
        varyans  = sum((s - ortalama) ** 2 for s in skorlar) / len(skorlar)

        return {
            "scores":    [round(s, 1) for s in skorlar],
            "aggregate": round(birlesik, 2),
            "mean":      round(ortalama, 2),
            "std":       round(varyans ** 0.5, 1),
            "flagged":   sum(1 for s in skorlar if s >= 50.0),
            "top_k":     ANIMATION_TOP_K,
            "best_raw":  ham[en_iyi[0]],
        }

    def classify_medium(self, img_np: np.ndarray) -> dict:
        """
        @brief Görselin türünü (fotoğraf / tablo / render / ekran) belirleyen genel API.
        @param img_np RGB uint8 görüntü dizisi.
        @return       MediumClassifier raporu.
        @see _run_medium_classification
        """
        return self._run_medium_classification(img_np)

    def _run_medium_classification(self, img_np: np.ndarray) -> dict:
        """
        @brief Medium sınıflandırmasını güvenli biçimde çalıştırır.
        @param img_np RGB uint8 görüntü dizisi.
        @return       Medium raporu; modül yoksa kararı etkilemeyen boş rapor.
        @note Hata hâlinde `blocks_ai_verdict=False` dönerek mevcut davranışı korur
              (Ek Madde-1 — çökme riski sıfırlanır).
        """
        if self._medium_classifier is None:
            return self._bos_medium_raporu()

        try:
            return self._medium_classifier.classify(img_np)
        except Exception:
            logger.exception("Medium sınıflandırması başarısız")
            return self._bos_medium_raporu()

    @staticmethod
    def _bos_medium_raporu() -> dict:
        """
        @brief Medium sınıflandırması yapılamadığında dönen güvenli rapor.
        @return Kararı etkilemeyen boş rapor.
        @note Tek yerde tanımlıdır; iki ayrı yerde kopyalanmış hâli
              birleştirildi (kod tekrarı giderildi).
        """
        return {
            "available":         False,
            "medium":            "unknown",
            "medium_label":      "Belirlenemedi",
            "confidence":        0.0,
            "scores":            {},
            "photographic":      True,
            "confident":         False,
            "blocks_ai_verdict": False,
            "caution":           "",
        }

    def detect_persons(self, img_np: np.ndarray) -> dict:
        """
        @brief Yalnızca kişi sayımı yapan genel API (US-5).
        @param img_np RGB uint8 görüntü dizisi.
        @return       PersonDetector raporu.
        @see _run_person_detection
        """
        return self._run_person_detection(img_np)

    def _run_person_detection(self, img_np: np.ndarray) -> dict:
        """
        @brief Görüntüdeki kişi sayısını tespit eder (US-5).
        @param img_np RGB uint8 görüntü dizisi.
        @return       PersonDetector raporu; modül/model yoksa `available=False`.
        @note Kişi tespiti hiçbir koşulda analizi düşürmemeli: hata durumunda
              boş rapor döndürülür (Ek Madde-1 — çökme riski sıfırlanır).
        """
        if self._person_detector is None:
            return {
                "available":    False,
                "person_count": 0,
                "body_count":   0,
                "face_count":   0,
                "persons":      [],
                "faces":        [],
                "summary":      "person_detector.py bulunamadı.",
            }

        try:
            return self._person_detector.detect(img_np)
        except Exception:
            logger.exception("Kişi tespiti başarısız")
            return {
                "available":    False,
                "person_count": 0,
                "body_count":   0,
                "face_count":   0,
                "persons":      [],
                "faces":        [],
                "summary":      "Kişi tespiti sırasında hata oluştu.",
            }

    def _render_person_overlay(self, img_np: np.ndarray,
                               person_report: dict,
                               age_report: dict = None) -> str:
        """
        @brief Kişi ve yüz kutularını görüntü üzerine çizer (US-5 görselleştirme).

        @param img_np        RGB uint8 görüntü.
        @param person_report `_run_person_detection` raporu.
        @param age_report    `_run_age_estimation` raporu (opsiyonel).
        @return              Base64 PNG; kişi yoksa veya hata olursa boş string.

        @details Gövde kutuları yeşil (`#1`, `#2`…), yüz kutuları turkuaz
                 (`Y1`, `Y2`…) çizilir; her etikette güven skoru bulunur.
                 Yaş tahmini varsa yüz etiketine eklenir: `Y1  29y  %94`.
        """
        if not person_report.get("available"):
            return ""
        persons = person_report.get("persons", [])
        faces   = person_report.get("faces", [])
        if not persons and not faces:
            return ""

        try:
            canvas = img_np.copy()

            # Etiketler küçük görsellerde okunamayacak kadar ufalmasın diye
            # ölçek alt sınırı yüksek tutulur.
            short_edge = min(img_np.shape[0], img_np.shape[1])
            scale  = max(0.75, min(short_edge / 900.0, 1.6))
            thick  = max(3, int(round(3 * scale)))
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.62 * scale
            ftick  = max(2, int(round(1.7 * scale)))

            for idx, person in enumerate(persons, start=1):
                x1, y1, x2, y2 = (int(round(v)) for v in person["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), PERSON_BOX_COLOR, thick)
                self._draw_tag(canvas, f"#{idx}  %{person['score'] * 100:.0f}",
                               x1, y1, PERSON_BOX_COLOR, font, fscale, ftick, thick)

            # Yüz sırası -> yaş tahmini eşlemesi (US-6)
            yas_eslemesi = {}
            if age_report and age_report.get("available"):
                for kayit in age_report.get("ages", []):
                    yas_eslemesi[kayit["face_index"]] = kayit

            for idx, face in enumerate(faces, start=1):
                x1, y1, x2, y2 = (int(round(v)) for v in face["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), FACE_BOX_COLOR,
                              max(2, thick - 1))
                yas = yas_eslemesi.get(idx - 1)
                etiket = (f"Y{idx}  {yas['age']}y  %{face['score'] * 100:.0f}"
                          if yas else f"Y{idx}  %{face['score'] * 100:.0f}")
                self._draw_tag(canvas, etiket,
                               x1, y1, FACE_BOX_COLOR, font,
                               fscale * 0.85, max(1, ftick - 1), thick,
                               below=True, box=(x1, y1, x2, y2))

            return self._np_to_b64(canvas)

        except Exception:
            logger.exception("Kişi kutuları çizilemedi")
            return ""

    @staticmethod
    def _draw_tag(canvas, text, x, y, color, font, fscale, ftick, border,
                  below: bool = False, box=None) -> None:
        """
        @brief Kutunun köşesine dolgulu, okunaklı bir numara etiketi çizer.

        @param canvas Üzerine çizilecek görüntü (yerinde değiştirilir).
        @param text   Etiket metni (örn. "#1  %99").
        @param x      Kutunun sol kenarı.
        @param y      Kutunun üst kenarı.
        @param color  Kutu rengi (etiket dolgusu da bu renktir).
        @param font   OpenCV font sabiti.
        @param fscale Yazı ölçeği.
        @param ftick  Yazı kalınlığı.
        @param border Kutu çizgi kalınlığı.
        @param below  True ise etiket kutunun üstüne değil içine/altına konur.
        @param box    `below` için kutu koordinatları `(x1, y1, x2, y2)`.

        @details Etiket koyu dolgu üzerine kutu renginde yazılır; böylece hem
                 açık hem koyu görüntülerde okunur. Kadraj dışına taşarsa
                 kutunun içine alınır.
        """
        (tw, th), base = cv2.getTextSize(text, font, fscale, ftick)
        pad = max(4, int(round(4 * fscale / 0.62)))
        h, w = canvas.shape[:2]

        tag_x = max(0, min(x, w - tw - 2 * pad))
        if below and box is not None:
            tag_y = min(h - th - 2 * pad, box[3] + 2)
            if tag_y + th + 2 * pad > h:
                tag_y = max(0, box[1] + 2)
        else:
            tag_y = y - th - 2 * pad - border // 2
            if tag_y < 0:
                tag_y = y + border

        x1, y1 = tag_x, tag_y
        x2, y2 = tag_x + tw + 2 * pad, tag_y + th + 2 * pad

        # Koyu dolgu + renkli çerçeve: her zemin üzerinde okunur
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (12, 16, 22), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(1, border // 2))
        cv2.putText(canvas, text, (x1 + pad, y2 - pad - base // 2),
                    font, fscale, color, ftick, cv2.LINE_AA)

    def inspect_metadata(self, img_bytes: bytes) -> dict:
        """
        @brief Yalnızca dosya yapısı incelemesi yapan genel API (piksel analizi yok).
        @param img_bytes Dosyanın ham baytları.
        @return MetadataInspector raporu.
        @see _run_metadata_inspection
        """
        return self._run_metadata_inspection(img_bytes)

    def _run_metadata_inspection(self, img_bytes: bytes) -> dict:
        """
        @brief Dosya yapısındaki AI etiketlerini inceler (US-4).
        @param img_bytes Dosyanın ham baytları.
        @return MetadataInspector raporu; modül yoksa `available=False` olan yedek rapor.
        @note Metadata analizi hiçbir koşulda analizi düşürmemeli: hata durumunda
              boş rapor döndürülür (Ek Madde-1 — çökme riski sıfırlanır).
        """
        if self._metadata_inspector is None:
            return {
                "available":     False,
                "verdict":       "unavailable",
                "verdict_label": "Metadata modülü yüklü değil",
                "summary":       "metadata_inspector.py bulunamadı.",
                "confidence":    0.0,
                "ai_tag_found":  False,
                "findings":      [],
            }

        try:
            return self._metadata_inspector.inspect(img_bytes)
        except Exception:
            logger.exception("Metadata incelemesi başarısız")
            return {
                "available":     False,
                "verdict":       "error",
                "verdict_label": "Metadata incelenemedi",
                "summary":       "Dosya yapısı çözümlenirken hata oluştu.",
                "confidence":    0.0,
                "ai_tag_found":  False,
                "findings":      [],
            }

    # ------------------------------------------------------------------
    # Algorithm: Copy-Move via Keypoint Detectors
    # ------------------------------------------------------------------

    def _run_keypoint_detector(self, algo_id: str, gray: np.ndarray,
                                img_np: np.ndarray) -> dict:
        """
        @brief Generic copy-move detector using a keypoint-based algorithm.

        @param algo_id  One of: 'sift', 'surf', 'akaze', 'orb'.
        @param gray     Grayscale image.
        @param img_np   Original color image (unused, kept for API consistency).
        @return         Dict with confidence, keypoint_count, match_count, regions.
        """
        try:
            detector = self._create_detector(algo_id)
            if detector is None:
                return self._empty_result(algo_id, "Detector not available")

            kps, descs = detector.detectAndCompute(gray, None)
            if descs is None or len(kps) < 10:
                return self._empty_result(algo_id, "Insufficient keypoints")

            # BRISK and ORB use binary descriptors → Hamming distance
            # SIFT and AKAZE use float descriptors → L2 distance
            use_hamming = algo_id in ("orb", "surf")  # surf = BRISK replacement
            matcher  = self.bf_hamming if use_hamming else self.bf_matcher
            matches  = matcher.knnMatch(descs, descs, k=3)
            ratio    = 0.75 if algo_id == "sift" else 0.80

            good_matches, regions = self._filter_matches(matches, kps, ratio)

            kp_count    = len(kps)
            match_count = len(good_matches)
            confidence  = min(100.0, (match_count / max(kp_count, 1)) * 400)

            return {
                "algorithm":      algo_id.upper(),
                "confidence":     round(confidence, 2),
                "keypoint_count": kp_count,
                "match_count":    match_count,
                "regions":        regions,
                "status":         "ok",
            }

        except Exception as e:
            logger.warning("%s failed: %s", algo_id, e)
            return self._empty_result(algo_id, str(e))

    def _filter_matches(self, matches: list, kps: list,
                        ratio: float) -> tuple:
        """
        @brief Filter descriptor matches using Lowe's ratio test and spatial guard.

        @param matches  Raw knnMatch results (k=3).
        @param kps      List of detected keypoints.
        @param ratio    Lowe's ratio threshold.
        @return         Tuple of (good_matches list, regions list).
        """
        good_matches = []
        regions      = []

        for match_group in matches:
            filtered = [m for m in match_group if m.distance > 0.01]
            if len(filtered) < 2:
                continue

            m, n = filtered[0], filtered[1]
            if m.distance >= ratio * n.distance:
                continue

            pt1  = kps[m.queryIdx].pt
            pt2  = kps[m.trainIdx].pt
            dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

            if dist > 20:
                good_matches.append(m)
                regions.append({
                    "x1":      int(pt1[0]),
                    "y1":      int(pt1[1]),
                    "x2":      int(pt2[0]),
                    "y2":      int(pt2[1]),
                    "strength": float(1.0 - m.distance / max(n.distance, 1e-6)),
                })

        return good_matches, regions

    def _create_detector(self, algo_id: str):
        """
        @brief Factory: create OpenCV feature detector by ID.
        @param algo_id Algorithm identifier string.
        @return OpenCV feature detector object or None.
        """
        if algo_id == "sift":
            return cv2.SIFT_create(nfeatures=2000)
        elif algo_id == "surf":
            # SURF is patent-restricted in standard OpenCV builds.
            # BRISK (Binary Robust Invariant Scalable Keypoints) is used as a
            # drop-in replacement: patent-free, similar accuracy and speed.
            return cv2.BRISK_create()
        elif algo_id == "akaze":
            return cv2.AKAZE_create()
        elif algo_id == "orb":
            return cv2.ORB_create(nfeatures=3000)
        return None

    # ------------------------------------------------------------------
    # Algorithm: Error Level Analysis (ELA)
    # ------------------------------------------------------------------

    def _run_ela(self, img_bytes: bytes, quality: int = 90) -> tuple:
        """
        @brief Error Level Analysis — orijinal JPEG kalitesine adaptif versiyon.

        @details
        Standart ELA sabit bir kalite (örn. 90) ile karşılaştırır.
        Sorun: Orijinal görüntü kalitesi 90'dan yüksekse (örn. telefon selfie ~97)
        büyük fark oluşur ve manipülasyon yokken %100 ELA skoru verir.

        Çözüm: Orijinal JPEG quantization tablosundan kaliteyi tahmin et,
        karşılaştırmayı orijinale yakın bir kalitede yap.
        Gerçek manipülasyon: orijinale benzer kalitede bile yüksek fark
        Sadece yeniden sıkıştırma: orijinaline yakın kalitede fark küçük

        @param img_bytes Raw image bytes.
        @param quality   Varsayılan karşılaştırma kalitesi (adaptif mod yoksa).
        @return          Tuple of (result_dict, base64_ELA_image_string).
        """
        try:
            pil = Image.open(BytesIO(img_bytes)).convert("RGB")

            # --- Orijinal JPEG kalitesini tahmin et ---
            orig_quality = self._estimate_jpeg_quality(img_bytes)

            # Adaptif karşılaştırma kalitesi:
            # Yüksek kaliteli orijinal (telefon selfie ~95-99) → çok yakın kalitede karşılaştır
            # Orta kalite (WhatsApp ~75-85) → standart 90 kullan
            if orig_quality is not None and orig_quality >= 94:
                compare_quality = max(90, orig_quality - 2)  # çok yakın: orig-2
                scale_factor    = 0.7                         # muhafazakâr
            elif orig_quality is not None and orig_quality >= 85:
                compare_quality = max(80, orig_quality - 5)
                scale_factor    = 1.0
            else:
                compare_quality = quality   # varsayılan 90
                scale_factor    = 1.5

            # Re-save at controlled quality
            buf = BytesIO()
            pil.save(buf, "JPEG", quality=compare_quality)
            buf.seek(0)
            recompressed = Image.open(buf).convert("RGB")

            ela_arr = ImageChops.difference(pil, recompressed)
            ela_np  = np.array(ela_arr).astype(np.float32)

            # Amplify for visibility
            scale   = 10
            ela_vis = np.clip(ela_np * scale, 0, 255).astype(np.uint8)

            # Confidence: top-5% brightest pixels
            brightness = ela_vis.mean(axis=2)
            threshold  = np.percentile(brightness, 95)
            hot_pixels = brightness[brightness > threshold]

            confidence = min(100.0, float(hot_pixels.mean()) * scale_factor) \
                         if len(hot_pixels) else 0.0

            # Heatmap
            hm      = brightness / 255.0

            ela_pil = Image.fromarray(ela_vis)
            ela_buf = BytesIO()
            ela_pil.save(ela_buf, "PNG")
            ela_b64 = base64.b64encode(ela_buf.getvalue()).decode("utf-8")

            logger.debug(f"ELA: orig_q={orig_quality} cmp_q={compare_quality} conf={confidence:.1f}")

            return (
                {"confidence": round(confidence, 2), "heatmap": hm},
                ela_b64
            )

        except Exception as e:
            logger.warning(f"ELA failed: {e}")
            return ({"confidence": 0.0, "heatmap": None}, "")

    @staticmethod
    def _jpeg_quality_from_avg(avg_q: float) -> int:
        """
        @brief Convert average quantization table value to JPEG quality estimate.
        @param avg_q  Average value from JPEG quantization table.
        @return       Estimated quality integer (30-100).
        """
        thresholds = [
            (1,  100), (2,  99), (4,  97), (8,  92),
            (12, 85),  (16, 80), (24, 72), (36, 65), (55, 55),
        ]
        for limit, quality in thresholds:
            if avg_q <= limit:
                return quality
        return max(30, int(100 - avg_q))

    @staticmethod
    def _estimate_jpeg_quality(img_bytes: bytes) -> int | None:
        """
        @brief JPEG quantization tablosundan orijinal kaliteyi tahmin et.
        @param img_bytes  Ham görüntü baytları.
        @return           Tahmini kalite (1-100) veya JPEG değilse None.
        """
        try:
            import struct
            data   = img_bytes
            tables = []
            i      = 0
            while i < len(data) - 3:
                if data[i] == 0xFF and data[i + 1] == 0xDB:
                    length = struct.unpack('>H', data[i + 2:i + 4])[0]
                    tables.extend(list(data[i + 5: i + 4 + length])[:64])
                    i += 2 + length
                else:
                    i += 1

            if not tables:
                return None

            avg_q = sum(tables[:64]) / 64.0
            return ForensicsEngine._jpeg_quality_from_avg(avg_q)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Algorithm: AI Generation Detection — 8-Signal Forensic Detector
    # ------------------------------------------------------------------

    def _detect_ai_generated_multisignal(self, img_np: np.ndarray,
                                          gray: np.ndarray) -> dict:
        """
        @brief Multi-signal AI image detector — orchestrates 8 independent forensic signals.
        @param img_np  Color image (RGB, uint8).
        @param gray    Grayscale image.
        @return        Dict with generated, confidence, signal_detail, verdict_label.
        """
        signals = {
            "dct_kurtosis":         self._signal_dct_kurtosis(gray),
            "noise_residual":       self._signal_noise_residual(gray),
            "lbp_entropy":          self._signal_lbp_entropy(gray),
            "fft_beta":             self._signal_fft_beta(gray),
            "glcm_homogeneity":     self._signal_glcm_homogeneity(gray),
            "chromatic_aberration": self._signal_chromatic_aberration(img_np),
            "blocking_artifact":    self._signal_blocking_artifact(gray),
            "saturation_kurtosis":  self._signal_saturation_kurtosis(img_np),
        }

        weights = {
            "dct_kurtosis":         0.20,
            "noise_residual":       0.18,
            "fft_beta":             0.20,
            "lbp_entropy":          0.10,
            "glcm_homogeneity":     0.08,
            "chromatic_aberration": 0.14,
            "blocking_artifact":    0.05,
            "saturation_kurtosis":  0.05,
        }

        weighted_sum   = sum(signals[k][0] * weights[k] for k in weights if k in signals)
        weight_total   = sum(weights[k] for k in weights if k in signals)
        final_conf     = weighted_sum / weight_total if weight_total > 0 else 50.0
        calib          = float(1.0 / (1.0 + np.exp(-(final_conf - 50.0) / 6.0)) * 100.0)

        ai_vote_count  = sum(1 for k in weights if k in signals and signals[k][0] >= 60.0)
        total_signals  = len([k for k in weights if k in signals])

        calib = self._calibrate_ai_score(calib, ai_vote_count)
        generated      = calib >= 55.0
        signal_detail  = [
            {"name": k.replace("_", " ").title(),
             "score": round(signals[k][0], 1),
             "label": signals[k][1]}
            for k in weights if k in signals
        ]
        verdict_label  = self._build_verdict_label(calib, ai_vote_count, total_signals)

        logger.info("AI Detection: %.1f%% [%s]", calib, verdict_label)
        return {
            "generated":     generated,
            "confidence":    round(calib, 2),
            "signal_detail": signal_detail,
            "verdict_label": verdict_label,
        }

    @staticmethod
    def _calibrate_ai_score(calib: float, ai_vote_count: int) -> float:
        """@brief Apply voting-based boost/suppression to raw AI confidence."""
        if ai_vote_count >= 5 and calib < 70:
            return min(95.0, calib + 12.0)
        if ai_vote_count >= 4 and calib < 60:
            return min(85.0, calib + 8.0)
        if ai_vote_count <= 1 and calib > 45:
            return max(10.0, calib - 10.0)
        return calib

    @staticmethod
    def _build_verdict_label(calib: float, votes: int, total: int) -> str:
        """@brief Build human-readable verdict string from calibrated confidence."""
        tag = f"({votes}/{total} sinyal)"
        if calib >= 80:
            return f"AI ÜRETİMİ — KESİN {tag}"
        if calib >= 65:
            return f"AI ÜRETİMİ — YÜKSEK {tag}"
        if calib >= 50:
            return f"AI ÜRETİMİ — ZAYIF {tag}"
        if calib >= 35:
            return f"GERÇEK — ZAYIF {tag}"
        if calib >= 20:
            return f"GERÇEK — YÜKSEK {tag}"
        return f"GERÇEK — KESİN {tag}"

    # ------------------------------------------------------------------
    # Individual signal extractors
    # ------------------------------------------------------------------

    def _signal_dct_kurtosis(self, gray: np.ndarray) -> tuple:
        """@brief Signal 1: DCT coefficient kurtosis."""
        try:
            h, w    = gray.shape
            cy, cx  = h // 2, w // 2
            ch, cw  = min(256, h), min(256, w)
            crop    = gray[cy-ch//2:cy+ch//2, cx-cw//2:cx+cw//2].astype(np.float32)
            dct     = cv2.dct(crop / 255.0)
            mid_dct = dct[4:ch//2, 4:cw//2].flatten()
            if len(mid_dct) <= 10:
                return (50.0, "DCT kurtosis: yetersiz veri")
            kurt = float(sstats.kurtosis(mid_dct, fisher=True))
            if not np.isfinite(kurt):
                return (50.0, "DCT kurtosis: hesaplanamadı")
            if kurt < 0.5:
                return (90.0, f"Çok düşük DCT kurtosis ({kurt:.2f}) — diffusion/GAN imzası")
            if kurt < 2.5:
                return (72.0, f"Düşük DCT kurtosis ({kurt:.2f}) — AI olası")
            if kurt < 5.0:
                return (45.0, f"Orta DCT kurtosis ({kurt:.2f}) — belirsiz")
            if kurt < 12.0:
                return (20.0, f"Yüksek DCT kurtosis ({kurt:.2f}) — doğal fotoğraf")
            return (10.0, f"Çok yüksek DCT kurtosis ({kurt:.2f}) — kesinlikle gerçek")
        except Exception as e:
            logger.warning("Signal 1 (DCT) failed: %s", e)
            return (50.0, "DCT kurtosis: hata")

    def _signal_noise_residual(self, gray: np.ndarray) -> tuple:
        """@brief Signal 2: High-pass noise residual and periodicity."""
        try:
            gray_f   = gray.astype(np.float32)
            blurred  = cv2.GaussianBlur(gray_f, (3, 3), 0)
            residual = gray_f - blurred
            noise_std = float(residual.std())
            res_norm  = residual - residual.mean()
            periodicity = 0.0
            if res_norm.std() > 0:
                row_means   = res_norm.mean(axis=0)
                ac_lag1     = float(np.corrcoef(row_means[:-1], row_means[1:])[0, 1])
                ac_lag4     = float(np.corrcoef(row_means[:-4], row_means[4:])[0, 1])
                periodicity = max(abs(ac_lag1), abs(ac_lag4))
            if noise_std < 2.0:
                return (88.0, f"Çok düşük gürültü ({noise_std:.2f}) — diffusion imzası")
            if noise_std < 4.5:
                return (68.0, f"Düşük gürültü ({noise_std:.2f}) — AI olası")
            if periodicity > 0.4:
                return (72.0, f"Periyodik gürültü (ac={periodicity:.2f}) — GAN upsampling")
            if noise_std > 14.0:
                return (15.0, f"Yüksek doğal gürültü ({noise_std:.2f}) — kamera sensörü")
            if noise_std > 8.0:
                return (25.0, f"Normal kamera gürültüsü ({noise_std:.2f})")
            return (42.0, f"Orta gürültü ({noise_std:.2f}) — sıkıştırılmış olabilir")
        except Exception as e:
            logger.warning("Signal 2 (noise) failed: %s", e)
            return (50.0, "Gürültü analizi: hata")

    def _signal_lbp_entropy(self, gray: np.ndarray) -> tuple:
        """@brief Signal 3: Local Binary Pattern texture entropy."""
        try:
            lbp  = local_binary_pattern(gray.astype(np.uint8), P=8, R=1, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
            hist    = np.clip(hist, 1e-9, None)
            norm_e  = float(-np.sum(hist * np.log2(hist))) / np.log2(10)
            if norm_e < 0.65:
                return (78.0, f"Düşük LBP entropi ({norm_e:.3f}) — yapay yüzey")
            if norm_e < 0.78:
                return (52.0, f"Orta LBP entropi ({norm_e:.3f})")
            if norm_e < 0.88:
                return (28.0, f"Yüksek LBP entropi ({norm_e:.3f}) — doğal doku")
            return (15.0, f"Çok yüksek LBP entropi ({norm_e:.3f}) — zengin doku")
        except Exception as e:
            logger.warning("Signal 3 (LBP) failed: %s", e)
            return (50.0, "LBP analizi: hata")

    def _signal_fft_beta(self, gray: np.ndarray) -> tuple:
        """@brief Signal 4: FFT radial power spectrum 1/f beta exponent."""
        try:
            h, w       = gray.shape
            fft_sh     = sfft.fftshift(sfft.fft2(gray.astype(np.float64)))
            power      = np.abs(fft_sh) ** 2
            cy2, cx2   = h // 2, w // 2
            Y, X       = np.ogrid[:h, :w]
            R          = np.sqrt((X - cx2)**2 + (Y - cy2)**2).astype(int)
            max_r      = min(cx2, cy2)
            rad_power  = np.array([power[R == r].mean()
                                   for r in range(1, max_r) if np.any(R == r)])
            if len(rad_power) <= 10:
                return (50.0, "FFT beta: çok küçük görüntü")
            log_f  = np.log10(np.arange(1, len(rad_power) + 1))
            log_p  = np.log10(rad_power + 1e-12)
            valid  = np.isfinite(log_p)
            if valid.sum() <= 5:
                return (50.0, "FFT beta: yetersiz veri")
            beta = float(np.polyfit(log_f[valid], log_p[valid], 1)[0])
            if beta > -0.8:
                return (86.0, f"Çok düz spektrum (β={beta:.2f}) — GAN imzası")
            if beta > -1.5:
                return (73.0, f"Düz spektrum (β={beta:.2f}) — diffusion imzası")
            if beta > -2.0:
                return (50.0, f"Sınırda spektrum (β={beta:.2f}) — belirsiz")
            if beta >= -2.8:
                return (18.0, f"Doğal 1/f² spektrum (β={beta:.2f}) — gerçek fotoğraf")
            if beta < -3.5:
                return (55.0, f"Aşırı dik spektrum (β={beta:.2f}) — GAN artefakt")
            return (35.0, f"Derin spektrum (β={beta:.2f})")
        except Exception as e:
            logger.warning("Signal 4 (FFT) failed: %s", e)
            return (50.0, "FFT spektrum: hata")

    def _signal_glcm_homogeneity(self, gray: np.ndarray) -> tuple:
        """@brief Signal 5: GLCM texture homogeneity."""
        try:
            g8          = (gray.astype(np.float32) / 32).astype(np.uint8).clip(0, 7)
            co          = np.zeros((8, 8), dtype=np.float64)
            co[g8[:-1, :].ravel(), g8[1:, :].ravel()] += 1
            total_co    = co.sum()
            if total_co > 0:
                co /= total_co
            i_idx, j_idx = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
            hom = float(np.sum(co / (1.0 + np.abs(i_idx - j_idx))))
            if hom > 0.75:
                return (80.0, f"Çok yüksek GLCM homojenlik ({hom:.3f}) — yapay yumuşaklık")
            if hom > 0.65:
                return (58.0, f"Yüksek GLCM homojenlik ({hom:.3f})")
            if hom > 0.45:
                return (25.0, f"Normal GLCM homojenlik ({hom:.3f})")
            return (15.0, f"Düşük GLCM homojenlik ({hom:.3f}) — zengin doku")
        except Exception as e:
            logger.warning("Signal 5 (GLCM) failed: %s", e)
            return (50.0, "GLCM: hata")

    def _signal_chromatic_aberration(self, img_np: np.ndarray) -> tuple:
        """@brief Signal 6: Chromatic aberration at image edges."""
        try:
            r_ch     = img_np[:, :, 0].astype(np.float32)
            g_ch     = img_np[:, :, 1].astype(np.float32)
            b_ch     = img_np[:, :, 2].astype(np.float32)
            edge_mask = cv2.Canny(g_ch.astype(np.uint8), 50, 150) > 0
            if edge_mask.sum() <= 100:
                return (50.0, "Kromatik aberasyon: kenar bulunamadı")
            ca = (float(np.abs((r_ch - g_ch)[edge_mask]).mean())
                  + float(np.abs((b_ch - g_ch)[edge_mask]).mean())) / 2.0
            if ca < 0.8:
                return (87.0, f"Sıfıra yakın CA ({ca:.2f}) — lens yok → AI")
            if ca < 1.8:
                return (63.0, f"Çok düşük CA ({ca:.2f}) — AI olası")
            if ca <= 7.0:
                return (16.0, f"Gerçekçi lens aberasyonu (CA={ca:.2f}) — kamera imzası")
            return (38.0, f"Yüksek CA ({ca:.2f}) — aşırı sıkıştırma")
        except Exception as e:
            logger.warning("Signal 6 (CA) failed: %s", e)
            return (50.0, "Kromatik aberasyon: hata")

    def _signal_blocking_artifact(self, gray: np.ndarray) -> tuple:
        """@brief Signal 7: Block boundary artifact metric (BAM)."""
        try:
            h, w   = gray.shape
            gray_f = gray.astype(np.float64)

            h_diffs = self._collect_boundary_diffs_h(gray_f, h, w)
            v_diffs = self._collect_boundary_diffs_v(gray_f, h, w)

            if not h_diffs or not v_diffs:
                return (40.0, "Blok analizi: yetersiz veri")

            bam = float(np.mean(h_diffs + v_diffs))
            if bam > 2.0:
                return (72.0, f"Güçlü blok artefakları (BAM={bam:.2f}) — VAE/GAN ızgara imzası")
            if bam > 1.4:
                return (48.0, f"Orta blok artefakları (BAM={bam:.2f})")
            return (20.0, f"Blok artefakı yok (BAM={bam:.2f})")
        except Exception as e:
            logger.warning("Signal 7 (BAM) failed: %s", e)
            return (50.0, "Blok analizi: hata")

    def _signal_saturation_kurtosis(self, img_np: np.ndarray) -> tuple:
        """@brief Signal 8: HSV saturation channel kurtosis."""
        try:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1].astype(np.float32).ravel()
            if sat.std() <= 1e-3:
                return (55.0, "Grayscale görüntü — doygunluk analizi sınırlı")
            sat_kurt = float(sstats.kurtosis(sat, fisher=True))
            if sat_kurt < 0.3:
                return (78.0, f"Düzgün doygunluk ({sat_kurt:.2f}) — AI renk paleti")
            if sat_kurt < 1.5:
                return (56.0, f"Orta doygunluk kurtosis ({sat_kurt:.2f}) — AI olası")
            if sat_kurt < 3.0:
                return (38.0, f"Normal doygunluk kurtosis ({sat_kurt:.2f}) — stüdyo fotoğrafı")
            return (16.0, f"Yüksek doygunluk kurtosis ({sat_kurt:.2f}) — doğal aydınlatma")
        except Exception as e:
            logger.warning("Signal 8 (saturation) failed: %s", e)
            return (50.0, "Doygunluk analizi: hata")

    # ------------------------------------------------------------------
    # Heatmap & Visualization
    # ------------------------------------------------------------------

    def _accumulate_heatmap(self, heatmap: np.ndarray,
                             regions: list) -> np.ndarray:
        """
        @brief Accumulate detected regions into the heatmap array.

        @param heatmap  Float32 array to accumulate into.
        @param regions  List of region dicts from algorithm results.
        @return         Updated heatmap array.
        """
        h, w = heatmap.shape
        for r in regions:
            strength = r.get("strength", 0.5)
            # Handle both (x1,y1,x2,y2) box and (x1,y1)+(x2,y2) point pairs
            if "x2" in r and "y2" in r:
                x1, y1 = max(0, r["x1"]), max(0, r["y1"])
                x2, y2 = min(w, r["x2"]), min(h, r["y2"])
                if x2 > x1 and y2 > y1:
                    heatmap[y1:y2, x1:x2] += strength
            else:
                # Point pair: draw small circle
                for key in [("x1","y1"), ("x2","y2")]:
                    xk, yk = key
                    cx = min(w-1, max(0, r.get(xk, 0)))
                    cy = min(h-1, max(0, r.get(yk, 0)))
                    cv2.circle(heatmap, (cx, cy), 8, strength, -1)
        return heatmap

    def _render_outputs(self, img_np: np.ndarray,
                        heatmap: np.ndarray) -> tuple:
        """
        @brief Render heatmap overlay and annotated output image.

        @param img_np   Original image (RGB).
        @param heatmap  Float32 heatmap array.
        @return         Tuple (heatmap_b64, annotated_b64) as PNG base64 strings.
        """
        h, w = heatmap.shape

        # Normalize heatmap
        if heatmap.max() > 0:
            hm_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            hm_norm = np.zeros((h, w), dtype=np.uint8)

        # Apply colormap (JET)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_color_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

        # Blend with original
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Draw contours on annotated output
        _, thresh = cv2.threshold(hm_norm, 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        annotated = overlay_rgb.copy()
        cv2.drawContours(annotated, contours, -1, (255, 50, 50), 2)

        # Encode both to base64 PNG
        heatmap_b64  = self._np_to_b64(hm_color_rgb)
        annotated_b64 = self._np_to_b64(annotated)

        return heatmap_b64, annotated_b64

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _np_to_b64(img_np: np.ndarray) -> str:
        """
        @brief Convert NumPy image array to base64-encoded PNG string.
        @param img_np  RGB uint8 array.
        @return        Base64 string.
        """
        pil = Image.fromarray(img_np.astype(np.uint8))
        buf = BytesIO()
        pil.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _detect_screenshot(self, img_np: np.ndarray,
                            gray: np.ndarray) -> bool:
        """
        @brief Görüntünün ekran görüntüsü / dijital UI içeriği olup olmadığını tespit et.

        @details
        Ekran görüntüleri (screenshot, diagram, UI mockup, yazılım arayüzü) istatistiksel
        olarak AI görsellerine benzer çünkü:
          - Düz renkli büyük alanlar (arka plan, butonlar) → sıfıra yakın gürültü
          - Sınırlı renk paleti (tema renkleri) → yüksek kanal korelasyonu
          - DCT kurtosis düşük → AI ile örtüşüyor

        Tespit yöntemi 3 bağımsız sinyal kullanır:
          1. Düz bölge oranı: Komşu piksel farkı < 3 olan piksellerin oranı
             UI'da büyük düz renkli alanlar bunu çok yüksek yapar (>0.55)
          2. Benzersiz renk oranı: Toplam piksel sayısına göre farklı renk sayısı
             Fotoğraflar çok renklidir; ekran görüntüleri az ve tekrarlayan renk içerir
          3. Histogram tepe noktası: Tek bir gri değerinin histogramdaki payı
             UI arka planları histogramda keskin bir tepe oluşturur (>0.04)

        @param img_np  RGB görüntü array.
        @param gray    Grayscale görüntü array.
        @return        True ekran görüntüsü ise, False değilse.
        """
        try:
            h, w = gray.shape

            # Sinyal 1: Düz bölge oranı
            diff_h = np.abs(
                gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)
            )
            flat_ratio = float((diff_h < 3).mean())

            # Sinyal 2: Benzersiz renk oranı
            unique_colors = len(np.unique(img_np.reshape(-1, 3), axis=0))
            unique_ratio  = unique_colors / (h * w)

            # Sinyal 3: Histogram tepe noktası
            hist      = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist_norm = hist / (hist.sum() + 1e-9)
            max_bin   = float(hist_norm.max())

            # Üç sinyal birlikte yüksekse → ekran görüntüsü
            # Eşikler: gerçek foto ve AI görsellerinde yanlış pozitif vermeyecek şekilde
            # Trello:     flat=0.928  unique=0.0076  maxbin=0.475  → SS ✓
            # Gerçek foto: flat=0.289  unique=0.064   maxbin=0.014  → ok ✓
            # AI görsel:  flat=0.75   unique=0.055    maxbin=0.089  → ok ✓
            is_ss = (flat_ratio > 0.70 and
                     unique_ratio < 0.05 and
                     max_bin > 0.05)

            if is_ss:
                logger.info(
                    f"Ekran görüntüsü tespit edildi: "
                    f"flat={flat_ratio:.3f} unique={unique_ratio:.4f} "
                    f"hist_peak={max_bin:.4f}"
                )

            return is_ss

        except Exception as e:
            logger.warning(f"Screenshot detection failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Metadata → AI skoru sınırlı artış (Karar K-12)
    # ------------------------------------------------------------------

    def _apply_metadata_boost(self, ai_result: dict,
                              metadata_report: dict) -> tuple:
        """
        @brief Dosya yapısında AI etiketi bulunduğunda AI skorunu sınırlı yükseltir.

        @param ai_result       Piksel analizinden gelen AI sonucu (değiştirilmez).
        @param metadata_report MetadataInspector raporu.
        @return                `(güncellenmiş_ai_result, boost_bilgisi_veya_None)`

        @details
        Kurallar `METADATA_BOOST_RULES` içinde tanımlıdır:
        kesin üretici imzası (`ai_signature`) veya C2PA AI beyanı (`c2pa_ai`)
        → +25 puan ve en az %75; difüzyon parametreleri (`ai_suspect`)
        → +12 puan ve en az %60. Sonuç `METADATA_BOOST_CAP` (%95) ile sınırlıdır.

        @note Skor **asla düşürülmez**: metadata'da AI izi olmaması görselin
              gerçek olduğunu kanıtlamaz (metadata silinebilir). Üst sınırın
              %100 olmaması da bilinçlidir — metadata başka dosyadan kopyalanabilir.
        """
        if not isinstance(metadata_report, dict):
            return ai_result, None

        verdict = metadata_report.get("verdict", "")
        rule    = METADATA_BOOST_RULES.get(verdict)
        if rule is None:
            return ai_result, None

        increment, floor = rule
        base    = float(ai_result.get("confidence", 0.0))
        boosted = min(METADATA_BOOST_CAP, max(base + increment, floor))

        if boosted <= base + 0.01:
            return ai_result, None

        generators = metadata_report.get("generators", []) or []
        label_src  = ", ".join(str(g) for g in generators[:2]) or \
                     metadata_report.get("verdict_label", "AI etiketi")

        updated = dict(ai_result)
        updated["confidence"]    = round(boosted, 2)
        updated["generated"]     = boosted > 50.0
        updated["verdict_label"] = (
            f"{self._verdict_label_for(boosted)} · metadata AI etiketli"
        )
        updated["signal_detail"] = [{
            "name":  "Metadata AI Etiketi (US-4)",
            "score": round(float(metadata_report.get("confidence", 0.0)), 1),
            "label": f"{label_src} — dosya yapısında bulundu "
                     f"(skor %{base:.1f} → %{boosted:.1f})",
        }] + list(ai_result.get("signal_detail", []))

        logger.info(
            "Metadata artışı uygulandı: verdict=%s, %.1f%% → %.1f%%",
            verdict, base, boosted,
        )

        return updated, {
            "applied":    True,
            "verdict":    verdict,
            "base":       round(base, 2),
            "final":      round(boosted, 2),
            "delta":      round(boosted - base, 2),
            "cap":        METADATA_BOOST_CAP,
            "generators": generators[:3],
            "note":       "Metadata kopyalanabildiği için artış sınırlıdır; "
                          "skor %95'i geçemez.",
        }

    @staticmethod
    def _verdict_label_for(conf: float) -> str:
        """
        @brief Güven skorundan Türkçe verdict etiketi üretir.
        @param conf 0-100 arası AI güven skoru.
        @return     Kullanıcıya gösterilecek etiket.
        """
        if conf >= 80:
            return "AI ÜRETİMİ — KESİN"
        if conf >= 65:
            return "AI ÜRETİMİ — YÜKSEK"
        if conf >= 50:
            return "AI ÜRETİMİ — ZAYIF"
        if conf >= 35:
            return "GERÇEK — ZAYIF"
        if conf >= 20:
            return "GERÇEK — YÜKSEK"
        return "GERÇEK — KESİN"

    def _ai_verdict_label(self, raw: dict) -> str:
        """
        @brief AIDetector sonucundan Türkçe verdict etiketi üret.
        @param raw  AIDetector.detect() dönüş dict'i.
        @return     Kullanıcıya gösterilecek etiket string'i.
        """
        return self._verdict_label_for(raw.get("confidence", 0))

    def _ai_signal_detail(self, raw: dict) -> list:
        """
        @brief AIDetector ham sonucunu sinyal detay listesine çevir.
        @param raw  AIDetector.detect() dönüş dict'i.
        @return     UI'ın beklediği signal_detail listesi.
        """
        # Model skorları buraya **konmaz**: arayüzde "Modeller" bölümü zaten
        # `model_scores` alanını gösteriyor. Burası yalnızca istatistiksel
        # kanalın tek tek ölçümlerini listeler (aynı bilgi iki kez yazılmaz).
        details = []

        # İstatistiksel sinyaller
        signals = raw.get("signals", {})
        notes   = raw.get("signal_notes", {})
        labels  = {
            "freq":    "Frekans Spektrumu (DCT)",
            "noise":   "Gürültü Düzenliliği",
            "color":   "Renk Kanalı Korelasyonu",
            "edge":    "Kenar Kalitesi",
            "texture": "Doku Düzenlilik Analizi",
        }
        for key, display in labels.items():
            if key in signals:
                details.append({
                    "name":  display,
                    "score": round(signals[key], 1),
                    "label": notes.get(key, "")
                })

        return details

    @staticmethod
    def _empty_result(algo_id: str, reason: str = "") -> dict:
        """
        @brief Return a zero-confidence placeholder result.
        @param algo_id  Algorithm identifier.
        @param reason   Human-readable failure reason.
        @return         Placeholder result dict.
        """
        return {
            "algorithm": algo_id.upper(),
            "confidence": 0.0,
            "keypoint_count": 0,
            "match_count": 0,
            "regions": [],
            "status": f"error: {reason}"
        }