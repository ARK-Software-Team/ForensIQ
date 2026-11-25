"""
@file tests/conftest.py
@brief pytest ortak yapılandırması.

@details Testler proje kökündeki modülleri içe aktarabilsin diye kök dizin
         `sys.path`'e eklenir. Böylece testler `pytest` komutuyla proje
         kökünden doğrudan çalıştırılabilir.

@note Bu dosyadaki testler **model dosyası gerektirmez**. Hepsi saf karar
      mantığını sınar; CLIP / SwinV2 / Faster R-CNN yüklenmez. Bu bilinçlidir:
      testlerin saniyeler içinde çalışması, her değişiklikten sonra
      koşturulabilmelerini sağlar.
"""

import sys
from pathlib import Path

PROJE_KOKU = Path(__file__).resolve().parent.parent
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))
