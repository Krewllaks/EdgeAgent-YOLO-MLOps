# EdgeAgent Teknik Strateji Dokumani

Tarih: 2026-03-12
Durum: Sprint 1 Tamamlandi, Model V2 + Post-Processing Hazirligi

---

## 1. Rakip Gruplarla Farkimiz

### Diger Gruplarin Yaklasimi
- Sadece veri coklama (augmentation) ile mevcut YOLO modelini egitme
- Tek katmanli tespit: YOLO ciktisi = nihai karar
- Donanim farkindaligindan yoksun (edge deployment plani yok)

### Bizim Yaklasimimiz: Fine-tuning + VLM Reasoning
1. **Fine-tuning:** Hatali goruntuleri toplayip modeli gercek kusurlarla egittik
2. **VLM Reasoning:** PaliGemma 3B ile "neden hatali?" sorusuna dogal dil cevabi
3. **Donanim Farkindaligi:** Jetson Orin Nano profiling, TensorRT plani, bellek tahmini
4. **Agentic Reasoning:** Sistem sadece "hatali/hatasiz" demez, hata nedenini aciklar

### Sonuc
- Diger grup: Augmentation -> YOLO -> OK/NOK (tek katman)
- Biz: Fine-tuning + Augmentation -> YOLO -> Belirsizse VLM -> Neden Analizi (iki katman)

Bu fark, hocanin "diger gruptan en buyuk farkiniz nedir?" sorusunun cevabi:
**Agentic Reasoning ve Donanim Farkindaligi.**

---

## 2. Neden Yeni Sinif Acmadik?

### Karar: "Egri Takilmis Vida" icin ayri sinif ACILMADI

**Gerekce:**
- **Basitlik:** 3 sinif (screw, missing_screw, missing_component) yeterli
- **Hiz:** Her ek sinif, detection head'de islem yukunu arttirir
- **Veri yetersizligi:** "Egri vida" icin yeterli etiketli veri yok
- **VLM cozumu:** Belirsiz durumlarda PaliGemma zaten detayli analiz yapar.
  "Bu vida egri mi?" sorusunu VLM katmani cevaplayabilir.

**Phase 3 Degerlendirmesi:** Eger operatorlerden gelen geri bildirimde
"egri vida" vakasi sik tekrar ederse, o zaman sinif acilabilir.

---

## 3. Data Leakage Onleme: Hash Kontrolu

### Problem
Veri coklama (augmentation) sirasinda, orijinal bir goruntunun augmented
versiyonu train setinde, orijinali ise test setinde olabilir. Bu durum
yapay yuksek metrikler uretir (leakage).

### Cozum: SHA-256 Hash Kontrolu

```
Akis:
1. Orijinal goruntulerin SHA-256 hash'leri hesaplanir
2. Augmented goruntulerin kaynak dosya adi metadata'dan okunur
3. Test setindeki her goruntunun hash'i, train setindeki orijinallerle karsilastirilir
4. Cakisma tespit edilirse -> goruntu test'ten cikarilir veya train'den cikarilir
```

### Mevcut Durum
- Sprint 1'de augmented veriler orijinallerinden turetildi
- Orijinal ve augmented verilerin ayni split icinde olmasi saglandi
- Hash-bazli dogrulama `src/data/augment_analysis.py` icerisinde yapilir

### Risk Degerlendirmesi
- Augmented veriler orijinallerinden uretildigi icin, orijinal test'te +
  augmented train'de olmasi leakage yaratir
- Mevcut pipeline'da bu kontrol edilmektedir

---

## 4. Asenkron VLM Tetikleme Mimarisi

### Problem
PaliGemma 3B @ 4-bit: ~500ms/frame isleme suresi.
YOLO hedef: <5ms/frame (TensorRT ile).
Senkron calissa ana hatti **100x yavaslatir**.

### Cozum: Async Kuyruk Mimarisi

```
Ana Hat (Hizli):                     VLM Hat (Yavas, Bagimsiz):
+----------+    +----------+         +----------------+
| Kamera   | -> | YOLO     | ------> | Dashboard      |
| Frame    |    | Inference |  OK     | Sonuc Goster   |
+----------+    +----+-----+         +----------------+
                     |
                     | conf < 0.40
                     v
                +----+-----+         +----------------+
                | Async    | ------> | PaliGemma      |
                | Queue    |         | VLM Worker     |
                +----------+         +-------+--------+
                                             |
                                     +-------v--------+
                                     | VLM Sonuc      |
                                     | Dashboard'a Yaz|
                                     +----------------+
```

### Teknik Detaylar

1. **Tetikleme Kosulu:**
   - YOLO confidence < 0.40 (ayarlanabilir esik)
   - Hedef siniflar: `missing_screw`, `missing_component`
   - Hic tespit yoksa -> yine VLM'e gonder (kacirilan kusur riski)

2. **Kuyruk Ozellikleri:**
   - Thread-safe `Queue` (maxsize=32)
   - Kuyruk doluysa frame drop edilir (hiz > dogruluk tercihi)
   - Drop sayisi loglarda izlenir

3. **VLM Worker:**
   - Daemon thread olarak calisir
   - Frame alir -> PaliGemma'ya gonderir -> sonucu JSONL dosyasina yazar
   - Dashboard bu dosyayi okuyarak operatore gosterir

4. **Performans Etkisi:**
   - Ana YOLO pipeline'i HICBIR ZAMAN beklemez
   - VLM latency'si sadece "yorum gecikmesi" olarak yansir
   - Fabrika bandi hizi etkilenmez

### Implementasyon Dosyalari
- `src/edge/vlm_trigger.py` - Trigger logic + async worker
- `src/edge/profiler.py` - Latency ve bellek tahmini

---

## 5. KRITIK RISK: Weak Labeling (Zayif Etiketleme)

### Problem
Augmented (coklanmis) verilerde tum goruntudeki kutu `(0.5, 0.5, 1.0, 1.0)`
olarak (yani resmin tamami) etiketlendiginde:

- YOLO, nesnenin arka plandan **ayrildigini** ogrenemez
- Model "metal dokunun tamamini" hata olarak algilamaya baslar
- Yuksek mAP raporu alditicildir cunku model resmin %100'unu eslestiriyor

### Cozum (UYGULANMIS)
- `augment_analysis.py` guncellendi: Artik weak label OLUSTURMUYOR
- Eger goruntunun yaninda uygun YOLO label dosyasi varsa -> kopyalar
- Eger label dosyasi yoksa -> goruntuyu atlar (egitim setine EKLEMEZ)
- `--clean-augmented` flagi ile eski weak label dosyalari temizlenebilir
- Ingilizce klasor isimleri de artik destekleniyor (screw, missing_screw, missing_component)

### Etki Degerlendirmesi
- Eski durum: Tum augmented goruntulere `(0.5, 0.5, 0.98, 0.98)` label veriliyordu -> TEHLIKELI
- Yeni durum: Sadece duzgun label'a sahip goruntular kullaniliyor -> GUVENLI

---

## 6. Canny Edge Enhancement (Onisleme)

### Problem
Metal yuzeyler uzerindeki yansimalar (specularity) YOLO'nun
kenar detaylarini algilamasini zorlastirir. Ozellikle parlak
vida baslarinda false negative veya dusuk confidence olabiliyor.

### Cozum: Canny Kenar Karistirma
```
blended = alpha * original + (1 - alpha) * canny_edges_rgb
```

- **alpha**: 0.7 (varsayilan) - orijinal goruntunun agirligi
- **Canny esikleri**: low=50, high=150 (ayarlanabilir)
- Kenar haritasi: grayscale -> Canny -> RGB'ye cevir -> harmanlama

### Kullanim
- Dashboard "Edge Enhancement" sayfasindan interaktif onizleme
- Batch isleme: `python src/data/edge_enhancer.py --input-dir ... --output-dir ...`
- YOLO karsilastirmasi: Orijinal vs Enhanced tespit sayisi

### Implementasyon
- `src/data/edge_enhancer.py` - enhance_single(), enhance_dataset(), preview_enhancement()

---

## 7. Geometrik Mekansal Kumeleme (Post-Processing)

### Problem
YOLO sadece bireysel nesneleri tespit eder, ama urunun **fiziksel geometrisini**
bilmez. Bir urun uzerinde 4 vida pozisyonu varsa (sol 2, sag 2), bu bilgiyi
kullanarak daha guvenilir karar verebiliriz.

### Cozum: Spatial Clustering + Karar Matrisi

1. YOLO tespitlerini K-Means ile 4 kumeye ayir (beklenen vida pozisyonlari)
2. Kumeleri sol/sag tarafa ata (x-koordinat median'i)
3. Her tarafin durumunu belirle: S (screw var) veya MS (missing_screw)
4. Karar matrisi uygula:

```
Sol S  + Sag S   -> OK (tum vidalar mevcut)
Sol MS + Sag S   -> missing_screw (sol taraf)
Sol S  + Sag MS  -> missing_screw (sag taraf)
Sol MS + Sag MS  -> missing_component (kesin)
Herhangi MC      -> missing_component (dogrudan tespit)
```

### Avantajlari
- Tek basina YOLO'dan daha guvenilir kararlar
- Fiziksel urun geometrisini kullanarak false positive azaltma
- "Iki tarafta da eksik = komponent eksik" mantigi otomatik

### Implementasyon
- `src/reasoning/spatial_logic.py` - SpatialAnalyzer, detections_from_yolo_result()
- Dashboard "Spatial Clustering" sayfasindan interaktif analiz

---

## 8. TensorRT Optimizasyonu

### Neden Gerekli?
- Fabrika hedefi: 200 urun/saniye -> frame basi 5ms butce
- PyTorch FP32 Orin tahmini: ~119ms (6.5x RTX scale)
- TensorRT FP16/INT8 ile 3-5x hizlanma beklenir -> ~24-40ms
- Hala 5ms'nin ustunde olabilir -> batch processing veya multi-stream gerekebilir

### Plan
1. `python scripts/export_tensorrt.py --half` ile TensorRT FP16 export
2. INT8 calibration icin test setinden 500 gorsel kullanilir
3. Benchmark: TensorRT vs PyTorch latency karsilastirmasi
4. Gerekirse: Model pruning veya daha kucuk backbone (YOLOv10-N)

### Export Scripti
`scripts/export_tensorrt.py` hazir. Kullanim:
```bash
python scripts/export_tensorrt.py --half              # FP16
python scripts/export_tensorrt.py --int8 --half       # INT8 + FP16 fallback
python scripts/export_tensorrt.py --dry-run           # Sadece config goster
```

---

## 9. Surekli Egitim (Continuous Training) Stratejisi

### Hedef
Model, her hafta yeni gelen hatali verilerle otomatik fine-tune edilir.

### Planlanan Akis
1. Operatorler yanlis tespitleri isaretler (dashboard "Yanlis" butonu)
2. Isaretlenen veriler `data/feedback/feedback_log.jsonl` dosyasina kaydedilir
3. Haftalik cron job: Yeni veriyle incremental fine-tune
4. Yeni model, eski modelle A/B testi yapilir (mAP karsilastirmasi)
5. Basarili ise otomatik deploy, degilse eski model korunur

### Active Learning Dongusu
Dashboard'daki inference playground'da operator her tahmin icin
"Dogru" / "Yanlis" butonlarina basar. Bu geri bildirim:
- `data/feedback/feedback_log.jsonl` -> JSONL formatinda kaydedilir
- Her hafta bu dosyadaki "incorrect" isareti olanlar fine-tune setine eklenir
- FP analizi sayfasindan istatistikler izlenir

### MLOps Level 2 Hedefi
- Egitim suresi: 4 saat -> 30 dakika (daha kucuk incremental set)
- Otomatik pipeline: Data -> Train -> Validate -> Deploy
- Model registry: MLflow ile versiyon takibi

---

## 10. Iyilestirme Oncelik Tablosu

| Oncelik | Madde | Durum |
|---------|-------|-------|
| KRITIK | Weak label'leri duzelt (augment_analysis.py) | TAMAMLANDI |
| KRITIK | TensorRT FP16 export + benchmark | SCRIPT HAZIR |
| YUKSEK | Dashboard inference playground | TAMAMLANDI |
| YUKSEK | FP orneklerini kaydet + analiz et | TAMAMLANDI |
| YUKSEK | Operator geri bildirim (Active Learning) | TAMAMLANDI |
| YUKSEK | Canny Edge Enhancement onisleme | TAMAMLANDI |
| YUKSEK | Geometrik Mekansal Kumeleme post-processing | TAMAMLANDI |
| YUKSEK | Model V2 egitimi (tum veri kaynaklari) | HAZIRLANIYOR |
| ORTA | Isik degisimi concept drift testi | PHASE 2 |
| ORTA | CA Neck ablation deneyi | PHASE 2 |
| ORTA | PaliGemma async zamanlama olcumu | PHASE 2 |
| DUSUK | DagsHub/W&B entegrasyonu | PHASE 3 |
| DUSUK | Setup.bat tek tikla kurulum | PHASE 3 |
| DUSUK | Model drift alarm sistemi | PHASE 3 |

### Elenen/Ertelenen Maddeler (Gerekce)
- **Hardcoded paths**: Sorun yok. Tum dosyalar `Path(__file__)` kullaniyor.
- **pip freeze**: Dev repo icin minimum version pin dogru yaklasim.
- **Early stopping patience**: Zaten `patience=25` ayarli.
- **VRAM stres testi**: Profiler zaten bellek tahmini yapiyor.
- **Modulerlik (OOP)**: Train script 250 satir, henuz bolunmeye gerek yok.
