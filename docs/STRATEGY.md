# EdgeAgent Teknik Strateji Dokumani

Tarih: 2026-03-10
Durum: Sprint 1 Tamamlandi, Phase 2 Hazirligi

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

## 5. TensorRT Optimizasyonu

### Neden Gerekli?
- Fabrika hedefi: 200 urun/saniye -> frame basi 5ms butce
- PyTorch FP32 Orin tahmini: ~119ms (6.5x RTX scale)
- TensorRT FP16/INT8 ile 3-5x hizlanma beklenir -> ~24-40ms
- Hala 5ms'nin ustunde olabilir -> batch processing veya multi-stream gerekebilir

### Plan
1. `model.export(format="engine", half=True)` ile TensorRT FP16 export
2. INT8 calibration icin test setinden 500 gorsel kullanilir
3. Benchmark: TensorRT vs PyTorch latency karsilastirmasi
4. Gerekirse: Model pruning veya daha kucuk backbone (YOLOv10-N)

---

## 6. Surekli Egitim (Continuous Training) Stratejisi

### Hedef
Model, her hafta yeni gelen hatali verilerle otomatik fine-tune edilir.

### Planlanan Akis
1. Operatorler yanlis tespitleri isaretler (dashboard butonu)
2. Isaretlenen veriler `data/feedback/` klasorune toplanir
3. Haftalik cron job: Yeni veriyle incremental fine-tune
4. Yeni model, eski modelle A/B testi yapilir (mAP karsilastirmasi)
5. Basarili ise otomatik deploy, degilse eski model korunur

### MLOps Level 2 Hedefi
- Egitim suresi: 4 saat -> 30 dakika (daha kucuk incremental set)
- Otomatik pipeline: Data -> Train -> Validate -> Deploy
- Model registry: MLflow ile versiyon takibi
