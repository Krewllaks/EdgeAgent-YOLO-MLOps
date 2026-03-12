# EdgeAgent YOLO MLOps - Proje Detay Dokumani

Tarih: 2026-03-10  
Durum: Sprint 1 tamamlandi, Phase 2 hazirligi aktif

---

## 1) Proje Nedir, Ne Ise Yarar?

Bu proje, endustriyel kalite kontrolde insan yorgunlugu kaynakli hata kacirma oranini azaltmak,
hat hizina uyumlu milisaniye seviyesinde karar vermek ve modeli uzun vadede canli tutmak icin
tasarlanmis iki katmanli bir AI + MLOps sistemidir.

Ana hedef:
- Uretim hattinda OK/NOK tespiti yapmak
- Model performansi dusunce otomatik iyilestirme dongusune girebilmek
- Belirsiz durumlarda semantik analiz (VLM) ile operatore neden aciklamasi sunmak

Kisa ifade ile: **hizli tespit + akilli yorum + surekli ogrenme**.

---

## 2) Problem Tanimi

Rapor ve saha gozlemine gore sistemin hedefledigi temel problemler:

1. Insan operator yorgunlugu ve dikkat kaybi nedeniyle kusur kacirma
2. Statik modellerin concept drift ile zamanla dogruluk kaybetmesi
3. Cloud bagimliliginin gecikme ve gizlilik riski olusturmasi
4. Hidden technical debt nedeniyle modelin bakim maliyetinin artmasi

Bu nedenle proje, edge odakli ve MLOps uyumlu bir mimari olarak kurgulanmistir.

---

## 3) Sistem Mimarisi (Yuksek Seviye)

## 3.1 Faz 1 - Hiz Katmani
- Model: YOLOv10-S
- Ek modul: Coordinate Attention (CA)
- Gorev: `screw`, `missing_screw`, `missing_component`
- Cikti: bbox + class + confidence
- Oncelik: dusuk latency, yuksek throughput

## 3.2 Faz 2 - Bilissel Katman
- Model: PaliGemma (planlanan)
- Calisma modu: asenkron tetikleme
- Tetikleme: YOLO confidence esik alti (baslangic politikasi: `< 0.40`)
- Cikti: dogal dilde kusur yorumu / root cause ipucu

## 3.3 MLOps Katmani
- Veri hazirlama / validasyon
- Egitim / yeniden egitim
- MLflow run takibi
- Raporlama ve threshold bazli operasyon

---

## 4) Fazlar ve Kapsam

Bu projenin akademik cekirdegi 2 ana fazdan olusur:

- **Phase 1 (operasyonel):** YOLOv10-S + CA ile hizli tespit
- **Phase 2 (kognitif):** VLM reasoning ve agentic yorum

Uygulama tarafinda sprint bazli alt asamalar da mevcuttur:
- Veri audit / ingestion
- Augmentation entegrasyonu
- Final training
- Dashboard + edge profiling

---

## 5) Veri Mimarisi ve Ingestion

## 5.1 Kaynaklar
- `erdogan1/`, `erdogan2/` (ham saha kaynaklari)
- `roboflowetiketlenen/` (etiketli COCO export)
- `coklanmis/` (augmentation takviyesi)

## 5.2 Canonical dataset cikisi
- `data/processed/phase1_multiclass_v1/`
- Split: `train/val/test`
- YAML: `data/processed/phase1_multiclass_v1/data.yaml`

## 5.3 Leakage onleme
`src/data/augment_analysis.py` icinde hash tabanli kontrol uygulanir:
- train duplicate engeli
- val/test ile cakisma engeli

Not: exact hash kontrolu kuvvetli bir temel saglar; gelecekte near-duplicate kontrolu (pHash vb.)
eklenmesi tavsiye edilir.

---

## 6) Model ve Egitim Konfigurasyonu

## 6.1 Mimari
- Konfig dosyasi: `configs/models/yolov10s_ca.yaml`
- Sinif sayisi: `nc=3`
- CA injection noktasi: backbone icinde coklu seviye

## 6.2 Final training scripti
- Script: `scripts/train_final_phase1.py`
- Varsayilanlar:
  - `epochs=100`
  - `batch=8`
  - `imgsz=640`
  - `amp=True`
  - `workers=4`
  - `device=0` (GPU varsa)

## 6.3 Otomasyon
Egitim bitince script:
- best checkpoint'i `models/phase1_final_ca.pt` altina kopyalar
- baseline vs final karsilastirma raporunu `reports/final_phase1_report.md` dosyasina yazar

---

## 7) Sprint 1 Sonuclari (Somut Metrikler)

Kaynak: `reports/final_phase1_report.md`

- Baseline mAP50(B): `0.494170`
- Final mAP50(B): `0.994210`
- Delta: `+0.500040`
- Best observed mAP50(B): `0.994300` (epoch 96)

Bu, modelin veri dengesi ve CA entegrasyonuyla ciddi bir iyilesme gosterdigini ortaya koyar.

---

## 8) Veri Dengesi Son Durum (Sprint 1)

Kaynak: `docs/STATUS_CURRENT.md`

- Train image: `4174`
- BBox instance:
  - `screw`: `2561`
  - `missing_screw`: `416`
  - `missing_component`: `203`
- Background image: `2059`

Bu dagilim, nadir siniflarin ogrenilmesini baseline'e gore anlamli sekilde desteklemistir.

---

## 9) UI / Dashboard Bilesenleri

Ana dosya: `src/ui/sprint1_dashboard.py`

Dashboard su teknik bolumleri kapsar:
- Canli inference playground
- Veri dengeleme ve class dagilimi
- "Neden CA?" teknik aciklama
- MLflow takibi
- VLM tetikleme strateji akisi
- Edge profiler gorunumu
- False positive analizi
- Operator kontrol paneli

Bu arayuz sadece "hata var" demek yerine karar mantigini ve veri etkisini teknik olarak gosterir.

---

## 10) Edge ve Donanim Farkindaligi

## 10.1 GPU dogrulama
- Script: `scripts/check_gpu.py`
- Hedef: CUDA 12.1 + RTX 3050 dogrulamasi

## 10.2 TensorRT export
- Script: `scripts/export_tensorrt.py`
- Amac: Jetson tarafinda FP16/INT8 hizlandirma

## 10.3 Edge profiler
- Script: `src/edge/profiler.py`
- Cikti: Orin gecikme tahmini + TensorRT gereksinim yorumu + bellek tahmini

Proje karar notu:
- 200 urun/s hedefi icin strict butce ~5ms/urun oldugu icin TensorRT optimizasyonu kritik gorunur.

---

## 11) Phase 2 Hazirligi (VLM Trigger)

Script: `src/edge/vlm_trigger.py`

Strateji:
1. YOLO her frame'i hizli isler
2. Belirsizlikte async queue olayi olusturur
3. VLM worker main loop'u bloklamadan yorum uretir
4. Sonuclar dashboard kanalina yansitilir

Bu sayede semantic depth artarken ana hat throughput'u korunur.

---

## 12) Rakip Gruplardan Fark

Bu projeyi ayiran noktalar:
- Sadece augmentation degil, defect-odakli fine-tuning
- MLOps + izlenebilirlik + raporlama
- Edge deployment farkindaligi
- VLM reasoning stratejisi ile agentic expansion

Tek satirlik ozet:
**Detection + Fine-tuning + Agentic Reasoning + Hardware Awareness**

---

## 13) Kod Gozden Gecirme Ozeti (Mevcut Repo)

Kod tabani genel olarak duzenli ve modulerdir. Kritik scriptler:
- `scripts/train_final_phase1.py`
- `src/models/coordatt.py`
- `src/data/augment_analysis.py`
- `src/edge/profiler.py`
- `src/edge/vlm_trigger.py`
- `src/ui/sprint1_dashboard.py`

Gucsuz noktalar / dikkat edilmesi gerekenler:
1. Augmented veri etiket kalitesi (weak bbox) mutlaka duzenli kontrol edilmelidir.
2. Leakage kontrolu exact hash ile basarili ama near-duplicate kontrolleri de eklenebilir.
3. VLM queue load senaryolari (dolu kuyruk, event drop) metrik olarak izlenmelidir.
4. Dashboard icindeki sabit metrikler zamanla rapor dosyalarindan dinamik okunacak hale getirilebilir.

---

## 14) Calistirma Akisi (Takim Icinde Standart)

1. Kurulum
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_gpu.py
```

2. Veri pipeline
```bash
python scripts/prepare_phase1_dataset.py
python src/data/augment_analysis.py
```

3. Dashboard
```bash
streamlit run src/ui/sprint1_dashboard.py
```

4. Final training (opsiyonel yeniden)
```bash
python scripts/train_final_phase1.py --data data/processed/phase1_multiclass_v1/data.yaml --model-cfg configs/models/yolov10s_ca.yaml --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0
```

5. Phase 2 prep
```bash
python src/edge/profiler.py --model models/phase1_final_ca.pt --source data/processed/phase1_multiclass_v1/test/images
python src/edge/vlm_trigger.py --model models/phase1_final_ca.pt --source data/processed/phase1_multiclass_v1/test/images --conf-threshold 0.40
```

### V2 veri hattini hizli hazirlama
V2 dataset icin tek komut:

```bash
python scripts/prepare_v2_dataset.py
```

Elle calistirmak istersen:

```bash
python scripts/prepare_phase1_dataset.py --output-dir data/processed/phase1_v2
python src/data/augment_analysis.py --dataset-dir data/processed/phase1_v2 --source-dir coklanmis --source-dir coklanmisacili --allow-folder-labels --fallback-bbox "0.5 0.5 0.8 0.8" --clean-augmented
```

Sonrasinda V2 egitimi:

```bash
python scripts/train_final_phase1.py --data data/processed/phase1_v2/data.yaml
```

---

## 15) Sonuc

Sprint 1 ile proje, raporda belirtilen Faz 1 hedeflerini teknik olarak gerceklestirmis durumdadir:
- yuksek mAP
- edge odakli egitim/operasyon kurgusu
- dashboard uzerinden teknik izlenebilirlik
- Phase 2 icin async VLM trigger ve edge profiler altyapisi

Bir sonraki buyuk adim, Phase 2'de VLM yorum katmanini production-safe sekilde aktif etmek,
concept drift ve continuous training mekaniklerini operasyonel hale getirmektir.
