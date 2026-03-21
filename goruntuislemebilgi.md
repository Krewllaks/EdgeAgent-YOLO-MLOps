# EdgeAgent-YOLO-MLOps — Proje Bilgi Dokumani

Tarih: 2026-03-21
Durum: Faz 0-2 tamamlandi, uretim pipeline'i hazir

---

## 1. Proje Ozeti

Bu proje, endustriyel kalite kontrolde insan yorgunlugu kaynakli hata kacirma oranini azaltmak,
hat hizina uyumlu milisaniye seviyesinde karar vermek ve modeli uzun vadede canli tutmak icin
tasarlanmis **iki katmanli AI + MLOps** sistemidir.

Tek satirlik ozet: **Detection + Fine-tuning + Agentic Reasoning + Hardware Awareness**

### Ana Hedefler
- Uretim hattinda OK/NOK tespiti (screw, missing_screw, missing_component)
- Belirsiz tespitlerde VLM (PaliGemma) ile semantik analiz ve neden aciklamasi
- Operator geri bildirimi ile surekli ogrenme (Active Learning)
- Donanim bagimsiz calisma (ONNX/TensorRT/PyTorch, herhangi bir kamera)
- Kurumsal fabrika entegrasyonu (OPC-UA, MQTT, audit trail, vardiya raporlama)

### Rakip Gruplardan Fark
- Diger grup: Augmentation -> YOLO -> OK/NOK (tek katman, offline batch isleme)
- Biz: Fine-tuning + Augmentation -> YOLO -> Belirsizse VLM -> Neden Analizi (iki katman)
- Ek olarak: Canli pipeline, edge deployment, MLOps, operator geri bildirim, OPC-UA/SCADA

---

## 2. Sistemi Calistirma Rehberi

Sistemin 3 farkli calisma modu vardir. Amacina gore birini sec:

### 2.1 Streamlit Dashboard (Demo & Ogrenme Modu)

**Amac:** Sistemin nasil calistigini gormek, anlamak ve test etmek isteyen icin.
Goruntu yukle, YOLO sonuclarini gor, VLM'i test et, spatial logic'i anla.
9 sayfalik interaktif demo arayuzu.

```bash
streamlit run src/ui/sprint1_dashboard.py
# Tarayici: http://localhost:8501
```

**Sayfalar:**
1. **Canli Analiz** — Goruntu yukle, YOLO + Edge Enhancement + Spatial + VLM sonuclarini gor
2. **Veri Dengeleme** — Sinif dagilimi grafikleri (V1/V3/V4 karsilastirma)
3. **Neden CA?** — Coordinate Attention mimarisinin teknik aciklamasi
4. **MLflow Takibi** — Egitim deneyleri ve metrik karsilastirmalari
5. **Edge Profiler** — Jetson Orin Nano gecikme ve bellek tahminleri
6. **VLM Merkezi** — VLM stratejisi, anomali galerisi, performans metrikleri
7. **FP Analizi** — False positive inceleme ve Active Learning istatistikleri
8. **Karar Tablosu** — Ozet KPI snapshot'i
9. **Operator Kontrol** — VLM yukle/kaldir, acil durdurma, model yeniden yukle

**Ne zaman kullan:** Sistemi tanimak, birine gostermek, tek goruntu uzerinde test etmek istediginde.

---

### 2.2 Production HMI — Pipeline AKTIF (Fabrika Modu)

**Amac:** Gercek fabrika uretim hattinda canli calisma. Kamera baglanir, model yuklenir,
surekli inference dongusu baslar. Her kareyi otomatik isler, karar verir, loglar.

```bash
python src/ui/production_hmi.py
# Tarayici: http://localhost:8080
```

**Bu mod calistiginda:**
- Kamera otomatik baglanir (USB/RTSP/GigE — `configs/camera_config.yaml`'dan)
- YOLO modeli yuklenir (`models/phase1_final_ca.pt` veya ONNX)
- Inference dongusu baslar (surekli, frame by frame)
- Spatial Logic + Conflict Resolver + VLM trigger aktif
- MQTT/OPC-UA uzerinden fabrika sistemlerine veri gonderir
- Audit log tutar (her karar kaydedilir)
- Vardiya istatistikleri otomatik hesaplanir
- Operator web arayuzunden geri bildirim verebilir

**Ozel konfigurasyon ile:**
```bash
python src/ui/production_hmi.py --config configs/production_config.yaml --host 0.0.0.0 --port 8080
```

**Web arayuzunde gorunenler:**
- Son karar (OK/NOK) buyuk gosterge
- Toplam denetim, OK/NOK sayaclari
- Kalite orani (%) ve gorsel bar
- Vardiya bilgisi (Sabah/Aksam/Gece) + PPM
- Sistem sagligi (GPU, disk, pipeline durumu)
- Son 10 karar tablosu
- Operator geri bildirim butonlari (Dogru / Kismi Dogru / Yanlis)

**API Endpoints:**
| Endpoint | Metod | Aciklama |
|----------|-------|----------|
| `/` | GET | Ana sayfa (HTML) |
| `/api/stats` | GET | Pipeline istatistikleri (JSON) |
| `/api/shift` | GET | Vardiya raporu |
| `/api/health` | GET | Sistem sagligi (GPU, disk, pipeline) |
| `/api/recent?n=10` | GET | Son N karar |
| `/api/feedback` | POST | Operator geri bildirimi kaydet |
| `/api/model/swap` | POST | Model degistir (hot-swap) |

---

### 2.3 Production HMI — Pipeline DEVRE DISI (Gelistirme Modu)

**Amac:** Sadece web arayuzunu gormek ve test etmek. Kamera, model, inference hicbiri calismaz.
API'ler bos/varsayilan deger dondurur.

```bash
python src/ui/production_hmi.py --no-pipeline
# Tarayici: http://localhost:8080
```

**Bu mod calistiginda:**
- Web arayuzu acilir, tasarimini gorebilirsin
- Kamera baglanmaz, model yuklenmez
- Stats, shift, recent hep bos/sifir doner
- Health endpoint sadece disk ve GPU kontrolu yapar
- Feedback endpoint calısır (JSONL'e yazar)

**Ne zaman kullan:** Arayuzu test etmek, CSS/layout denemek, demo gostermek, kamera/GPU olmayan makinede calismak istediginde.

---

### 2.4 Mod Karsilastirmasi

| Ozellik | Streamlit (Demo) | HMI Pipeline Aktif | HMI No-Pipeline |
|---------|-------------------|--------------------|-----------------|
| **Calistirma** | `streamlit run src/ui/sprint1_dashboard.py` | `python src/ui/production_hmi.py` | `python src/ui/production_hmi.py --no-pipeline` |
| **Port** | 8501 | 8080 | 8080 |
| **Kamera** | Yok (dosya yukle) | Canli (USB/RTSP/GigE) | Yok |
| **Model yukleme** | Manuel (sidebar) | Otomatik (config'den) | Yok |
| **Inference** | Tek goruntu (butonla) | Surekli (otomatik dongu) | Yok |
| **VLM** | Manuel tetikleme | Otomatik (conf < 0.40) | Yok |
| **MQTT/OPC-UA** | Yok | Aktif (config'e gore) | Yok |
| **Audit log** | Yok | Aktif (JSONL + SQLite) | Yok |
| **Vardiya** | Yok | Otomatik (saat bazli) | Yok |
| **Geri bildirim** | Dashboard icinde | Web arayuzunde | Web arayuzunde (kayit aktif) |
| **Coklu kullanici** | Tek | Birden fazla (web) | Birden fazla (web) |
| **7/24 uygun** | Hayir | Evet | — |
| **Amac** | Anla, test et, demo goster | Fabrika uretime koy | Arayuzu gelistir/test et |

---

## 3. Sistem Mimarisi

### 3.1 Katman 1 — Hiz Katmani
- **Model:** YOLOv10-S + Coordinate Attention (CA)
- **Konfig:** `configs/models/yolov10s_ca.yaml`
- **Siniflar (nc=3):** screw (0), missing_screw (1), missing_component (2)
- **Oncelik:** Dusuk latency, yuksek throughput

### 3.2 Katman 2 — Bilissel Katman
- **Model:** PaliGemma 3B (google/paligemma-3b-mix-224)
- **Quantization:** NF4 (BitsAndBytes) veya float16 fallback (Windows)
- **Calisma modu:** Asenkron tetikleme (YOLO confidence < 0.40 -> VLM'e gonder)
- **Cikti:** Dogal dilde kusur yorumu + DEFECT_TYPE + REASON formati

### 3.3 Uretim Pipeline Akisi
```
[Kamera] (USB / RTSP / GigE / File)
    |
    v
[YOLO v10-S + CA] (conf_threshold=0.25)
    |
    v
[Spatial Logic] (K-means 4-kumeleme, sol/sag taraf analizi)
    |
    v
[Guven Kontrolu]
    |-- conf >= 0.40 --> [Conflict Resolver] --> Nihai Karar
    |-- conf < 0.40 --> [VLM Kuyrugu] (async, ana hatti yavalatmaz)
    |-- Belirsiz    --> [Uncertain Collector] (frame kaydet)
    |
    v
[MQTT / OPC-UA Yayini] --> PLC / SCADA / MES
    |
    v
[Audit Logger] + [Shift Logger] --> Izlenebilirlik
    |
    v
[Operator HMI] --> Geri Bildirim --> [Active Learning]
```

### 3.4 Conflict Resolution (Catisma Cozumu)
- YOLO + Spatial ayni sonuca ulasirsa -> consensus (VLM atlanir, hiz kazanilir)
- YOLO + Spatial celisirse -> VLM hakem olarak devreye girer (VLM-as-Judge)
- VLM mevcut degilse/basarisizsa -> Spatial kazanir (konservatif, dusuk FP)

---

## 4. Donanim ve Ortam

- **Gelistirme:** Windows 10/11, Python 3.12, RTX 3050 (4GB VRAM)
- **CUDA:** 12.1
- **Hedef Edge:** Jetson Orin Nano (TensorRT FP16/INT8) veya herhangi GPU
- **Donanim bagimsiz:** ONNX Runtime ile CPU, CUDA veya TensorRT uzerinde calisir
- **Kamera bagimsiz:** USB, RTSP (IP kamera), GigE Vision (endustriyel), dosya, mock

### GPU Kurulumu
```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python scripts/check_gpu.py  # Dogrulama
```

---

## 5. Veri Mimarisi

### 5.1 Kaynaklar
| Kaynak | Yol | Aciklama |
|--------|-----|----------|
| Erdogan 1 | `erdogan1/` | Tek sinifli YOLO export (vida-ok), NOK ham goruntuler |
| Erdogan 2 | `erdogan2/MLOps Calismalari/` | NOK/OK fotolar, modeller, kodlar |
| Roboflow | `roboflowetiketlenen/` | COCO format, 897 goruntu, 3 sinif |
| Augmented 1 | `coklanmis/` | aparatsiz->MC, eksik_vida->MS, ok->S, diger->bg |
| Augmented 2 | `coklanmisacili/` | Ek acili augmentation verileri |
| Augmented 3 | `coklanmis1000/` | 1000 ek crop goruntu |
| Augmented 4 | `coklanmisyeni/` | Yeni ek crop goruntuler |

### 5.2 Veri Setleri
| Versiyon | Yol | Train | Val | Test | Aciklama |
|----------|-----|-------|-----|------|----------|
| V1 | `data/processed/phase1_multiclass_v1/` | 717 | 89 | 91 | Roboflow + augmentation |
| V2 | `data/processed/phase1_v2/` | — | — | — | Tum kaynaklardan |
| V4 | `data/processed/phase1_v4_full/` | 3287 | 89 | 91 | V1+erdogan1+tum crop kaynaklari |
| Prod | `data/processed/production_v1/` | 3247 | 619 | 619 | %60/20/20 stratified (uretim baseline) |

### 5.3 Leakage Onleme
- SHA-256 hash tabanli kontrol: `src/data/augment_analysis.py`
- Train duplicate engeli
- val/test ile cakisma engeli
- Val/test'te augmented veri OLMAMALI — sadece orijinal fabrika goruntusleri

---

## 6. Model Egitimleri ve Metrikler

### 6.1 V1 — Sprint 1 Final
- **Sonuc:** mAP50 = 0.9943 (epoch 96)
- **Model:** `models/phase1_final_ca.pt`

### 6.2 V4 — Tum Verilerle
- **Sonuc:** mAP50 = 0.953 (kucuk val seti uzerinde)
- **Gercek baseline (production val=619 goruntu):** mAP50 = 0.918
  - screw: P=0.966, R=0.987
  - missing_screw: P=0.848, R=0.719
  - missing_component: P=0.750, R=0.833

### 6.3 Egitim Komutu
```bash
python scripts/train_final_phase1.py \
    --data data/processed/phase1_v4_full/data.yaml \
    --model-cfg configs/models/yolov10s_ca.yaml \
    --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0
```

---

## 7. Uretim Pipeline Modulleri

### 7.1 Kamera Soyutlama Katmani
**Dosya:** `src/camera/capture.py`

Tum kamera tiplerini tek API ile destekler:
- `USBCamera` — OpenCV VideoCapture (webcam, USB kamera)
- `RTSPCamera` — GStreamer/OpenCV ile IP kamera
- `GigECamera` — pypylon (Basler) veya Harvester/GenICam (endustriyel)
- `FileCamera` — Kayitli goruntu/video dosyalarindan okuma (test icin)
- `MockCamera` — Sahte kamera (unit test icin)

```python
from src.camera.capture import create_camera
cam = create_camera("usb", device_id=0, width=1280, height=1024)
cam.connect()
frame = cam.grab_frame()
cam.release()
```

**Konfigurasyon:** `configs/camera_config.yaml`

### 7.2 Donanim Bagimsiz Model Runner
**Dosya:** `src/pipeline/model_runner.py`

ONNX Runtime ile donanim bagimsiz calistirma:
- `UltralyticsRunner` — PyTorch .pt modeli (gelistirme)
- `ONNXRunner` — ONNX Runtime (TensorRT EP > CUDA EP > CPU otomatik secim)
- `create_model_runner(path)` — Dosya uzantisina gore otomatik secim

```python
from src.pipeline.model_runner import create_model_runner
runner = create_model_runner("models/phase1_final_ca.onnx")  # veya .pt
runner.load()
result = runner.predict(frame, conf=0.25, imgsz=640)
```

### 7.3 ONNX Export
**Dosya:** `scripts/export_onnx.py`

PyTorch modelini ONNX'e donusturur (platform bagimsiz):
```bash
python scripts/export_onnx.py                          # Varsayilan
python scripts/export_onnx.py --model models/phase1_final_ca.pt --half  # FP16
python scripts/export_onnx.py --imgsz 640 --opset 17   # Ozel ayarlar
```

### 7.4 Model Registry
**Dosya:** `src/pipeline/model_registry.py`

Model versiyonlama ve yasam dongusu yonetimi:
- `register_model()` — Yeni model kaydet (SHA256 hash)
- `promote_to_champion()` — Modeli aktif uretim modeline yukselt
- `promote_to_challenger()` — Shadow mode icin challenger olarak ata
- `hot_swap()` — Uretim durmadan model degistir
- `rollback()` — Onceki champion'a geri don

Model durumlari: `staged → challenger → champion → retired`

### 7.5 Inference Pipeline
**Dosya:** `src/pipeline/inference_pipeline.py`

Tum modulleri orkestre eden ana dongu:
```python
from src.pipeline.inference_pipeline import InferencePipeline
pipeline = InferencePipeline.from_config("configs/production_config.yaml")
pipeline.on_event(my_callback)  # Her karar icin callback
pipeline.start()                # Donguyu baslat (ayri thread)
pipeline.stop()                 # Durdur
```

Shadow mode destegi: Champion + challenger model paralel calisir.

### 7.6 Watchdog (Saglik Izleme)
**Dosya:** `src/pipeline/watchdog.py`

- GPU bellek ve sicaklik izleme
- Disk alani kontrolu
- Kamera baglanti durumu
- Heartbeat (5sn goruntu gelmezse alarm)
- MQTT uzerinden saglik durumu yayini (her 30sn)

### 7.7 Audit Logger
**Dosya:** `src/pipeline/audit_logger.py`

Kurumsal izlenebilirlik (FDA 21 CFR Part 11 uyumlu):
- Append-only JSONL dosyalar (gunluk rotasyon)
- SQLite veritabani (sorgulanabilir)
- SHA256 event hash (kurcalama onleme)
- Her inference olayinin tam kaydi

### 7.8 Shift Logger (Vardiya Yonetimi)
**Dosya:** `src/pipeline/shift_logger.py`

- Otomatik vardiya tespiti (Sabah 08-16, Aksam 16-24, Gece 24-08)
- Vardiya bazli OK/NOK sayaclari
- PPM (milyon parcada hata) hesaplama
- JSON rapor dosyalari

### 7.9 OPC-UA Sunucusu
**Dosya:** `src/integration/opcua_server.py`

IEC 62541 uyumlu endustriyel protokol:
- SCADA sistemlerinin (Siemens WinCC, TIA Portal) dogrudan sorgulayabildigi veri
- Yayinlanan degiskenler: LastVerdict, Confidence, TotalInspected, QualityRate, ModelVersion, SystemHealthy
- `opc.tcp://0.0.0.0:4840/edgeagent` adresinde calisir

### 7.10 Surekli Egitim
**Dosya:** `src/mlops/continuous_trainer.py`

Kosul bazli otomatik retrain + shadow deployment:
```bash
python src/mlops/continuous_trainer.py --check    # Retrain gerekli mi?
python src/mlops/continuous_trainer.py --status   # Sistem durumu
python src/mlops/continuous_trainer.py --run      # Retrain dongusunu baslat
```

Tetikleme kosullari:
- 100+ belirsiz kare VEYA 50+ operator duzeltmesi biriktiginde
- VE son retrain'den en az 1 gun gectiginde
- Yeni model "challenger" olarak shadow mode'da test edilir
- Metrikler iyilesirse hot-swap, kotulesirse rollback

---

## 8. Coordinate Attention (CA) Modulu

**Dosya:** `src/models/coordatt.py`

### Calisma Mantigi
- Giris feature map'ini X ve Y ekseni boyunca ayri ayri average pooling yapar
- 1x1 bottleneck convolution ile ortak temsil olusturur
- Iki carpimsal attention haritasi (Ax, Ay) uretir
- Cikis = identity * Ax * Ay (pozisyon duyarli dikkat)

### Kayit (Model yuklenmeden ONCE cagrilmali)
```python
from src.models.coordatt import register_coordatt
register_coordatt()
```

---

## 9. Edge Enhancement (Canny Onisleme)

**Dosya:** `src/data/edge_enhancer.py`

Metal yuzeyler uzerindeki yansimlari bastirmak icin Canny kenar harmanlama:
```python
blended = alpha * original + (1 - alpha) * canny_edges_rgb
```

Domain-Aware Auto-Tune:
```python
from src.data.edge_enhancer import auto_tune_fast
result = auto_tune_fast(img_bgr, yolo_model)
```

---

## 10. Geometrik Mekansal Kumeleme

**Dosya:** `src/reasoning/spatial_logic.py`

YOLO tespitlerini K-Means ile 4 kumeye ayirir, sol/sag taraf analizi yapar:
- 2 screw/taraf = OK
- 1 screw + 1 missing_screw = missing_screw
- 2 missing_screw = missing_component (komponent yok!)

---

## 11. VLM Reasoning Engine (PaliGemma)

**Dosya:** `src/reasoning/vlm_reasoner.py`

```python
from src.reasoning.vlm_reasoner import VLMReasoner
reasoner = VLMReasoner()
reasoner.load_model()   # NF4 veya float16 fallback
result = reasoner.reason(cropped_image_rgb)
```

- Warm Standby: Model VRAM'de yuklu kalir
- Latency: ~3-4sn (RTX 3050)
- HuggingFace token gerekli: `huggingface-cli login`

---

## 12. MLOps Bilesenleri

### 12.1 Active Learning Pipeline
**Dosya:** `src/mlops/active_learning.py`
- Operator geri bildirimini (JSONL) toplar
- Per-detection feedback: her tespit icin dogru/yanlis isareti
- Retrain tetikleyici: 50+ duzeltme + 14+ gun

### 12.2 Concept Drift Detector
**Dosya:** `src/mlops/drift_detector.py`
- SSIM tabanli goruntu dagilim izleme
- SSIM dususu > %15 = retrain onerisi

### 12.3 MLflow
```bash
mlflow ui --port 5000
# Tarayici: http://localhost:5000
```

---

## 13. Edge Deployment

### 13.1 TensorRT Export
```bash
python scripts/export_tensorrt.py --half          # FP16
python scripts/export_tensorrt.py --int8 --half   # INT8
```

### 13.2 ONNX Export (Platform Bagimsiz)
```bash
python scripts/export_onnx.py --model models/phase1_final_ca.pt --half
```

### 13.3 MQTT Bridge
**Dosya:** `src/edge/mqtt_bridge.py`
- Topic: `edgeagent/factory/*`
- QoS: 1, simulasyon modu destegi

---

## 14. Konfigurasyon Dosyalari

| Dosya | Aciklama |
|-------|----------|
| `configs/models/yolov10s_ca.yaml` | YOLOv10-S + CA model mimarisi |
| `configs/phase2_config.yaml` | VLM, queue, drift, AL, MQTT ayarlari |
| `configs/production_config.yaml` | Uretim yapilandirmasi (kamera, model, pipeline, MQTT, OPC-UA, watchdog, vardiya) |
| `configs/camera_config.yaml` | Kamera ayarlari (tip, cozunurluk, FPS, trigger modu) |
| `configs/rules.yaml` | Urun bazli dinamik kurallar |

---

## 15. Dosya Yapisi

```
Goruntuisleme/
|
|-- configs/
|   |-- models/yolov10s_ca.yaml           # YOLOv10-S + CA model konfig
|   |-- phase2_config.yaml                # VLM, queue, drift, AL, MQTT
|   |-- production_config.yaml            # Uretim yapilandirmasi
|   |-- camera_config.yaml                # Kamera ayarlari
|   |-- rules.yaml                        # Dinamik kurallar
|
|-- scripts/
|   |-- check_gpu.py                      # GPU/CUDA dogrulama
|   |-- export_tensorrt.py                # TensorRT FP16/INT8 export
|   |-- export_onnx.py                    # ONNX export (platform bagimsiz)
|   |-- prepare_phase1_dataset.py         # Roboflow COCO -> YOLO donusum
|   |-- prepare_v2_dataset.py             # V2 dataset
|   |-- prepare_v4_dataset.py             # V4 dataset (tum kaynaklar)
|   |-- prepare_production_dataset.py     # Uretim dataseti (%60/20/20)
|   |-- prepare_v3_copypaste.py           # Copy-paste augmentation
|   |-- train_final_phase1.py             # YOLO egitim + rapor
|   |-- generate_vlm_captions.py          # VLM caption uretimi
|   |-- backfill_mlflow.py                # MLflow gecmis kayit
|   |-- start_mlflow_server.py            # MLflow sunucusu
|
|-- src/
|   |-- camera/
|   |   |-- capture.py                    # Kamera soyutlama (USB/RTSP/GigE/File/Mock)
|   |
|   |-- pipeline/
|   |   |-- inference_pipeline.py         # End-to-end inference dongusu
|   |   |-- model_runner.py               # Donanim bagimsiz model (ONNX/TensorRT/PT)
|   |   |-- model_registry.py             # Model versiyonlama + hot-swap
|   |   |-- watchdog.py                   # GPU/disk/kamera saglik izleme
|   |   |-- audit_logger.py               # Kurumsal audit trail (JSONL+SQLite)
|   |   |-- shift_logger.py               # Vardiya yonetimi + PPM raporlama
|   |
|   |-- integration/
|   |   |-- opcua_server.py               # OPC-UA sunucusu (SCADA entegrasyonu)
|   |
|   |-- data/
|   |   |-- augment_analysis.py           # Augmentation ingestion + leakage check
|   |   |-- edge_enhancer.py              # Canny enhancement + auto-tune
|   |   |-- label_validator.py            # Etiket dogrulama
|   |   |-- uncertain_collector.py        # Belirsiz kare toplama
|   |   |-- vlm_augmentor.py              # VLM tabanli augmentation
|   |
|   |-- edge/
|   |   |-- profiler.py                   # Jetson Orin Nano latency tahmini
|   |   |-- vlm_trigger.py               # VLM async tetikleme
|   |   |-- mqtt_bridge.py               # MQTT IoT entegrasyonu
|   |
|   |-- models/
|   |   |-- coordatt.py                   # Coordinate Attention modulu
|   |
|   |-- reasoning/
|   |   |-- vlm_reasoner.py               # PaliGemma VLM inference engine
|   |   |-- spatial_logic.py              # Geometrik mekansal kumeleme
|   |   |-- conflict_resolver.py          # YOLO vs Spatial vs VLM arbitrasyon
|   |   |-- rca_templates.py              # 10 Turkce kok neden sablonu
|   |   |-- dynamic_rules.py             # Dinamik urun kurallari
|   |
|   |-- mlops/
|   |   |-- active_learning.py            # Operator feedback + retrain
|   |   |-- drift_detector.py             # SSIM concept drift
|   |   |-- continuous_trainer.py         # Shadow deployment + otomatik retrain
|   |
|   |-- evaluation/
|   |   |-- accuracy_report.py            # Model degerlendirme raporu
|   |
|   |-- agent/                            # Agent chat modulleri
|   |
|   |-- ui/
|       |-- sprint1_dashboard.py          # Streamlit demo dashboard (9 sayfa)
|       |-- production_hmi.py             # FastAPI uretim arayuzu
|
|-- models/
|   |-- phase1_final_ca.pt                # V1 best (mAP50=0.9943)
|   |-- registry/                         # Model versiyonlama kayitlari
|
|-- data/
|   |-- processed/                        # Islenmmis YOLO dataset'leri
|   |-- feedback/feedback_log.jsonl       # Operator geri bildirim kayitlari
|   |-- audit/                            # Audit log dosyalari
|
|-- reports/
|   |-- final_phase1_report.md            # Karsilastirma raporu
```

---

## 16. Kurulum

### 16.1 Temel Kurulum
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python scripts/check_gpu.py   # GPU dogrulama
```

### 16.2 Bagimliliklarin Ozeti
| Paket | Amac |
|-------|------|
| ultralytics >= 8.4.0 | YOLOv10 egitim/inference |
| torch == 2.5.1+cu121 | PyTorch (CUDA destekli) |
| transformers >= 4.40 | PaliGemma VLM |
| onnxruntime-gpu >= 1.17 | ONNX Runtime (donanim bagimsiz) |
| fastapi >= 0.111 | Uretim HMI backend |
| uvicorn >= 0.30 | ASGI server |
| asyncua >= 1.1 | OPC-UA sunucusu |
| paho-mqtt >= 2.0 | MQTT IoT iletisimi |
| mlflow >= 2.13 | Deney takibi |
| streamlit >= 1.55 | Demo dashboard |

---

## 17. Fabrikadaki Mevcut Uygulama vs Bizim Sistem

`erdogan2/MLOps Calismalari/Codes/` altinda fabrikada kullanilan uygulama var:

### Fabrikadaki Exe (model-deployment.exe)
- **GUI:** Tkinter (600x400 pencere)
- **Ne yapiyor:** Kullanici model (.pt) secer, goruntu klasoru secer, toplu inference yapar, annotated goruntu kaydeder
- **Calisma modu:** Offline batch — fotograf cek, exe ac, isle, sonuclari gor
- **Sinif:** Tek sinif (vida-ok)
- **Kamera:** Yok (dosyadan okur)
- **PyInstaller ile .exe'ye paketlenmis** (Python 3.11 + PyTorch 2.5.1+cu121 gomulu)

### Karsilastirma
| Ozellik | Fabrikadaki Exe | Bizim EdgeAgent |
|---------|-----------------|-----------------|
| **Calisma** | Offline batch | **Canli** (surekli kamera akisi) |
| **Siniflar** | 1 (vida-ok) | 3 (screw, missing_screw, missing_component) |
| **Karar** | Sadece YOLO bbox ciz | YOLO + Spatial + VLM + Conflict Resolver |
| **Kamera** | Yok | USB/RTSP/GigE (herhangi bir kamera) |
| **Donanim** | Sadece PyTorch .pt | ONNX + TensorRT + PyTorch (bagimsiz) |
| **Geri bildirim** | Yok | Active Learning + operator feedback |
| **Fabrika entegrasyon** | Yok | MQTT + OPC-UA + audit trail |
| **Model guncelleme** | Manuel | Shadow deployment + hot-swap |
| **Raporlama** | Yok | Vardiya, PPM, kalite orani |
| **7/24** | Uygun degil | Uygun (watchdog, reconnect) |

---

## 18. Teknik Kararlar

### Neden YOLOv10-S + Coordinate Attention?
- v10 NMS-free = dusuk latency, CA pozisyon bilgisi korur (vida lokasyonu icin kritik)

### Neden ONNX Runtime?
- Tek model dosyasi her yerde calisir (CPU, CUDA, TensorRT EP)
- Donanim bagimsizligi: fabrika hangi GPU'yu kullanirsa kullansin

### Neden FastAPI (Streamlit degil)?
- Streamlit: bellek sizintisi (7/24 uygun degil), tek kullanici, demo icin
- FastAPI: coklu kullanici, 7/24 calisir, API destegi, dokunmatik ekrana uygun

### Neden Shadow Deployment?
- Yeni model uretimi durdurmadan test edilir
- Champion hep kontrol eder, challenger sadece gozlem yapar
- Metrikler iyilesirse atomik hot-swap, kotulesirse rollback

### Neden OPC-UA?
- Kurumsal fabrikalarin standardi (IEC 62541)
- Siemens, Beckhoff, Bosch gibi firmalar OPC-UA kullanir
- MQTT kalir (IoT, dashboard icin) ama OPC-UA ana uretim protokolu

---

## 19. Sorun Giderme

| Sorun | Cozum |
|-------|-------|
| `cuda_available: False` | GPU driver guncelle, CUDA 12.1 wheel yukle |
| `ModuleNotFoundError: src` | Proje kokunden calistir |
| `CoordAtt` hatasi | `register_coordatt()` model yuklenmeden once cagrilmali |
| MLflow UI acilmiyor | `pip install mlflow>=2.13`, port cakismasi kontrol |
| VLM OOM | Model unload et, batch size dusur |
| HMI acilmiyor | `pip install fastapi uvicorn`, port 8080 musait mi kontrol et |
| ONNX export hatasi | `register_coordatt()` export oncesi cagrilmali |
| Kamera baglanmiyor | `configs/camera_config.yaml`'da dogru tip ve device_id sec |
| HF token hatasi | `huggingface-cli login` veya `python -c "from huggingface_hub import login; login()"` |

---

## 20. Proje Durumu

| Faz | Durum | Aciklama |
|-----|-------|----------|
| Faz 0: Veri Temeli | TAMAMLANDI | %60/20/20 production dataset, gercek baseline mAP50=0.918 |
| Faz 1: Uretim Pipeline | TAMAMLANDI | Kamera, model runner, inference pipeline, registry, watchdog |
| Faz 2: Kurumsal Entegrasyon | TAMAMLANDI | Audit trail, OPC-UA, vardiya, HMI |
| Faz 3: Shadow Mode Dogrulama | BEKLIYOR | Gercek bantta 7 gun shadow calistirma |
| Faz 4: Canli Uretim | BEKLIYOR | Reject mekanizmasi aktif, operator gozetimli |
| Faz 5: Surekli Iyilestirme | BEKLIYOR | Drift algilama + otomatik retrain canli |

---

## 21. Gitignore'daki Buyuk Dosyalar

Repo'da YER ALMAYAN buyuk/yerel dosyalar:
- `erdogan1/`, `erdogan2/`, `coklanmis/`, `coklanmisacili/`, `coklanmis1000/`, `coklanmisyeni/`, `roboflowetiketlenen/`
- `data/processed/`, `data/augmented/`, `data/audit/`
- `runs/`, `mlruns/`
- `*.pt`, `*.onnx`, `*.engine`
- `reports/generated/`
