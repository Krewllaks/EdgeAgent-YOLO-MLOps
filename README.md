# EdgeAgent-YOLO-MLOps

Endustriyel kalite kontrol icin iki katmanli AI + MLOps sistemi.

**Hiz Katmani:** YOLOv10-S + Coordinate Attention (mAP50: 0.9943)
**Bilissel Katman:** PaliGemma 3B VLM — belirsiz tespitlerde dogal dil aciklamasi
**MLOps:** Active Learning, Concept Drift, Operator Feedback, MLflow

---

## Hizli Baslangic

### 1) Kurulum

```bash
git clone <repo-url>
cd Goruntuisleme
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

GPU gorunmuyorsa CUDA 12.1 wheel yukle:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

GPU dogrulama:

```bash
python scripts/check_gpu.py
```

### 2) Dataset Hazirlama

```bash
# V1 (Roboflow COCO -> YOLO)
python scripts/prepare_phase1_dataset.py

# Augmentation entegrasyonu
python src/data/augment_analysis.py --clean-augmented
python src/data/augment_analysis.py --source-dir coklanmis --source-dir coklanmisacili

# V2 tek komut
python scripts/prepare_v2_dataset.py
```

### 3) Egitim

```bash
python scripts/train_final_phase1.py \
    --data data/processed/phase1_multiclass_v1/data.yaml \
    --model-cfg configs/models/yolov10s_ca.yaml \
    --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0
```

Egitim bitince otomatik olarak:
- Best model -> `models/phase1_final_ca.pt`
- Karsilastirma raporu -> `reports/final_phase1_report.md`

### 4) Dashboard

```bash
streamlit run src/ui/sprint1_dashboard.py
```

### 5) MLflow

```bash
mlflow ui --port 5000
```

Tarayicida ac: [http://localhost:5000](http://localhost:5000)

Tum egitim deneyleri, hyperparametreler ve metrikler burada goruntulenir.
`mlruns/` klasoru `.gitignore`'da oldugu icin her takim uyesi kendi yerel MLflow veritabanini olusturur.

### 6) VLM (PaliGemma) Kurulumu

```bash
pip install huggingface_hub
python -c "from huggingface_hub import login; login()"
```

Token'i girdikten sonra dashboard uzerinden "VLM Yukle" butonuyla model VRAM'e alinir.

### 7) Edge Profiling

```bash
# Jetson Orin Nano latency tahmini
python src/edge/profiler.py --model models/phase1_final_ca.pt \
    --source data/processed/phase1_multiclass_v1/test/images

# VLM tetikleme simulasyonu
python src/edge/vlm_trigger.py --model models/phase1_final_ca.pt \
    --source data/processed/phase1_multiclass_v1/test/images --conf-threshold 0.40

# TensorRT export
python scripts/export_tensorrt.py --half          # FP16
python scripts/export_tensorrt.py --int8 --half   # INT8
```

---

## Sistem Mimarisi

```
Kamera Frame --> [YOLO + CA] --OK--> Dashboard
                     |
                     | conf < 0.40
                     v
               [Async Queue] --> [PaliGemma VLM] --> Yorum
                     |
                     v
            [Conflict Resolver] --> YOLO vs Spatial vs VLM
                     |
                     v
              [RCA Template] --> Turkce neden analizi
                     |
                     v
               [Operator Feedback] --> Active Learning --> Retrain
```

---

## Sprint 1 Sonuclari

| Metrik | Baseline | Final | Delta |
|--------|----------|-------|-------|
| mAP50(B) | 0.4942 | 0.9942 | +0.5000 |

- Model: YOLOv10-S + Coordinate Attention
- 3 sinif: `screw`, `missing_screw`, `missing_component`
- Train: 4174 goruntu, 3180 bbox

---

## Donanim Gereksinimleri

- OS: Windows 10/11
- Python: 3.10+
- GPU: NVIDIA RTX 3050 (4 GB VRAM)
- CUDA: 12.1

VRAM butcesi:
- YOLO: ~500 MB
- PaliGemma float16: ~2800 MB
- Toplam: ~3700 MB

---

## Repository Yapisi

```
src/
  data/
    augment_analysis.py       # Augmentation + leakage check
    edge_enhancer.py          # Canny enhancement + domain-aware auto-tune
  edge/
    profiler.py               # Jetson latency tahmini
    vlm_trigger.py            # VLM async tetikleme
    mqtt_bridge.py            # MQTT IoT bridge
  models/
    coordatt.py               # Coordinate Attention (tek kaynak)
  reasoning/
    vlm_reasoner.py           # PaliGemma VLM engine
    spatial_logic.py          # Geometrik mekansal kumeleme
    conflict_resolver.py      # YOLO vs Spatial vs VLM arbitrasyon
    rca_templates.py          # 10 Turkce RCA sablonu
  mlops/
    active_learning.py        # Operator feedback -> retrain
    drift_detector.py         # SSIM concept drift
  ui/
    sprint1_dashboard.py      # Streamlit dashboard (9 sayfa)

scripts/
  check_gpu.py                # GPU dogrulama
  train_final_phase1.py       # Final egitim + rapor
  prepare_phase1_dataset.py   # COCO -> YOLO donusum
  prepare_v2_dataset.py       # V2 dataset
  export_tensorrt.py          # TensorRT FP16/INT8
  generate_vlm_captions.py    # VLM caption uretimi

configs/
  models/yolov10s_ca.yaml     # Model mimari konfig
  phase2_config.yaml          # VLM, queue, drift, AL, MQTT
```

---

## Dashboard Sayfalari

1. **Canli Analiz** — 3 sutun: YOLO Tespit | Edge Enhancement | Mekansal Kumeleme + VLM
2. **Veri Dengeleme** — Sinif dagilimi oncesi/sonrasi
3. **Neden CA?** — Coordinate Attention aciklama
4. **MLflow Takibi** — Deney metrikleri
5. **Edge Profiler** — Jetson Orin Nano tahminleri
6. **VLM Merkezi** — Strateji, Anomali Galerisi, Performans Metrikleri
7. **FP Analizi** — False positive + Active Learning istatistikleri
8. **Karar Tablosu** — Ozet KPI'lar
9. **Operator Kontrol** — VLM yukle/kaldir, acil durdurma

---

## Detayli Bilgi

Tum teknik detaylar, kod aciklamalari, konfigurasyon parametreleri ve sorun giderme icin:

**[goruntuislemebilgi.md](goruntuislemebilgi.md)**
