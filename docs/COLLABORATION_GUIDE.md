# Collaboration Guide - Projeyi Ayaga Kaldirma

Bu rehber, projeye yeni katilan takim uyesinin yerel ortamda projeyi
calistirmasini ve MLflow sayfasini ayaga kaldirmasini anlatir.

---

## 1. Repoyu Klonla

```bash
git clone <repo-url>
cd Goruntuisleme
```

## 2. Python Ortami Kur

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

## 3. Bagimliliklari Yukle

```bash
pip install -r requirements.txt
```

**Not:** Windows'ta CUDA 12.1 PyTorch otomatik yuklenir. Eger GPU
gorunmuyorsa:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

GPU dogrulamasi:

```bash
python scripts/check_gpu.py
```

Beklenen cikti: `cuda_available: True`, GPU adi ve matmul testi basarili.

## 4. MLflow Kurulumu

MLflow requirements.txt icerisinde yer alir. Eger eksikse:

```bash
pip install mlflow>=2.13
```

### MLflow UI Baslatma

```bash
mlflow ui --port 5000
```

Tarayicida ac: `http://localhost:5000`

Burada tum egitim deneyleri, hyperparametreler ve metrikler gorunur.

### MLflow Experiment Yapisi

```
mlruns/
  0/           <- Default experiment
    <run-id>/
      params/   <- epoch, batch, imgsz, lr...
      metrics/  <- mAP50, loss, precision, recall
      artifacts/ <- model checkpoints, plots
```

**Not:** `mlruns/` klasoru `.gitignore`'da oldugu icin git'e eklenmez.
Her takim uyesi kendi yerel MLflow veritabanini olusturur.

## 5. Dataset Hazirlama

Dataset dosyalari buyuk oldugu icin git'te yer almaz. Asagidaki
klasorleri proje kokune yerlestiriniz:

```
Goruntuisleme/
  erdogan1/          <- Ham etiketli veri (kaynak 1)
  erdogan2/          <- Ham etiketli veri (kaynak 2)
  roboflowetiketlenen/  <- Roboflow export
  coklanmis/         <- Augmented veriler
```

Sonra canonical YOLO formatina donusturun:

```bash
python scripts/prepare_phase1_dataset.py
```

Bu komut `data/processed/phase1_multiclass_v1/` altinda train/val/test
split'ini olusturur.

## 6. Augmentation Analizi

```bash
python src/data/augment_analysis.py
```

Ciktilar:
- `reports/generated/augmentation_imbalance_latest.json`
- `reports/generated/augmentation_imbalance_latest.png`

## 7. Dashboard Calistirma

```bash
streamlit run src/ui/sprint1_dashboard.py
```

Dashboard bolumlerinde sunlar yer alir:
- Veri dengeleme KPI'lari
- Neden CA kullandik aciklamasi
- MLflow entegrasyon bilgisi
- Phase 2 VLM tetikleme stratejisi
- Edge profiler sonuclari
- Karar tablosu
- Operator kontrolleri

## 8. Egitimi Tekrar Calistirma (Opsiyonel)

```bash
python scripts/train_final_phase1.py \
    --data data/processed/phase1_multiclass_v1/data.yaml \
    --model-cfg configs/models/yolov10s_ca.yaml \
    --epochs 100 --batch 8 --imgsz 640 --amp --device 0
```

Egitim sonunda:
- `models/phase1_final_ca.pt` -> en iyi model
- `reports/final_phase1_report.md` -> karsilastirma raporu

## 9. Edge Profiler (Phase 2 Hazirlik)

```bash
python src/edge/profiler.py \
    --model models/phase1_final_ca.pt \
    --source data/processed/phase1_multiclass_v1/test/images
```

Jetson Orin Nano latency tahmini ve TensorRT gereksinim raporu uretir.

## 10. VLM Trigger Testi (Phase 2 Hazirlik)

```bash
python src/edge/vlm_trigger.py \
    --model models/phase1_final_ca.pt \
    --source data/processed/phase1_multiclass_v1/test/images \
    --conf-threshold 0.40
```

Dusuk confidence tespitlerinde PaliGemma tetikleme simulasyonu yapar.

---

## Sorun Giderme

| Sorun | Cozum |
|-------|-------|
| `cuda_available: False` | GPU driver guncelle, CUDA 12.1 wheel yukle |
| `ModuleNotFoundError: src` | Proje kokunden calistir: `python -m src.edge.profiler` |
| MLflow UI acilmiyor | `pip install mlflow>=2.13` ve port cakismasi kontrol et |
| Dataset bulunamadi | Klasorleri proje kokune yerlestir, prepare script calistir |
| `CoordAtt` hatasi | `register_coordatt()` cagrisinin model yuklenmeden once yapildigindan emin ol |
