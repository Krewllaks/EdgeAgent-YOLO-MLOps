# Sprint 1 Execution Plan

Bu plan, buyuk egitim oncesi teknik altyapiyi tamamlama adimlarini listeler.

## A) GPU Hazirlik

1. CUDA 12.1 PyTorch kurulumunu tamamla:
   - `docs/GPU_SETUP_WINDOWS.md`
2. Test et:
   - `python scripts/check_gpu.py`

Beklenen: `cuda_available=True` ve RTX 3050 gorunumu.

## B) Augmented Data + Imbalance Raporu

1. Entegrasyon scripti:
   - `python src/data/augment_analysis.py`
2. Ciktilar:
   - `reports/generated/augmentation_imbalance_latest.json`
   - `reports/generated/augmentation_imbalance_latest.png`

## C) Teknik Dashboard

1. Arayuzu baslat:
   - `streamlit run src/ui/sprint1_dashboard.py`
2. Dashboard icerigi:
   - Sinif dagilimi oncesi/sonrasi
   - CA teknik aciklama
   - Rapor artifact yollari

## D) Final Phase-1 Training (YOLOv10-S + CA)

Komut:

```bash
python scripts/train_final_phase1.py --data data/processed/phase1_multiclass_v1/data.yaml --model-cfg configs/models/yolov10s_ca.yaml --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0
```

Otomatik ciktilar:

- Best weights kopyasi: `models/phase1_final_ca.pt`
- Kiyas raporu: `reports/final_phase1_report.md`
