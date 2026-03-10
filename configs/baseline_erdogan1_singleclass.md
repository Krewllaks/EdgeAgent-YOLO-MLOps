# Baseline Dataset (Existing)

Bu baseline, mevcut tek sinifli Roboflow export'u hizli smoke-test icin kullanir.

- Data YAML: `C:/Users/bahti/Desktop/Goruntuisleme/erdogan1/model/data.yaml`
- Sinif: `vida-ok`

Ornek komut:

```bash
python scripts/train_yolo.py --data "C:/Users/bahti/Desktop/Goruntuisleme/erdogan1/model/data.yaml" --model yolov8n.pt --epochs 10
```

> Not: Bu dataset final multiclass hedefini karsilamaz; sadece pipeline dogrulama icin kullanilir.
