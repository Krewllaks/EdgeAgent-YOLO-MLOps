# GPU Setup (Windows, RTX 3050, CUDA 12.1)

Bu adimlar CUDA 12.1 uyumlu PyTorch kurulumunu hedefler.

## 1) Mevcut CPU PyTorch paketlerini temizle

```bash
pip uninstall -y torch torchvision torchaudio
```

## 2) CUDA 12.1 build kur

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## 3) Dogrulama

```bash
python scripts/check_gpu.py
```

Beklenen:

- `cuda_available: True`
- `device[0] name: NVIDIA GeForce RTX 3050 ...`
- `matmul_ok: torch.Size([1024, 1024])`

## Notlar

- `nvidia-smi` komutu driver tarafini gosterir, PyTorch CUDA kullanimi ayrica test edilmelidir.
- `torch.cuda.is_available()` false ise virtual environment ve pip hedefi kontrol edilmelidir.
