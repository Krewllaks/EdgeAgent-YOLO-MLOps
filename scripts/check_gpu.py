import platform

import torch


def main() -> None:
    print("=== Environment ===")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")

    print("\n=== CUDA Visibility ===")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("[WARN] CUDA gorunmuyor. CUDA destekli PyTorch kurulumunu kontrol edin.")
        return

    idx = 0
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)

    print(f"device[{idx}] name: {name}")
    print(f"device[{idx}] capability: {cap}")
    print(f"device[{idx}] total_memory_gb: {total_mem_gb:.2f}")

    print("\n=== Quick CUDA Op Test ===")
    x = torch.randn((1024, 1024), device="cuda")
    y = torch.randn((1024, 1024), device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    print(f"matmul_ok: {z.shape}")
    print("[OK] GPU test basarili.")


if __name__ == "__main__":
    main()
