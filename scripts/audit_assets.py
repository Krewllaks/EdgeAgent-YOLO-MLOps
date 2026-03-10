import json
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "generated"


def image_suffix(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def audit_folder(folder: Path) -> dict:
    result = {
        "path": str(folder),
        "exists": folder.exists(),
        "files_total": 0,
        "file_types": {},
        "images_total": 0,
        "image_sizes_top5": [],
        "duplicate_filenames": 0,
        "top_level_breakdown": {},
    }
    if not folder.exists():
        return result

    ext_counter = Counter()
    size_counter = Counter()
    name_counter = Counter()
    top_counter = Counter()

    for p in folder.rglob("*"):
        if not p.is_file():
            continue

        result["files_total"] += 1
        ext_counter[p.suffix.lower()] += 1
        name_counter[p.name] += 1

        rel = p.relative_to(folder)
        top = rel.parts[0] if rel.parts else "."
        top_counter[top] += 1

        if image_suffix(p):
            result["images_total"] += 1
            try:
                with Image.open(p) as im:
                    size_counter[f"{im.width}x{im.height}"] += 1
            except Exception:
                size_counter["unreadable"] += 1

    dupes = [k for k, v in name_counter.items() if v > 1]

    result["file_types"] = dict(ext_counter.most_common())
    result["image_sizes_top5"] = size_counter.most_common(5)
    result["duplicate_filenames"] = len(dupes)
    result["top_level_breakdown"] = dict(top_counter.most_common())
    return result


def main() -> None:
    sources = {
        "erdogan1": ROOT / "erdogan1",
        "erdogan2": ROOT / "erdogan2",
        "roboflowetiketlenen": ROOT / "roboflowetiketlenen",
        "erdogan1_model": ROOT / "erdogan1" / "model",
        "erdogan1_nok": ROOT / "erdogan1" / "NOK fotoğraflar",
        "erdogan2_mlops": ROOT / "erdogan2" / "MLOps Çalışmaları",
    }

    report = {name: audit_folder(path) for name, path in sources.items()}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "data_audit.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Audit report written: {out_path}")
    for name, data in report.items():
        print(
            f"- {name}: exists={data['exists']} files={data['files_total']} images={data['images_total']}"
        )


if __name__ == "__main__":
    main()
