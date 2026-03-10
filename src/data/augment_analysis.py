import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_ID_TO_NAME = {
    0: "screw",
    1: "missing_screw",
    2: "missing_component",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and ingest augmented data into phase1 train split"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "coklanmis",
        help="Folder that contains aparatsiz/eksik_vida/vida(ok)/diger",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1",
        help="Canonical YOLO dataset root (train/val/test)",
    )
    parser.add_argument(
        "--max-background",
        type=int,
        default=2500,
        help="Hard cap for background images to ingest from 'diger'",
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=1.5,
        help="Background cap ratio relative to positive candidates",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def folder_to_class_id(folder_name: str):
    key = normalize_name(folder_name)
    class_map = {
        "aparatsiz": 2,
        "eksik_vida": 1,
        "eksikvida": 1,
        "vida": 0,
        "ok": 0,
    }
    background_keys = {"diger", "background", "arka_plan", "arkaplan"}
    if key in class_map:
        return class_map[key]
    if key in background_keys:
        return None
    return "ignore"


def iter_images(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_split_hashes(dataset_dir: Path):
    train_hashes = set()
    valtest_hashes = set()
    split_image_counts = {}

    for split in ["train", "val", "test"]:
        image_dir = dataset_dir / split / "images"
        imgs = list(iter_images(image_dir))
        split_image_counts[split] = len(imgs)
        for p in imgs:
            h = md5_file(p)
            if split == "train":
                train_hashes.add(h)
            else:
                valtest_hashes.add(h)

    return train_hashes, valtest_hashes, split_image_counts


def read_label_distribution(labels_dir: Path):
    instance_counts = Counter({0: 0, 1: 0, 2: 0})
    image_presence_counts = Counter({0: 0, 1: 0, 2: 0})
    background_images = 0
    total_images = 0

    for txt in labels_dir.glob("*.txt"):
        total_images += 1
        raw_lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not raw_lines:
            background_images += 1
            continue

        seen_in_image = set()
        for ln in raw_lines:
            cls = int(float(ln.split()[0]))
            if cls in instance_counts:
                instance_counts[cls] += 1
                seen_in_image.add(cls)

        for cls in seen_in_image:
            image_presence_counts[cls] += 1

    return {
        "total_images": total_images,
        "background_images": background_images,
        "instance_counts": dict(instance_counts),
        "image_presence_counts": dict(image_presence_counts),
    }


def make_safe_stem(folder_name: str, source_stem: str, file_hash: str) -> str:
    raw = f"aug_{normalize_name(folder_name)}_{source_stem}".replace("__", "_")
    raw = raw.replace(" ", "_")
    if len(raw) > 96:
        raw = raw[:96]
    return f"{raw}_{file_hash[:10]}"


def write_bar_chart(before: dict, after: dict, out_png: Path) -> None:
    cls_order = [0, 1, 2]
    cls_labels = [CLASS_ID_TO_NAME[i] for i in cls_order]

    before_instances = [before["instance_counts"].get(str(i), before["instance_counts"].get(i, 0)) for i in cls_order]
    after_instances = [after["instance_counts"].get(str(i), after["instance_counts"].get(i, 0)) for i in cls_order]

    before_img_presence = [
        before["image_presence_counts"].get(str(i), before["image_presence_counts"].get(i, 0))
        for i in cls_order
    ]
    after_img_presence = [
        after["image_presence_counts"].get(str(i), after["image_presence_counts"].get(i, 0))
        for i in cls_order
    ]

    x = range(len(cls_labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)

    axes[0].bar([i - width / 2 for i in x], before_instances, width=width, label="Before")
    axes[0].bar([i + width / 2 for i in x], after_instances, width=width, label="After")
    axes[0].set_xticks(list(x), cls_labels)
    axes[0].set_title("BBox Instance Counts")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x], before_img_presence, width=width, label="Before")
    axes[1].bar([i + width / 2 for i in x], after_img_presence, width=width, label="After")
    axes[1].set_xticks(list(x), cls_labels)
    axes[1].set_title("Train Images Containing Class")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle("Augmented Data Integration - Imbalance Improvement")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random_gen = random.Random(args.seed)

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {args.source_dir}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")

    train_img_dir = args.dataset_dir / "train" / "images"
    train_lbl_dir = args.dataset_dir / "train" / "labels"
    val_img_dir = args.dataset_dir / "val" / "images"
    test_img_dir = args.dataset_dir / "test" / "images"
    for p in [train_img_dir, train_lbl_dir, val_img_dir, test_img_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Required folder missing: {p}")

    before = read_label_distribution(train_lbl_dir)

    train_hashes, valtest_hashes, split_counts = collect_split_hashes(args.dataset_dir)

    class_files = {0: [], 1: [], 2: []}
    background_files = []
    ignored_folders = []

    for child in sorted(args.source_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        mapped = folder_to_class_id(child.name)
        imgs = list(iter_images(child))
        if mapped == "ignore":
            ignored_folders.append({"folder": child.name, "images": len(imgs)})
            continue
        if mapped is None:
            background_files.extend(imgs)
        else:
            class_files[mapped].extend(imgs)

    positive_candidates = sum(len(v) for v in class_files.values())
    background_cap_from_ratio = int(positive_candidates * args.background_ratio)
    target_bg = min(len(background_files), max(0, background_cap_from_ratio), max(0, args.max_background))

    for k in class_files:
        random_gen.shuffle(class_files[k])
    random_gen.shuffle(background_files)
    background_selected = background_files[:target_bg]

    stats = {
        "added": Counter({"screw": 0, "missing_screw": 0, "missing_component": 0, "background": 0}),
        "skipped_duplicate_train": 0,
        "skipped_duplicate_batch": 0,
        "skipped_leakage_valtest": 0,
    }
    added_hashes = set()

    def process_one(src_img: Path, class_id):
        file_hash = md5_file(src_img)
        if file_hash in valtest_hashes:
            stats["skipped_leakage_valtest"] += 1
            return
        if file_hash in train_hashes:
            stats["skipped_duplicate_train"] += 1
            return
        if file_hash in added_hashes:
            stats["skipped_duplicate_batch"] += 1
            return

        stem = make_safe_stem(src_img.parent.name, src_img.stem, file_hash)
        dst_img = train_img_dir / f"{stem}.jpg"
        dst_lbl = train_lbl_dir / f"{stem}.txt"

        idx = 1
        while dst_img.exists() or dst_lbl.exists():
            dst_img = train_img_dir / f"{stem}_{idx}.jpg"
            dst_lbl = train_lbl_dir / f"{stem}_{idx}.txt"
            idx += 1

        shutil.copy2(src_img, dst_img)
        if class_id is None:
            dst_lbl.write_text("", encoding="utf-8")
            stats["added"]["background"] += 1
        else:
            line = f"{class_id} 0.500000 0.500000 0.980000 0.980000\n"
            dst_lbl.write_text(line, encoding="utf-8")
            stats["added"][CLASS_ID_TO_NAME[class_id]] += 1

        train_hashes.add(file_hash)
        added_hashes.add(file_hash)

    for class_id, files in class_files.items():
        for img in files:
            process_one(img, class_id)

    for img in background_selected:
        process_one(img, None)

    after = read_label_distribution(train_lbl_dir)

    report_dir = ROOT / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = report_dir / f"augmentation_imbalance_{ts}.png"
    json_path = report_dir / f"augmentation_imbalance_{ts}.json"
    latest_json = report_dir / "augmentation_imbalance_latest.json"
    latest_png = report_dir / "augmentation_imbalance_latest.png"

    write_bar_chart(before, after, chart_path)
    shutil.copy2(chart_path, latest_png)

    report = {
        "source_dir": str(args.source_dir),
        "dataset_dir": str(args.dataset_dir),
        "split_image_counts_before": split_counts,
        "before_train_distribution": before,
        "after_train_distribution": after,
        "source_candidates": {
            "screw": len(class_files[0]),
            "missing_screw": len(class_files[1]),
            "missing_component": len(class_files[2]),
            "background_raw": len(background_files),
            "background_selected": len(background_selected),
        },
        "integration_stats": {
            "added": dict(stats["added"]),
            "skipped_duplicate_train": stats["skipped_duplicate_train"],
            "skipped_duplicate_batch": stats["skipped_duplicate_batch"],
            "skipped_leakage_valtest": stats["skipped_leakage_valtest"],
        },
        "ignored_folders": ignored_folders,
        "params": {
            "seed": args.seed,
            "max_background": args.max_background,
            "background_ratio": args.background_ratio,
        },
        "artifacts": {
            "chart": str(chart_path),
            "latest_chart": str(latest_png),
        },
    }

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Augmented data ingestion complete")
    print(f"- source: {args.source_dir}")
    print(f"- dataset: {args.dataset_dir}")
    print(f"- added: {dict(stats['added'])}")
    print(
        "- skipped: "
        f"duplicate_train={stats['skipped_duplicate_train']}, "
        f"duplicate_batch={stats['skipped_duplicate_batch']}, "
        f"leakage_valtest={stats['skipped_leakage_valtest']}"
    )
    print(f"- report json: {json_path}")
    print(f"- report chart: {chart_path}")


if __name__ == "__main__":
    main()
