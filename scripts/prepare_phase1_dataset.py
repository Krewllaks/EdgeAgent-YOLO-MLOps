import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


TARGET_CLASS_ORDER = ["screw", "missing_screw", "missing_component"]


def coco_bbox_to_yolo(bbox, img_w: float, img_h: float):
    x, y, w, h = [float(v) for v in bbox]
    img_w = float(img_w)
    img_h = float(img_h)
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h

    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare phase1 YOLO dataset from Roboflow COCO export")
    parser.add_argument(
        "--coco-json",
        type=Path,
        default=ROOT / "roboflowetiketlenen" / "train" / "_annotations.coco.json",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "roboflowetiketlenen" / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def ensure_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def main() -> None:
    args = parse_args()
    ensure_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if not args.coco_json.exists():
        raise FileNotFoundError(f"COCO json not found: {args.coco_json}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {args.images_dir}")

    coco = json.loads(args.coco_json.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    cat_name_by_id = {c["id"]: c["name"] for c in categories}
    class_id_by_name = {name: idx for idx, name in enumerate(TARGET_CLASS_ORDER)}

    ann_by_image = defaultdict(list)
    for ann in annotations:
        cat_name = cat_name_by_id.get(ann["category_id"], "")
        if cat_name not in class_id_by_name:
            continue
        ann_by_image[ann["image_id"]].append(ann)

    image_items = sorted(images, key=lambda x: x["file_name"])
    random.Random(args.seed).shuffle(image_items)

    total = len(image_items)
    n_train = int(total * args.train_ratio)
    n_val = int(total * args.val_ratio)
    n_test = total - n_train - n_val

    splits = {
        "train": image_items[:n_train],
        "val": image_items[n_train : n_train + n_val],
        "test": image_items[n_train + n_val :],
    }

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    for split in ["train", "val", "test"]:
        (args.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    class_counter = Counter()
    split_class_counter = {"train": Counter(), "val": Counter(), "test": Counter()}
    split_counter = {}

    for split_name, split_images in splits.items():
        split_counter[split_name] = len(split_images)
        for img in split_images:
            img_name = img["file_name"]
            img_id = img["id"]
            w = img["width"]
            h = img["height"]

            src_img = args.images_dir / img_name
            if not src_img.exists():
                raise FileNotFoundError(f"Image missing in export: {src_img}")

            dst_img = args.output_dir / split_name / "images" / img_name
            shutil.copy2(src_img, dst_img)

            label_path = args.output_dir / split_name / "labels" / f"{Path(img_name).stem}.txt"
            lines = []
            for ann in ann_by_image.get(img_id, []):
                cat_name = cat_name_by_id[ann["category_id"]]
                cls_id = class_id_by_name[cat_name]
                cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], w, h)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                class_counter[cat_name] += 1
                split_class_counter[split_name][cat_name] += 1

            label_path.write_text("\n".join(lines), encoding="utf-8")

    data_yaml = {
        "path": str(args.output_dir).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(TARGET_CLASS_ORDER),
        "names": TARGET_CLASS_ORDER,
    }
    (args.output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    summary = {
        "source_coco": str(args.coco_json),
        "source_images": str(args.images_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "split_sizes": split_counter,
        "class_counts": dict(class_counter),
        "class_counts_by_split": {k: dict(v) for k, v in split_class_counter.items()},
        "target_class_order": TARGET_CLASS_ORDER,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[OK] Phase-1 dataset is ready")
    print(f"- output: {args.output_dir}")
    print(f"- split sizes: {split_counter}")
    print(f"- class counts: {dict(class_counter)}")


if __name__ == "__main__":
    main()
