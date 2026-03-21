"""
Per-class accuracy report for EdgeAgent YOLO models.

Hocanın direktifi: "Screw'leri ne kadar doğru tespit etti,
hatalı screw'leri ne kadar doğru tespit etti bunu görmeliyiz."

Bu modül YOLO modelini val/test seti üzerinde çalıştırıp
per-class Precision, Recall, F1 ve confusion matrix üretir.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["screw", "missing_screw", "missing_component"]


# ── Data Structures ──────────────────────────────────────────────


@dataclass
class ClassMetrics:
    class_name: str
    class_id: int
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    support: int = 0  # ground truth count


@dataclass
class AccuracyReport:
    timestamp: str = ""
    model_path: str = ""
    data_path: str = ""
    split: str = "val"
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    total_images: int = 0
    total_gt_boxes: int = 0
    total_pred_boxes: int = 0
    per_class: list = field(default_factory=list)
    confusion_matrix: list = field(default_factory=list)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    mAP50: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# EdgeAgent — Per-Class Accuracy Report",
            "",
            f"- **Tarih:** {self.timestamp}",
            f"- **Model:** `{self.model_path}`",
            f"- **Veri:** `{self.data_path}`",
            f"- **Split:** {self.split}",
            f"- **IoU Eşiği:** {self.iou_threshold}",
            f"- **Confidence Eşiği:** {self.conf_threshold}",
            "",
            "## Genel İstatistikler",
            "",
            f"| Metrik | Değer |",
            f"|--------|-------|",
            f"| Toplam Görüntü | {self.total_images} |",
            f"| Toplam GT Kutu | {self.total_gt_boxes} |",
            f"| Toplam Tahmin Kutu | {self.total_pred_boxes} |",
            f"| mAP50 | {self.mAP50:.4f} |",
            f"| Macro Precision | {self.macro_precision:.4f} |",
            f"| Macro Recall | {self.macro_recall:.4f} |",
            f"| Macro F1 | {self.macro_f1:.4f} |",
            f"| Weighted F1 | {self.weighted_f1:.4f} |",
            "",
            "## Per-Class Metrikler",
            "",
            "| Sınıf | Precision | Recall | F1 | TP | FP | FN | GT Sayısı |",
            "|-------|-----------|--------|-----|----|----|-----|-----------|",
        ]

        for cm in self.per_class:
            if isinstance(cm, dict):
                lines.append(
                    f"| {cm['class_name']} | {cm['precision']:.4f} | "
                    f"{cm['recall']:.4f} | {cm['f1']:.4f} | "
                    f"{cm['true_positives']} | {cm['false_positives']} | "
                    f"{cm['false_negatives']} | {cm['support']} |"
                )
            else:
                lines.append(
                    f"| {cm.class_name} | {cm.precision:.4f} | "
                    f"{cm.recall:.4f} | {cm.f1:.4f} | "
                    f"{cm.true_positives} | {cm.false_positives} | "
                    f"{cm.false_negatives} | {cm.support} |"
                )

        lines.extend([
            "",
            "## Confusion Matrix",
            "",
            "Satır = Gerçek (GT), Sütun = Tahmin (Pred)",
            "",
        ])

        header = "| | " + " | ".join(CLASS_NAMES) + " | Kaçırılan (FN) |"
        separator = "|---|" + "|".join(["---"] * (len(CLASS_NAMES) + 1)) + "|"
        lines.append(header)
        lines.append(separator)

        cm = self.confusion_matrix
        for i, name in enumerate(CLASS_NAMES):
            if i < len(cm):
                row_vals = [str(int(cm[i][j])) for j in range(len(CLASS_NAMES))]
                fn = sum(cm[i]) - (cm[i][i] if i < len(cm[i]) else 0)
                # FN = GT objects not detected (last column in extended matrix)
                fn_count = cm[i][-1] if len(cm[i]) > len(CLASS_NAMES) else 0
                row_vals.append(str(int(fn_count)))
                lines.append(f"| **{name}** | " + " | ".join(row_vals) + " |")

        lines.extend([
            "",
            "## Kritik Metrikler (Hocanın İstediği)",
            "",
            "| Soru | Cevap |",
            "|------|-------|",
        ])

        for cm_item in self.per_class:
            d = cm_item if isinstance(cm_item, dict) else asdict(cm_item)
            if d["class_name"] == "screw":
                lines.append(
                    f"| Screw'leri ne kadar doğru tespit etti? | "
                    f"Precision: {d['precision']:.1%}, Recall: {d['recall']:.1%} |"
                )
            elif d["class_name"] == "missing_screw":
                lines.append(
                    f"| Hatalı (missing) screw'leri ne kadar doğru tespit etti? | "
                    f"Precision: {d['precision']:.1%}, Recall: {d['recall']:.1%} |"
                )
            elif d["class_name"] == "missing_component":
                lines.append(
                    f"| Eksik komponent tespiti ne kadar doğru? | "
                    f"Precision: {d['precision']:.1%}, Recall: {d['recall']:.1%} |"
                )

        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Markdown rapor: {path}")


# ── IoU Calculation ──────────────────────────────────────────────


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(box: list, img_w: int, img_h: int) -> np.ndarray:
    """Convert YOLO normalized xywh to pixel xyxy."""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.array([x1, y1, x2, y2])


# ── Core Evaluation ─────────────────────────────────────────────


def load_gt_labels(label_dir: Path, image_files: list) -> dict:
    """Load ground truth labels from YOLO format txt files."""
    gt_by_image = {}
    for img_file in image_files:
        stem = Path(img_file).stem
        label_path = label_dir / f"{stem}.txt"
        boxes = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:5]]
                boxes.append({"class_id": class_id, "bbox_xywh": coords})
        gt_by_image[stem] = boxes
    return gt_by_image


def evaluate_predictions(
    gt_by_image: dict,
    pred_by_image: dict,
    num_classes: int = 3,
    iou_threshold: float = 0.5,
    img_size: int = 640,
) -> AccuracyReport:
    """Match predictions to ground truth and compute per-class metrics."""

    # Confusion matrix: rows=GT class, cols=Pred class
    # Extra column for FN (missed), extra row for FP (background pred)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_class_support = defaultdict(int)

    total_gt = 0
    total_pred = 0

    for stem, gt_boxes in gt_by_image.items():
        pred_boxes = pred_by_image.get(stem, [])
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        # Track which GT and Pred boxes are matched
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        # Convert GT to xyxy
        gt_xyxy = []
        for gt in gt_boxes:
            gt_xyxy.append(xywh_to_xyxy(gt["bbox_xywh"], img_size, img_size))
            per_class_support[gt["class_id"]] += 1

        # Convert Pred to xyxy
        pred_xyxy = []
        for pred in pred_boxes:
            pred_xyxy.append(pred["bbox_xyxy"])

        # Match predictions to GT (greedy, highest IoU first)
        iou_pairs = []
        for pi, p_box in enumerate(pred_xyxy):
            for gi, g_box in enumerate(gt_xyxy):
                iou = compute_iou(p_box, g_box)
                if iou >= iou_threshold:
                    iou_pairs.append((iou, pi, gi))

        iou_pairs.sort(key=lambda x: -x[0])  # highest IoU first

        for iou_val, pi, gi in iou_pairs:
            if pred_matched[pi] or gt_matched[gi]:
                continue
            pred_matched[pi] = True
            gt_matched[gi] = True

            pred_cls = pred_boxes[pi]["class_id"]
            gt_cls = gt_boxes[gi]["class_id"]

            if pred_cls == gt_cls:
                per_class_tp[gt_cls] += 1
                cm[gt_cls][pred_cls] += 1
            else:
                # Class mismatch: FP for pred_cls, FN for gt_cls
                per_class_fp[pred_cls] += 1
                per_class_fn[gt_cls] += 1
                cm[gt_cls][pred_cls] += 1

        # Unmatched predictions = FP
        for pi, matched in enumerate(pred_matched):
            if not matched:
                pred_cls = pred_boxes[pi]["class_id"]
                per_class_fp[pred_cls] += 1
                cm[num_classes][pred_cls] += 1  # background row

        # Unmatched GT = FN
        for gi, matched in enumerate(gt_matched):
            if not matched:
                gt_cls = gt_boxes[gi]["class_id"]
                per_class_fn[gt_cls] += 1
                cm[gt_cls][num_classes] += 1  # missed column

    # Build per-class metrics
    class_metrics = []
    for cls_id in range(num_classes):
        tp = per_class_tp[cls_id]
        fp = per_class_fp[cls_id]
        fn = per_class_fn[cls_id]
        support = per_class_support[cls_id]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics.append(ClassMetrics(
            class_name=CLASS_NAMES[cls_id],
            class_id=cls_id,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            support=support,
        ))

    # Macro averages
    valid_classes = [m for m in class_metrics if m.support > 0]
    macro_p = np.mean([m.precision for m in valid_classes]) if valid_classes else 0.0
    macro_r = np.mean([m.recall for m in valid_classes]) if valid_classes else 0.0
    macro_f1 = np.mean([m.f1 for m in valid_classes]) if valid_classes else 0.0

    total_support = sum(m.support for m in valid_classes)
    weighted_f1 = (
        sum(m.f1 * m.support for m in valid_classes) / total_support
        if total_support > 0 else 0.0
    )

    report = AccuracyReport(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        total_images=len(gt_by_image),
        total_gt_boxes=total_gt,
        total_pred_boxes=total_pred,
        per_class=[asdict(m) for m in class_metrics],
        confusion_matrix=cm.tolist(),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
    )

    return report


# ── YOLO Model Evaluation ───────────────────────────────────────


def run_yolo_evaluation(
    model_path: Path,
    data_yaml: Path,
    split: str = "val",
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str = "0",
) -> AccuracyReport:
    """Run YOLO model on val/test split and compute per-class metrics."""
    from src.models.coordatt import register_coordatt
    register_coordatt()

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # Run validation
    results = model.val(
        data=str(data_yaml),
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    # Extract per-class metrics from YOLO results
    report = AccuracyReport(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model_path=str(model_path),
        data_path=str(data_yaml),
        split=split,
        iou_threshold=iou,
        conf_threshold=conf,
    )

    # YOLO results object has per-class metrics
    box_results = results.box
    report.mAP50 = float(box_results.map50)
    report.total_images = int(results.seen) if hasattr(results, "seen") else 0

    # Per-class from YOLO results
    ap_per_class = box_results.ap50  # shape: (num_classes,)
    p_per_class = box_results.p  # shape: (num_classes,) — precision at best conf
    r_per_class = box_results.r  # shape: (num_classes,) — recall at best conf
    f1_per_class = box_results.f1  # shape: (num_classes,)

    # Confusion matrix from YOLO
    if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
        cm_obj = results.confusion_matrix
        cm_matrix = cm_obj.matrix if hasattr(cm_obj, "matrix") else None
        if cm_matrix is not None:
            report.confusion_matrix = cm_matrix.tolist()

    class_metrics = []
    total_gt = 0
    total_pred = 0
    for cls_id in range(len(CLASS_NAMES)):
        if cls_id < len(p_per_class):
            p = float(p_per_class[cls_id])
            r = float(r_per_class[cls_id])
            f1 = float(f1_per_class[cls_id]) if cls_id < len(f1_per_class) else 0.0

            # Extract TP/FP/FN from confusion matrix if available
            tp, fp, fn = 0, 0, 0
            support = 0
            if report.confusion_matrix and cls_id < len(report.confusion_matrix):
                cm_row = report.confusion_matrix[cls_id]
                tp = int(cm_row[cls_id]) if cls_id < len(cm_row) else 0
                # FP = sum of column[cls_id] except diagonal
                fp = sum(
                    int(report.confusion_matrix[r][cls_id])
                    for r in range(len(report.confusion_matrix))
                    if r != cls_id and cls_id < len(report.confusion_matrix[r])
                )
                # FN = sum of row[cls_id] except diagonal
                fn = sum(int(v) for j, v in enumerate(cm_row) if j != cls_id)
                support = tp + fn

            cm = ClassMetrics(
                class_name=CLASS_NAMES[cls_id],
                class_id=cls_id,
                precision=p,
                recall=r,
                f1=f1,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                support=support,
            )
            class_metrics.append(cm)
            total_gt += support
        else:
            class_metrics.append(ClassMetrics(
                class_name=CLASS_NAMES[cls_id], class_id=cls_id,
            ))

    report.per_class = [asdict(m) for m in class_metrics]
    report.total_gt_boxes = total_gt

    # Macro averages
    valid = [m for m in class_metrics if m.support > 0]
    report.macro_precision = float(np.mean([m.precision for m in valid])) if valid else 0.0
    report.macro_recall = float(np.mean([m.recall for m in valid])) if valid else 0.0
    report.macro_f1 = float(np.mean([m.f1 for m in valid])) if valid else 0.0
    total_s = sum(m.support for m in valid)
    report.weighted_f1 = float(
        sum(m.f1 * m.support for m in valid) / total_s
    ) if total_s > 0 else 0.0

    return report


def run_offline_evaluation(
    label_dir: Path,
    pred_dir: Path,
    num_classes: int = 3,
    iou_threshold: float = 0.5,
    img_size: int = 640,
) -> AccuracyReport:
    """Evaluate from saved prediction files (no model needed).

    Useful when predictions are already exported as YOLO-format txt files.
    """
    image_stems = [p.stem for p in label_dir.glob("*.txt")]

    gt_by_image = {}
    for stem in image_stems:
        gt_path = label_dir / f"{stem}.txt"
        boxes = []
        if gt_path.exists():
            for line in gt_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:5]]
                boxes.append({"class_id": class_id, "bbox_xywh": coords})
        gt_by_image[stem] = boxes

    pred_by_image = {}
    for stem in image_stems:
        pred_path = pred_dir / f"{stem}.txt"
        boxes = []
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:5]]
                xyxy = xywh_to_xyxy(coords, img_size, img_size)
                boxes.append({"class_id": class_id, "bbox_xyxy": xyxy})
        pred_by_image[stem] = boxes

    report = evaluate_predictions(gt_by_image, pred_by_image, num_classes, iou_threshold, img_size)
    return report


# ── Summary Printing ─────────────────────────────────────────────


def print_summary(report: AccuracyReport) -> None:
    """Print a concise summary to console."""
    print("\n" + "=" * 60)
    print("  EdgeAgent — Per-Class Accuracy Report")
    print("=" * 60)
    print(f"  Model : {report.model_path}")
    print(f"  Split : {report.split}")
    print(f"  mAP50 : {report.mAP50:.4f}")
    print(f"  Images: {report.total_images}")
    print("-" * 60)
    print(f"  {'Sınıf':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'GT':>6}")
    print("-" * 60)

    for cm in report.per_class:
        d = cm if isinstance(cm, dict) else asdict(cm)
        print(
            f"  {d['class_name']:<20} "
            f"{d['precision']:>10.4f} "
            f"{d['recall']:>10.4f} "
            f"{d['f1']:>10.4f} "
            f"{d['support']:>6}"
        )

    print("-" * 60)
    print(f"  {'Macro Avg':<20} {report.macro_precision:>10.4f} {report.macro_recall:>10.4f} {report.macro_f1:>10.4f}")
    print(f"  {'Weighted F1':<20} {'':>10} {'':>10} {report.weighted_f1:>10.4f}")
    print("=" * 60 + "\n")


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EdgeAgent per-class accuracy evaluation"
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to YOLO model (.pt file)",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1" / "data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports" / "generated",
        help="Output directory for reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    print(f"[INFO] Evaluating {args.model.name} on {args.split} split...")
    report = run_yolo_evaluation(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    print_summary(report)

    # Save reports
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.stem

    md_path = args.output_dir / f"accuracy_{model_name}_{args.split}_{timestamp}.md"
    json_path = args.output_dir / f"accuracy_{model_name}_{args.split}_{timestamp}.json"

    report.to_markdown(md_path)
    report.to_json(json_path)

    print(f"[OK] JSON rapor: {json_path}")


if __name__ == "__main__":
    main()
