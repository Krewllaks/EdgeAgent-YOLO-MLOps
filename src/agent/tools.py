"""
Agent Tools — Callable functions for the Orchestrator Agent.

Her tool, agent'in chat uzerinden cagirabilecegi bir pipeline
islemini temsil eder. Hocanin direktifi:
"Sen chat'le sorarsin: 'Benim ne kadar datam var?
Datayi su kadar dataya cokla.' Train et dersin."
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.constants import CLASS_NAMES


# ── Tool Registry ────────────────────────────────────────────────

TOOLS = {}


def register_tool(func):
    """Register a function as an agent tool."""
    TOOLS[func.__name__] = {
        "function": func,
        "description": func.__doc__.strip().split("\n")[0] if func.__doc__ else "",
        "full_doc": func.__doc__ or "",
    }
    return func


# ── Data Analysis Tools ──────────────────────────────────────────


@register_tool
def analyze_data(dataset_path: str = None) -> dict:
    """Veri seti durumunu analiz et: sinif dagilimi, label kalitesi, goruntuler.

    Ornek kullanim: "Benim ne kadar datam var?"
    """
    if dataset_path is None:
        dataset_path = str(ROOT / "data" / "processed" / "phase1_multiclass_v1")

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        return {"error": f"Dataset bulunamadi: {dataset_path}"}

    result = {
        "dataset": dataset_dir.name,
        "path": str(dataset_dir),
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "class_distribution": {},
    }

    for cls_name in CLASS_NAMES.values():
        result["class_distribution"][cls_name] = 0

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists():
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        labels = [p for p in lbl_dir.iterdir() if p.suffix == ".txt"] if lbl_dir.exists() else []

        split_classes = {}
        for cls_name in CLASS_NAMES.values():
            split_classes[cls_name] = 0

        total_boxes = 0
        for lbl_path in labels:
            text = lbl_path.read_text(encoding="utf-8").strip()
            for line in text.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
                    split_classes[cls_name] = split_classes.get(cls_name, 0) + 1
                    result["class_distribution"][cls_name] = result["class_distribution"].get(cls_name, 0) + 1
                    total_boxes += 1

        result["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": total_boxes,
            "class_distribution": split_classes,
        }
        result["total_images"] += len(images)
        result["total_labels"] += len(labels)

    return result


@register_tool
def validate_labels(dataset_path: str = None) -> dict:
    """Label kalitesini kontrol et: weak label ve etiketsiz veri tespiti.

    Ornek kullanim: "Label'lar temiz mi? Egitim oncesi kontrol yap."
    """
    from src.data.label_validator import validate_dataset, check_training_ready

    if dataset_path is None:
        dataset_path = str(ROOT / "data" / "processed" / "phase1_multiclass_v1")

    dataset_dir = Path(dataset_path)
    results = validate_dataset(dataset_dir)

    summary = {
        "dataset": dataset_dir.name,
        "training_ready": check_training_ready(results),
        "splits": {},
    }

    for split, stats in results.items():
        summary["splits"][split] = {
            "total_images": stats.total_images,
            "weak_labeled": stats.weak_labeled_images,
            "unlabeled": stats.unlabeled_images,
            "strong_labeled": stats.strong_labeled_images,
            "weak_boxes": stats.weak_boxes,
            "total_boxes": stats.total_boxes,
        }

    return summary


# ── Augmentation Tools ───────────────────────────────────────────


@register_tool
def augment_data(
    num_images: int = 200,
    strategy: str = "smart",
    dataset_path: str = None,
    output_path: str = None,
) -> dict:
    """VLM-guided augmentation ile yeni labeled egitim verisi uret.

    Ornek kullanim: "Datayi 200 tane daha cogalt."
    Stratejiler: smart (VLM-guided), copypaste (crop yapistirma)
    """
    from src.data.vlm_augmentor import smart_augment_dataset, extract_crops, copy_paste_augment

    if dataset_path is None:
        dataset_path = str(ROOT / "data" / "processed" / "phase1_multiclass_v1")

    dataset_dir = Path(dataset_path)
    source_images = dataset_dir / "train" / "images"
    source_labels = dataset_dir / "train" / "labels"

    if output_path is None:
        output_path = str(ROOT / "data" / "augmented" / f"vlm_{strategy}_{datetime.now().strftime('%Y%m%d')}")

    output_dir = Path(output_path)

    if strategy == "smart":
        result = smart_augment_dataset(
            source_image_dir=source_images,
            source_label_dir=source_labels,
            output_dir=output_dir,
            num_augmented=num_images,
        )
    elif strategy == "copypaste":
        crops = extract_crops(source_images, source_labels)
        result = copy_paste_augment(
            background_dir=source_images,
            crops=crops,
            output_dir=output_dir,
            num_images=num_images,
        )
    else:
        return {"error": f"Bilinmeyen strateji: {strategy}. 'smart' veya 'copypaste' kullanin."}

    return {
        "strategy": result.strategy,
        "total_generated": result.total_generated,
        "output_dir": result.output_dir,
    }


# ── Training Tools ───────────────────────────────────────────────


@register_tool
def train_model(
    data_yaml: str = None,
    epochs: int = 100,
    batch: int = 8,
    imgsz: int = 640,
    device: str = "0",
    name: str = None,
) -> dict:
    """YOLO modelini egit.

    Ornek kullanim: "Modeli yeniden egit" veya "100 epoch ile egit"
    """
    if data_yaml is None:
        data_yaml = str(ROOT / "data" / "processed" / "phase1_multiclass_v1" / "data.yaml")

    if name is None:
        name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Build training command
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_final_phase1.py"),
        "--data", data_yaml,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--imgsz", str(imgsz),
        "--device", device,
        "--name", name,
    ]

    return {
        "command": " ".join(cmd),
        "status": "ready",
        "message": (
            f"Egitim komutu hazir. {epochs} epoch, batch={batch}, imgsz={imgsz}.\n"
            f"Baslatmak icin onay verin."
        ),
        "data_yaml": data_yaml,
        "name": name,
    }


@register_tool
def evaluate_model(
    model_path: str = None,
    data_yaml: str = None,
    split: str = "val",
) -> dict:
    """Model performansini degerlendir: per-class precision/recall/F1.

    Ornek kullanim: "Missing screw accuracy nedir?"
    """
    if model_path is None:
        model_path = str(ROOT / "models" / "phase1_final_ca.pt")

    if data_yaml is None:
        data_yaml = str(ROOT / "data" / "processed" / "phase1_multiclass_v1" / "data.yaml")

    model_p = Path(model_path)
    data_p = Path(data_yaml)

    if not model_p.exists():
        return {"error": f"Model bulunamadi: {model_path}"}
    if not data_p.exists():
        return {"error": f"data.yaml bulunamadi: {data_yaml}"}

    return {
        "status": "ready",
        "message": (
            f"Degerlendirme hazir.\n"
            f"Model: {model_p.name}\n"
            f"Split: {split}\n"
            f"Calistirmak icin:\n"
            f"  python src/evaluation/accuracy_report.py "
            f"--model {model_path} --data {data_yaml} --split {split}"
        ),
        "model": model_path,
        "data": data_yaml,
        "split": split,
    }


# ── Deployment Tools ─────────────────────────────────────────────


@register_tool
def deploy_model(model_path: str = None, half: bool = True, int8: bool = False) -> dict:
    """Modeli TensorRT'ye export et (Jetson deployment icin).

    Ornek kullanim: "Modeli Jetson'a hazirla"
    """
    if model_path is None:
        model_path = str(ROOT / "models" / "phase1_final_ca.pt")

    cmd_parts = [
        sys.executable, str(ROOT / "scripts" / "export_tensorrt.py"),
    ]
    if half:
        cmd_parts.append("--half")
    if int8:
        cmd_parts.append("--int8")
    cmd_parts.append("--simplify")

    return {
        "command": " ".join(cmd_parts),
        "status": "ready",
        "message": (
            f"TensorRT export komutu hazir.\n"
            f"Model: {model_path}\n"
            f"FP16: {half}, INT8: {int8}\n"
            f"Baslatmak icin onay verin."
        ),
    }


@register_tool
def profile_edge(model_path: str = None) -> dict:
    """Edge cihaz (Jetson Orin Nano) icin gecikme ve bellek profillemesi.

    Ornek kullanim: "Jetson'da ne kadar hizli calisir?"
    """
    return {
        "command": f"{sys.executable} {ROOT / 'src' / 'edge' / 'profiler.py'}",
        "status": "ready",
        "message": "Edge profiler hazir. Calistirmak icin onay verin.",
    }


# ── Rule Management Tools ───────────────────────────────────────


@register_tool
def list_rules() -> dict:
    """Mevcut urun denetim kurallarini listele.

    Ornek kullanim: "Hangi urun tipleri tanimli?"
    """
    from src.reasoning.dynamic_rules import RuleEngine

    engine = RuleEngine()
    products = {}
    for name in engine.list_products():
        rule = engine.get_rule(name)
        products[name] = {
            "description": rule.description,
            "total_screws": rule.total_screw_positions,
            "sides": rule.sides,
            "per_side": rule.screws_per_side,
        }

    return {"products": products}


@register_tool
def add_rule(spec: str) -> dict:
    """Yeni urun tipi icin kural olustur.

    Ornek kullanim: "Yeni urun tipi ekle: 6 vidali panel, sol 3 sag 3"
    """
    from src.reasoning.dynamic_rules import RuleEngine

    engine = RuleEngine()
    rule = engine.generate_rule_from_spec(spec)
    engine.add_product(rule)

    return {
        "name": rule.name,
        "description": rule.description,
        "total_screws": rule.total_screw_positions,
        "sides": rule.sides,
        "per_side": rule.screws_per_side,
        "saved": True,
    }


# ── Feedback Tools ───────────────────────────────────────────────


@register_tool
def check_feedback() -> dict:
    """Operator geri bildirimlerini analiz et, retrain gerekli mi kontrol et.

    Ornek kullanim: "Son geri bildirimleri analiz et"
    """
    feedback_path = ROOT / "data" / "feedback" / "feedback_log.jsonl"

    if not feedback_path.exists():
        return {"error": "Geri bildirim dosyasi bulunamadi", "total": 0}

    entries = []
    for line in feedback_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    correct = sum(1 for e in entries if e.get("label") == "correct")
    incorrect = sum(1 for e in entries if e.get("label") == "incorrect")
    partial = sum(1 for e in entries if e.get("label") == "partial")

    corrective = incorrect + partial
    needs_retrain = corrective >= 50

    return {
        "total_feedback": len(entries),
        "correct": correct,
        "incorrect": incorrect,
        "partial": partial,
        "corrective_total": corrective,
        "retrain_threshold": 50,
        "needs_retrain": needs_retrain,
        "recommendation": (
            f"Retrain gerekli! {corrective} duzeltici geri bildirim var (esik: 50)."
            if needs_retrain
            else f"Retrain henuz gerekli degil. {corrective}/50 duzeltici geri bildirim."
        ),
    }


# ── Utility ──────────────────────────────────────────────────────


def get_tool_list() -> list[dict]:
    """Return list of available tools with descriptions."""
    return [
        {"name": name, "description": info["description"]}
        for name, info in TOOLS.items()
    ]


def call_tool(tool_name: str, **kwargs) -> dict:
    """Call a tool by name with given arguments."""
    if tool_name not in TOOLS:
        return {"error": f"Bilinmeyen tool: {tool_name}. Mevcut: {list(TOOLS.keys())}"}

    try:
        func = TOOLS[tool_name]["function"]
        result = func(**kwargs)
        return result
    except Exception as e:
        return {"error": f"Tool hatasi ({tool_name}): {str(e)}"}
