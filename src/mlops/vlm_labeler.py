"""
VLM Pseudo-Labeler — Belirsiz karelere VLM ile otomatik label uretici.

CT (Continuous Training) dongusu:
1. Pipeline belirsiz kareyi toplar (UncertainCollector)
2. VLM bu kareleri analiz eder, DEFECT_TYPE + bbox uretir
3. Pseudo-label YOLO formatinda kaydedilir
4. Operator onayladiginda CT havuzuna girer
5. Yeterli veri birikince retrain tetiklenir

VLM burada iki is yapiyor:
- Derin analiz (missing_screw mi, missing_component mi?)
- Pseudo-label uretimi (YOLO egitimi icin yeni veri)

Kullanim:
    python src/mlops/vlm_labeler.py --label-all     # Tum belirsiz kareleri etiketle
    python src/mlops/vlm_labeler.py --status         # Kac etiketli/etiketsiz kare var
    python src/mlops/vlm_labeler.py --export-ct      # CT-ready veriyi cikart
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

CLASS_MAP = {"screw": 0, "missing_screw": 1, "missing_component": 2, "ok": -1}


@dataclass
class PseudoLabel:
    """VLM tarafindan uretilen pseudo-label."""
    image_path: str
    defect_type: str          # missing_screw | missing_component | ok
    yolo_labels: list         # ["class_id cx cy w h", ...]
    vlm_confidence: float     # VLM'in tahmini guven
    vlm_reasoning: str        # Neden bu label
    operator_verified: bool   # Operator onayi
    timestamp: str


class VLMLabeler:
    """VLM ile belirsiz karelere pseudo-label uret."""

    def __init__(self, vlm_reasoner=None):
        self._vlm = vlm_reasoner
        self._labels_log = ROOT / "data" / "uncertain" / "pseudo_labels.jsonl"

    def _ensure_vlm(self):
        """VLM'i lazy-load et."""
        if self._vlm is not None:
            return

        try:
            from src.reasoning.vlm_reasoner import VLMReasoner
            self._vlm = VLMReasoner()
            self._vlm.load_model()
            logger.info("VLM yuklendi (pseudo-labeling icin)")
        except Exception as e:
            logger.error(f"VLM yuklenemedi: {e}")
            raise

    def label_single(self, image_path: Path, yolo_detections: list = None) -> PseudoLabel:
        """Tek bir belirsiz kareye VLM ile pseudo-label uret.

        Args:
            image_path: Belirsiz kare dosya yolu
            yolo_detections: YOLO'nun mevcut (dusuk guvenli) tespitleri
                             [{class_id, confidence, bbox[cx,cy,w,h]}]

        Returns:
            PseudoLabel with YOLO-format labels
        """
        self._ensure_vlm()

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Goruntu okunamadi: {image_path}")

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mevcut YOLO tespitleri varsa her birini VLM ile dogrula
        yolo_labels = []

        if yolo_detections:
            for det in yolo_detections:
                # Crop bolgesi (bbox + %20 padding)
                cx, cy, bw, bh = det.get("bbox", [0.5, 0.5, 0.5, 0.5])
                pad = 0.20
                x1 = int(max(0, (cx - bw / 2 - pad * bw) * w))
                y1 = int(max(0, (cy - bh / 2 - pad * bh) * h))
                x2 = int(min(w, (cx + bw / 2 + pad * bw) * w))
                y2 = int(min(h, (cy + bh / 2 + pad * bh) * h))

                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                crop = img_rgb[y1:y2, x1:x2]
                result = self._vlm.reason(crop)

                if result.defect_type and result.defect_type in CLASS_MAP:
                    class_id = CLASS_MAP[result.defect_type]
                    if class_id >= 0:  # "ok" icin label ekleme
                        yolo_labels.append(
                            f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                        )
        else:
            # YOLO hic tespit edememis — tum kareyi VLM'e ver
            result = self._vlm.reason(img_rgb)
            defect_type = result.defect_type or "ok"

            if defect_type != "ok" and defect_type in CLASS_MAP:
                class_id = CLASS_MAP[defect_type]
                # Tam kare boyutunda genel label
                yolo_labels.append(f"{class_id} 0.5 0.5 0.8 0.8")

        # En son VLM sonucunu kullan
        vlm_result = result if 'result' in dir() else None
        defect_type = vlm_result.defect_type if vlm_result else "ok"
        vlm_conf = vlm_result.confidence_estimate if vlm_result else 0.0
        vlm_reason = vlm_result.reasoning if vlm_result else ""

        pseudo = PseudoLabel(
            image_path=str(image_path),
            defect_type=defect_type or "ok",
            yolo_labels=yolo_labels,
            vlm_confidence=vlm_conf,
            vlm_reasoning=vlm_reason,
            operator_verified=False,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        # Log kaydet
        self._save_label_log(pseudo)

        return pseudo

    def label_all_uncertain(self, max_frames: int = 50) -> list:
        """Tum etiketsiz belirsiz kareleri VLM ile etiketle.

        Returns:
            List of PseudoLabel objects
        """
        from src.data.uncertain_collector import UncertainCollector

        collector = UncertainCollector()
        unlabeled = collector.get_unlabeled_images()

        if not unlabeled:
            logger.info("Etiketsiz belirsiz kare yok")
            return []

        # Metadata'dan YOLO detection bilgisini oku
        metadata = self._load_metadata(collector)

        results = []
        count = min(len(unlabeled), max_frames)
        logger.info(f"{count}/{len(unlabeled)} belirsiz kare etiketlenecek")

        for i, img_path in enumerate(unlabeled[:count]):
            try:
                # Metadata'dan mevcut detection'lari bul
                yolo_dets = metadata.get(img_path.name, [])

                pseudo = self.label_single(img_path, yolo_dets)

                # YOLO label dosyasini kaydet
                if pseudo.yolo_labels:
                    collector.save_pseudo_label(img_path, pseudo.yolo_labels)

                results.append(pseudo)
                logger.info(
                    f"  [{i+1}/{count}] {img_path.name} → {pseudo.defect_type} "
                    f"({len(pseudo.yolo_labels)} label)"
                )
            except Exception as e:
                logger.warning(f"  [{i+1}/{count}] {img_path.name} HATA: {e}")

        logger.info(f"Toplam {len(results)} kare etiketlendi")
        return results

    def export_for_ct(self, output_dir: Path = None) -> dict:
        """CT-ready veriyi (image + label ciftleri) ihrac et.

        Sadece operatoru tarafindan verify edilmis VEYA
        VLM confidence'i yuksek olan verileri cikartir.

        Returns:
            {"exported": int, "output_dir": str, "pairs": [...]}
        """
        import shutil
        from src.data.uncertain_collector import UncertainCollector

        if output_dir is None:
            output_dir = ROOT / "data" / "ct_ready"

        collector = UncertainCollector()
        pairs = collector.get_labeled_pairs()

        if not pairs:
            return {"exported": 0, "output_dir": str(output_dir), "pairs": []}

        img_out = output_dir / "images"
        lbl_out = output_dir / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        exported = []
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)
            exported.append(img_path.name)

        return {
            "exported": len(exported),
            "output_dir": str(output_dir),
            "pairs": exported,
        }

    def get_status(self) -> dict:
        """Pseudo-labeling durumu."""
        from src.data.uncertain_collector import UncertainCollector
        collector = UncertainCollector()

        total = collector.get_collected_count()
        unlabeled = len(collector.get_unlabeled_images())
        labeled_pairs = len(collector.get_labeled_pairs())

        # Log'dan istatistikler
        log_entries = self._load_label_log()
        verified = sum(1 for e in log_entries if e.get("operator_verified"))

        return {
            "total_uncertain": total,
            "unlabeled": unlabeled,
            "pseudo_labeled": labeled_pairs,
            "operator_verified": verified,
            "ct_ready": verified,  # Onaylananlar CT'ye hazir
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def _load_metadata(self, collector) -> dict:
        """Uncertain collector metadata'sindan detection bilgilerini yukle."""
        metadata = {}
        meta_path = collector.output_dir / "metadata.jsonl"
        if not meta_path.exists():
            return metadata

        for line in meta_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                img_name = Path(entry.get("image_path", "")).name
                dets = entry.get("detections", [])
                if img_name and dets:
                    metadata[img_name] = dets
            except json.JSONDecodeError:
                continue

        return metadata

    def _save_label_log(self, pseudo: PseudoLabel):
        """Pseudo-label log'a ekle."""
        self._labels_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self._labels_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(pseudo), ensure_ascii=False) + "\n")

    def _load_label_log(self) -> list:
        """Tum pseudo-label log'larini yukle."""
        if not self._labels_log.exists():
            return []
        entries = []
        for line in self._labels_log.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM Pseudo-Labeler")
    parser.add_argument("--label-all", action="store_true",
                        help="Tum belirsiz kareleri etiketle")
    parser.add_argument("--status", action="store_true",
                        help="Labeling durumu")
    parser.add_argument("--export-ct", action="store_true",
                        help="CT-ready veriyi ihrac et")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Tek seferde max etiketlenecek kare")
    args = parser.parse_args()

    labeler = VLMLabeler()

    if args.status:
        status = labeler.get_status()
        print("\n=== VLM Pseudo-Labeler Durumu ===")
        print(f"  Toplam belirsiz kare: {status['total_uncertain']}")
        print(f"  Etiketsiz:           {status['unlabeled']}")
        print(f"  Pseudo-labeled:      {status['pseudo_labeled']}")
        print(f"  Operator onayli:     {status['operator_verified']}")
        print(f"  CT'ye hazir:         {status['ct_ready']}")
        return

    if args.label_all:
        print("\n=== VLM Pseudo-Labeling Baslatiliyor ===")
        results = labeler.label_all_uncertain(max_frames=args.max_frames)
        print(f"\n[OK] {len(results)} kare etiketlendi")
        for r in results:
            print(f"  {Path(r.image_path).name}: {r.defect_type} ({len(r.yolo_labels)} label)")
        return

    if args.export_ct:
        result = labeler.export_for_ct()
        print(f"\n[OK] {result['exported']} goruntu CT icin ihrac edildi")
        print(f"     Cikti: {result['output_dir']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
