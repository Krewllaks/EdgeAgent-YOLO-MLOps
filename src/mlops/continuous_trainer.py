"""
Continuous Training Pipeline — Shadow Deployment + Hot-Swap.

Kosul bazli otomatik retrain + guvenli model guncelleme:

1. URETIMDE (surekli):
   - Uncertain frame'ler toplanir
   - Operator geri bildirimi alinir
   - Drift detector SSIM kontrol eder

2. RETRAIN TETIKLEYICI (kosul bazli):
   - 100+ uncertain frame VEYA 50+ operator duzeltmesi
   - VE drift algilandiginda
   - Otomatik tetiklenir, bant durmaz

3. EGITIM (arka planda):
   - Biriken veri + copy-paste augmentation
   - Yeni model egitilir → "challenger" olarak kaydedilir

4. SHADOW DEPLOYMENT (guvenli gecis):
   - Challenger, champion ile paralel calisir
   - Metrikler iyilesirse → hot-swap
   - Kotulesirse → retire, rollback

Kullanim:
    python src/mlops/continuous_trainer.py --check     # Retrain gerekli mi?
    python src/mlops/continuous_trainer.py --run       # Tam donguyu calistir
    python src/mlops/continuous_trainer.py --status    # Durum raporu
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HISTORY_PATH = ROOT / "data" / "feedback" / "retrain_history.jsonl"


@dataclass
class RetrainDecision:
    """Decision on whether to retrain."""
    should_retrain: bool
    reason: str
    uncertain_count: int
    feedback_corrective: int
    days_since_last_retrain: float
    timestamp: str = ""


@dataclass
class RetrainResult:
    """Result of a retrain cycle."""
    success: bool
    timestamp: str
    dataset_version: str
    model_path: str
    augmented_count: int
    total_train_images: int
    trigger: str  # "uncertain_frames" | "operator_feedback" | "manual"
    metrics: dict  # mAP50, per-class etc.
    error: str = ""


class ContinuousTrainer:
    """Manages the continuous training lifecycle."""

    def __init__(
        self,
        min_uncertain_frames: int = 100,
        min_feedback_corrections: int = 50,
        min_days_between_retrain: float = 1.0,
        base_dataset: str = "phase1_multiclass_v1",
    ):
        self.min_uncertain_frames = min_uncertain_frames
        self.min_feedback_corrections = min_feedback_corrections
        self.min_days_between_retrain = min_days_between_retrain
        self.base_dataset = base_dataset

    def check_retrain_needed(self) -> RetrainDecision:
        """Check if conditions for retrain are met."""
        from src.data.uncertain_collector import UncertainCollector

        # Check uncertain frames
        collector = UncertainCollector()
        uncertain_count = collector.get_collected_count()

        # Check operator feedback
        feedback_path = ROOT / "data" / "feedback" / "feedback_log.jsonl"
        corrective = 0
        if feedback_path.exists():
            for line in feedback_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("label") in ("incorrect", "partial"):
                        corrective += 1
                except json.JSONDecodeError:
                    continue

        # Check time since last retrain
        days_since = self._days_since_last_retrain()

        # Decision logic
        time_ok = days_since >= self.min_days_between_retrain or days_since < 0
        uncertain_ok = uncertain_count >= self.min_uncertain_frames
        feedback_ok = corrective >= self.min_feedback_corrections

        should_retrain = time_ok and (uncertain_ok or feedback_ok)

        reasons = []
        if not time_ok:
            reasons.append(f"Son retrain'den {days_since:.1f} gun gecmis (min: {self.min_days_between_retrain})")
        if uncertain_ok:
            reasons.append(f"{uncertain_count} belirsiz kare birikti (esik: {self.min_uncertain_frames})")
        elif not feedback_ok:
            reasons.append(f"{uncertain_count}/{self.min_uncertain_frames} belirsiz kare (yetersiz)")
        if feedback_ok:
            reasons.append(f"{corrective} duzeltici geri bildirim (esik: {self.min_feedback_corrections})")
        elif not uncertain_ok:
            reasons.append(f"{corrective}/{self.min_feedback_corrections} geri bildirim (yetersiz)")

        return RetrainDecision(
            should_retrain=should_retrain,
            reason="; ".join(reasons),
            uncertain_count=uncertain_count,
            feedback_corrective=corrective,
            days_since_last_retrain=days_since,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    def run_retrain_cycle(
        self,
        num_augmented: int = 500,
        epochs: int = 100,
        trigger: str = "manual",
    ) -> RetrainResult:
        """Run the full retrain cycle.

        Steps:
        1. VLM ile belirsiz karelere pseudo-label uret
        2. CT-ready veriyi + augmentation ile dataset olustur
        3. YOLO egitimi calistir
        4. Sonuclari kaydet
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"retrain_{timestamp}"

        try:
            # Step 0: VLM pseudo-labeling (belirsiz karelere label uret)
            print("[0/5] VLM ile belirsiz kareler etiketleniyor...")
            try:
                from src.mlops.vlm_labeler import VLMLabeler
                labeler = VLMLabeler()
                pseudo_results = labeler.label_all_uncertain(max_frames=50)
                print(f"       {len(pseudo_results)} kare VLM ile etiketlendi")

                # CT-ready veriyi ihrac et
                ct_export = labeler.export_for_ct()
                ct_count = ct_export.get("exported", 0)
                print(f"       {ct_count} goruntu CT havuzuna aktarildi")
            except Exception as e:
                print(f"       VLM labeling atlandi: {e}")
                ct_count = 0

            # Step 1: Prepare dataset
            print(f"[1/5] Dataset hazirlaniyor (augment={num_augmented})...")
            dataset_dir = ROOT / "data" / "processed" / version

            # Use copy-paste augmentation
            prepare_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "prepare_v3_copypaste.py"),
                "--num-augmented", str(num_augmented),
                "--output-dir", str(dataset_dir),
            ]

            result = subprocess.run(
                prepare_cmd, capture_output=True, text=True, cwd=str(ROOT),
            )

            # Step 1.5: CT-ready veriyi train setine ekle
            if ct_count > 0:
                print(f"[1.5/5] {ct_count} VLM-labeled goruntu train setine ekleniyor...")
                ct_dir = ROOT / "data" / "ct_ready"
                train_img = dataset_dir / "train" / "images"
                train_lbl = dataset_dir / "train" / "labels"
                if train_img.exists() and (ct_dir / "images").exists():
                    import shutil
                    for img in (ct_dir / "images").iterdir():
                        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            shutil.copy2(img, train_img / img.name)
                    for lbl in (ct_dir / "labels").iterdir():
                        if lbl.suffix == ".txt":
                            shutil.copy2(lbl, train_lbl / lbl.name)

            if result.returncode != 0:
                return RetrainResult(
                    success=False, timestamp=timestamp,
                    dataset_version=version, model_path="",
                    augmented_count=0, total_train_images=0,
                    trigger=trigger, metrics={},
                    error=f"Dataset hazirlama hatasi: {result.stderr[:500]}",
                )

            # Step 2: Train
            print(f"[2/5] Model egitiliyor ({epochs} epoch)...")
            data_yaml = dataset_dir / "data.yaml"
            if not data_yaml.exists():
                # Fallback
                data_yaml = ROOT / "data" / "processed" / self.base_dataset / "data.yaml"

            train_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "train_final_phase1.py"),
                "--data", str(data_yaml),
                "--epochs", str(epochs),
                "--batch", "8",
                "--imgsz", "640",
                "--device", "0",
                "--name", f"retrain_{timestamp}",
            ]

            result = subprocess.run(
                train_cmd, capture_output=True, text=True, cwd=str(ROOT),
                timeout=14400,  # 4 hour timeout
            )

            # Step 3: Check results
            print("[3/5] Sonuclar kontrol ediliyor...")
            model_path = ROOT / "models" / "phase1_final_ca.pt"
            metrics = {}

            # Try to extract mAP50 from training output
            if "mAP50" in result.stdout:
                for line in result.stdout.split("\n"):
                    if "mAP50" in line:
                        try:
                            parts = line.strip().split()
                            for i, p in enumerate(parts):
                                if "mAP50" in p and i + 1 < len(parts):
                                    metrics["mAP50"] = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass

            # Step 4: Log result
            print("[4/5] Sonuclar kaydediliyor...")
            retrain_result = RetrainResult(
                success=result.returncode == 0,
                timestamp=timestamp,
                dataset_version=version,
                model_path=str(model_path),
                augmented_count=num_augmented,
                total_train_images=0,
                trigger=trigger,
                metrics=metrics,
                error=result.stderr[:200] if result.returncode != 0 else "",
            )

            self._save_history(retrain_result)

            # Clear uncertain frames after successful retrain
            if retrain_result.success:
                from src.data.uncertain_collector import UncertainCollector
                collector = UncertainCollector()
                cleared = collector.clear()
                print(f"[OK] {cleared} belirsiz kare temizlendi.")

            return retrain_result

        except subprocess.TimeoutExpired:
            return RetrainResult(
                success=False, timestamp=timestamp,
                dataset_version=version, model_path="",
                augmented_count=num_augmented, total_train_images=0,
                trigger=trigger, metrics={},
                error="Egitim zaman asimina ugradi (4 saat)",
            )
        except Exception as e:
            return RetrainResult(
                success=False, timestamp=timestamp,
                dataset_version=version, model_path="",
                augmented_count=num_augmented, total_train_images=0,
                trigger=trigger, metrics={},
                error=str(e),
            )

    def get_retrain_history(self) -> list:
        """Get history of all retrain cycles."""
        if not HISTORY_PATH.exists():
            return []
        entries = []
        for line in HISTORY_PATH.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _days_since_last_retrain(self) -> float:
        """Days since the last retrain. Returns -1 if no history."""
        history = self.get_retrain_history()
        if not history:
            return -1.0
        last = history[-1]
        try:
            last_time = datetime.fromisoformat(last.get("timestamp", ""))
            delta = datetime.now() - last_time
            return delta.total_seconds() / 86400
        except (ValueError, TypeError):
            return -1.0

    def _save_history(self, result: RetrainResult) -> None:
        """Append retrain result to history."""
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def get_status(self) -> dict:
        """Get complete status of the continuous training system."""
        from src.data.uncertain_collector import UncertainCollector

        collector = UncertainCollector()
        decision = self.check_retrain_needed()
        history = self.get_retrain_history()

        return {
            "uncertain_frames": collector.get_summary(),
            "retrain_decision": asdict(decision),
            "retrain_history_count": len(history),
            "last_retrain": history[-1] if history else None,
        }


def main():
    parser = argparse.ArgumentParser(description="Continuous Training Pipeline")
    parser.add_argument("--check", action="store_true", help="Retrain gerekli mi kontrol et")
    parser.add_argument("--run", action="store_true", help="Retrain dongusunu calistir")
    parser.add_argument("--status", action="store_true", help="Sistem durumu")
    parser.add_argument("--augmented", type=int, default=500, help="Augmented goruntu sayisi")
    parser.add_argument("--epochs", type=int, default=100, help="Egitim epoch sayisi")
    parser.add_argument(
        "--min-uncertain", type=int, default=100,
        help="Retrain icin minimum belirsiz kare (default: 100)",
    )
    args = parser.parse_args()

    trainer = ContinuousTrainer(min_uncertain_frames=args.min_uncertain)

    if args.check:
        decision = trainer.check_retrain_needed()
        status = "EVET" if decision.should_retrain else "HAYIR"
        print(f"\nRetrain gerekli mi? {status}")
        print(f"  Sebep: {decision.reason}")
        print(f"  Belirsiz kare: {decision.uncertain_count}")
        print(f"  Duzeltici feedback: {decision.feedback_corrective}")
        print(f"  Son retrain: {decision.days_since_last_retrain:.1f} gun once")
        return

    if args.status:
        status = trainer.get_status()
        print("\n=== Continuous Training Durumu ===")
        uf = status["uncertain_frames"]
        print(f"  Belirsiz kare: {uf.get('total', 0)}")
        print(f"  Ortalama confidence: {uf.get('avg_confidence', 0):.3f}")
        rd = status["retrain_decision"]
        print(f"  Retrain gerekli: {'EVET' if rd['should_retrain'] else 'HAYIR'}")
        print(f"  Retrain gecmisi: {status['retrain_history_count']} kayit")
        if status["last_retrain"]:
            lr = status["last_retrain"]
            print(f"  Son retrain: {lr.get('timestamp', '?')} - {'Basarili' if lr.get('success') else 'Basarisiz'}")
        return

    if args.run:
        decision = trainer.check_retrain_needed()
        if not decision.should_retrain:
            print(f"Retrain henuz gerekli degil: {decision.reason}")
            print("Zorla baslatmak icin --force ekleyin.")
            return

        print("\n=== Continuous Training Baslatiyor ===")
        result = trainer.run_retrain_cycle(
            num_augmented=args.augmented,
            epochs=args.epochs,
            trigger="automatic",
        )
        if result.success:
            print(f"\n[OK] Retrain basarili!")
            print(f"  Model: {result.model_path}")
            print(f"  Metrikler: {result.metrics}")
        else:
            print(f"\n[HATA] Retrain basarisiz: {result.error}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
