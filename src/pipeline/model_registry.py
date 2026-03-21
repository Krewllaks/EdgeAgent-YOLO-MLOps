"""
Model Registry — Versiyon yonetimi, shadow deployment, hot-swap, rollback.

Her model versiyonu:
  - Benzersiz hash (SHA256)
  - Egitim metrikleri (mAP50, per-class P/R/F1)
  - Durum: staged | champion | challenger | retired

Shadow deployment:
  - Champion (uretimde aktif) + Challenger (test ediliyor) paralel calisir
  - Challenger metrikleri daha iyiyse → hot-swap (champion olur)
  - Kotulesirse → retire edilir, champion devam eder

Kullanim:
    from src.pipeline.model_registry import ModelRegistry

    registry = ModelRegistry()
    registry.register_model("models/v4.pt", metrics={...})
    registry.promote_to_challenger("models/v5.pt")
    registry.hot_swap()  # challenger → champion
    registry.rollback()  # onceki champion'a don
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Tek bir model versiyonu."""
    model_path: str
    model_hash: str
    version_tag: str
    status: str  # staged | champion | challenger | retired
    registered_at: str
    metrics: dict = field(default_factory=dict)
    training_data: str = ""
    notes: str = ""


class ModelRegistry:
    """Model versiyon yonetimi ve deployment."""

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.registry_dir / "registry.json"
        self._versions: list[ModelVersion] = []
        self._load()

    def _load(self):
        """Registry'yi diskten yukle."""
        if self._registry_file.exists():
            data = json.loads(self._registry_file.read_text(encoding="utf-8"))
            self._versions = [ModelVersion(**v) for v in data.get("versions", [])]
        else:
            self._versions = []

    def _save(self):
        """Registry'yi diske yaz."""
        data = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "versions": [asdict(v) for v in self._versions],
        }
        self._registry_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _hash_file(path: str) -> str:
        """Dosyanin SHA256 hash'ini hesapla."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def register_model(self, model_path: str, version_tag: str = "",
                       metrics: Optional[dict] = None,
                       training_data: str = "", notes: str = "") -> ModelVersion:
        """Yeni model versiyonu kaydet."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model bulunamadi: {model_path}")

        model_hash = self._hash_file(model_path)

        # Zaten kayitli mi?
        for v in self._versions:
            if v.model_hash == model_hash:
                logger.info(f"Model zaten kayitli: {v.version_tag} ({model_hash})")
                return v

        if not version_tag:
            version_tag = f"v{len(self._versions) + 1}_{model_hash[:8]}"

        # Modeli registry dizinine kopyala
        dst = self.registry_dir / f"{version_tag}{path.suffix}"
        shutil.copy2(model_path, dst)

        version = ModelVersion(
            model_path=str(dst),
            model_hash=model_hash,
            version_tag=version_tag,
            status="staged",
            registered_at=datetime.now().isoformat(timespec="seconds"),
            metrics=metrics or {},
            training_data=training_data,
            notes=notes,
        )
        self._versions.append(version)
        self._save()
        logger.info(f"Model kaydedildi: {version_tag} (hash: {model_hash})")
        return version

    def get_champion(self) -> Optional[ModelVersion]:
        """Aktif uretim modelini getir."""
        for v in reversed(self._versions):
            if v.status == "champion":
                return v
        return None

    def get_challenger(self) -> Optional[ModelVersion]:
        """Shadow test modelini getir."""
        for v in reversed(self._versions):
            if v.status == "challenger":
                return v
        return None

    def promote_to_champion(self, version_tag: str):
        """Modeli champion (uretim) olarak ata. Onceki champion retired olur."""
        target = None
        for v in self._versions:
            if v.version_tag == version_tag:
                target = v
                break
        if target is None:
            raise ValueError(f"Versiyon bulunamadi: {version_tag}")

        # Onceki champion'i retire et
        for v in self._versions:
            if v.status == "champion":
                v.status = "retired"

        target.status = "champion"
        self._save()
        logger.info(f"Champion olarak atandi: {version_tag}")

    def promote_to_challenger(self, version_tag: str):
        """Modeli challenger (shadow test) olarak ata."""
        target = None
        for v in self._versions:
            if v.version_tag == version_tag:
                target = v
                break
        if target is None:
            raise ValueError(f"Versiyon bulunamadi: {version_tag}")

        # Onceki challenger'i retire et
        for v in self._versions:
            if v.status == "challenger":
                v.status = "retired"

        target.status = "challenger"
        self._save()
        logger.info(f"Challenger olarak atandi: {version_tag}")

    def hot_swap(self) -> bool:
        """Challenger → Champion (uretim durmadan model degistir)."""
        challenger = self.get_challenger()
        champion = self.get_champion()

        if challenger is None:
            logger.warning("Hot-swap: challenger yok")
            return False

        if champion is not None:
            champion.status = "retired"
            logger.info(f"Onceki champion retire: {champion.version_tag}")

        challenger.status = "champion"
        self._save()
        logger.info(f"Hot-swap tamamlandi: {challenger.version_tag} artik champion")
        return True

    def rollback(self) -> bool:
        """Son retired champion'a geri don."""
        current = self.get_champion()
        # Son retired'i bul
        last_retired = None
        for v in reversed(self._versions):
            if v.status == "retired" and v != current:
                last_retired = v
                break

        if last_retired is None:
            logger.warning("Rollback: geri donulecek versiyon yok")
            return False

        if current is not None:
            current.status = "retired"

        last_retired.status = "champion"
        self._save()
        logger.info(f"Rollback: {last_retired.version_tag} tekrar champion")
        return True

    def list_versions(self) -> list[ModelVersion]:
        """Tum versiyonlari listele."""
        return list(self._versions)

    def get_version(self, version_tag: str) -> Optional[ModelVersion]:
        for v in self._versions:
            if v.version_tag == version_tag:
                return v
        return None
