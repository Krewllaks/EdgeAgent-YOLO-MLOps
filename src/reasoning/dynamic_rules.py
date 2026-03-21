"""
Dynamic Rules Engine — Config-driven product inspection rules.

Hocanin direktifi: "Hard coded olmasin. Rule'lara dikkat et,
rule'lar coklanabilir rule'lar olsun. Orada da LLM olabilir."

Bu modul:
1. rules.yaml'dan urun tipine gore kurallari yukler
2. Yeni urun tipi icin LLM ile otomatik rule olusturur
3. spatial_logic.py'nin hard-coded degerlerini config'den alir
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs" / "rules.yaml"


@dataclass
class ProductRule:
    """A product type's inspection rules loaded from config."""
    name: str
    description: str = ""
    total_screw_positions: int = 4
    sides: int = 2
    screws_per_side: int = 2
    max_missing_components: int = 2
    classes: list = field(default_factory=list)
    side_rules: dict = field(default_factory=dict)
    decision_matrix: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    clustering: dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return self.clustering.get("n_clusters", self.total_screw_positions)

    @property
    def min_detections(self) -> int:
        return self.clustering.get("min_detections", 1)

    @property
    def max_total_detections(self) -> int:
        return self.constraints.get("max_total_detections", self.total_screw_positions * 2)

    @property
    def reject_if_exceeds(self) -> bool:
        return self.constraints.get("reject_if_total_exceeds", True)

    @property
    def auto_tune_on_reject(self) -> bool:
        return self.constraints.get("auto_tune_on_reject", True)


class RuleEngine:
    """Loads and manages product inspection rules from config."""

    def __init__(self, rules_path: Path = RULES_PATH):
        self.rules_path = rules_path
        self._config = {}
        self._products: dict[str, ProductRule] = {}
        self._default_product: str = ""
        self.load()

    def load(self) -> None:
        """Load rules from YAML config."""
        if not self.rules_path.exists():
            print(f"[WARN] Rules config not found: {self.rules_path}")
            self._load_defaults()
            return

        with open(self.rules_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        self._default_product = self._config.get("default_product", "")
        products = self._config.get("products", {})

        for name, spec in products.items():
            if spec is None:
                continue
            rule = ProductRule(
                name=name,
                description=spec.get("description", ""),
                total_screw_positions=spec.get("total_screw_positions", 4),
                sides=spec.get("sides", 2),
                screws_per_side=spec.get("screws_per_side", 2),
                max_missing_components=spec.get("max_missing_components", 2),
                classes=spec.get("classes", []),
                side_rules=spec.get("side_rules", {}),
                decision_matrix=spec.get("decision_matrix", {}),
                constraints=spec.get("constraints", {}),
                clustering=spec.get("clustering", {}),
            )
            self._products[name] = rule

    def _load_defaults(self) -> None:
        """Load default 4-screw product rule (backward compatibility)."""
        self._default_product = "4_screw_component"
        self._products["4_screw_component"] = ProductRule(
            name="4_screw_component",
            description="4 vidali standart komponent (sol 2, sag 2)",
            total_screw_positions=4,
            sides=2,
            screws_per_side=2,
            max_missing_components=2,
            clustering={"n_clusters": 4, "min_detections": 1},
            constraints={"max_total_detections": 8, "reject_if_total_exceeds": True},
        )

    def get_rule(self, product_name: Optional[str] = None) -> ProductRule:
        """Get rules for a product type."""
        name = product_name or self._default_product
        if name in self._products:
            return self._products[name]
        # Fallback to default
        if self._default_product in self._products:
            return self._products[self._default_product]
        # Ultimate fallback
        self._load_defaults()
        return self._products["4_screw_component"]

    def list_products(self) -> list[str]:
        """List available product types."""
        return list(self._products.keys())

    def add_product(self, rule: ProductRule) -> None:
        """Add a new product rule and save to config."""
        self._products[rule.name] = rule
        self._save()

    def generate_rule_from_spec(self, spec_text: str) -> ProductRule:
        """Generate a ProductRule from a natural language specification.

        This is the LLM-driven rule generation entry point.
        For now, it parses structured specs. In the future,
        an LLM can be used to extract rules from free-form text.

        Args:
            spec_text: Product specification, e.g.:
                "6 vidali panel, sol 3 sag 3, metal yuzey"
                or structured format:
                "screws=6, sides=2, per_side=3"
        """
        # Parse structured format
        params = {}
        for part in spec_text.replace(",", " ").split():
            if "=" in part:
                key, val = part.split("=", 1)
                params[key.strip()] = val.strip()

        # Extract from parsed params or try to infer from text
        total = int(params.get("screws", params.get("total", 4)))
        sides = int(params.get("sides", 2))
        per_side = int(params.get("per_side", total // sides))

        # Try to extract from Turkish text
        if not params:
            import re
            # "6 vidali" pattern
            m = re.search(r"(\d+)\s*vida", spec_text, re.IGNORECASE)
            if m:
                total = int(m.group(1))
            # "sol 3 sag 3" pattern
            m_sides = re.findall(r"(sol|sag|left|right)\s*(\d+)", spec_text, re.IGNORECASE)
            if m_sides:
                sides = len(m_sides)
                per_side = int(m_sides[0][1])

        name = f"{total}_screw_custom"
        rule = ProductRule(
            name=name,
            description=f"Otomatik olusturulmus: {spec_text}",
            total_screw_positions=total,
            sides=sides,
            screws_per_side=per_side,
            max_missing_components=sides,
            classes=[
                {"id": 0, "name": "screw", "role": "fastener"},
                {"id": 1, "name": "missing_screw", "role": "defect_fastener"},
                {"id": 2, "name": "missing_component", "role": "defect_structural"},
            ],
            side_rules={
                "all_present": "S",
                "partial_missing": "MS",
                "all_missing": "MC",
            },
            decision_matrix={
                "S_S": "OK",
                "S_MS": "missing_screw",
                "MS_S": "missing_screw",
                "MS_MS": "missing_screw",
                "MC_any": "missing_component",
                "any_MC": "missing_component",
            },
            constraints={
                "max_total_detections": total * 2,
                "reject_if_total_exceeds": True,
                "auto_tune_on_reject": True,
            },
            clustering={
                "n_clusters": total,
                "min_detections": 1,
                "side_assignment": "median_x",
            },
        )

        return rule

    def _save(self) -> None:
        """Save current rules back to YAML config."""
        config = {
            "default_product": self._default_product,
            "products": {},
        }
        for name, rule in self._products.items():
            config["products"][name] = {
                "description": rule.description,
                "total_screw_positions": rule.total_screw_positions,
                "sides": rule.sides,
                "screws_per_side": rule.screws_per_side,
                "max_missing_components": rule.max_missing_components,
                "classes": rule.classes,
                "side_rules": rule.side_rules,
                "decision_matrix": rule.decision_matrix,
                "constraints": rule.constraints,
                "clustering": rule.clustering,
            }

        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rules_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def apply_decision_with_rules(
    rule: ProductRule,
    left_detections: list[dict],
    right_detections: list[dict],
) -> tuple[str, str]:
    """Apply the decision matrix from a rule to detection results.

    Args:
        rule: Product inspection rule
        left_detections: List of detections on left side [{"class_name": ..., ...}]
        right_detections: List of detections on right side

    Returns:
        (verdict, reason)
    """
    left_status = _compute_side_status(left_detections, rule)
    right_status = _compute_side_status(right_detections, rule)

    dm = rule.decision_matrix
    key = f"{left_status}_{right_status}"

    # Check exact match
    if key in dm:
        verdict = dm[key]
    # Check wildcard matches
    elif f"{left_status}_any" in dm:
        verdict = dm[f"{left_status}_any"]
    elif f"any_{right_status}" in dm:
        verdict = dm[f"any_{right_status}"]
    else:
        verdict = "OK"

    reason = f"Rule '{rule.name}': Sol={left_status}, Sag={right_status} -> {verdict}"
    return verdict, reason


def _compute_side_status(detections: list[dict], rule: ProductRule) -> str:
    """Compute side status based on rule's side_rules."""
    if not detections:
        return "unknown"

    s_count = sum(1 for d in detections if d.get("class_name") == "screw")
    ms_count = sum(1 for d in detections if d.get("class_name") == "missing_screw")
    mc_count = sum(1 for d in detections if d.get("class_name") == "missing_component")

    if mc_count > 0:
        return rule.side_rules.get("all_missing", "MC")
    if s_count > 0 and ms_count == 0:
        return rule.side_rules.get("all_present", "S")
    if s_count > 0 and ms_count > 0:
        return rule.side_rules.get("partial_missing", "MS")
    if ms_count > 0 and s_count == 0:
        return rule.side_rules.get("all_missing", "MC")

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic Rules Engine")
    parser.add_argument("--list", action="store_true", help="List available product types")
    parser.add_argument("--show", type=str, default=None, help="Show rules for a product type")
    parser.add_argument(
        "--generate", type=str, default=None,
        help="Generate rules from spec (e.g., 'screws=6, sides=2, per_side=3')",
    )
    parser.add_argument("--save", action="store_true", help="Save generated rule to config")
    args = parser.parse_args()

    engine = RuleEngine()

    if args.list:
        print("Available product types:")
        for name in engine.list_products():
            rule = engine.get_rule(name)
            print(f"  - {name}: {rule.description}")
        return

    if args.show:
        rule = engine.get_rule(args.show)
        print(f"Product: {rule.name}")
        print(f"  Description: {rule.description}")
        print(f"  Total screws: {rule.total_screw_positions}")
        print(f"  Sides: {rule.sides}")
        print(f"  Per side: {rule.screws_per_side}")
        print(f"  Clusters: {rule.n_clusters}")
        print(f"  Max detections: {rule.max_total_detections}")
        print(f"  Decision matrix: {json.dumps(rule.decision_matrix, indent=4)}")
        return

    if args.generate:
        rule = engine.generate_rule_from_spec(args.generate)
        print(f"Generated rule: {rule.name}")
        print(f"  Description: {rule.description}")
        print(f"  Total screws: {rule.total_screw_positions}")
        print(f"  Sides: {rule.sides}")
        print(f"  Per side: {rule.screws_per_side}")
        print(f"  Clusters: {rule.n_clusters}")

        if args.save:
            engine.add_product(rule)
            print(f"[OK] Rule saved to {engine.rules_path}")
        return

    # Default: show current default
    rule = engine.get_rule()
    print(f"Default product: {rule.name}")
    print(f"  {rule.description}")
    print(f"  Screws: {rule.total_screw_positions} ({rule.screws_per_side} per side)")


if __name__ == "__main__":
    main()
