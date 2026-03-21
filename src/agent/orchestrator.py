"""
Orchestrator Agent — Chat-based pipeline controller.

Hocanin direktifi: "Bir tane agent koyun uste. Sen chat'le sorarsin:
'Benim ne kadar datam var? Datayi su kadar dataya cokla.'
Muhammet'in kismi calisir. Ondan sonra bunu train et dersin.
Bu pipeline'i da calistirirsin ki bu bir agent haline gelsin."

Bu agent:
1. Chat arayuzu uzerinden dogal dil komutlari alir
2. Uygun tool'u secip calistirir
3. Sonuclari operatore raporlar
4. Tam pipeline kontrolu saglar: veri analizi -> augmentation -> egitim -> deployment
"""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.tools import TOOLS, call_tool, get_tool_list


@dataclass
class ConversationMessage:
    role: str  # "user" or "agent"
    content: str
    timestamp: str = ""
    tool_used: Optional[str] = None
    tool_result: Optional[dict] = None


class OrchestratorAgent:
    """Chat-based orchestrator agent for the EdgeAgent pipeline."""

    def __init__(self):
        self.history: list[ConversationMessage] = []
        self.docs_dir = ROOT / "docs" / "agent"

    def process_message(self, user_input: str) -> str:
        """Process a user message and return the agent response.

        This is the main entry point. It:
        1. Parses the user intent
        2. Routes to the appropriate tool
        3. Formats the response
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.history.append(ConversationMessage(
            role="user", content=user_input, timestamp=timestamp,
        ))

        # Route to appropriate tool
        intent = self._detect_intent(user_input)
        tool_name = intent.get("tool")
        tool_args = intent.get("args", {})

        if tool_name == "help":
            response = self._format_help()
        elif tool_name == "history":
            response = self._format_history()
        elif tool_name in TOOLS:
            result = call_tool(tool_name, **tool_args)
            response = self._format_tool_result(tool_name, result)
            self.history.append(ConversationMessage(
                role="agent", content=response, timestamp=timestamp,
                tool_used=tool_name, tool_result=result,
            ))
            return response
        else:
            response = self._handle_unknown(user_input)

        self.history.append(ConversationMessage(
            role="agent", content=response, timestamp=timestamp,
        ))
        return response

    def _detect_intent(self, text: str) -> dict:
        """Detect user intent from natural language input.

        Uses keyword matching for now. Can be upgraded to LLM-based
        intent detection in the future.
        """
        text_lower = text.lower().strip()

        # Help
        if text_lower in ("help", "yardim", "?", "komutlar"):
            return {"tool": "help"}

        # History
        if text_lower in ("gecmis", "history", "log"):
            return {"tool": "history"}

        # Data analysis
        if any(kw in text_lower for kw in [
            "ne kadar data", "kac tane", "veri seti", "dataset",
            "data analiz", "veri analiz", "sinif dagilim",
            "kac goruntu", "kac resim",
        ]):
            # Extract dataset path if provided
            args = {}
            path_match = re.search(r'path[=:]\s*["\']?([^\s"\']+)', text_lower)
            if path_match:
                args["dataset_path"] = path_match.group(1)
            return {"tool": "analyze_data", "args": args}

        # Label validation
        if any(kw in text_lower for kw in [
            "label", "etiket", "weak", "kontrol", "dogrula",
            "egitim oncesi", "hazir mi", "temiz mi",
        ]):
            args = {}
            path_match = re.search(r'path[=:]\s*["\']?([^\s"\']+)', text_lower)
            if path_match:
                args["dataset_path"] = path_match.group(1)
            return {"tool": "validate_labels", "args": args}

        # Augmentation
        if any(kw in text_lower for kw in [
            "cogalt", "augment", "cokla", "uret", "generate",
            "yeni veri", "data uret",
        ]):
            args = {}
            # Extract number
            num_match = re.search(r'(\d+)\s*(tane|adet|goruntu|resim|image)?', text_lower)
            if num_match:
                args["num_images"] = int(num_match.group(1))

            if "copypaste" in text_lower or "copy-paste" in text_lower or "yapistir" in text_lower:
                args["strategy"] = "copypaste"
            else:
                args["strategy"] = "smart"

            return {"tool": "augment_data", "args": args}

        # Training
        if any(kw in text_lower for kw in [
            "egit", "train", "retrain", "yeniden egit",
            "model egit", "ogret",
        ]):
            args = {}
            epoch_match = re.search(r'(\d+)\s*epoch', text_lower)
            if epoch_match:
                args["epochs"] = int(epoch_match.group(1))
            batch_match = re.search(r'batch\s*[=:]?\s*(\d+)', text_lower)
            if batch_match:
                args["batch"] = int(batch_match.group(1))
            return {"tool": "train_model", "args": args}

        # Evaluation
        if any(kw in text_lower for kw in [
            "accuracy", "degerlendir", "evaluate", "performans",
            "precision", "recall", "f1", "basari", "dogruluk",
            "ne kadar dogru", "ne kadar basarili",
        ]):
            args = {}
            if "test" in text_lower:
                args["split"] = "test"
            return {"tool": "evaluate_model", "args": args}

        # Deployment
        if any(kw in text_lower for kw in [
            "deploy", "dagit", "export", "tensorrt", "jetson",
            "hazirla", "production",
        ]):
            args = {}
            if "int8" in text_lower:
                args["int8"] = True
            return {"tool": "deploy_model", "args": args}

        # Profiling
        if any(kw in text_lower for kw in [
            "profil", "gecikme", "latency", "hiz", "benchmark",
            "ne kadar hizli",
        ]):
            return {"tool": "profile_edge"}

        # Rules
        if any(kw in text_lower for kw in [
            "kural", "rule", "urun tipi", "product",
        ]):
            if any(kw in text_lower for kw in ["ekle", "add", "olustur", "yeni"]):
                # Extract spec
                spec_match = re.search(r'["\']([^"\']+)["\']', text)
                spec = spec_match.group(1) if spec_match else text
                return {"tool": "add_rule", "args": {"spec": spec}}
            return {"tool": "list_rules"}

        # Feedback
        if any(kw in text_lower for kw in [
            "geri bildirim", "feedback", "operator", "duzeltme",
            "retrain gerekli", "retrain lazim",
        ]):
            return {"tool": "check_feedback"}

        return {"tool": None}

    def _format_tool_result(self, tool_name: str, result: dict) -> str:
        """Format tool result for display."""
        if "error" in result:
            return f"Hata: {result['error']}"

        if tool_name == "analyze_data":
            lines = [f"Veri Seti: {result.get('dataset', '?')}"]
            lines.append(f"Toplam Goruntu: {result.get('total_images', 0)}")
            lines.append("")
            for split, info in result.get("splits", {}).items():
                lines.append(f"  [{split}] {info['images']} goruntu, {info['boxes']} kutu")
                for cls, cnt in info.get("class_distribution", {}).items():
                    if cnt > 0:
                        lines.append(f"    - {cls}: {cnt}")
            lines.append("")
            lines.append("Toplam Sinif Dagilimi:")
            for cls, cnt in result.get("class_distribution", {}).items():
                lines.append(f"  - {cls}: {cnt}")
            return "\n".join(lines)

        if tool_name == "validate_labels":
            ready = result.get("training_ready", False)
            status = "HAZIR" if ready else "ENGELLENDI"
            lines = [f"Label Dogrulama: {status}"]
            for split, info in result.get("splits", {}).items():
                weak = info.get("weak_labeled", 0)
                total = info.get("total_images", 0)
                lines.append(f"  [{split}] {total} goruntu, {weak} weak label")
            return "\n".join(lines)

        if tool_name == "augment_data":
            return (
                f"Augmentation tamamlandi!\n"
                f"  Strateji: {result.get('strategy')}\n"
                f"  Uretilen: {result.get('total_generated')} goruntu\n"
                f"  Cikti: {result.get('output_dir')}"
            )

        if tool_name == "check_feedback":
            return (
                f"Geri Bildirim Ozeti:\n"
                f"  Toplam: {result.get('total_feedback', 0)}\n"
                f"  Dogru: {result.get('correct', 0)}\n"
                f"  Yanlis: {result.get('incorrect', 0)}\n"
                f"  Kismi: {result.get('partial', 0)}\n"
                f"  ---\n"
                f"  {result.get('recommendation', '')}"
            )

        if tool_name == "list_rules":
            lines = ["Tanimli Urun Tipleri:"]
            for name, info in result.get("products", {}).items():
                lines.append(
                    f"  - {name}: {info['description']} "
                    f"({info['total_screws']} vida, {info['sides']} taraf)"
                )
            return "\n".join(lines)

        if tool_name == "add_rule":
            return (
                f"Yeni kural olusturuldu:\n"
                f"  Ad: {result.get('name')}\n"
                f"  Aciklama: {result.get('description')}\n"
                f"  Vidalar: {result.get('total_screws')} ({result.get('per_side')} per side)"
            )

        # Generic formatting for other tools
        if "message" in result:
            return result["message"]

        return json.dumps(result, indent=2, ensure_ascii=False)

    def _format_help(self) -> str:
        """Format help message."""
        lines = [
            "EdgeAgent Orchestrator - Kullanilabilir Komutlar:",
            "",
            "Veri Yonetimi:",
            '  "Benim ne kadar datam var?"     -> Veri seti analizi',
            '  "Label\'lar temiz mi?"            -> Weak label kontrolu',
            '  "200 tane daha cogalt"           -> VLM-guided augmentation',
            "",
            "Model Islemleri:",
            '  "Modeli egit"                    -> YOLO egitimi baslat',
            '  "Accuracy nedir?"                -> Per-class metrikler',
            '  "Modeli Jetson\'a hazirla"        -> TensorRT export',
            '  "Ne kadar hizli calisir?"        -> Edge profiling',
            "",
            "Kurallar:",
            '  "Hangi urun tipleri var?"        -> Kural listesi',
            '  "Yeni urun ekle: 6 vidali"      -> Dinamik kural',
            "",
            "Geri Bildirim:",
            '  "Geri bildirimleri analiz et"    -> Retrain gerekli mi?',
            "",
            "Diger:",
            '  "help" veya "yardim"             -> Bu mesaj',
            '  "gecmis"                         -> Konusma gecmisi',
        ]
        return "\n".join(lines)

    def _format_history(self) -> str:
        """Format conversation history."""
        if not self.history:
            return "Henuz konusma gecmisi yok."

        lines = ["Son 10 mesaj:"]
        for msg in self.history[-10:]:
            prefix = "Siz" if msg.role == "user" else "Agent"
            tool_info = f" [{msg.tool_used}]" if msg.tool_used else ""
            lines.append(f"  [{msg.timestamp}] {prefix}{tool_info}: {msg.content[:80]}")
        return "\n".join(lines)

    def _handle_unknown(self, text: str) -> str:
        """Handle unrecognized input."""
        return (
            f"Anlayamadim: '{text[:50]}...'\n"
            f"Yardim icin 'help' yazin.\n"
            f"Mevcut yetenekler: veri analizi, augmentation, egitim, "
            f"degerlendirme, deployment, kural yonetimi, geri bildirim."
        )


# ── CLI Mode ─────────────────────────────────────────────────────


def main():
    """Run the orchestrator agent in CLI chat mode."""
    agent = OrchestratorAgent()

    print("=" * 60)
    print("  EdgeAgent Orchestrator")
    print("  Cikis icin 'quit' yazin, yardim icin 'help' yazin")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nSiz > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoruntuleme agent kapatiliyor...")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "cikis", "q"):
            print("Gorusuruz!")
            break

        response = agent.process_message(user_input)
        print(f"\nAgent > {response}")


if __name__ == "__main__":
    main()
