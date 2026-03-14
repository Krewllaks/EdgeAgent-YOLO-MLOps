"""Generate VLM Training Captions via Gemini API.

Offline data preparation tool: sends NOK (defective) images to Google
Gemini API for automated industrial captioning. The resulting captions
can be used to fine-tune PaliGemma for domain-specific reasoning.

Output format (JSONL):
    {"image_path": "img_001.jpg", "caption": "...", "defect_type": "missing_screw", "model": "gemini-2.0-flash"}

Usage:
    export GOOGLE_API_KEY=your_key_here
    python scripts/generate_vlm_captions.py \
        --source data/processed/phase1_multiclass_v1/test/images \
        --output data/vlm_captions/captions.jsonl \
        --max-images 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CAPTION_PROMPT = (
    "You are an industrial quality control expert. Analyze this image of a "
    "manufactured part and describe any defects you see. Focus on:\n"
    "1. Are all screws present and properly seated?\n"
    "2. Are any components or brackets missing?\n"
    "3. Describe the condition of the metal surface.\n\n"
    "Respond in this exact format:\n"
    "DEFECT_TYPE: [missing_screw|missing_component|ok]\n"
    "DESCRIPTION: [detailed description of what you observe]\n"
    "SEVERITY: [critical|moderate|minor|none]"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate VLM captions via Gemini API")
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1" / "test" / "images",
        help="Directory containing images to caption",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "vlm_captions" / "captions.jsonl",
        help="Output JSONL file path",
    )
    p.add_argument("--max-images", type=int, default=200)
    p.add_argument("--rpm-limit", type=int, default=15,
                   help="Requests per minute (free tier: 15)")
    p.add_argument("--model", type=str, default="gemini-2.0-flash")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without making API calls")
    return p.parse_args()


def load_existing_captions(output_path: Path) -> set[str]:
    """Load already-captioned image names for idempotency."""
    if not output_path.exists():
        return set()
    captioned = set()
    for line in output_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            entry = json.loads(line)
            captioned.add(entry.get("image_path", ""))
    return captioned


def caption_image(genai_model, image_path: Path) -> dict:
    """Send image to Gemini and parse response."""
    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    response = genai_model.generate_content([CAPTION_PROMPT, img])
    text = response.text.strip()

    # Parse structured response
    defect_type = None
    description = text
    severity = None

    for line in text.split("\n"):
        if line.startswith("DEFECT_TYPE:"):
            dt = line.split(":", 1)[1].strip().lower()
            if dt in ("missing_screw", "missing_component", "ok"):
                defect_type = dt
        elif line.startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
        elif line.startswith("SEVERITY:"):
            severity = line.split(":", 1)[1].strip().lower()

    return {
        "defect_type": defect_type,
        "caption": description,
        "severity": severity,
        "raw_response": text,
    }


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        sys.exit(f"[ERR] Source not found: {args.source}")

    # Find images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in args.source.iterdir() if p.suffix.lower() in exts])
    images = images[:args.max_images]

    if not images:
        sys.exit(f"[ERR] No images in {args.source}")

    # Check for already captioned (idempotent)
    existing = load_existing_captions(args.output)
    to_caption = [img for img in images if img.name not in existing]

    print(f"[INFO] Found {len(images)} images, {len(existing)} already captioned, "
          f"{len(to_caption)} to process")

    if args.dry_run:
        print("[DRY-RUN] Would caption these images:")
        for img in to_caption[:10]:
            print(f"  - {img.name}")
        if len(to_caption) > 10:
            print(f"  ... and {len(to_caption) - 10} more")
        return

    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "[ERR] Set GOOGLE_API_KEY environment variable.\n"
            "Get a key at: https://aistudio.google.com/apikey"
        )

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    # Process images with rate limiting
    args.output.parent.mkdir(parents=True, exist_ok=True)
    interval = 60.0 / args.rpm_limit  # seconds between requests

    captioned = 0
    errors = 0

    with args.output.open("a", encoding="utf-8") as f:
        for i, img_path in enumerate(to_caption):
            try:
                result = caption_image(model, img_path)
                entry = {
                    "image_path": img_path.name,
                    "caption": result["caption"],
                    "defect_type": result["defect_type"],
                    "severity": result["severity"],
                    "model": args.model,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                captioned += 1

                print(f"[{captioned}/{len(to_caption)}] {img_path.name} -> "
                      f"{result['defect_type'] or '?'}")

            except Exception as e:
                errors += 1
                print(f"[ERR] {img_path.name}: {e}")

            # Rate limiting
            if i < len(to_caption) - 1:
                time.sleep(interval)

    print(f"\n[OK] Captioning complete"
          f" - processed: {captioned}"
          f" - errors: {errors}"
          f" - output: {args.output}")


if __name__ == "__main__":
    main()
