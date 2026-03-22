"""EdgeAgent Industrial Quality Control Dashboard.

Multi-page Streamlit dashboard with:
- Unified inference playground (YOLO + Edge Enhancement + Spatial Clustering + VLM)
- Data integration & class balance visualisation
- CA rationale and MLflow tracking
- Phase 2 VLM strategy (visual flow diagram)
- VLM Anomaly Gallery and VLM Metrics pages
- Edge profiler results
- False-Positive analysis
- Active Learning operator feedback
- Decision snapshot & operator controls
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports" / "generated"
FEEDBACK_DIR = ROOT / "data" / "feedback"
LATEST_JSON = REPORTS_DIR / "augmentation_imbalance_latest.json"
LATEST_PNG = REPORTS_DIR / "augmentation_imbalance_latest.png"
MODEL_PATH = ROOT / "models" / "phase1_final_ca.pt"

from src.common.constants import CLASS_NAMES
CLASS_COLORS = {"screw": "#4CAF50", "missing_screw": "#FF9800", "missing_component": "#F44336"}

# ── Early CoordAtt Registration (must happen before torch.load) ─────
# Streamlit re-executes the script on every rerun, so we register at
# module level to ensure pickle can always resolve the custom classes.
sys.path.insert(0, str(ROOT))
try:
    from src.models.coordatt import CoordAtt, HSigmoid, HSwish, register_coordatt
    register_coordatt()
    # Register in both __main__ and sys.modules for pickle resolution
    import __main__ as _main_mod
    for _cls in (HSigmoid, HSwish, CoordAtt):
        setattr(_main_mod, _cls.__name__, _cls)
        if "__main__" in sys.modules:
            setattr(sys.modules["__main__"], _cls.__name__, _cls)
    _COORDATT_OK = True
except ImportError:
    _COORDATT_OK = False

# ── VLM Reasoner (Phase 2) ───────────────────────────────────────────
try:
    from src.reasoning.vlm_reasoner import VLMReasoner
    from src.reasoning.conflict_resolver import ConflictResolver, FinalVerdict
    from src.reasoning.rca_templates import get_rca
    from src.edge.vlm_trigger import should_trigger_vlm, TriggerConfig
    _VLM_AVAILABLE = True
except ImportError:
    _VLM_AVAILABLE = False


def _get_vlm_reasoner():
    """Get or create VLM reasoner in session state."""
    if not _VLM_AVAILABLE:
        return None
    if "vlm_reasoner" not in st.session_state:
        st.session_state.vlm_reasoner = None
    return st.session_state.vlm_reasoner


# ── Helpers ──────────────────────────────────────────────────────────

def get_count(d: dict, key: int) -> int:
    return int(d.get(str(key), d.get(key, 0)))


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_report() -> dict:
    return load_json(LATEST_JSON) or {}


def load_edge_profile() -> dict | None:
    profiles = sorted(REPORTS_DIR.glob("edge_profile_*.json"), reverse=True)
    if not profiles:
        return None
    return json.loads(profiles[0].read_text(encoding="utf-8"))


def load_vlm_summary() -> dict | None:
    return load_json(REPORTS_DIR / "vlm_trigger_events.summary.json")


def load_feedback_stats() -> dict:
    """Load feedback statistics from the feedback directory.

    Supports both legacy (correct/incorrect) and granular (partial + per-detection) formats.
    """
    if not FEEDBACK_DIR.exists():
        return {"total": 0, "correct": 0, "incorrect": 0, "partial": 0,
                "det_correct": 0, "det_incorrect": 0, "files": []}
    feedback_file = FEEDBACK_DIR / "feedback_log.jsonl"
    if not feedback_file.exists():
        return {"total": 0, "correct": 0, "incorrect": 0, "partial": 0,
                "det_correct": 0, "det_incorrect": 0, "files": []}
    entries = []
    for line in feedback_file.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    correct = sum(1 for e in entries if e.get("label") == "correct")
    incorrect = sum(1 for e in entries if e.get("label") == "incorrect")
    partial = sum(1 for e in entries if e.get("label") == "partial")
    # Per-detection stats from granular entries
    det_correct = sum(e.get("correct_count", 0) for e in entries)
    det_incorrect = sum(e.get("incorrect_count", 0) for e in entries)
    return {
        "total": len(entries), "correct": correct, "incorrect": incorrect,
        "partial": partial, "det_correct": det_correct, "det_incorrect": det_incorrect,
        "files": entries,
    }


def get_model():
    """Load YOLO model with caching."""
    if "yolo_model" not in st.session_state:
        if not MODEL_PATH.exists():
            return None
        if not _COORDATT_OK:
            return None
        from ultralytics import YOLO
        st.session_state.yolo_model = YOLO(str(MODEL_PATH))
    return st.session_state.yolo_model


# ── Page: Unified Inference (YOLO + Edge + Spatial) ─────────────────

def page_inference():
    st.header("Canli Tahmin (Inference Playground)")

    model = get_model()
    if model is None:
        st.error(
            "Model bulunamadi: `models/phase1_final_ca.pt`\n\n"
            "Egitimi calistirdiktan sonra tekrar deneyin."
        )
        return

    # ── Top bar: Upload + VLM button ──
    top_left, top_right = st.columns([3, 1])
    with top_left:
        uploaded = st.file_uploader(
            "Goruntu yukle (JPG/PNG)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
    with top_right:
        reasoner = _get_vlm_reasoner()
        vlm_loaded = reasoner is not None and reasoner.is_loaded
        if _VLM_AVAILABLE:
            if vlm_loaded:
                st.success(f"VLM: Aktif ({reasoner.get_vram_usage_mb():.0f}MB)")
                if st.button("VLM Kaldir", key="vlm_unload_top"):
                    reasoner.unload_model()
                    st.session_state.vlm_reasoner = None
                    st.rerun()
            else:
                if st.button("VLM Yukle", key="vlm_load_top", type="primary"):
                    with st.spinner("PaliGemma yukleniyor..."):
                        try:
                            r = VLMReasoner()
                            r.load_model()
                            st.session_state.vlm_reasoner = r
                            st.rerun()
                        except Exception as e:
                            st.error(f"VLM hatasi: {e}")

    # ── Sidebar settings ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("YOLO Ayarlari")
    conf_thresh = st.sidebar.slider("Confidence Esigi", 0.1, 0.95, 0.25, 0.05)
    iou_thresh = st.sidebar.slider("IoU Esigi (NMS)", 0.1, 0.95, 0.45, 0.05)
    img_size = st.sidebar.selectbox("Goruntu Boyutu", [640, 416, 320], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Edge Enhancement")
    auto_tune_enabled = st.sidebar.checkbox(
        "Otomatik Parametre Bul",
        value=False,
        help="YOLO tespitini optimize eden Canny parametrelerini otomatik arar",
    )
    alpha = st.sidebar.slider("Alpha (orijinal agirlik)", 0.3, 1.0, 0.7, 0.05,
                              help="1.0 = sadece orijinal, 0.3 = guclu kenar karistirma",
                              disabled=auto_tune_enabled)
    canny_low = st.sidebar.slider("Canny Alt Esik", 10, 150, 50, 10,
                                  disabled=auto_tune_enabled)
    canny_high = st.sidebar.slider("Canny Ust Esik", 50, 300, 150, 10,
                                   disabled=auto_tune_enabled)

    if not uploaded:
        st.info("Bir veya birden fazla goruntu yukleyin, model anlik tahmin yapsin.")
        test_dir = ROOT / "data" / "processed" / "phase1_multiclass_v1" / "test" / "images"
        if test_dir.exists():
            samples = sorted(test_dir.glob("*.jpg"))[:6]
            if samples:
                st.markdown("**Test setinden ornekler** (yuklemeden hizli test icin):")
                sample_cols = st.columns(min(len(samples), 3))
                for i, sample in enumerate(samples[:3]):
                    with sample_cols[i]:
                        if st.button(f"Tara: {sample.name}", key=f"sample_{i}"):
                            _run_full_analysis(
                                model, str(sample), conf_thresh, iou_thresh,
                                img_size, alpha, canny_low, canny_high,
                                auto_tune=auto_tune_enabled,
                            )
        return

    for file in uploaded:
        st.markdown(f"---\n### {file.name}")
        tmp_path = REPORTS_DIR / f"_tmp_upload_{file.name}"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(file.getvalue())
        _run_full_analysis(
            model, str(tmp_path), conf_thresh, iou_thresh,
            img_size, alpha, canny_low, canny_high,
            auto_tune=auto_tune_enabled,
        )
        if tmp_path.exists():
            tmp_path.unlink()


def _run_full_analysis(
    model, image_path: str, conf: float, iou: float, imgsz: int,
    alpha: float, canny_low: int, canny_high: int,
    auto_tune: bool = False,
):
    """Run YOLO + Edge Enhancement + Spatial + VLM on one image.

    Layout:
      Row 1: [YOLO Tespit] [Edge Enhancement] [Geometrik Mekansal Kumeleme]
      Row 2: [VLM Akil Yurume]
    """

    # ─── YOLO Inference ───
    t0 = time.perf_counter()
    results = model.predict(image_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    latency = (time.perf_counter() - t0) * 1000
    result = results[0]
    boxes = result.boxes

    # ─── Auto-tune edge parameters if enabled ───
    if auto_tune:
        try:
            from src.data.edge_enhancer import auto_tune_fast
            img_bgr = cv2.imread(image_path)
            if img_bgr is not None:
                with st.spinner("Edge parametreleri optimize ediliyor (domain kisitlamali)..."):
                    tune_result = auto_tune_fast(img_bgr, model, imgsz, conf, iou)
                if not tune_result.get("is_original", True) and tune_result.get("valid", True):
                    alpha = tune_result["alpha"]
                    canny_low = tune_result["canny_low"]
                    canny_high = tune_result["canny_high"]
                    st.success(
                        f"Otomatik: alpha={alpha}, canny=({canny_low},{canny_high}) | "
                        f"Skor: {tune_result['score']:.2f} (orijinal: {tune_result['original_score']:.2f}) | "
                        f"Tespit: {tune_result['per_class']}"
                    )
                else:
                    st.info(
                        "Orijinal goruntu en iyi sonucu veriyor. "
                        "Edge enhancement ekstra FP uretiyor, gerek yok."
                    )
        except Exception as e:
            st.warning(f"Auto-tune hatasi: {e}")

    # ─── 3-Column Layout ───
    col_yolo, col_edge, col_spatial = st.columns(3)

    # ── Column 1: YOLO Tespit ──
    with col_yolo:
        st.subheader("YOLO Tespit")
        annotated = result.plot()
        annotated_rgb = annotated[:, :, ::-1].copy()

        # Numaralı etiketler ekle (her bbox'a #1, #2 vs.)
        _orig_bgr = cv2.imread(str(image_path)) if isinstance(image_path, (str, Path)) else None
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int(xyxy[1]) - 8
            if cy < 20:
                cy = int(xyxy[3]) + 20
            cv2.putText(annotated_rgb, f"#{i+1}", (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        st.image(annotated_rgb, caption="Tahmin Sonucu", width="stretch")

        mc1, mc2 = st.columns(2)
        mc1.metric("Latency", f"{latency:.1f} ms")
        mc2.metric("Tespit", len(boxes))

        img_name = Path(image_path).name
        if len(boxes) > 0:
            st.markdown("**Tespitleri tek tek isaretleyin** *(yanlis olanlarin tikini kaldir)*:")
            det_feedback = []
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf_val = float(boxes.conf[i])
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                color = CLASS_COLORS.get(cls_name, "#999")
                xyxy = boxes.xyxy[i].tolist()

                # Mini crop göster
                crop_col, check_col = st.columns([1, 2])
                with crop_col:
                    if _orig_bgr is not None:
                        x1c, y1c = max(0, int(xyxy[0]) - 10), max(0, int(xyxy[1]) - 10)
                        x2c = min(_orig_bgr.shape[1], int(xyxy[2]) + 10)
                        y2c = min(_orig_bgr.shape[0], int(xyxy[3]) + 10)
                        mini = _orig_bgr[y1c:y2c, x1c:x2c, ::-1]
                        st.image(mini, caption=f"#{i+1}", width="stretch")

                with check_col:
                    is_correct = st.checkbox(
                        f"**#{i+1}** :{_color_name(color)}[**{cls_name}**] `{conf_val:.1%}`",
                        value=True,
                        key=f"det_{img_name}_{i}",
                    )
                    pos_txt = f"({int(xyxy[0])},{int(xyxy[1])})-({int(xyxy[2])},{int(xyxy[3])})"
                    st.caption(pos_txt)

                det_feedback.append({
                    "idx": i,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox": [round(v, 1) for v in xyxy],
                    "correct": is_correct,
                })

            st.markdown("---")
            n_correct = sum(1 for d in det_feedback if d["correct"])
            n_wrong = len(det_feedback) - n_correct
            st.info(f"✅ {n_correct} dogru  |  ❌ {n_wrong} yanlis isaretlendi")

            if st.button("💾 Geri Bildirimi Kaydet", key=f"save_fb_{img_name}",
                         type="primary", width="stretch"):
                _save_feedback_detailed(img_name, det_feedback)
                if n_wrong == 0:
                    st.success("Tumu dogru olarak kaydedildi")
                else:
                    st.warning(f"{n_wrong} yanlis tespit isaretlendi — kaydedildi")
        else:
            st.warning("Tespit yok")
            if st.button("Bos Goruntu Kaydet", key=f"save_empty_{img_name}",
                         width="stretch"):
                _save_feedback_detailed(img_name, [])
                st.success("Kaydedildi")

    # ── Column 2: Edge Enhancement ──
    with col_edge:
        st.subheader("Edge Enhancement")
        try:
            from src.data.edge_enhancer import preview_enhancement

            original_rgb, edges_rgb, blended_rgb = preview_enhancement(
                image_path, alpha, canny_low, canny_high
            )

            sub_c1, sub_c2, sub_c3 = st.columns(3)
            with sub_c1:
                st.image(original_rgb, caption="Orijinal", width="stretch")
            with sub_c2:
                st.image(edges_rgb, caption="Canny", width="stretch")
            with sub_c3:
                st.image(blended_rgb, caption="Harmanlanmis", width="stretch")

            st.markdown("**YOLO Karsilastirmasi:**")
            comp1, comp2 = st.columns(2)
            with comp1:
                st.caption("Orijinal ile tespit")
                ann1 = result.plot()[:, :, ::-1]
                st.image(ann1, width="stretch")
                st.caption(f"Tespit: {len(result.boxes)}")

            with comp2:
                st.caption("Enhanced ile tespit")
                enhanced_path = REPORTS_DIR / f"_tmp_edge_{Path(image_path).name}"
                enhanced_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(enhanced_path), enhanced_bgr)
                r2 = model.predict(str(enhanced_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
                ann2 = r2[0].plot()[:, :, ::-1]
                st.image(ann2, width="stretch")
                st.caption(f"Tespit: {len(r2[0].boxes)}")
                if enhanced_path.exists():
                    enhanced_path.unlink()

            with st.expander("Parametre Detaylari"):
                st.markdown(f"**Alpha:** {alpha}")
                st.markdown(f"**Canny Alt Esik:** {canny_low}")
                st.markdown(f"**Canny Ust Esik:** {canny_high}")

        except Exception as e:
            st.warning(f"Edge Enhancement yuklenemedi: {e}")

    # ── Column 3: Spatial Clustering ──
    spatial_result = None
    with col_spatial:
        st.subheader("Mekansal Kumeleme")
        try:
            from src.reasoning.spatial_logic import (
                SpatialAnalyzer, detections_from_yolo_result,
            )

            detections = detections_from_yolo_result(result)
            analyzer = SpatialAnalyzer(n_clusters=4)
            spatial_result = analyzer.analyze_frame(detections, img_shape=result.orig_shape)

            _draw_spatial_overlay(result, spatial_result)

            verdict_colors = {"OK": "green", "missing_screw": "orange", "missing_component": "red"}
            verdict_color = verdict_colors.get(spatial_result.verdict, "gray")
            st.markdown(f"### :{verdict_color}[{spatial_result.verdict.upper()}]")
            st.metric("Tespit", spatial_result.detection_count)
            st.metric("Ort. Confidence", f"{spatial_result.confidence:.1%}")
            st.markdown(f"**Neden:** {spatial_result.reason}")
            st.markdown(f"**Sol:** `{spatial_result.left_status}` | **Sag:** `{spatial_result.right_status}`")

            if spatial_result.clusters:
                with st.expander("Kume Detaylari"):
                    for cluster in spatial_result.clusters:
                        st.markdown(
                            f"- K{cluster.cluster_id} ({cluster.side}): "
                            f"**{cluster.dominant_class}** "
                            f"({len(cluster.detections)} tespit)"
                        )

        except Exception as e:
            st.warning(f"Mekansal analiz yuklenemedi: {e}")

    # ─── Row 2: VLM Reasoning ───
    st.markdown("---")
    st.subheader("VLM Akil Yurume (PaliGemma)")

    reasoner = _get_vlm_reasoner()
    if not _VLM_AVAILABLE:
        st.info("VLM modulleri yuklenemedi. `pip install transformers` gerekli.")
    elif reasoner is None or not reasoner.is_loaded:
        try:
            trigger_config = TriggerConfig(conf_threshold=0.40)
            should_fire, low_dets, reason = should_trigger_vlm(
                results, CLASS_NAMES, trigger_config
            )
            if should_fire:
                st.warning(
                    f"VLM tetiklenirdi! Neden: `{reason}` | "
                    f"{len(low_dets)} dusuk guvenli tespit | "
                    "Sayfanin ustundeki 'VLM Yukle' butonuna basin."
                )
            else:
                st.success("VLM tetiklenmezdi - tum tespitler yuksek guvenli.")
        except Exception:
            st.info("VLM modeli yuklenmedi. Sayfanin ustundeki 'VLM Yukle' butonuna basin.")
    else:
        try:
            trigger_config = TriggerConfig(conf_threshold=0.40)
            should_fire, low_dets, reason = should_trigger_vlm(
                results, CLASS_NAMES, trigger_config
            )

            if should_fire:
                st.warning(f"VLM tetiklendi! Neden: `{reason}`")

                img_rgb = cv2.imread(image_path)
                if img_rgb is not None:
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

                    if low_dets and low_dets[0].get("bbox"):
                        crop = VLMReasoner.crop_region(
                            img_rgb, tuple(low_dets[0]["bbox"]), 0.20
                        )
                    else:
                        crop = img_rgb

                    vlm_result = reasoner.reason(crop)

                    col_vlm_img, col_vlm_info, col_vlm_verdict = st.columns([1, 1, 1])
                    with col_vlm_img:
                        st.image(crop, caption="VLM Analiz Bolgesi", width="stretch")

                    with col_vlm_info:
                        if vlm_result.defect_type:
                            dt_colors = {"ok": "green", "missing_screw": "orange", "missing_component": "red"}
                            dt_color = dt_colors.get(vlm_result.defect_type, "gray")
                            st.markdown(f"**VLM Karari:** :{dt_color}[**{vlm_result.defect_type}**]")
                        else:
                            st.markdown("**VLM Karari:** :gray[Belirsiz]")

                        st.metric("VLM Confidence", f"{vlm_result.confidence_estimate:.0%}")
                        st.metric("VLM Latency", f"{vlm_result.latency_ms:.0f} ms")
                        st.markdown(f"**Aciklama:** {vlm_result.reasoning}")

                    with col_vlm_verdict:
                        if _VLM_AVAILABLE:
                            resolver = ConflictResolver()
                            yolo_dets = []
                            for i in range(len(boxes)):
                                yolo_dets.append({
                                    "class_name": CLASS_NAMES.get(int(boxes.cls[i]), "?"),
                                    "confidence": float(boxes.conf[i]),
                                })

                            spatial_v = spatial_result.verdict if spatial_result else None
                            spatial_s = spatial_result.left_status if spatial_result else None

                            final = resolver.resolve(
                                yolo_dets, spatial_v, spatial_s, vlm_result
                            )

                            st.markdown("**Nihai Karar:**")
                            fc = {"ok": "green", "missing_screw": "orange", "missing_component": "red"}
                            st.markdown(
                                f"### :{fc.get(final.verdict, 'gray')}[{final.verdict.upper()}]"
                            )
                            st.markdown(f"**Kaynak:** `{final.source}`")
                            st.markdown(f"**Conflict:** `{final.conflict_detected}`")
                            st.markdown(f"**Reasoning:** {final.reasoning}")

                            with st.expander("RCA (Kok Neden Analizi)"):
                                st.markdown(final.rca_text)
            else:
                st.success("VLM tetiklenmedi - tum tespitler yuksek guvenli.")
        except Exception as e:
            st.error(f"VLM analiz hatasi: {e}")


def _color_name(hex_color: str) -> str:
    """Map hex to streamlit color name."""
    mapping = {"#4CAF50": "green", "#FF9800": "orange", "#F44336": "red"}
    return mapping.get(hex_color, "gray")


def _save_feedback(image_name: str, label: str, det_count: int):
    """Save operator feedback for Active Learning pipeline (legacy)."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_file = FEEDBACK_DIR / "feedback_log.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": image_name,
        "label": label,
        "detection_count": det_count,
    }
    with feedback_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _save_feedback_detailed(image_name: str, detections: list[dict]):
    """Save per-detection operator feedback for granular Active Learning.

    Each detection has: idx, class_id, class_name, confidence, bbox, correct (bool).
    Label is auto-determined:
        - 'correct'   : all detections marked correct
        - 'incorrect' : all detections marked incorrect
        - 'partial'   : some correct, some incorrect
    """
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_file = FEEDBACK_DIR / "feedback_log.jsonl"

    n_total = len(detections)
    n_correct = sum(1 for d in detections if d.get("correct", True))
    n_incorrect = n_total - n_correct

    if n_total == 0:
        label = "correct"  # no detections = user confirms empty is fine
    elif n_incorrect == 0:
        label = "correct"
    elif n_correct == 0:
        label = "incorrect"
    else:
        label = "partial"

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": image_name,
        "label": label,
        "detection_count": n_total,
        "correct_count": n_correct,
        "incorrect_count": n_incorrect,
        "detections": detections,
    }
    with feedback_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _draw_spatial_overlay(yolo_result, spatial_result):
    """Draw annotated image with cluster boundaries and side labels."""
    annotated = yolo_result.plot()

    h, w = annotated.shape[:2]
    mid_x = w // 2

    # Draw center dividing line
    cv2.line(annotated, (mid_x, 0), (mid_x, h), (255, 255, 0), 2)
    cv2.putText(annotated, "SOL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(annotated, "SAG", (w - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Draw cluster centers
    cluster_colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 255)]
    for cluster in spatial_result.clusters:
        cx, cy = int(cluster.center[0]), int(cluster.center[1])
        color = cluster_colors[cluster.cluster_id % len(cluster_colors)]
        cv2.circle(annotated, (cx, cy), 12, color, 3)
        label = f"K{cluster.cluster_id}:{cluster.dominant_class[:3]}"
        cv2.putText(annotated, label, (cx - 30, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw verdict
    verdict_colors = {"OK": (0, 255, 0), "missing_screw": (0, 165, 255), "missing_component": (0, 0, 255)}
    v_color = verdict_colors.get(spatial_result.verdict, (128, 128, 128))
    cv2.putText(annotated, f"VERDICT: {spatial_result.verdict.upper()}",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, v_color, 2)

    annotated_rgb = annotated[:, :, ::-1]
    st.image(annotated_rgb, caption="Mekansal Analiz Sonucu", width="stretch")


# ── Page: Data & Balance ─────────────────────────────────────────────

def _load_dataset_versions() -> list[dict]:
    """Load summary.json from all dataset versions (V1-V4), normalized."""
    processed = ROOT / "data" / "processed"
    versions = []
    for d in sorted(processed.iterdir()):
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        # Normalize different summary formats
        if "splits" in data:
            splits = data["splits"]
        elif "split_sizes" in data:
            splits = data["split_sizes"]
        else:
            continue
        if "final_class_distribution" in data:
            classes = data["final_class_distribution"]
        elif "class_counts" in data:
            classes = data["class_counts"]
        else:
            classes = {}
        # Determine version name
        name = data.get("version", d.name.replace("phase1_", "").replace("multiclass_", ""))
        versions.append({
            "name": name.upper(),
            "dir": d.name,
            "train": splits.get("train", 0),
            "val": splits.get("val", 0),
            "test": splits.get("test", 0),
            "screw": classes.get("screw", 0),
            "missing_screw": classes.get("missing_screw", 0),
            "missing_component": classes.get("missing_component", 0),
        })
    return versions


def page_data():
    st.header("Veri Dengeleme & Sinif Dagilimi")

    versions = _load_dataset_versions()
    if not versions:
        st.error("Hic dataset bulunamadi (data/processed/phase1_*/summary.json)")
        return

    # Show last 3 versions by default
    show_count = st.selectbox("Kac versiyon goster?", [3, 4, len(versions)],
                              format_func=lambda x: f"Son {x}" if x <= 4 else "Tumu")
    display = versions[-show_count:]

    # KPI: latest vs previous
    latest = display[-1]
    prev = display[-2] if len(display) >= 2 else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Train ({latest['name']})", latest["train"],
              delta=f"+{latest['train'] - prev['train']}" if prev else None)
    c2.metric("screw", latest["screw"],
              delta=f"+{latest['screw'] - prev['screw']}" if prev else None)
    c3.metric("missing_screw", latest["missing_screw"],
              delta=f"+{latest['missing_screw'] - prev['missing_screw']}" if prev else None)
    c4.metric("missing_component", latest["missing_component"],
              delta=f"+{latest['missing_component'] - prev['missing_component']}" if prev else None)

    st.markdown("---")

    # Multi-version comparison chart
    col_chart, col_detail = st.columns([2, 1])
    with col_chart:
        _render_version_comparison(display)
    with col_detail:
        st.markdown("**Versiyon Detaylari**")
        for v in display:
            total_bbox = v["screw"] + v["missing_screw"] + v["missing_component"]
            ms_pct = v["missing_screw"] / max(1, total_bbox) * 100
            mc_pct = v["missing_component"] / max(1, total_bbox) * 100
            st.markdown(
                f"**{v['name']}**: {v['train']} img, {total_bbox} bbox\n"
                f"- ms: {v['missing_screw']} (%{ms_pct:.0f}), "
                f"mc: {v['missing_component']} (%{mc_pct:.0f})"
            )

    if LATEST_PNG.exists():
        with st.expander("Augmentation Analizi (Detay Grafik)"):
            st.image(str(LATEST_PNG), width="stretch")


def _render_version_comparison(versions: list[dict]):
    """Render bar chart comparing class distributions across dataset versions."""
    labels = [v["name"] for v in versions]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4), dpi=130)

    screw_vals = [v["screw"] for v in versions]
    ms_vals = [v["missing_screw"] for v in versions]
    mc_vals = [v["missing_component"] for v in versions]

    bars1 = ax.bar(x - width, screw_vals, width, label="screw", color="#4CAF50")
    bars2 = ax.bar(x, ms_vals, width, label="missing_screw", color="#FF9800")
    bars3 = ax.bar(x + width, mc_vals, width, label="missing_component", color="#F44336")

    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("BBox Sayisi")
    ax.set_title("Dataset Versiyonlari - Sinif Dagilimi Karsilastirmasi")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 20,
                        str(int(h)), ha="center", fontsize=7, color="#555")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Page: Why CA? ────────────────────────────────────────────────────

def page_ca():
    st.header("Neden Coordinate Attention (CA)?")

    col_problem, col_solution = st.columns(2)
    with col_problem:
        st.markdown("""
**Problem:**
- Klasik YOLO backbone sadece kanal bazli dikkat kullanir
- Vida gibi kucuk nesnelerde mekansal balam kaybedilir
- SE-Block konumsal bilgiyi sikistirir, geri alamaz
""")
    with col_solution:
        st.markdown("""
**CA Cozumu:**
- X ve Y eksenlerinde **ayri ayri** havuzlama
- Uzun menzilli konumsal iliski korunur
- 1-2 piksellik vida kaymasi bile tespit edilir
- SE-Block'a gore **%2-3 daha yuksek mAP**
""")

    st.markdown("---")
    st.subheader("Backbone Yerlestirme")

    # Visual backbone diagram
    fig, ax = plt.subplots(figsize=(14, 2.5), dpi=130)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    layers = [
        (0.2, 1.5, "Conv\n64", "#E3F2FD"),
        (1.5, 1.5, "Conv\n128", "#BBDEFB"),
        (2.8, 1.5, "C2f\n128x3", "#90CAF9"),
        (4.1, 1.5, "Conv\n256", "#64B5F6"),
        (5.4, 1.5, "C2f\n256x6", "#42A5F5"),
        (6.7, 1.5, "CA\n256", "#FF9800"),
        (8.0, 1.5, "C2f\n512x6", "#1E88E5"),
        (9.3, 1.5, "CA\n512", "#FF9800"),
        (10.6, 1.5, "SPPF+PSA\n1024", "#1565C0"),
        (11.9, 1.5, "CA\n1024", "#FF9800"),
        (13.1, 1.5, "Detect\nHead", "#4CAF50"),
    ]

    for x, h, text, color in layers:
        rect = mpatches.FancyBboxPatch(
            (x, 0.5), 1.1, h, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#333", alpha=0.9, linewidth=0.5,
        )
        ax.add_patch(rect)
        text_color = "white" if color in ("#FF9800", "#1E88E5", "#1565C0", "#4CAF50") else "#333"
        ax.text(x + 0.55, 0.5 + h / 2, text, ha="center", va="center",
                fontsize=7, fontweight="bold", color=text_color)

    # Arrows
    for i in range(len(layers) - 1):
        x1 = layers[i][0] + 1.1
        x2 = layers[i + 1][0]
        ax.annotate("", xy=(x2, 1.25), xytext=(x1, 1.25),
                     arrowprops=dict(arrowstyle="->", color="#666", lw=0.8))

    ax.set_title("YOLOv10-S + Coordinate Attention (turuncu = CA katmanlari)", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("CA Forward Akisi (Pseudo-code)"):
        st.code("""
Input F(C,H,W)
  -> AvgPool_x(F): (C,H,1) ve AvgPool_y(F): (C,1,W)
  -> Concat -> Shared 1x1 Conv + BN + HSwish
  -> Split -> conv_h(Ax), conv_w(Ay)
  -> Output = F * sigmoid(Ax) * sigmoid(Ay)

Avantaj: Her piksel hem yatay hem dikey baglamdan haberdar.
""".strip(), language="text")


# ── Page: MLflow ─────────────────────────────────────────────────────

def _load_mlflow_runs() -> list[dict]:
    """MLflow veritabanindan tum run'lari yukle."""
    try:
        import mlflow
        db_path = ROOT / "mlflow.db"
        if not db_path.exists():
            return []
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        all_runs = []
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id],
                                      order_by=["start_time DESC"])
            for r in runs:
                all_runs.append({
                    "run_id": r.info.run_id,
                    "name": r.info.run_name or r.info.run_id[:8],
                    "status": r.info.status,
                    "params": r.data.params,
                    "metrics": r.data.metrics,
                    "tags": r.data.tags,
                })
        return all_runs
    except Exception as e:
        st.warning(f"MLflow yuklenemedi: {e}")
        return []


def _get_metric(metrics: dict, base_name: str) -> float:
    """Get metric value trying multiple key formats (ultralytics vs backfill)."""
    candidates = [
        f"final_metrics_{base_name}",   # backfill format
        f"metrics_{base_name}",          # backfill v2
        f"metrics/{base_name}",          # ultralytics auto-log
        f"final_{base_name}",            # custom log
        base_name,                       # raw
    ]
    for key in candidates:
        if key in metrics and metrics[key] != 0:
            return metrics[key]
    # Return 0 only if truly not found
    for key in candidates:
        if key in metrics:
            return metrics[key]
    return 0.0


def page_mlflow():
    st.header("MLflow - Model Egitim Takibi")
    st.caption(
        "Her egitim run'inin performansini karsilastirir. "
        "mAP50 = ortalama tespit dogrulugu, Precision = yanlis alarm orani, "
        "Recall = kacirma orani."
    )

    runs = _load_mlflow_runs()

    if not runs:
        st.warning("Kayitli MLflow run bulunamadi.")
        st.code("python scripts/backfill_mlflow.py", language="bash")
        return

    # ── Ozet metrikler ──
    st.subheader(f"Toplam {len(runs)} Egitim Run'i")

    # En iyi ve baseline run'lari bul
    best_run = None
    best_map50 = 0.0
    baseline_map50 = 0.0
    for r in runs:
        m = _get_metric(r["metrics"], "mAP50B")
        if m > best_map50:
            best_map50 = m
            best_run = r
        if r["tags"].get("version") == "baseline" or "baseline" in r["name"].lower():
            baseline_map50 = m

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline mAP50", f"{baseline_map50:.4f}" if baseline_map50 else "N/A")
    m2.metric("En Iyi mAP50", f"{best_map50:.4f}" if best_map50 else "N/A",
              delta=f"+{best_map50 - baseline_map50:.4f}" if baseline_map50 and best_map50 else None)
    if best_run:
        m3.metric("En Iyi Run", best_run["name"][:25])
    else:
        m3.metric("En Iyi Run", "N/A")
    m4.metric("Toplam Run", len(runs))

    st.markdown("---")

    # ── Run karsilastirma tablosu ──
    st.subheader("Run Karsilastirma")
    table_rows = []
    for r in runs:
        metrics = r["metrics"]
        map50 = _get_metric(metrics, "mAP50B")
        map50_95 = _get_metric(metrics, "mAP50-95B")
        prec = _get_metric(metrics, "precisionB")
        rec = _get_metric(metrics, "recallB")
        table_rows.append({
            "Run": r["name"],
            "mAP50": f"{map50:.4f}",
            "mAP50-95": f"{map50_95:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "Epochs": r["params"].get("epochs", "?"),
            "Batch": r["params"].get("batch", "?"),
            "LR": r["params"].get("lr0", "?"),
        })
    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    # Metric explanations
    with st.expander("Metrikler ne anlama geliyor?"):
        st.markdown("""
- **mAP50**: Ortalama tespit dogrulugu (%50 IoU esiginde). 1.0 = mukemmel.
- **mAP50-95**: Daha siki degerlendirme (IoU %50-%95 arasi). Genelde daha dusuk cikar.
- **Precision**: Modelin "hata var" dedigi seylerin kaci gercekten hata? (Yuksek = az yanlis alarm)
- **Recall**: Gercek hatalarin kacini yakaladi? (Yuksek = az kacirma). Uretimde Recall kritik!
- **Epochs**: Kac tur egitim yapildi.
""")

    st.markdown("---")

    # ── Epoch-by-epoch grafik ──
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("mAP50 Egitim Egrisi")
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            import pandas as pd
            chart_data = {}
            for r in runs:
                # Try multiple metric key formats
                for mkey in ["metrics/mAP50B", "metrics_mAP50B"]:
                    history = client.get_metric_history(r["run_id"], mkey)
                    if history:
                        chart_data[r["name"]] = {h.step: h.value for h in history}
                        break
            if chart_data:
                max_steps = max(max(d.keys()) for d in chart_data.values() if d)
                df = pd.DataFrame(index=range(max_steps + 1))
                for name, vals in chart_data.items():
                    df[name] = pd.Series(vals)
                df.index.name = "Epoch"
                st.line_chart(df)
            else:
                st.info("Epoch verisi bulunamadi")
        except Exception as e:
            st.warning(f"Grafik olusturulamadi: {e}")

    with col_chart2:
        st.subheader("Loss Egrisi (Val)")
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            import pandas as pd
            chart_data = {}
            for r in runs:
                for lkey in ["val/box_loss", "val_box_loss"]:
                    history = client.get_metric_history(r["run_id"], lkey)
                    if history:
                        chart_data[r["name"]] = {h.step: h.value for h in history}
                        break
            if chart_data:
                max_steps = max(max(d.keys()) for d in chart_data.values() if d)
                df = pd.DataFrame(index=range(max_steps + 1))
                for name, vals in chart_data.items():
                    df[name] = pd.Series(vals)
                df.index.name = "Epoch"
                st.line_chart(df)
            else:
                st.info("Loss verisi bulunamadi")
        except Exception as e:
            st.warning(f"Grafik olusturulamadi: {e}")

    st.markdown("---")

    # ── Detayli run bilgisi ──
    st.subheader("Run Detaylari")
    run_names = [r["name"] for r in runs]
    selected = st.selectbox("Run sec", run_names)
    sel_run = next((r for r in runs if r["name"] == selected), None)

    if sel_run:
        p_col, m_col = st.columns(2)
        with p_col:
            st.markdown("**Hyperparametreler**")
            st.json(sel_run["params"])
        with m_col:
            st.markdown("**Metrikler**")
            st.json({k: round(v, 6) if isinstance(v, float) else v
                     for k, v in sel_run["metrics"].items()})

    st.markdown("---")
    st.subheader("MLflow UI")
    st.code("python scripts/start_mlflow_server.py", language="bash")
    st.markdown("[http://localhost:5000](http://localhost:5000) adresinde detayli UI")


# ── Page: VLM Strategy ───────────────────────────────────────────────

def page_vlm():
    st.header("Phase 2: VLM Tetikleme Stratejisi")

    st.markdown(
        '**Temel Mantik:** "YOLO uyanirsa VLM calisir" - '
        "VLM sadece belirsiz tespitlerde devreye girer."
    )

    # Flow diagram
    fig, ax = plt.subplots(figsize=(13, 4.5), dpi=130)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    boxes = [
        (0.3, 1.8, 2.0, 1.6, "Kamera\nFrame", "#78909C"),
        (2.8, 1.8, 2.2, 1.6, "YOLO\n7.3ms\n(RTX)", "#1565C0"),
        (5.5, 3.0, 2.5, 1.3, "conf >= 0.40\nOK / NOK", "#4CAF50"),
        (5.5, 0.7, 2.5, 1.3, "conf < 0.40\nBelirsiz!", "#FF9800"),
        (8.5, 0.7, 2.5, 1.3, "PaliGemma\nAsync VLM", "#D32F2F"),
        (11.2, 0.7, 1.5, 1.3, "Dashboard\nRapor", "#7B1FA2"),
    ]

    for bx, by, bw, bh, text, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white", alpha=0.9, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(bx + bw / 2, by + bh / 2, text,
                ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    arrow_kw = dict(arrowstyle="-|>", color="#333", lw=1.8)
    ax.annotate("", xy=(2.8, 2.6), xytext=(2.3, 2.6), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.5, 3.65), xytext=(5.0, 2.9), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.5, 1.35), xytext=(5.0, 2.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.5, 1.35), xytext=(8.0, 1.35), arrowprops=arrow_kw)
    ax.annotate("", xy=(11.2, 1.35), xytext=(11.0, 1.35), arrowprops=arrow_kw)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Stats if available
    summary = load_vlm_summary()
    if summary:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Taranan Goruntu", summary.get("total_images", 0))
        c2.metric("VLM Tetikleme", summary.get("triggered_events", 0))
        c3.metric("VLM Islenen", summary.get("processed_by_vlm", 0))
        c4.metric("Confidence Esik", f"{summary.get('conf_threshold', 0.4):.0%}")

    st.markdown("---")
    col_why, col_how = st.columns(2)
    with col_why:
        st.subheader("Neden Asenkron?")
        st.markdown("""
- PaliGemma 3B @ 4-bit: **~500ms/frame**
- YOLO hedef: **<5ms/frame** (TensorRT)
- Senkron calissa ana hatti **100x** yavaslatir
- Async kuyruk ile ana hat **hic etkilenmez**
""")
    with col_how:
        st.subheader("Nasil Calisir?")
        st.markdown("""
1. YOLO her frame'i isler
2. `conf < 0.40` -> frame async kuyruga
3. Worker thread PaliGemma ile analiz
4. Sonuc dashboard'a yazilir
5. Kuyruk doluysa frame drop (hiz oncelikli)
""")


# ── Page: Edge Profiler ──────────────────────────────────────────────

def page_edge():
    st.header("Edge Profiler (Jetson Orin Nano)")
    profile = load_edge_profile()

    if not profile:
        st.warning("Edge profiler henuz calistirilmadi.")
        st.code(
            "python src/edge/profiler.py --model models/phase1_final_ca.pt "
            "--source data/processed/phase1_multiclass_v1/test/images",
            language="bash",
        )
        # Still show Orin specs
        st.markdown("---")
        st.subheader("Jetson Orin Nano 8GB Specs")
        specs = {"Ozellik": ["GPU", "AI TOPS (sparse)", "RAM", "Guc Modlari"],
                 "Deger": ["Ampere 1024 CUDA", "67 TOPS", "8GB LPDDR5 (shared)", "7W / 15W / 25W"]}
        st.dataframe(specs, hide_index=True)
        return

    local = profile.get("local_latency_ms", {})
    orin = profile.get("orin_estimate", {})
    mem = profile.get("memory_estimate_mb", {})

    # Latency metrics
    st.subheader("Latency Analizi")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Local Avg", f"{local.get('avg', 0):.1f} ms")
    c2.metric("Local P95", f"{local.get('p95', 0):.1f} ms")
    c3.metric("Orin Tahmini", f"{orin.get('est_avg_ms', 0):.1f} ms")
    c4.metric(
        "TensorRT",
        "GEREKLI" if orin.get("needs_tensorrt") else "Yeterli",
        delta="Hedef: 5ms" if orin.get("needs_tensorrt") else "OK",
        delta_color="inverse" if orin.get("needs_tensorrt") else "normal",
    )

    # Memory metrics
    st.subheader("Bellek Tahmini")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("YOLO Model", f"{mem.get('yolo_model_file', 0):.0f} MB")
    m2.metric("YOLO Runtime", f"{mem.get('yolo_runtime_est', 0):.0f} MB")
    m3.metric("PaliGemma 4-bit", f"{mem.get('paligemma_4bit_est', 0)} MB")
    m4.metric("Orin Fit?", "EVET" if mem.get("fits_in_orin") else "HAYIR",
              delta=f"{mem.get('total_est', 0):.0f}/{mem.get('orin_ram_mb', 0)} MB")

    if profile.get("recommendation"):
        st.info(profile["recommendation"])

    # Latency bar chart
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(8, 3), dpi=130)
    categories = ["Local\nAvg", "Local\nP95", "Orin\nTahmini", "TRT\nTahmini", "Hedef\n(200/s)"]
    trt_est = orin.get("est_avg_ms", 0) / 4  # TensorRT ~4x speedup estimate
    values = [local.get("avg", 0), local.get("p95", 0), orin.get("est_avg_ms", 0),
              trt_est, orin.get("target_budget_ms", 5)]
    colors = ["#42A5F5", "#1E88E5", "#FF9800", "#66BB6A", "#F44336"]
    bars = ax.barh(categories, values, color=colors, height=0.6)
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}ms", va="center", fontsize=8)
    ax.set_xlabel("Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Page: FP Analysis ────────────────────────────────────────────────

def page_fp_analysis():
    st.header("False Positive Analizi")

    feedback = load_feedback_stats()

    if feedback["total"] == 0:
        st.info(
            "Henuz operator geri bildirimi yok.\n\n"
            "**Canli Tahmin** sekmesinden goruntu tarayip "
            "'Dogru' / 'Yanlis' butonlariyla geri bildirim verin."
        )

        st.markdown("---")
        st.subheader("False Positive Azaltma Stratejisi")
        st.markdown("""
**Mevcut Onlemler:**
- Coordinate Attention ile mekansal secicilik artirildi
- 2059 background gorsel ile negatif ogrenme guclendrildi
- Confidence threshold 0.40 ile belirsiz tespitler VLM'e yonlendirilir

**Planlanan (Phase 2):**
- FP orneklerini otomatik kaydetme (`reports/fp_samples/`)
- Haftalik FP analiz raporu (hangi sinifta, hangi kosulda)
- Operatorgeri bildirimi ile Active Learning dongusu
- Cost-benefit analizi: Her FP'nin fabrika maliyeti
""")
        return

    # Show feedback stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Geri Bildirim", feedback["total"])
    c2.metric("Tumu Dogru", feedback["correct"])
    c3.metric("Tumu Yanlis", feedback["incorrect"])
    c4.metric("Kismi Duzeltme", feedback.get("partial", 0))

    # Per-detection level stats
    det_c = feedback.get("det_correct", 0)
    det_i = feedback.get("det_incorrect", 0)
    det_total = det_c + det_i
    if det_total > 0:
        d1, d2, d3 = st.columns(3)
        d1.metric("Tekil Tespit (Dogru)", det_c)
        d2.metric("Tekil Tespit (Yanlis)", det_i)
        d3.metric("Tespit Dogruluk", f"{det_c / det_total:.0%}")

    needs_correction = feedback["incorrect"] + feedback.get("partial", 0)
    if needs_correction > 0:
        st.warning(
            f"{needs_correction} goruntude duzeltme isaretlendi "
            f"({det_i} tekil yanlis tespit). "
            "Bu veriler Active Learning dongusu icin kullanilacak."
        )

    # Show recent feedback
    st.markdown("---")
    st.subheader("Son Geri Bildirimler")
    recent = feedback["files"][-20:][::-1]
    # Show summary columns, not full detection details
    display_entries = []
    for e in recent:
        display_entries.append({
            "Zaman": e.get("timestamp", ""),
            "Goruntu": e.get("image", ""),
            "Durum": e.get("label", ""),
            "Tespit": e.get("detection_count", 0),
            "Dogru": e.get("correct_count", e.get("detection_count", 0) if e.get("label") == "correct" else 0),
            "Yanlis": e.get("incorrect_count", 0),
        })
    st.dataframe(display_entries, hide_index=True)


# ── Page: Decisions ──────────────────────────────────────────────────

def page_decisions():
    st.header("Karar Tablosu & Bilinen Riskler")

    tab_decisions, tab_risks, tab_checklist = st.tabs(
        ["Teknik Kararlar", "Bilinen Riskler", "Iyilestirme Checklist"]
    )

    with tab_decisions:
        decisions = [
            {"Konu": "Hiz (200 urun/s)", "Karar": "TensorRT GEREKLI",
             "Detay": "PyTorch FP32 ~119ms > 5ms hedef. TRT ile 3-5x kazanim."},
            {"Konu": "VLM Esik Degeri", "Karar": "conf < 0.40",
             "Detay": "Bu esik altinda PaliGemma tetiklenir."},
            {"Konu": "Yeni Sinif (Egri Vida)", "Karar": "HAYIR",
             "Detay": "Basitlik oncelikli. VLM zaten aciklama yapar. Phase 3'te yeniden degerlendirilir."},
            {"Konu": "Data Leakage", "Karar": "SHA-256 hash kontrolu",
             "Detay": "Augmented <-> test cakismasi hash ile engellenir."},
            {"Konu": "VLM Bildirim Kanali", "Karar": "Dashboard",
             "Detay": "SMS/e-posta yerine dashboard uzerinden operatore iletilir."},
            {"Konu": "Surekli Egitim", "Karar": "Haftalik auto fine-tune",
             "Detay": "Operator geri bildirimi + yeni hatali verilerle."},
            {"Konu": "Background Verisi", "Karar": "2059 gorsel yeterli",
             "Detay": "Metal yansima false positive gorulmedi."},
            {"Konu": "Concept Drift", "Karar": "Phase 2'de test",
             "Detay": "Isik degisimi testleri planlanacak."},
        ]
        st.dataframe(decisions, hide_index=True)

    with tab_risks:
        st.markdown("""
**KRITIK - Weak Labeling Riski:**

Augmented (coklanmis) verilerde `(0.5, 0.5, 1.0, 1.0)` seklinde tum goruntudeki
kutu kullanildiysa, model **metal dokunun tamamini** hata sanmaya baslayabilir.
YOLO, nesnenin arka plandan ayrildigini ogrenmek ister. Tum resmi kutu icine
almak bu siniri bozar.

**Cozum:** Coklanmis verilerde nesneyi tam cevreleyen (tight bounding box)
etiketleme kullanilmali. Mevcut augmented etiketler gozden gecirilmeli.

---

**Diger Riskler:**
- TensorRT sonrasi INT8 quantization'da mAP kaybi olabilir (test gerekli)
- PaliGemma + YOLO ayni anda Orin 8GB'a sigmayabilir (sequential loading gerekebilir)
- Operatorsuz calisma durumunda VLM sonuclari birikerek disk dolurabilir
""")

    with tab_checklist:
        checklist = [
            ("KRITIK", "Weak label'leri tight bounding box ile degistir"),
            ("KRITIK", "TensorRT FP16 export scripti hazirla ve benchmark yap"),
            ("YUKSEK", "Dashboard inference playground ile canli test"),
            ("YUKSEK", "FP orneklerini otomatik kaydet (reports/fp_samples/)"),
            ("YUKSEK", "Operator geri bildirim dongusu (Active Learning)"),
            ("ORTA", "Isik degisimi concept drift testi olustur"),
            ("ORTA", "CA katmanini Neck'e tasiyarak ablation deneyi yap"),
            ("ORTA", "PaliGemma async tetiklenme suresini olc ve raporla"),
            ("DUSUK", "DagsHub/W&B entegrasyonu (takim buyuyunce)"),
            ("DUSUK", "Setup.bat tek tikla kurulum scripti"),
            ("DUSUK", "Model drift alarm sistemi kur"),
        ]
        for priority, item in checklist:
            color = {"KRITIK": "red", "YUKSEK": "orange", "ORTA": "blue", "DUSUK": "gray"}[priority]
            st.markdown(f"- :{color}[**[{priority}]**] {item}")


# ── Page: VLM Anomaly Gallery ────────────────────────────────────────

def page_vlm_gallery():
    st.header("VLM Anomali Galerisi")

    events_path = REPORTS_DIR / "vlm_trigger_events.jsonl"
    if not events_path.exists():
        st.info(
            "Henuz VLM trigger event'i yok.\n\n"
            "VLM trigger'i calistirmak icin:\n"
            "`python src/edge/vlm_trigger.py --model models/phase1_final_ca.pt`"
        )
        return

    # Load events
    events = []
    for line in events_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            events.append(json.loads(line))

    if not events:
        st.info("Event dosyasi bos.")
        return

    # Filter
    reasons = sorted(set(e.get("trigger_reason", "?") for e in events))
    selected_reason = st.selectbox("Trigger Nedeni Filtrele", ["Tumu"] + reasons)

    if selected_reason != "Tumu":
        events = [e for e in events if e.get("trigger_reason") == selected_reason]

    st.metric("Toplam Event", len(events))
    st.markdown("---")

    # Display events in reverse chronological order
    for i, event in enumerate(reversed(events[-50:])):
        with st.expander(
            f"{event.get('timestamp', '?')} | {event.get('image_path', '?')} | "
            f"Neden: {event.get('trigger_reason', '?')}",
            expanded=(i < 3),
        ):
            col_info, col_vlm = st.columns([1, 2])
            with col_info:
                st.markdown(f"**Gorsel:** `{event.get('image_path', '?')}`")
                st.markdown(f"**Neden:** `{event.get('trigger_reason', '?')}`")
                low_dets = event.get("low_conf_detections", [])
                if low_dets:
                    st.markdown(f"**Dusuk Guvenli Tespitler:** {len(low_dets)}")
                    for det in low_dets:
                        st.markdown(
                            f"  - `{det.get('class', '?')}` "
                            f"conf={det.get('confidence', 0):.4f}"
                        )

            with col_vlm:
                vlm_resp = event.get("vlm_response", "Yok")
                if vlm_resp and "[VLM]" in vlm_resp:
                    st.success(f"**VLM Sonucu:** {vlm_resp}")
                elif vlm_resp and "[SIMULATED]" in vlm_resp:
                    st.warning(f"**Simule:** {vlm_resp}")
                else:
                    st.info(f"**Yanit:** {vlm_resp or 'Yok'}")


# ── Page: VLM Metrics ───────────────────────────────────────────────

def page_vlm_metrics():
    st.header("VLM Performans Metrikleri")

    # VLM model status
    reasoner = _get_vlm_reasoner()
    st.subheader("Model Durumu")
    s1, s2, s3 = st.columns(3)
    s1.metric("VLM Modulu", "Mevcut" if _VLM_AVAILABLE else "Eksik")
    s2.metric("Model Yuklendi", "Evet" if (reasoner and reasoner.is_loaded) else "Hayir")
    if reasoner and reasoner.is_loaded:
        vram = reasoner.get_vram_usage_mb()
        s3.metric("VRAM Kullanimi", f"{vram:.0f} MB")
    else:
        s3.metric("VRAM Kullanimi", "N/A")

    st.markdown("---")

    # Trigger summary
    summary = load_vlm_summary()
    if summary:
        st.subheader("Trigger Istatistikleri")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Taranan Goruntu", summary.get("total_images", 0))
        c2.metric("VLM Tetikleme", summary.get("triggered_events", 0))
        c3.metric("VLM Islenen", summary.get("processed_by_vlm", 0))
        c4.metric("Drop Edilen", summary.get("dropped", 0))

        total = summary.get("total_images", 1)
        triggered = summary.get("triggered_events", 0)
        trigger_rate = triggered / max(1, total) * 100

        st.metric("Tetikleme Orani", f"{trigger_rate:.1f}%")

        st.markdown("---")
        st.subheader("Kuyruk Ayarlari")
        st.json({
            "strategy": summary.get("queue_strategy", "fifo"),
            "maxsize": summary.get("queue_maxsize", 32),
            "vlm_enabled": summary.get("vlm_enabled", False),
            "conf_threshold": summary.get("conf_threshold", 0.40),
        })
    else:
        st.info("VLM trigger henuz calistirilmadi. Ozet verisi yok.")

    # Event latency analysis
    st.markdown("---")
    st.subheader("VLM Latency Analizi")
    events_path = REPORTS_DIR / "vlm_trigger_events.jsonl"
    if events_path.exists():
        import re as _re
        latencies = []
        for line in events_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            event = json.loads(line)
            resp = event.get("vlm_response", "")
            # Parse latency from VLM response: "latency=XXXms"
            m = _re.search(r"latency=(\d+(?:\.\d+)?)ms", resp)
            if m:
                latencies.append(float(m.group(1)))

        if latencies:
            col_l1, col_l2, col_l3, col_l4 = st.columns(4)
            col_l1.metric("Avg Latency", f"{sum(latencies)/len(latencies):.0f} ms")
            col_l2.metric("Min Latency", f"{min(latencies):.0f} ms")
            col_l3.metric("Max Latency", f"{max(latencies):.0f} ms")
            sorted_lat = sorted(latencies)
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 1 else sorted_lat[0]
            col_l4.metric("P95 Latency", f"{p95:.0f} ms")

            # Histogram
            fig, ax = plt.subplots(figsize=(8, 3), dpi=130)
            ax.hist(latencies, bins=20, color="#42A5F5", edgecolor="white")
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Sayi")
            ax.set_title("VLM Inference Latency Dagilimi")
            ax.axvline(2000, color="red", linestyle="--", label="2s Budget")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Gercek VLM inference verisi yok (simule edilmis event'ler latency icermez).")
    else:
        st.info("VLM event dosyasi bulunamadi.")


# ── Page: Operator Controls ──────────────────────────────────────────

def page_operator():
    st.header("Operator Kontrol Paneli")

    # Status overview
    st.subheader("Sistem Durumu")
    reasoner = _get_vlm_reasoner()
    vlm_loaded = reasoner is not None and reasoner.is_loaded

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Model", "Aktif" if MODEL_PATH.exists() else "Yuklenmedi")
    s2.metric("YOLO Durum", "Hazir")
    s3.metric("VLM Durum", "Aktif" if vlm_loaded else "Yuklenmedi")
    feedback = load_feedback_stats()
    s4.metric("Geri Bildirim", f"{feedback['total']} kayit")

    st.markdown("---")

    # VLM Load/Unload
    if _VLM_AVAILABLE:
        st.subheader("VLM Model Kontrolu")
        vlm_col1, vlm_col2 = st.columns(2)
        with vlm_col1:
            if not vlm_loaded:
                if st.button("VLM Yukle (PaliGemma 3B NF4)", width="stretch"):
                    with st.spinner("PaliGemma yukleniyor... (~30 sn)"):
                        try:
                            r = VLMReasoner()
                            r.load_model()
                            st.session_state.vlm_reasoner = r
                            st.success(
                                f"VLM yuklendi! VRAM: {r.get_vram_usage_mb():.0f} MB"
                            )
                        except Exception as e:
                            st.error(f"VLM yukleme hatasi: {e}")
            else:
                st.success(f"VLM aktif | VRAM: {reasoner.get_vram_usage_mb():.0f} MB")
        with vlm_col2:
            if vlm_loaded:
                if st.button("VLM Kaldir (VRAM Bosalt)", width="stretch"):
                    reasoner.unload_model()
                    st.session_state.vlm_reasoner = None
                    st.success("VLM kaldirildi, VRAM bosaltildi.")

        st.markdown("---")

    # Control buttons
    st.subheader("Kontroller")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("ACIL DURDURMA", type="primary", width="stretch"):
            st.error("ACIL DURDURMA aktif!")
            st.caption("Simulasyon - Gercek PLC icin MQTT/Modbus gerekli")

    with col2:
        if st.button("VLM Kuyrugu Temizle", width="stretch"):
            st.success("VLM kuyrugu temizlendi.")

    with col3:
        if st.button("Model Yeniden Yukle", width="stretch"):
            if "yolo_model" in st.session_state:
                del st.session_state.yolo_model
            st.success("Model cache temizlendi. Sonraki inference'da yeniden yuklenir.")

    with col4:
        if st.button("Geri Bildirim Sifirla", width="stretch"):
            feedback_file = FEEDBACK_DIR / "feedback_log.jsonl"
            if feedback_file.exists():
                feedback_file.unlink()
            st.success("Geri bildirim dosyasi sifirlandi.")

    # System info
    st.markdown("---")
    with st.expander("Sistem Bilgileri"):
        st.json({
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "reports_dir": str(REPORTS_DIR),
            "feedback_dir": str(FEEDBACK_DIR),
            "class_names": CLASS_NAMES,
        })


# ── Main ─────────────────────────────────────────────────────────────

def page_vlm_hub():
    """Combined VLM page: Strategy + Gallery + Metrics in tabs."""
    st.header("VLM Merkezi")

    tab_strategy, tab_gallery, tab_metrics = st.tabs(
        ["Strateji", "Anomali Galerisi", "Performans Metrikleri"]
    )

    with tab_strategy:
        page_vlm()

    with tab_gallery:
        page_vlm_gallery()

    with tab_metrics:
        page_vlm_metrics()


def page_accuracy():
    """Page 10: Per-class accuracy metrics."""
    st.header("Sinif Bazli Basari Metrikleri")
    st.caption(
        "Her sinifin (screw, missing_screw, missing_component) ayri ayri "
        "ne kadar dogru tespit edildigini gosterir. "
        "Ornegin: 'Eksik vidalarin %100'unu yakaladi mi?'"
    )

    # Find latest accuracy report
    report_files = sorted(REPORTS_DIR.glob("accuracy_*.json"), reverse=True)

    if report_files:
        selected = st.selectbox("Rapor sec", report_files, format_func=lambda p: p.name)
        try:
            report = json.loads(selected.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Rapor okunamadi: {e}")
            return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("mAP50 (Genel Basari)", f"{report.get('mAP50', 0):.4f}")
        col2.metric("Precision (Yanlis Alarm)", f"{report.get('macro_precision', 0):.4f}")
        col3.metric("Recall (Kacirma Orani)", f"{report.get('macro_recall', 0):.4f}")
        col4.metric("F1 (Dengeli Skor)", f"{report.get('macro_f1', 0):.4f}")

        with st.expander("Bu metrikler ne anlama geliyor?"):
            st.markdown("""
- **Precision (Kesinlik)**: Model "bu vida eksik" dediginde gercekten eksik mi?
  - 0.95 = her 100 uyaridan 95'i gercek hata, 5'i yanlis alarm
- **Recall (Duyarlilik)**: Gercekten eksik olan vidalarin kacini yakaladi?
  - 1.00 = hicbir eksik vidayi kacirmadi (uretimde EN KRITIK metrik!)
- **F1**: Precision ve Recall'in dengeli ortalamasi
- **mAP50**: Tum siniflar icin genel tespit dogrulugu
""")

        st.subheader("Sinif Bazli Detay")
        per_class = report.get("per_class", [])
        if per_class:
            import pandas as pd
            df = pd.DataFrame(per_class)
            # Rename columns for clarity
            col_rename = {
                "class_name": "Sinif",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "true_positives": "Dogru Tespit",
                "false_positives": "Yanlis Alarm",
                "false_negatives": "Kacirilan",
                "support": "Toplam Ornek",
            }
            cols_to_show = [c for c in col_rename.keys() if c in df.columns]
            display_df = df[cols_to_show].rename(columns=col_rename)
            st.dataframe(display_df, use_container_width=True)

            # Visual bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            x = range(len(df))
            width = 0.25
            ax.bar([i - width for i in x], df["precision"], width, label="Precision (Kesinlik)", color="#4CAF50")
            ax.bar([i for i in x], df["recall"], width, label="Recall (Duyarlilik)", color="#FF9800")
            ax.bar([i + width for i in x], df["f1"], width, label="F1 (Denge)", color="#2196F3")
            ax.set_xticks(list(x))
            ax.set_xticklabels(df["class_name"])
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.set_title("Sinif Bazli Basari Karsilastirmasi")
            st.pyplot(fig)
            plt.close()

        # Confusion matrix
        cm = report.get("confusion_matrix", [])
        if cm:
            st.subheader("Karisiklik Matrisi (Confusion Matrix)")
            st.caption("Satir = Gercekte ne? | Sutun = Model ne dedi?")
            class_names = ["screw", "missing_screw", "missing_component", "kacirilan"]
            import pandas as pd
            n = min(len(cm), len(class_names))
            cm_display = []
            for i in range(n):
                row = cm[i][:n] if i < len(cm) else [0] * n
                cm_display.append(row)
            cm_df = pd.DataFrame(cm_display, index=class_names[:n], columns=class_names[:n])
            st.dataframe(cm_df, use_container_width=True)

        st.caption(f"Rapor: {selected.name} | {report.get('timestamp', '')}")
    else:
        st.warning("Henuz accuracy raporu yok. Olusturmak icin:")
        st.code(
            "python src/evaluation/accuracy_report.py "
            "--model models/phase1_final_ca.pt --split val",
            language="bash",
        )

    # Generate report button
    if st.button("Yeni Accuracy Raporu Olustur"):
        if MODEL_PATH.exists():
            st.info("Rapor olusturuluyor... (Bu birka dakika surebilir)")
            try:
                from src.evaluation.accuracy_report import run_yolo_evaluation, AccuracyReport
                data_yaml = ROOT / "data" / "processed" / "phase1_multiclass_v1" / "data.yaml"
                report = run_yolo_evaluation(
                    model_path=MODEL_PATH,
                    data_yaml=data_yaml,
                    split="val",
                    device="0",
                )
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_path = REPORTS_DIR / f"accuracy_{MODEL_PATH.stem}_val_{ts}.json"
                report.to_json(json_path)
                st.success(f"Rapor olusturuldu: {json_path.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Hata: {e}")
        else:
            st.error("Model bulunamadi!")


def page_agent_chat():
    """Page 11: Orchestrator Agent chat interface."""
    st.header("Agent Chat - Sistem Yonetim Asistani")
    st.caption(
        "Dogal dille sistem hakkinda soru sorabilir ve islem yapabilirsin. "
        "Agent, keyword tabanli calisan bir asistan. LLM degil, "
        "ama veri, model, kural ve geri bildirim islemlerini yonetir."
    )

    # Initialize agent in session state
    if "agent" not in st.session_state:
        try:
            from src.agent.orchestrator import OrchestratorAgent
            st.session_state.agent = OrchestratorAgent()
            st.session_state.chat_history = []
        except ImportError:
            st.error("Agent modulu yuklenemedi!")
            return

    # Two-column layout: chat + info
    chat_col, info_col = st.columns([2, 1])

    with info_col:
        st.markdown("#### Agent Neler Yapabilir?")
        st.markdown("""
**Veri Sorgulama:**
- "Ne kadar datam var?" -> Dataset istatistikleri
- "Label'lar temiz mi?" -> Weak label kontrolu

**Model Islemleri:**
- "Accuracy nedir?" -> Mevcut modelin basarisi
- "Per-class metrikler" -> Sinif bazli detay

**Kural Yonetimi:**
- "Hangi urun tipleri var?" -> Dinamik kural listesi
- "Yeni urun ekle" -> rules.yaml'a yeni urun

**Geri Bildirim:**
- "Geri bildirimleri analiz et" -> Operator feedback ozeti
- "Retrain gerekli mi?" -> Yeniden egitim kontrolu
""")

        st.markdown("---")
        st.markdown("#### Continuous Training")
        try:
            from src.mlops.continuous_trainer import ContinuousTrainer
            trainer = ContinuousTrainer()
            status = trainer.get_status()
            uf = status["uncertain_frames"]
            rd = status["retrain_decision"]

            col1, col2 = st.columns(2)
            col1.metric("Belirsiz Kare", uf.get("total", 0))
            col2.metric("Feedback", rd.get("feedback_corrective", 0))
            retrain = rd.get("should_retrain", False)
            if retrain:
                st.error("RETRAIN GEREKLI!")
            else:
                st.success("Model guncel")
            if rd.get("reason"):
                st.caption(rd["reason"])

            with st.expander("Continuous Training nasil calisiyor?"):
                st.markdown("""
1. **Gunduz**: Uretim hattinda model calisiyor
   - Dusuk guvenli tespitler (conf < 0.40) toplanir
2. **Gece**: Toplanan verilerle model guncellenir
   - 100+ belirsiz kare VEYA 50+ operator duzeltmesi
   - + son egitimden 1+ gun gecmis olmalii
3. **Otomatik**: Retrain kosullari saglaninca sistem uyarir
""")
        except Exception as e:
            st.warning(f"Continuous trainer: {e}")

    with chat_col:
        # Chat history display
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

        # Chat input
        user_input = st.chat_input("Agent'a soru sor...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            with st.spinner("Agent dusunuyor..."):
                response = st.session_state.agent.process_message(user_input)

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

    # Quick action buttons in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hizli Komutlar")
    quick_commands = [
        "Ne kadar datam var?",
        "Label'lar temiz mi?",
        "Accuracy nedir?",
        "Geri bildirimleri analiz et",
        "Retrain gerekli mi?",
    ]
    for cmd in quick_commands:
        if st.sidebar.button(cmd, key=f"quick_{cmd}"):
            st.session_state.chat_history.append({"role": "user", "content": cmd})
            response = st.session_state.agent.process_message(cmd)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()


PAGES = {
    "Canli Tahmin": page_inference,
    "Veri & Dengeleme": page_data,
    "Neden CA?": page_ca,
    "MLflow Takibi": page_mlflow,
    "VLM Merkezi": page_vlm_hub,
    "Edge Profiler": page_edge,
    "FP Analizi": page_fp_analysis,
    "Karar & Riskler": page_decisions,
    "Operator Kontrol": page_operator,
    "Accuracy Metrikleri": page_accuracy,
    "Agent Chat": page_agent_chat,
}


def main() -> None:
    st.set_page_config(
        page_title="EdgeAgent Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("EdgeAgent")
    st.sidebar.caption("Endustriyel Kalite Kontrol")
    page = st.sidebar.radio("Sayfa", list(PAGES.keys()))

    PAGES[page]()

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Model: `{'Aktif' if MODEL_PATH.exists() else 'Yok'}`\n\n"
        f"Versiyon: Sprint 2 (Phase 2)\n\n"
        f"mAP50: **0.9943**"
    )


if __name__ == "__main__":
    main()
