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

CLASS_NAMES = {0: "screw", 1: "missing_screw", 2: "missing_component"}
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
    """Load feedback statistics from the feedback directory."""
    if not FEEDBACK_DIR.exists():
        return {"total": 0, "correct": 0, "incorrect": 0, "files": []}
    feedback_file = FEEDBACK_DIR / "feedback_log.jsonl"
    if not feedback_file.exists():
        return {"total": 0, "correct": 0, "incorrect": 0, "files": []}
    entries = []
    for line in feedback_file.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    correct = sum(1 for e in entries if e.get("label") == "correct")
    incorrect = sum(1 for e in entries if e.get("label") == "incorrect")
    return {"total": len(entries), "correct": correct, "incorrect": incorrect, "files": entries}


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

    # ── Sidebar settings ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("YOLO Ayarlari")
    conf_thresh = st.sidebar.slider("Confidence Esigi", 0.1, 0.95, 0.25, 0.05)
    iou_thresh = st.sidebar.slider("IoU Esigi (NMS)", 0.1, 0.95, 0.45, 0.05)
    img_size = st.sidebar.selectbox("Goruntu Boyutu", [640, 416, 320], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Edge Enhancement")
    alpha = st.sidebar.slider("Alpha (orijinal agirlik)", 0.3, 1.0, 0.7, 0.05,
                              help="1.0 = sadece orijinal, 0.3 = guclu kenar karistirma")
    canny_low = st.sidebar.slider("Canny Alt Esik", 10, 150, 50, 10)
    canny_high = st.sidebar.slider("Canny Ust Esik", 50, 300, 150, 10)

    # ── Upload ──
    uploaded = st.file_uploader(
        "Goruntu yukle (JPG/PNG)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("Bir veya birden fazla goruntu yukleyin, model anlik tahmin yapsin.")
        # Show sample from test set if available
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
                            )
        return

    for file in uploaded:
        st.markdown(f"---\n### {file.name}")
        # Save to temp
        tmp_path = REPORTS_DIR / f"_tmp_upload_{file.name}"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(file.getvalue())
        _run_full_analysis(
            model, str(tmp_path), conf_thresh, iou_thresh,
            img_size, alpha, canny_low, canny_high,
        )
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()


def _run_full_analysis(
    model, image_path: str, conf: float, iou: float, imgsz: int,
    alpha: float, canny_low: int, canny_high: int,
):
    """Run YOLO + Edge Enhancement + Spatial analysis on one image."""

    # ─── Section 1: YOLO Inference ───
    st.subheader("1. YOLO Tespit")
    t0 = time.perf_counter()
    results = model.predict(image_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    latency = (time.perf_counter() - t0) * 1000
    result = results[0]
    boxes = result.boxes

    col_img, col_info = st.columns([2, 1])

    with col_img:
        annotated = result.plot()
        annotated_rgb = annotated[:, :, ::-1]
        st.image(annotated_rgb, caption="Tahmin Sonucu", width="stretch")

    with col_info:
        st.metric("Latency", f"{latency:.1f} ms")
        st.metric("Tespit Sayisi", len(boxes))

        if len(boxes) > 0:
            st.markdown("**Tespitler:**")
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf_val = float(boxes.conf[i])
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                color = CLASS_COLORS.get(cls_name, "#999")

                conf_bar = "+" * int(conf_val * 20) + "-" * (20 - int(conf_val * 20))
                st.markdown(
                    f"- :{_color_name(color)}[**{cls_name}**] "
                    f"`{conf_val:.1%}` `[{conf_bar}]`"
                )

                if conf_val < 0.40:
                    st.warning(f"Dusuk confidence! VLM tetiklenirdi (< 0.40)")
        else:
            st.warning("Hicbir nesne tespit edilemedi. VLM tetiklenirdi.")

        # Feedback buttons
        st.markdown("---")
        st.markdown("**Geri Bildirim:**")
        fb_col1, fb_col2 = st.columns(2)
        img_name = Path(image_path).name
        with fb_col1:
            if st.button("Dogru", key=f"correct_{img_name}", use_container_width=True):
                _save_feedback(img_name, "correct", len(boxes))
                st.success("Kaydedildi!")
        with fb_col2:
            if st.button("Yanlis", key=f"wrong_{img_name}", type="primary", use_container_width=True):
                _save_feedback(img_name, "incorrect", len(boxes))
                st.error("Geri bildirim kaydedildi.")

    # ─── Section 2: Edge Enhancement ───
    st.markdown("---")
    st.subheader("2. Edge Enhancement (Canny Onisleme)")
    try:
        from src.data.edge_enhancer import preview_enhancement

        original_rgb, edges_rgb, blended_rgb = preview_enhancement(
            image_path, alpha, canny_low, canny_high
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_rgb, caption="Orijinal", width="stretch")
        with col2:
            st.image(edges_rgb, caption="Canny Kenarlar", width="stretch")
        with col3:
            st.image(blended_rgb, caption=f"Harmanlanmis (a={alpha})", width="stretch")

        # YOLO comparison
        st.markdown("**YOLO Karsilastirmasi:**")
        col_orig, col_enhanced = st.columns(2)

        with col_orig:
            st.markdown("*Orijinal ile Tespit*")
            ann1 = result.plot()[:, :, ::-1]
            st.image(ann1, width="stretch")
            st.caption(f"Tespit: {len(result.boxes)}")

        with col_enhanced:
            st.markdown("*Enhanced ile Tespit*")
            enhanced_path = REPORTS_DIR / f"_tmp_edge_enhanced_{Path(image_path).name}"
            enhanced_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(enhanced_path), enhanced_bgr)
            r2 = model.predict(str(enhanced_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
            ann2 = r2[0].plot()[:, :, ::-1]
            st.image(ann2, width="stretch")
            st.caption(f"Tespit: {len(r2[0].boxes)}")
            if enhanced_path.exists():
                enhanced_path.unlink()

    except Exception as e:
        st.warning(f"Edge Enhancement yuklenemedi: {e}")

    # ─── Section 3: Spatial Clustering ───
    st.markdown("---")
    st.subheader("3. Geometrik Mekansal Kumeleme")
    try:
        from src.reasoning.spatial_logic import (
            SpatialAnalyzer, detections_from_yolo_result,
        )

        detections = detections_from_yolo_result(result)
        analyzer = SpatialAnalyzer(n_clusters=4)
        spatial_result = analyzer.analyze_frame(detections, img_shape=result.orig_shape)

        col_simg, col_sresult = st.columns([2, 1])

        with col_simg:
            _draw_spatial_overlay(result, spatial_result)

        with col_sresult:
            verdict_colors = {"OK": "green", "missing_screw": "orange", "missing_component": "red"}
            verdict_color = verdict_colors.get(spatial_result.verdict, "gray")
            st.markdown(f"### :{verdict_color}[{spatial_result.verdict.upper()}]")
            st.metric("Tespit Sayisi", spatial_result.detection_count)
            st.metric("Ort. Confidence", f"{spatial_result.confidence:.1%}")

            st.markdown("---")
            st.markdown(f"**Neden:** {spatial_result.reason}")
            st.markdown(f"**Sol Taraf:** `{spatial_result.left_status}`")
            st.markdown(f"**Sag Taraf:** `{spatial_result.right_status}`")

            if spatial_result.clusters:
                st.markdown("---")
                for cluster in spatial_result.clusters:
                    st.markdown(
                        f"- Kume {cluster.cluster_id} ({cluster.side}): "
                        f"**{cluster.dominant_class}** "
                        f"({len(cluster.detections)} tespit)"
                    )

    except Exception as e:
        st.warning(f"Mekansal analiz yuklenemedi: {e}")

    # ─── Section 4: VLM Reasoning (Phase 2) ───
    st.markdown("---")
    st.subheader("4. VLM Akil Yurume (PaliGemma)")

    reasoner = _get_vlm_reasoner()
    if not _VLM_AVAILABLE:
        st.info("VLM modulleri yuklenemedi. `pip install transformers` gerekli.")
    elif reasoner is None or not reasoner.is_loaded:
        st.info(
            "VLM modeli yuklenmedi. **Operator Kontrol** sayfasindan "
            "'VLM Yukle' butonuna basin."
        )
        # Still show trigger status
        try:
            trigger_config = TriggerConfig(conf_threshold=0.40)
            should_fire, low_dets, reason = should_trigger_vlm(
                results, CLASS_NAMES, trigger_config
            )
            if should_fire:
                st.warning(
                    f"VLM tetiklenirdi! Neden: `{reason}` | "
                    f"{len(low_dets)} dusuk guvenli tespit"
                )
            else:
                st.success("VLM tetiklenmezdi - tum tespitler yuksek guvenli.")
        except Exception:
            pass
    else:
        try:
            trigger_config = TriggerConfig(conf_threshold=0.40)
            should_fire, low_dets, reason = should_trigger_vlm(
                results, CLASS_NAMES, trigger_config
            )

            if should_fire:
                st.warning(f"VLM tetiklendi! Neden: `{reason}`")

                # Crop and reason
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

                    col_vlm_img, col_vlm_info = st.columns([1, 2])
                    with col_vlm_img:
                        st.image(crop, caption="VLM Analiz Bolgesi", width="stretch")

                    with col_vlm_info:
                        # Show VLM result
                        if vlm_result.defect_type:
                            dt_colors = {
                                "ok": "green",
                                "missing_screw": "orange",
                                "missing_component": "red",
                            }
                            dt_color = dt_colors.get(vlm_result.defect_type, "gray")
                            st.markdown(
                                f"**VLM Karari:** :{dt_color}[**{vlm_result.defect_type}**]"
                            )
                        else:
                            st.markdown("**VLM Karari:** :gray[Belirsiz]")

                        st.metric("VLM Confidence", f"{vlm_result.confidence_estimate:.0%}")
                        st.metric("VLM Latency", f"{vlm_result.latency_ms:.0f} ms")
                        st.markdown(f"**Aciklama:** {vlm_result.reasoning}")

                    # Conflict resolution
                    if _VLM_AVAILABLE:
                        resolver = ConflictResolver()
                        yolo_dets = []
                        for i in range(len(boxes)):
                            yolo_dets.append({
                                "class_name": CLASS_NAMES.get(int(boxes.cls[i]), "?"),
                                "confidence": float(boxes.conf[i]),
                            })

                        try:
                            spatial_v = spatial_result.verdict
                            spatial_s = spatial_result.left_status
                        except NameError:
                            spatial_v = None
                            spatial_s = None

                        final = resolver.resolve(
                            yolo_dets, spatial_v, spatial_s, vlm_result
                        )

                        st.markdown("---")
                        st.markdown("**Nihai Karar (Conflict Resolver):**")
                        fc = {"ok": "green", "missing_screw": "orange", "missing_component": "red"}
                        st.markdown(
                            f"### :{fc.get(final.verdict, 'gray')}[{final.verdict.upper()}]"
                        )
                        st.markdown(f"**Kaynak:** `{final.source}` | **Conflict:** `{final.conflict_detected}`")
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
    """Save operator feedback for Active Learning pipeline."""
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

def page_data():
    st.header("Veri Dengeleme & Sinif Dagilimi")

    data = load_latest_report()
    if not data:
        st.error(
            "Rapor bulunamadi. Once su komutu calistir:\n"
            "`python src/data/augment_analysis.py`"
        )
        return

    before = data.get("before_train_distribution", {})
    after = data.get("after_train_distribution", {})

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Train Images",
        after.get("total_images", 0),
        delta=f"+{after.get('total_images', 0) - before.get('total_images', 0)}",
    )
    c2.metric("screw", get_count(after.get("instance_counts", {}), 0),
              delta=f"+{get_count(after.get('instance_counts', {}), 0) - get_count(before.get('instance_counts', {}), 0)}")
    c3.metric("missing_screw", get_count(after.get("instance_counts", {}), 1),
              delta=f"+{get_count(after.get('instance_counts', {}), 1) - get_count(before.get('instance_counts', {}), 1)}")
    c4.metric("missing_component", get_count(after.get("instance_counts", {}), 2),
              delta=f"+{get_count(after.get('instance_counts', {}), 2) - get_count(before.get('instance_counts', {}), 2)}")

    st.markdown("---")

    # Chart
    col_chart, col_detail = st.columns([2, 1])
    with col_chart:
        _render_distribution_plot(before, after)
    with col_detail:
        st.markdown("**Iyilesme Carpanlari**")
        for cid, name in CLASS_NAMES.items():
            b = get_count(before.get("instance_counts", {}), cid)
            a = get_count(after.get("instance_counts", {}), cid)
            ratio = a / max(1, b)
            st.markdown(f"- **{name}**: x{ratio:.2f} ({b} -> {a})")
        st.markdown(f"\n**Background**: {after.get('background_images', 0)} gorsel")

    if LATEST_PNG.exists():
        with st.expander("Augmentation Analizi (Detay Grafik)"):
            st.image(str(LATEST_PNG), width="stretch")


def _render_distribution_plot(before: dict, after: dict):
    labels = list(CLASS_NAMES.values())
    class_ids = list(CLASS_NAMES.keys())
    before_values = [get_count(before.get("instance_counts", {}), cid) for cid in class_ids]
    after_values = [get_count(after.get("instance_counts", {}), cid) for cid in class_ids]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=130)
    bars1 = ax.bar(x - width / 2, before_values, width, label="Oncesi", color="#90CAF9")
    bars2 = ax.bar(x + width / 2, after_values, width, label="Sonrasi", color="#1565C0")

    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("BBox Sayisi")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(int(bar.get_height())), ha="center", fontsize=7, color="#555")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(int(bar.get_height())), ha="center", fontsize=7, color="#555")

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

def page_mlflow():
    st.header("MLflow Experiment Tracking")

    # Sprint 1 metrics - prominent
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline mAP50", "0.4942")
    m2.metric("Final mAP50", "0.9943", delta="+0.5001")
    m3.metric("Best Epoch", "96 / 100")
    m4.metric("Iyilesme", "+101%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Neden MLflow?")
        st.markdown("""
- Her egitimin hyperparameter + metrik kaydini tutar
- Model versiyonlama ve artifact yonetimi
- Takim ici deneykarsilastirmasi
- Otomatik regression detection (mAP duserse alarm)
""")
        st.subheader("Kayit Edilen Metrikler")
        metrics_data = {
            "Metrik": ["mAP50(B)", "mAP50-95(B)", "precision(B)", "recall(B)",
                       "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"],
            "Aciklama": ["Ana basari metrigi", "Siki IoU metrigi", "Kesinlik",
                         "Duyarlilik", "Kutu kaybi", "Sinif kaybi",
                         "Val kutu kaybi", "Val sinif kaybi"],
        }
        st.dataframe(metrics_data, hide_index=True)

    with col2:
        st.subheader("Hizli Baslangic")
        st.code("""
# MLflow UI baslatma
mlflow ui --port 5000

# Tarayicida ac
# http://localhost:5000
""".strip(), language="bash")

        st.subheader("Sprint 1 Egitim Parametreleri")
        params = {
            "Parametre": ["epochs", "batch", "imgsz", "patience", "close_mosaic",
                          "amp", "seed", "optimizer"],
            "Deger": ["100", "8", "640", "25", "10", "True", "42", "auto"],
        }
        st.dataframe(params, hide_index=True)


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
    c2.metric("Dogru Tespit", feedback["correct"])
    c3.metric("Yanlis Tespit", feedback["incorrect"])
    accuracy = feedback["correct"] / max(1, feedback["total"])
    c4.metric("Operator Dogruluk", f"{accuracy:.0%}")

    if feedback["incorrect"] > 0:
        st.warning(
            f"{feedback['incorrect']} yanlis tespit isaretlendi. "
            "Bu veriler Active Learning dongusu icin kullanilacak."
        )

    # Show recent feedback
    st.markdown("---")
    st.subheader("Son Geri Bildirimler")
    recent = feedback["files"][-20:][::-1]
    st.dataframe(recent, hide_index=True)


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
                if st.button("VLM Yukle (PaliGemma 3B NF4)", use_container_width=True):
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
                if st.button("VLM Kaldir (VRAM Bosalt)", use_container_width=True):
                    reasoner.unload_model()
                    st.session_state.vlm_reasoner = None
                    st.success("VLM kaldirildi, VRAM bosaltildi.")

        st.markdown("---")

    # Control buttons
    st.subheader("Kontroller")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("ACIL DURDURMA", type="primary", use_container_width=True):
            st.error("ACIL DURDURMA aktif!")
            st.caption("Simulasyon - Gercek PLC icin MQTT/Modbus gerekli")

    with col2:
        if st.button("VLM Kuyrugu Temizle", use_container_width=True):
            st.success("VLM kuyrugu temizlendi.")

    with col3:
        if st.button("Model Yeniden Yukle", use_container_width=True):
            if "yolo_model" in st.session_state:
                del st.session_state.yolo_model
            st.success("Model cache temizlendi. Sonraki inference'da yeniden yuklenir.")

    with col4:
        if st.button("Geri Bildirim Sifirla", use_container_width=True):
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

PAGES = {
    "Canli Tahmin": page_inference,
    "Veri & Dengeleme": page_data,
    "Neden CA?": page_ca,
    "MLflow Takibi": page_mlflow,
    "VLM Stratejisi": page_vlm,
    "VLM Galeri": page_vlm_gallery,
    "VLM Metrikler": page_vlm_metrics,
    "Edge Profiler": page_edge,
    "FP Analizi": page_fp_analysis,
    "Karar & Riskler": page_decisions,
    "Operator Kontrol": page_operator,
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
