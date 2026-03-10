import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports" / "generated"
LATEST_JSON = REPORTS_DIR / "augmentation_imbalance_latest.json"
LATEST_PNG = REPORTS_DIR / "augmentation_imbalance_latest.png"


def get_count(d: dict, key: int) -> int:
    return int(d.get(str(key), d.get(key, 0)))


def load_latest_report() -> dict:
    if not LATEST_JSON.exists():
        return {}
    return json.loads(LATEST_JSON.read_text(encoding="utf-8"))


def render_distribution_plot(before: dict, after: dict):
    labels = ["screw", "missing_screw", "missing_component"]
    class_ids = [0, 1, 2]

    before_values = [get_count(before["instance_counts"], cid) for cid in class_ids]
    after_values = [get_count(after["instance_counts"], cid) for cid in class_ids]

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
    ax.bar([i - width / 2 for i in x], before_values, width=width, label="Before")
    ax.bar([i + width / 2 for i in x], after_values, width=width, label="After")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("BBox instances")
    ax.set_title("Class Distribution Improvement (Train)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_ca_explanation():
    st.subheader("Coordinate Attention (CA) - Teknik Not")
    st.markdown(
        """
CA modulu, klasik kanal dikkatini mekansal baglamla birlestirir.

- **X-ekseni havuzlama** ve **Y-ekseni havuzlama** ayri ayri yapilir.
- Kanal bilgisini koruyarak uzun menzilli konumsal iliskiyi yakalar.
- Mikro kusurlar (vida yuvasi, eksik komponent bolgesi) icin daha stabil aktivasyon uretir.

Bu projede CA, YOLOv10-S backbone icine eklenerek modelin **mekansal seciciligini** artirir.
"""
    )
    st.code(
        """
Input F(C,H,W)
  -> AvgPool_x(F) and AvgPool_y(F)
  -> Shared 1x1 transform + non-linearity
  -> Split to attention maps Ax and Ay
  -> Output = F * Ax * Ay
""".strip(),
        language="text",
    )


def render_kpi_cards(before: dict, after: dict):
    b_screw = get_count(before["instance_counts"], 0)
    b_miss_screw = get_count(before["instance_counts"], 1)
    b_miss_comp = get_count(before["instance_counts"], 2)

    a_screw = get_count(after["instance_counts"], 0)
    a_miss_screw = get_count(after["instance_counts"], 1)
    a_miss_comp = get_count(after["instance_counts"], 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("screw (bbox)", f"{a_screw}", delta=f"+{a_screw - b_screw}")
    c2.metric("missing_screw (bbox)", f"{a_miss_screw}", delta=f"+{a_miss_screw - b_miss_screw}")
    c3.metric(
        "missing_component (bbox)",
        f"{a_miss_comp}",
        delta=f"+{a_miss_comp - b_miss_comp}",
    )

    ratio_screw = a_screw / max(1, b_screw)
    ratio_ms = a_miss_screw / max(1, b_miss_screw)
    ratio_mc = a_miss_comp / max(1, b_miss_comp)
    st.info(
        f"Iyilesme carpani -> screw: x{ratio_screw:.2f}, "
        f"missing_screw: x{ratio_ms:.2f}, missing_component: x{ratio_mc:.2f}"
    )


def main() -> None:
    st.set_page_config(page_title="Sprint 1 Technical Dashboard", layout="wide")
    st.title("Sprint 1 - Veri Dengeleme ve CA Teknik Izleme Paneli")

    data = load_latest_report()
    if not data:
        st.error(
            "Rapor bulunamadi. Once su komutu calistir:\n"
            "`python src/data/augment_analysis.py`"
        )
        return

    before = data.get("before_train_distribution", {})
    after = data.get("after_train_distribution", {})

    st.subheader("Data Integration Ozet")
    left, right = st.columns([2, 1])
    with left:
        render_kpi_cards(before, after)
    with right:
        st.markdown("**Train Image Count**")
        st.write(f"Before: {before.get('total_images', 0)}")
        st.write(f"After : {after.get('total_images', 0)}")
        st.write(f"Background: {after.get('background_images', 0)}")

    st.subheader("Imbalance Analizi")
    render_distribution_plot(before, after)

    if LATEST_PNG.exists():
        st.image(str(LATEST_PNG), caption="Generated imbalance chart (latest)", use_container_width=True)

    st.subheader("Teknik Bulgular")
    b_ms = get_count(before.get("instance_counts", {}), 1)
    a_ms = get_count(after.get("instance_counts", {}), 1)
    b_mc = get_count(before.get("instance_counts", {}), 2)
    a_mc = get_count(after.get("instance_counts", {}), 2)
    st.markdown(
        f"- `missing_screw`: **{b_ms} -> {a_ms}**\n"
        f"- `missing_component`: **{b_mc} -> {a_mc}**\n"
        f"- Bu artis, modelin nadir kusur siniflarinda recall toplamasini hedefler."
    )

    render_ca_explanation()

    st.subheader("Rapor Dosyalari")
    st.code(
        "\n".join(
            [
                str(LATEST_JSON),
                str(LATEST_PNG),
                str(ROOT / "reports" / "final_phase1_report.md"),
            ]
        ),
        language="text",
    )


if __name__ == "__main__":
    main()
