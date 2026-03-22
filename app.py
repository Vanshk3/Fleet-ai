import json
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from utils.inference import load_model, predict_image, predict_batch

st.set_page_config(
    page_title="Fleet AI — Tyre Defect Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .result-card {
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
    }
    .metric-card {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .conf-bar-wrap {
        background: rgba(128,128,128,0.15);
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
        margin-top: 6px;
    }
    .action-box {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.75rem;
        font-size: 14px;
        font-weight: 500;
    }
    [data-testid="stSidebar"] { background: #f8f8f6; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError as e:
        return None, str(e)


model, meta = get_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Fleet AI")
    st.markdown("**Tyre Defect Detection System**")
    st.markdown("---")

    page = st.radio(
        "Mode",
        ["Single inspection", "Batch processing", "Model performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if meta and not isinstance(meta, str):
        st.markdown("### Model info")
        st.markdown(f"**Architecture:** MobileNetV2")
        st.markdown(f"**Test accuracy:** {meta.get('test_accuracy', 'N/A'):.1%}")
        st.markdown(f"**Val accuracy:** {meta.get('best_val_accuracy', 'N/A'):.1%}")
    else:
        st.warning("Model not loaded. Run `python train.py` first.")

    st.markdown("---")
    st.markdown("[GitHub ↗](https://github.com/vanshk3/fleet-ai)")

# ── Model not loaded guard ────────────────────────────────────────────────────
if model is None:
    st.error(f"Model not found. {meta}")
    st.info("Run `python train.py` to train the model, then restart the app.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE INSPECTION
# ═══════════════════════════════════════════════════════════════════
if page == "Single inspection":
    st.markdown("## Single tyre inspection")
    st.markdown("Upload a tyre image and get an instant defect assessment with confidence score.")

    uploaded = st.file_uploader(
        "Upload tyre image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)

        with col_result:
            with st.spinner("Analysing..."):
                result = predict_image(model, meta, image)

            color = result["color"]
            label = result["label"]
            conf = result["confidence"]

            st.markdown(f"""
            <div class="result-card" style="background: {color}18; border: 1.5px solid {color};">
                <div style="font-size:13px;color:#666;margin-bottom:4px">VERDICT</div>
                <div style="font-size:28px;font-weight:700;color:{color}">{label}</div>
                <div style="font-size:14px;color:#555;margin-top:6px">{result['description']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Confidence score: {conf}%**")

            fig = go.Figure(go.Bar(
                x=list(result["all_probs"].values()),
                y=[c.title() for c in result["all_probs"].keys()],
                orientation="h",
                marker_color=[
                    "#1D9E75" if c == "good" else "#D85A30"
                    for c in result["all_probs"].keys()
                ],
                text=[f"{v}%" for v in result["all_probs"].values()],
                textposition="outside"
            ))
            fig.update_layout(
                height=140,
                margin=dict(l=10, r=40, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 110], showticklabels=False, showgrid=False),
                yaxis=dict(showgrid=False),
                font=dict(family="Arial", size=13),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            action_color = "#0F6E56" if result["class"] == "good" else "#993C1D"
            action_bg = "#e1f5ee" if result["class"] == "good" else "#faece7"
            st.markdown(f"""
            <div class="action-box" style="background:{action_bg};color:{action_color}">
                Recommended action: {result['action']}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Upload a tyre image above to begin inspection.")

# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════
elif page == "Batch processing":
    st.markdown("## Fleet batch inspection")
    st.markdown("Upload multiple tyre images at once to get a full fleet health report.")

    uploaded_files = st.file_uploader(
        "Upload tyre images (multiple)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        images = []
        paths = []
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            images.append(img)
            paths.append(f.name)

        with st.spinner(f"Analysing {len(images)} tyres..."):
            results = []
            for img, name in zip(images, paths):
                r = predict_image(model, meta, img)
                r["filename"] = name
                results.append(r)

        n_total = len(results)
        n_defective = sum(1 for r in results if r["class"] == "defective")
        n_good = n_total - n_defective
        avg_conf = sum(r["confidence"] for r in results) / n_total

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, color in [
            (c1, "Tyres inspected", str(n_total), "#1a1a1a"),
            (c2, "Good", str(n_good), "#1D9E75"),
            (c3, "Defective", str(n_defective), "#D85A30"),
            (c4, "Avg confidence", f"{avg_conf:.1f}%", "#7F77DD"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:12px;color:#888">{label}</div>
                    <div style="font-size:26px;font-weight:600;color:{color}">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart, col_table = st.columns([1, 2], gap="large")

        with col_chart:
            fig = go.Figure(go.Pie(
                labels=["Good", "Defective"],
                values=[n_good, n_defective],
                marker_colors=["#1D9E75", "#D85A30"],
                hole=0.5,
                textinfo="label+percent"
            ))
            fig.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(family="Arial", size=13)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            df = pd.DataFrame([{
                "Filename": r["filename"],
                "Result": r["label"],
                "Confidence": f"{r['confidence']}%",
                "Action": "Inspect immediately" if r["class"] == "defective" else "Routine check"
            } for r in results])

            def highlight(row):
                color = "#faece7" if row["Result"] == "Defective" else "#e1f5ee"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                df.style.apply(highlight, axis=1),
                use_container_width=True,
                height=260
            )

        st.markdown("---")

        if n_defective > 0:
            st.markdown("### Defective tyres requiring attention")
            cols = st.columns(min(4, n_defective))
            defective_results = [r for r in results if r["class"] == "defective"]
            for i, r in enumerate(defective_results):
                with cols[i % len(cols)]:
                    idx = paths.index(r["filename"])
                    st.image(images[idx], caption=f"{r['filename']} ({r['confidence']}%)", use_column_width=True)

        csv = df.to_csv(index=False)
        st.download_button(
            "Download inspection report (CSV)",
            data=csv,
            file_name="fleet_inspection_report.csv",
            mime="text/csv"
        )

    else:
        st.info("Upload multiple tyre images to run a full fleet inspection.")

# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
elif page == "Model performance":
    st.markdown("## Model performance metrics")
    st.markdown("Full evaluation of the trained MobileNetV2 classifier on the held-out test set.")

    if not meta:
        st.error("No model metadata found. Train the model first.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    report = meta.get("classification_report", {})
    for col, label, value in [
        (c1, "Test accuracy", f"{meta.get('test_accuracy', 0):.1%}"),
        (c2, "Val accuracy", f"{meta.get('best_val_accuracy', 0):.1%}"),
        (c3, "Defective F1", f"{report.get('defective', {}).get('f1-score', 0):.2f}"),
        (c4, "Good F1", f"{report.get('good', {}).get('f1-score', 0):.2f}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:12px;color:#888">{label}</div>
                <div style="font-size:26px;font-weight:600;color:#1a1a1a">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_hist, col_cm = st.columns(2, gap="large")

    with col_hist:
        st.markdown("#### Training history")
        history = meta.get("history", {})
        if history:
            epochs = list(range(1, len(history["train_acc"]) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train acc",
                                     line=dict(color="#7F77DD", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val acc",
                                     line=dict(color="#1D9E75", width=2)))
            fig.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial", size=12),
                legend=dict(orientation="h", y=1.1),
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_cm:
        st.markdown("#### Confusion matrix")
        cm = meta.get("confusion_matrix")
        classes = meta.get("classes", ["defective", "good"])
        if cm:
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=[c.title() for c in classes],
                y=[c.title() for c in classes],
                color_continuous_scale=[[0, "#f8f8f6"], [1, "#1D9E75"]],
                text_auto=True
            )
            fig.update_layout(
                height=300,
                margin=dict(l=60, r=20, t=20, b=60),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial", size=13),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Per-class metrics")
    rows = []
    for cls in classes:
        r = report.get(cls, {})
        rows.append({
            "Class": cls.title(),
            "Precision": f"{r.get('precision', 0):.3f}",
            "Recall": f"{r.get('recall', 0):.3f}",
            "F1-score": f"{r.get('f1-score', 0):.3f}",
            "Support": int(r.get("support", 0))
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Architecture")
    st.markdown(f"""
    - **Base model:** MobileNetV2 (pretrained on ImageNet)
    - **Fine-tuning:** Custom classifier head (Dropout → Linear 256 → ReLU → Dropout → Linear 2)
    - **Training device:** {meta.get('device', 'CPU')}
    - **Epochs trained:** {meta.get('epochs', 'N/A')}
    - **Data augmentation:** Horizontal/vertical flip, rotation ±15°, colour jitter, affine translation
    """)
