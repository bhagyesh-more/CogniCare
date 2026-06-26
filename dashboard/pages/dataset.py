"""
dashboard/pages/dataset.py
Page 5 - Dataset Intelligence Center.
"""

import pandas as pd
import streamlit as st

from dashboard.components import (
    C, correlation_heatmap, glass_card, histogram_chart,
    kpi_card, label_donut, neon_divider, scatter_chart, section_header,
)

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability",
]


def _label_row(label: str, count: int, total: int, color: str) -> str:
    pct = count / total * 100
    label_map = {"baseline":"1","stress":"2","amusement":"3"}
    label_code = label_map.get(label, "?")
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:10px 16px;border-radius:8px;margin-bottom:8px;'
        f'background:rgba(255,255,255,0.03);border:1px solid {color}22">'
        f'<div>'
        f'<span style="font-family:var(--font-body);font-size:0.85rem;color:{color};font-weight:600">'
        f'{label.capitalize()}</span>'
        f'<span style="font-size:0.6rem;color:{C["secondary"]};margin-left:8px;'
        f'letter-spacing:0.1em;text-transform:uppercase;font-family:var(--font-data)">'
        f'WESAD label {label_code}</span>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<span style="font-family:var(--font-display);font-size:1.1rem;font-weight:700;color:{color}">'
        f'{count}</span>'
        f'<span style="font-size:0.65rem;color:{C["secondary"]};margin-left:6px">'
        f'windows ({pct:.1f}%)</span>'
        f'</div>'
        f'</div>'
    )


def render(df: pd.DataFrame, session_state: dict) -> None:
    st.markdown(section_header("Dataset Intelligence Center", "WESAD Feature Dataset Analysis", "▦"),
                unsafe_allow_html=True)

    if df.empty:
        st.error("No dataset loaded.")
        return

    # ── Overview KPIs ────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(kpi_card(str(len(df)), "Total Windows", "▦", C["cyan"]), unsafe_allow_html=True)
    with k2:
        n_subj = df["subject_id"].nunique() if "subject_id" in df.columns else "-"
        st.markdown(kpi_card(str(n_subj), "Subjects", "👤", C["purple"]), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card(str(len(FEATURE_COLS)), "Features", "◈", C["green"]), unsafe_allow_html=True)
    with k4:
        n_labels = df["label"].nunique() if "label" in df.columns else "-"
        st.markdown(kpi_card(str(n_labels), "Affect Classes", "◉", C["orange"]), unsafe_allow_html=True)
    with k5:
        nan_pct = df[FEATURE_COLS].isna().sum().sum() / (len(df) * len(FEATURE_COLS)) * 100
        st.markdown(kpi_card(f"{nan_pct:.1f}%", "Missing Values", "⚠", C["pink"]), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Label Distribution ───────────────────────────────────────
    st.markdown(section_header("Label Distribution", "WESAD Affect States", "◉"), unsafe_allow_html=True)

    lc1, lc2 = st.columns([1, 2])
    with lc1:
        label_counts = df["label"].value_counts()
        st.plotly_chart(label_donut(label_counts), use_container_width=True,
                        config={"displayModeBar": False})

    with lc2:
        color_map = {"baseline": C["cyan"], "stress": C["pink"], "amusement": C["green"]}
        wesad_id  = {"baseline": "1", "stress": "2", "amusement": "3"}
        rows_html = ""
        for label, count in label_counts.items():
            color = color_map.get(label, C["purple"])
            wid   = wesad_id.get(label, "?")
            pct   = count / len(df) * 100
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:10px 16px;border-radius:8px;margin-bottom:8px;'
                f'background:rgba(255,255,255,0.03);border:1px solid {color}22">'
                f'<div>'
                f'<span style="font-family:var(--font-body);font-size:0.85rem;'
                f'color:{color};font-weight:600">{label.capitalize()}</span>'
                f'<span style="font-size:0.6rem;color:{C["secondary"]};margin-left:8px;'
                f'letter-spacing:0.1em;text-transform:uppercase;font-family:var(--font-data)">'
                f'WESAD label {wid}</span>'
                f'</div>'
                f'<div style="text-align:right">'
                f'<span style="font-family:var(--font-display);font-size:1.1rem;'
                f'font-weight:700;color:{color}">{count}</span>'
                f'<span style="font-size:0.65rem;color:{C["secondary"]};margin-left:6px">'
                f'windows ({pct:.1f}%)</span>'
                f'</div>'
                f'</div>'
            )
        st.markdown(glass_card(rows_html, "purple"), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Feature Statistics ───────────────────────────────────────
    st.markdown(section_header("Feature Statistics", "Descriptive Stats per Feature", "◈"),
                unsafe_allow_html=True)
    stats = df[FEATURE_COLS].describe().T
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(4)
    st.dataframe(stats, use_container_width=True, height=330)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Correlation Matrix ───────────────────────────────────────
    st.markdown(section_header("Correlation Matrix", "Pearson Correlation Between Features", "⬡"),
                unsafe_allow_html=True)
    corr = df[FEATURE_COLS].corr().round(3)
    st.plotly_chart(correlation_heatmap(corr), use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Feature Histograms ───────────────────────────────────────
    st.markdown(section_header("Feature Distributions", "Per-Feature Histogram", "▦"),
                unsafe_allow_html=True)
    hist_colors = [C["cyan"], C["purple"], C["pink"], C["green"],
                   C["orange"], C["cyan"], C["purple"], C["green"], C["pink"]]
    cols = st.columns(3)
    for i, col in enumerate(FEATURE_COLS):
        with cols[i % 3]:
            st.plotly_chart(histogram_chart(df[col], col, hist_colors[i]),
                            use_container_width=True, config={"displayModeBar": False})

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Scatter Analysis ─────────────────────────────────────────
    st.markdown(section_header("Scatter Analysis", "Feature-Pair Exploration", "◎"),
                unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        x_col = st.selectbox("X Axis", FEATURE_COLS, index=4, key="scatter_x")
    with sc2:
        y_col = st.selectbox("Y Axis", FEATURE_COLS, index=0, key="scatter_y")

    if "label" in df.columns:
        st.plotly_chart(scatter_chart(df, x_col, y_col, "label", f"{x_col} vs {y_col}"),
                        use_container_width=True, config={"displayModeBar": False})

    if "label" in df.columns:
        st.markdown(neon_divider(), unsafe_allow_html=True)
        st.markdown(section_header("Per-Label Feature Means", "Group Statistics by Affect State", "◉"),
                    unsafe_allow_html=True)
        label_group = df.groupby("label")[FEATURE_COLS].mean().round(4)
        st.dataframe(label_group, use_container_width=True)
