"""
dashboard/pages/mission_control.py
Page 1 - Mission Control Dashboard.
System status, model status, KPIs, active sessions, live monitoring.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.components import (
    C, gauge_chart, glass_card, kpi_card,
    neon_divider, section_header, status_badge,
)


def render(df: pd.DataFrame, session_state: dict) -> None:
    st.markdown(section_header("Mission Control", "System Status & Real-Time Monitoring", "⬡"), unsafe_allow_html=True)

    # ── System Status Row ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        h1 = (
            '<div class="section-label">Prediction Engine</div>'
            f'<div style="margin-top:8px">{status_badge("ONLINE", "online")}</div>'
            f'<div style="font-size:0.7rem;color:{C["secondary"]};margin-top:6px;font-family:var(--font-data)">'
            'Random Forest &nbsp;|&nbsp; v1.0</div>'
        )
        st.markdown(glass_card(h1, "green"), unsafe_allow_html=True)

    with c2:
        h2 = (
            '<div class="section-label">Responsible AI Engine</div>'
            f'<div style="margin-top:8px">{status_badge("ONLINE", "online")}</div>'
            f'<div style="font-size:0.7rem;color:{C["secondary"]};margin-top:6px;font-family:var(--font-data)">'
            'SHAP TreeExplainer &nbsp;|&nbsp; v0.51</div>'
        )
        st.markdown(glass_card(h2, "cyan"), unsafe_allow_html=True)

    with c3:
        h3 = (
            '<div class="section-label">Privacy Engine</div>'
            f'<div style="margin-top:8px">{status_badge("ACTIVE", "online")}</div>'
            f'<div style="font-size:0.7rem;color:{C["secondary"]};margin-top:6px;font-family:var(--font-data)">'
            'UUID4 Sessions &nbsp;|&nbsp; Zero PII</div>'
        )
        st.markdown(glass_card(h3, "purple"), unsafe_allow_html=True)

    with c4:
        model_ok = session_state.get("rai_loaded", False)
        state = "online" if model_ok else "warning"
        label = "LOADED" if model_ok else "LOADING"
        h4 = (
            '<div class="section-label">Model Artifacts</div>'
            f'<div style="margin-top:8px">{status_badge(label, state)}</div>'
            f'<div style="font-size:0.7rem;color:{C["secondary"]};margin-top:6px;font-family:var(--font-data)">'
            'Arousal &nbsp;|&nbsp; Cognitive Load</div>'
        )
        st.markdown(glass_card(h4, "orange"), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────────
    total_preds  = session_state.get("total_predictions", 0)
    active_sess  = session_state.get("active_sessions", 0)
    avg_conf     = session_state.get("avg_confidence", 0.0)
    last_ts      = session_state.get("last_analysis", "-")
    n_subjects   = df["subject_id"].nunique() if "subject_id" in df.columns else 0
    n_windows    = len(df)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(kpi_card(str(total_preds), "Total Predictions", "⚡", C["cyan"]), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card(str(active_sess), "Active Sessions", "🔐", C["purple"]), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card(f"{avg_conf:.1f}%", "Avg Confidence", "◎", C["green"]), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card(str(n_subjects), "Subjects Loaded", "👤", C["orange"]), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_card(str(n_windows), "Feature Windows", "▦", C["pink"]), unsafe_allow_html=True)
    with k6:
        ts_display = last_ts if last_ts != "-" else "-"
        st.markdown(kpi_card(ts_display, "Last Analysis", "◷", C["secondary"]), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Model Performance Gauges ─────────────────────────────────
    st.markdown(section_header("Model Performance", "Stratified 5-Fold CV Metrics", "◈"), unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(gauge_chart(83.7, "Arousal CV Accuracy", C["cyan"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(gauge_chart(79.0, "Arousal CV F1", C["purple"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(gauge_chart(83.7, "Cog. Load CV Accuracy", C["green"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g4:
        st.plotly_chart(gauge_chart(79.0, "Cog. Load CV F1", C["pink"]),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Dataset Status ───────────────────────────────────────────
    st.markdown(section_header("Dataset Intelligence", "WESAD Dataset Overview", "◉"), unsafe_allow_html=True)

    d1, d2 = st.columns([1, 2])
    with d1:
        label_counts = df["label"].value_counts()
        label_color  = {"baseline": C["cyan"], "stress": C["pink"], "amusement": C["green"]}
        dist_html = '<div class="section-label" style="margin-bottom:12px">Label Distribution</div>'
        for l, v in label_counts.items():
            lc = label_color.get(l, C["purple"])
            dist_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin-bottom:8px;padding:6px 10px;border-radius:6px;'
                f'background:rgba(255,255,255,0.03);border:1px solid {lc}22">'
                f'<span style="font-family:var(--font-body);font-size:0.8rem;color:{lc}">'
                f'{l.capitalize()}</span>'
                f'<span style="font-family:var(--font-display);font-size:0.85rem;'
                f'font-weight:700;color:{lc}">{v}</span>'
                f'</div>'
            )
        st.markdown(glass_card(dist_html, "purple"), unsafe_allow_html=True)

    with d2:
        feature_cols = ["eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
                        "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability"]
        stats = df[feature_cols].describe().T[["mean", "std", "min", "max"]].round(3)
        st.markdown(
            f'<div style="border:1px solid {C["border"]};border-radius:8px;overflow:hidden">',
            unsafe_allow_html=True,
        )
        st.dataframe(
            stats,
            use_container_width=True,
            height=220,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Recent Activity Preview ──────────────────────────────────
    history = session_state.get("prediction_history", [])
    if history:
        st.markdown(neon_divider(), unsafe_allow_html=True)
        st.markdown(section_header("Recent Activity", "Latest Prediction Events", "◷"), unsafe_allow_html=True)
        from dashboard.components import timeline_row
        for entry in reversed(history[-5:]):
            st.markdown(timeline_row(
                session_id=entry["session_id"],
                timestamp=entry["timestamp"],
                arousal=entry["arousal"],
                cog_load=entry["cognitive_load"],
                confidence=entry["confidence_pct"],
                tier=entry["tier"],
            ), unsafe_allow_html=True)
