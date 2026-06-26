"""
dashboard/pages/timeline.py
Page 6 - Activity Timeline.
Full session prediction history with filtering and export.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    C, glass_card, kpi_card, neon_divider, section_header, timeline_row,
)


def render(session_state: dict) -> None:
    st.markdown(section_header("Activity Timeline", "Prediction Session History", "◷"), unsafe_allow_html=True)

    history: list[dict] = session_state.get("prediction_history", [])

    if not history:
        st.markdown(glass_card(f"""
            <div style="text-align:center;padding:30px 0">
                <div style="font-size:2rem;margin-bottom:12px">◷</div>
                <div style="font-family:var(--font-display);font-size:0.85rem;
                             color:{C['secondary']};letter-spacing:0.15em">
                    NO ACTIVITY RECORDED
                </div>
                <div style="font-size:0.72rem;color:{C['secondary']};margin-top:8px;
                             font-family:var(--font-data)">
                    Run analyses from the Cognitive Analysis Center to populate this timeline.
                </div>
            </div>
        """, "purple"), unsafe_allow_html=True)
        return

    # ── Summary KPIs ─────────────────────────────────────────────
    total      = len(history)
    avg_conf   = sum(e["confidence_pct"] for e in history) / total
    high_count = sum(1 for e in history if e["tier"] == "High")
    low_count  = sum(1 for e in history if e["tier"] == "Low")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card(str(total), "Total Sessions", "◷", C["cyan"]), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card(f"{avg_conf:.1f}%", "Avg Confidence", "◎", C["green"]), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card(str(high_count), "High Confidence", "▲", C["purple"]), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card(str(low_count), "Low Confidence (Review)", "⚠", C["orange"]), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Filter Controls ──────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_tier = st.selectbox("Filter by Tier", ["All", "High", "Medium", "Low"], key="tl_tier")
    with fc2:
        filter_arousal = st.selectbox("Filter by Arousal", ["All", "High", "Medium", "Low"], key="tl_arousal")
    with fc3:
        filter_load = st.selectbox("Filter by Cognitive Load", ["All", "High", "Medium", "Low"], key="tl_load")

    filtered = history.copy()
    if filter_tier != "All":
        filtered = [e for e in filtered if e["tier"] == filter_tier]
    if filter_arousal != "All":
        filtered = [e for e in filtered if e["arousal"] == filter_arousal]
    if filter_load != "All":
        filtered = [e for e in filtered if e["cognitive_load"] == filter_load]

    st.markdown(
        f'<div style="font-size:0.65rem;color:{C["secondary"]};letter-spacing:0.15em;'
        f'text-transform:uppercase;font-family:var(--font-data);margin-bottom:10px">'
        f'Showing <span style="color:{C["cyan"]}">{len(filtered)}</span> of '
        f'<span style="color:{C["white"]}">{total}</span> entries</div>',
        unsafe_allow_html=True,
    )

    # ── Timeline Rows ─────────────────────────────────────────────
    for entry in reversed(filtered):
        st.markdown(timeline_row(
            session_id=entry["session_id"],
            timestamp=entry["timestamp"],
            arousal=entry["arousal"],
            cog_load=entry["cognitive_load"],
            confidence=entry["confidence_pct"],
            tier=entry["tier"],
        ), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Confidence Trend Chart ────────────────────────────────────
    if len(filtered) >= 2:
        st.markdown(section_header("Confidence Trend", "Over Session History", "◎"), unsafe_allow_html=True)
        hist_df = pd.DataFrame(filtered)
        tier_colors = {"High": C["green"], "Medium": C["orange"], "Low": C["red"]}
        point_colors = [tier_colors.get(t, C["cyan"]) for t in hist_df["tier"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(hist_df))),
            y=hist_df["confidence_pct"],
            mode="lines+markers",
            line=dict(color=C["cyan"], width=2, shape="spline"),
            fill="tozeroy",
            fillcolor=f"rgba(0,229,255,0.05)",
            marker=dict(color=point_colors, size=8, line=dict(width=1, color=C["bg"])),
            hovertemplate="Session %{x}<br>Confidence: %{y:.1f}%<extra></extra>",
        ))
        # Reference lines
        for val, color, label in [(85, C["green"], "High"), (60, C["orange"], "Medium")]:
            fig.add_hline(
                y=val, line=dict(color=color, width=1, dash="dot"),
                annotation_text=label, annotation_font=dict(size=9, color=color),
            )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk", color=C["secondary"]),
            margin=dict(l=10, r=10, t=20, b=10),
            height=250,
            xaxis=dict(gridcolor=C["grid"], tickfont=dict(size=9)),
            yaxis=dict(gridcolor=C["grid"], ticksuffix="%", range=[0, 110], tickfont=dict(size=9)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Export ────────────────────────────────────────────────────
    if filtered:
        st.markdown(neon_divider(), unsafe_allow_html=True)
        export_df  = pd.DataFrame(filtered)
        csv_bytes  = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Export Timeline CSV",
            data=csv_bytes,
            file_name="cogniarousal_timeline.csv",
            mime="text/csv",
        )
