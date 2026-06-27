"""
dashboard/pages/demo.py
Demo Mode - Page 0.
Six built-in physiological profiles for instant platform demonstration.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.components import (
    C, gauge_chart, glass_card, kpi_card, narrative_box,
    neon_divider, prediction_card, probability_bar_chart,
    section_header, shap_feature_row, status_badge,
)
from src.demo_profiles import PROFILES, list_profiles
from src.report_generator import ReportGenerator
from src.validator import Validator

_validator = Validator()

_PROFILE_ORDER = [
    "low_cognitive_load",
    "medium_cognitive_load",
    "high_cognitive_load",
    "low_emotional_arousal",
    "medium_emotional_arousal",
    "high_emotional_arousal",
]


def render(df: pd.DataFrame, session_state: dict, rai) -> None:
    st.markdown(
        section_header("Demo Mode", "Built-in Physiological Profiles - Instant Analysis", "◉"),
        unsafe_allow_html=True,
    )

    banner_html = (
        '<div style="display:flex;align-items:center;gap:16px">'
        '<div style="font-size:1.6rem">&#9673;</div>'
        '<div>'
        f'<div style="font-family:var(--font-body);font-size:0.9rem;color:{C["white"]};font-weight:600">'
        'Demonstration Mode - No Real Subject Data Required</div>'
        f'<div style="font-size:0.72rem;color:{C["secondary"]};font-family:var(--font-data);margin-top:3px">'
        'Select any of the 6 pre-built physiological profiles to instantly run the full inference pipeline '
        '- prediction, confidence scoring, SHAP explanation, and transparent AI narrative '
        '- without uploading any data.</div>'
        '</div></div>'
    )
    st.markdown(glass_card(banner_html, "cyan"), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Profile Cards Grid ───────────────────────────────────────
    st.markdown(
        section_header("Select a Profile", "Click any profile card to run instant analysis", "◈"),
        unsafe_allow_html=True,
    )

    profiles = [PROFILES[k] for k in _PROFILE_ORDER]
    col_groups = [profiles[:3], profiles[3:]]

    for group in col_groups:
        cols = st.columns(3)
        for col, profile in zip(cols, group):
            with col:
                _render_profile_card(profile, session_state, rai)

    # ── Results area ─────────────────────────────────────────────
    if st.session_state.get("demo_result"):
        st.markdown(neon_divider(), unsafe_allow_html=True)
        _render_demo_results(session_state, rai)


def _render_profile_card(profile, session_state: dict, rai) -> None:
    accent = profile.accent
    card_html = (
        f'<div class="glass-card" style="border-color:{accent}40;text-align:center;padding:18px 16px;min-height:160px">'
        f'<div style="font-size:1.8rem;margin-bottom:6px">{profile.icon}</div>'
        f'<div style="font-family:var(--font-display);font-size:0.75rem;font-weight:700;'
        f'color:{accent};letter-spacing:0.08em;margin-bottom:4px">{profile.name.upper()}</div>'
        f'<div style="font-size:0.65rem;color:{C["secondary"]};font-family:var(--font-data);'
        f'line-height:1.5;margin-bottom:10px">{profile.description}</div>'
        f'<div style="display:flex;justify-content:center;gap:8px">'
        f'<span class="status-badge" style="background:{accent}18;border:1px solid {accent}44;'
        f'color:{accent};font-size:0.55rem">Arousal: {profile.expected_arousal}</span>'
        f'<span class="status-badge" style="background:{accent}18;border:1px solid {accent}44;'
        f'color:{accent};font-size:0.55rem">Load: {profile.expected_cog_load}</span>'
        f'</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if st.button(f"Run  {profile.name}", key=f"demo_{profile.key}", use_container_width=True):
        with st.spinner(f"Analysing {profile.name}..."):
            _execute_demo(profile, session_state, rai)
        st.rerun()


def _execute_demo(profile, session_state: dict, rai) -> None:
    """Run full pipeline for a demo profile and store result in session state."""
    sample = profile.features

    # Validate
    val = _validator.validate_sample(sample)
    if not val.valid:
        session_state["demo_result"] = {"error": val.errors}
        return

    # Inference + explanation
    with rai.privacy.session() as session_id:
        a_res  = rai.prediction.predict_emotional_arousal(sample)
        c_res  = rai.prediction.predict_cognitive_load(sample)

        a_proba = {
            cls: float(a_res[f"prob_{cls}"].iloc[0])
            for cls in rai.prediction._artifacts["arousal"]["label_encoder"].classes_
        }
        c_proba = {
            cls: float(c_res[f"prob_{cls}"].iloc[0])
            for cls in rai.prediction._artifacts["cognitive_load"]["label_encoder"].classes_
        }
        a_conf = rai._confidence["arousal"].evaluate(list(a_proba.values()))
        c_conf = rai._confidence["cognitive_load"].evaluate(list(c_proba.values()))

        rai_result = rai.explain_prediction(
            session_id=session_id,
            target="cognitive_load",
            data=sample,
            top_n=9,
            sanitise=False,
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_avg  = (a_conf.confidence_pct + c_conf.confidence_pct) / 2

    # Update global session state
    session_state["total_predictions"] = session_state.get("total_predictions", 0) + 2
    session_state["active_sessions"]   = session_state.get("active_sessions", 0) + 1
    session_state["last_analysis"]     = datetime.now().strftime("%H:%M:%S")
    total = session_state["total_predictions"]
    prev  = session_state.get("avg_confidence", 0.0)
    session_state["avg_confidence"] = round(
        (prev * (total - 2) + conf_avg * 2) / total, 2
    ) if total > 2 else round(conf_avg, 2)

    history = session_state.get("prediction_history", [])
    history.append({
        "session_id":    session_id,
        "timestamp":     datetime.now().strftime("%H:%M:%S"),
        "arousal":       a_res["predicted_class"].iloc[0],
        "cognitive_load":c_res["predicted_class"].iloc[0],
        "confidence_pct":round(conf_avg, 1),
        "tier":          a_conf.tier,
    })
    session_state["prediction_history"] = history[-100:]

    session_state["demo_result"] = {
        "profile_name":   profile.name,
        "profile_icon":   profile.icon,
        "profile_accent": profile.accent,
        "clinical_notes": profile.clinical_notes,
        "session_id":     session_id,
        "timestamp":      timestamp,
        "sample":         sample,
        "a_class":        a_res["predicted_class"].iloc[0],
        "a_conf":         a_conf,
        "a_proba":        a_proba,
        "c_class":        c_res["predicted_class"].iloc[0],
        "c_conf":         c_conf,
        "c_proba":        c_proba,
        "rai_result":     rai_result,
        "warnings":       val.warnings,
    }


def _render_demo_results(session_state: dict, rai) -> None:
    r = session_state["demo_result"]

    if "error" in r:
        for e in r["error"]:
            st.error(e)
        return

    accent = r["profile_accent"]
    st.markdown(
        section_header(
            f"Results: {r['profile_name']}",
            f"Session {r['session_id'][:16]}...  |  {r['timestamp']}",
            r["profile_icon"],
        ),
        unsafe_allow_html=True,
    )

    # Validation warnings
    for w in r.get("warnings", []):
        st.warning(w)

    # ── Prediction cards ─────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(prediction_card(
            predicted_class=r["a_class"],
            target_label="EMOTIONAL AROUSAL",
            confidence_pct=r["a_conf"].confidence_pct,
            tier=r["a_conf"].tier,
            flag_review=r["a_conf"].flag_review,
        ), unsafe_allow_html=True)
        st.plotly_chart(
            probability_bar_chart(r["a_proba"], "Arousal Class Probabilities"),
            use_container_width=True, config={"displayModeBar": False},
        )

    with col_b:
        st.markdown(prediction_card(
            predicted_class=r["c_class"],
            target_label="COGNITIVE LOAD",
            confidence_pct=r["c_conf"].confidence_pct,
            tier=r["c_conf"].tier,
            flag_review=r["c_conf"].flag_review,
        ), unsafe_allow_html=True)
        st.plotly_chart(
            probability_bar_chart(r["c_proba"], "Cognitive Load Class Probabilities"),
            use_container_width=True, config={"displayModeBar": False},
        )

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── SHAP + clinical notes ────────────────────────────────────
    shap_col, notes_col = st.columns([3, 2])

    with shap_col:
        st.markdown(
            section_header("SHAP Local Explanation", "Feature Contributions to Cognitive Load", "◈"),
            unsafe_allow_html=True,
        )
        rai_res    = r["rai_result"]
        all_contribs = rai._explainers["cognitive_load"].explain_local(
            sample=r["sample"],
            predicted_class=r["c_class"],
            top_n=9,
        ).feature_contributions
        max_abs = max((abs(fc.shap_value) for fc in all_contribs), default=1e-9)
        shap_html = "".join([
            shap_feature_row(fc.label, fc.shap_value, max_abs, fc.direction)
            for fc in all_contribs
        ])
        st.markdown(glass_card(shap_html, "cyan"), unsafe_allow_html=True)

    with notes_col:
        st.markdown(
            section_header("Clinical Profile Notes", "", "◎"),
            unsafe_allow_html=True,
        )
        notes_html = "".join([
            f'<div style="display:flex;gap:8px;margin-bottom:8px;padding:8px 10px;'
            f'border-radius:6px;background:rgba(255,255,255,0.03);'
            f'border-left:2px solid {accent}">'
            f'<span style="color:{accent};font-size:0.7rem">◈</span>'
            f'<span style="font-size:0.75rem;color:{C["white"]};font-family:var(--font-body)">{note}</span>'
            f'</div>'
            for note in r["clinical_notes"]
        ])
        st.markdown(glass_card(notes_html, "purple"), unsafe_allow_html=True)

        # Confidence gauges
        st.plotly_chart(
            gauge_chart(r["a_conf"].confidence_pct, "Arousal Confidence", C["pink"]),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.plotly_chart(
            gauge_chart(r["c_conf"].confidence_pct, "Load Confidence", C["orange"]),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── Narrative ────────────────────────────────────────────────
    st.markdown(neon_divider(), unsafe_allow_html=True)
    st.markdown(
        section_header("AI Transparency Narrative", "", "◎"),
        unsafe_allow_html=True,
    )
    st.markdown(narrative_box(rai_res.narrative), unsafe_allow_html=True)

    # ── Report downloads ─────────────────────────────────────────
    st.markdown(neon_divider(), unsafe_allow_html=True)
    st.markdown(
        section_header("Download Report", "PDF and CSV formats available", "⬇"),
        unsafe_allow_html=True,
    )

    top_features_dicts = [
        {
            "feature":    fc.feature,
            "label":      fc.label,
            "direction":  fc.direction,
            "shap_value": fc.shap_value,
        }
        for fc in rai_res.top_features
    ]

    rg = ReportGenerator(
        session_id=r["session_id"],
        timestamp=r["timestamp"],
        input_data=r["sample"],
        arousal_result={
            "predicted_class": r["a_class"],
            "confidence_pct":  r["a_conf"].confidence_pct,
            "tier":            r["a_conf"].tier,
            "class_probs":     r["a_proba"],
        },
        cog_load_result={
            "predicted_class": r["c_class"],
            "confidence_pct":  r["c_conf"].confidence_pct,
            "tier":            r["c_conf"].tier,
            "class_probs":     r["c_proba"],
        },
        narrative=rai_res.narrative,
        top_features=top_features_dicts,
    )

    dl1, dl2 = st.columns(2)
    with dl1:
        pdf_bytes = rg.generate_pdf()
        st.download_button(
            label="⬇  Download PDF Report",
            data=pdf_bytes,
            file_name=f"cogniarousal_report_{r['session_id'][:8]}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl2:
        csv_str = rg.generate_csv()
        st.download_button(
            label="⬇  Download CSV Report",
            data=csv_str.encode("utf-8"),
            file_name=f"cogniarousal_report_{r['session_id'][:8]}.csv",
            mime="text/csv",
            use_container_width=True,
        )
