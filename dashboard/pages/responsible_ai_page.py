"""
dashboard/pages/responsible_ai_page.py
Page 4 - Responsible AI Center.
SHAP explanations, confidence analysis, transparency narratives.
"""

import pandas as pd
import streamlit as st

from dashboard.components import (
    C, gauge_chart, glass_card, global_shap_bar,
    kpi_card, narrative_box, neon_divider,
    prediction_card, probability_bar_chart, section_header, shap_feature_row,
)

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability",
]


def render(df: pd.DataFrame, session_state: dict, rai) -> None:
    st.markdown(section_header("Responsible AI Center", "Explainability · Confidence · Transparency", "◈"), unsafe_allow_html=True)

    tab_local, tab_global = st.tabs(["  Local Explanation  ", "  Global Importance  "])

    # ── Tab: Local Explanation ───────────────────────────────────
    with tab_local:
        sel_info_html = (
            '<div class="section-label" style="margin-bottom:6px">Sample Selection</div>'
            f'<div style="font-size:0.72rem;color:{C["secondary"]};font-family:var(--font-data)">'
            'Select a sample from the feature dataset to generate a fully transparent '
            'SHAP-based explanation with narrative.</div>'
        )
        st.markdown(glass_card(sel_info_html, "cyan"), unsafe_allow_html=True)

        col_idx, col_lbl = st.columns([1, 2])
        with col_idx:
            sample_idx = st.number_input(
                "Sample Index", min_value=0, max_value=len(df) - 1, value=0, step=1, key="rai_idx"
            )
        with col_lbl:
            target_sel = st.selectbox(
                "Target", ["arousal", "cognitive_load"], key="rai_target",
                format_func=lambda x: "Emotional Arousal" if x == "arousal" else "Cognitive Load"
            )
            top_n = st.selectbox("Top N Features", [3, 5, 9], index=0, key="rai_topn")

        sample_row = df[FEATURE_COLS].iloc[sample_idx]
        label_row  = df["label"].iloc[sample_idx] if "label" in df.columns else "unknown"

        feat_spans = "".join(
            f'<span style="font-size:0.7rem;color:{C["secondary"]};font-family:var(--font-data)">'
            f'<span style="color:{C["cyan"]}">{k}</span>: '
            f'<span style="color:{C["white"]};font-weight:600">{v:.4f}</span></span> '
            for k, v in sample_row.items()
        )
        sample_html = (
            f'<div style="display:flex;gap:16px;flex-wrap:wrap">{feat_spans}'
            f'<span style="font-size:0.7rem;color:{C["orange"]};font-family:var(--font-data)">'
            f'WESAD Label: <b>{label_row}</b></span></div>'
        )
        st.markdown(glass_card(sample_html, "purple"), unsafe_allow_html=True)

        run_btn = st.button("⚡  Generate Explanation", type="primary", key="rai_explain_btn")

        if run_btn:
            with st.spinner("Computing SHAP values..."):
                with rai.privacy.session() as session_id:
                    result = rai.explain_prediction(
                        session_id=session_id,
                        target=target_sel,
                        data=sample_row.to_dict(),
                        top_n=top_n,
                        sanitise=False,
                    )

            st.markdown(neon_divider(), unsafe_allow_html=True)

            # Results layout
            res1, res2 = st.columns([1, 1])

            with res1:
                st.markdown(prediction_card(
                    predicted_class=result.predicted_class,
                    target_label="EMOTIONAL AROUSAL" if target_sel == "arousal" else "COGNITIVE LOAD",
                    confidence_pct=result.confidence.confidence_pct,
                    tier=result.confidence.tier,
                    flag_review=result.flag_review,
                ), unsafe_allow_html=True)

                # Confidence gauges
                st.plotly_chart(
                    gauge_chart(result.confidence.confidence_pct, "Prediction Confidence",
                                C["cyan"] if result.confidence.tier == "High"
                                else C["orange"] if result.confidence.tier == "Medium"
                                else C["red"]),
                    use_container_width=True, config={"displayModeBar": False},
                )

                # Probability bar
                st.plotly_chart(
                    probability_bar_chart(result.confidence.class_probs,
                                          "Class Probabilities"),
                    use_container_width=True, config={"displayModeBar": False},
                )

            with res2:
                st.markdown(section_header("SHAP Local Explanation", "Feature Attributions for Predicted Class", "◈"), unsafe_allow_html=True)

                # SHAP feature rows
                all_contribs = rai._explainers[target_sel].explain_local(
                    sample=sample_row.to_dict(),
                    predicted_class=result.predicted_class,
                    top_n=9,
                ).feature_contributions
                # Rebuild full 9-feature list (explain_local may be capped to top_n)
                # Re-explain with top_n=9 for full visualization
                max_abs = max((abs(fc.shap_value) for fc in all_contribs), default=1e-9)
                shap_html = "".join([
                    shap_feature_row(fc.label, fc.shap_value, max_abs, fc.direction)
                    for fc in all_contribs
                ])
                st.markdown(glass_card(shap_html, "cyan"), unsafe_allow_html=True)

            # Narrative
            st.markdown(neon_divider(), unsafe_allow_html=True)
            st.markdown(section_header("AI Transparency Narrative", "", "◎"), unsafe_allow_html=True)
            st.markdown(narrative_box(result.narrative), unsafe_allow_html=True)

            # Entropy & metadata
            em1, em2, em3, em4 = st.columns(4)
            with em1:
                st.markdown(kpi_card(f"{result.confidence.entropy:.4f}", "Prediction Entropy", "⊗", C["purple"]), unsafe_allow_html=True)
            with em2:
                st.markdown(kpi_card(result.confidence.tier, "Confidence Tier", "◎", C["cyan"]), unsafe_allow_html=True)
            with em3:
                flag_color = C["red"] if result.flag_review else C["green"]
                st.markdown(kpi_card("YES" if result.flag_review else "NO", "Flag Review", "⚠", flag_color), unsafe_allow_html=True)
            with em4:
                st.markdown(kpi_card(result.session_id[:8] + "...", "Session ID", "🔐", C["secondary"]), unsafe_allow_html=True)

    # ── Tab: Global Importance ───────────────────────────────────
    with tab_global:
        global_info_html = (
            '<div class="section-label" style="margin-bottom:6px">Global SHAP Feature Importance</div>'
            f'<div style="font-size:0.72rem;color:{C["secondary"]};font-family:var(--font-data)">'
            'Mean absolute SHAP values computed across the full feature dataset. '
            'Higher values indicate greater average contribution to predictions.</div>'
        )
        st.markdown(glass_card(global_info_html, "purple"), unsafe_allow_html=True)

        g_target = st.selectbox(
            "Target", ["arousal", "cognitive_load"], key="global_shap_target",
            format_func=lambda x: "Emotional Arousal" if x == "arousal" else "Cognitive Load"
        )

        if st.button("⚡  Compute Global SHAP", type="primary", key="global_shap_btn"):
            with st.spinner("Computing global SHAP importances across dataset..."):
                global_df = rai.global_importance(g_target, df)

            st.markdown(neon_divider(), unsafe_allow_html=True)
            st.markdown(section_header("Global Feature Importance", "Mean |SHAP| Across Dataset", "◈"), unsafe_allow_html=True)

            # Bar chart
            st.plotly_chart(
                global_shap_bar(global_df,
                                f"Global SHAP Importance - {'Emotional Arousal' if g_target == 'arousal' else 'Cognitive Load'}"),
                use_container_width=True, config={"displayModeBar": False},
            )

            # Top feature narrative
            top_feat = global_df.iloc[0]
            st.markdown(narrative_box(
                f"The most influential physiological signal for {g_target.replace('_', ' ')} prediction "
                f"is <b>{top_feat['feature']}</b> with a mean absolute SHAP value of "
                f"<b>{top_feat['mean_abs_shap']:.4f}</b>, followed by "
                f"<b>{global_df.iloc[1]['feature']}</b> and <b>{global_df.iloc[2]['feature']}</b>."
            ), unsafe_allow_html=True)

            # Table
            st.dataframe(
                global_df.rename(columns={"mean_abs_shap": "Mean |SHAP|", "rank": "Rank"}),
                use_container_width=True,
                height=320,
            )

        else:
            # Show pre-computed from disk if available
            import pandas as _pd
            from pathlib import Path
            shap_path = Path("models") / g_target / "shap_global_importance.csv"
            if shap_path.exists():
                precomputed = _pd.read_csv(shap_path)
                st.markdown(neon_divider(), unsafe_allow_html=True)
                st.markdown(f'<div class="section-label" style="margin-bottom:8px;color:{C["secondary"]}">Pre-computed Global SHAP (from last explain.py run)</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    global_shap_bar(precomputed, f"Global SHAP - {g_target.replace('_', ' ').title()}"),
                    use_container_width=True, config={"displayModeBar": False},
                )
