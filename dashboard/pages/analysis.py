"""
dashboard/pages/analysis.py
Page 2 - Cognitive Analysis Center.
Manual physiological input and CSV batch upload linked to active patient profile.
Persists assessments in the clinical SQLite database.
"""

import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components import (
    C, gauge_chart, glass_card, kpi_card, neon_divider,
    prediction_card, probability_bar_chart, section_header, status_badge
)
from src.validator import Validator
from src.report_generator import ReportGenerator
from src.data_cleaner import DataCleaner

_validator = Validator()

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability",
]


def preprocess_raw_features(data: dict | pd.DataFrame) -> dict | pd.DataFrame:
    """
    Applies the preprocessing pipeline used during training to raw physiological inputs.
    Converts RMSSD and SDNN from milliseconds (ms) to seconds (s).
    Standardizes each feature using the baseline WESAD statistics and DataCleaner.
    """
    import numpy as np

    WESAD_STATS = {
        "eda_mean": {"mean": 5.03021690, "std": 3.00460941},
        "eda_std": {"mean": 0.09633747, "std": 0.09998216},
        "eda_peak_count": {"mean": 5243.71348315, "std": 4414.93447238},
        "eda_peak_amplitude": {"mean": 0.02966375, "std": 0.00897963},
        "heart_rate_bpm": {"mean": 69.18039864, "std": 10.48383984},
        "rmssd": {"mean": 0.06474043, "std": 0.03874872},
        "sdnn": {"mean": 0.08138460, "std": 0.03333514},
        "resp_rate_bpm": {"mean": 32.02247191, "std": 2.00268095},
        "resp_variability": {"mean": 0.45839002, "std": 0.06614139},
    }

    if isinstance(data, dict):
        df = pd.DataFrame([data])
        # Convert RMSSD and SDNN from ms to seconds
        df["rmssd"] = df["rmssd"] / 1000.0
        df["sdnn"] = df["sdnn"] / 1000.0

        cleaner = DataCleaner()
        cleaner._feature_cols = FEATURE_COLS
        cleaner.scaler.mean_ = np.array([WESAD_STATS[c]["mean"] for c in FEATURE_COLS])
        cleaner.scaler.scale_ = np.array([WESAD_STATS[c]["std"] for c in FEATURE_COLS])
        cleaner.scaler.var_ = cleaner.scaler.scale_ ** 2
        cleaner.scaler.n_samples_seen_ = 178

        df_scaled = cleaner.transform(df)
        return df_scaled.iloc[0].to_dict()
    else:
        df = data.copy()
        # Convert RMSSD and SDNN from ms to seconds
        df["rmssd"] = df["rmssd"] / 1000.0
        df["sdnn"] = df["sdnn"] / 1000.0

        cleaner = DataCleaner()
        cleaner._feature_cols = FEATURE_COLS
        cleaner.scaler.mean_ = np.array([WESAD_STATS[c]["mean"] for c in FEATURE_COLS])
        cleaner.scaler.scale_ = np.array([WESAD_STATS[c]["std"] for c in FEATURE_COLS])
        cleaner.scaler.var_ = cleaner.scaler.scale_ ** 2
        cleaner.scaler.n_samples_seen_ = 178

        df_scaled = cleaner.transform(df)
        return df_scaled[FEATURE_COLS]


def _run_prediction(data: dict, rai, session_state: dict, db_service, patient_id: str, doctor_id: int) -> tuple:
    monitor = session_state.get("monitor")
    ctx     = monitor.time_inference() if monitor else _null_ctx()

    with ctx:
        with rai.privacy.session() as session_id:
            arousal_res = rai.prediction.predict_emotional_arousal(data)
            cog_res     = rai.prediction.predict_cognitive_load(data)

            arousal_proba = {
                c: float(arousal_res[f"prob_{c}"].iloc[0])
                for c in rai.prediction._artifacts["arousal"]["label_encoder"].classes_
            }
            cog_proba = {
                c: float(cog_res[f"prob_{c}"].iloc[0])
                for c in rai.prediction._artifacts["cognitive_load"]["label_encoder"].classes_
            }

            a_conf = rai._confidence["arousal"].evaluate(list(arousal_proba.values()))
            c_conf = rai._confidence["cognitive_load"].evaluate(list(cog_proba.values()))

            # Responsible AI explanation
            rai_res = rai.explain_prediction(
                session_id=session_id, target="cognitive_load",
                data=data, top_n=9, sanitise=False
            )
            top_feats = [
                {"feature": fc.feature, "label": fc.label,
                 "direction": fc.direction, "shap_value": fc.shap_value}
                for fc in rai_res.top_features
            ]

    # Save to SQL database
    flag_review = 1 if (a_conf.flag_review or c_conf.flag_review) else 0
    db_service.log_assessment(
        patient_id=patient_id,
        doctor_id=doctor_id,
        session_id=session_id,
        inputs=data,
        a_pred=arousal_res["predicted_class"].iloc[0],
        a_conf=a_conf.confidence_pct / 100,
        a_tier=a_conf.tier,
        a_narrative=rai_res.narrative,
        c_pred=cog_res["predicted_class"].iloc[0],
        c_conf=c_conf.confidence_pct / 100,
        c_tier=c_conf.tier,
        c_narrative=rai_res.narrative,
        shap_list=top_feats,
        flag_review=flag_review
    )

    # Update session state metrics
    session_state["total_predictions"] = session_state.get("total_predictions", 0) + 2
    session_state["active_sessions"]   = session_state.get("active_sessions", 0) + 1
    conf_avg = (a_conf.confidence_pct + c_conf.confidence_pct) / 2
    prev_avg = session_state.get("avg_confidence", 0.0)
    total    = session_state.get("total_predictions", 2)
    session_state["avg_confidence"] = round(
        (prev_avg * (total - 2) + conf_avg * 2) / total, 2
    ) if total > 2 else round(conf_avg, 2)
    session_state["last_analysis"] = datetime.now().strftime("%H:%M:%S")

    # Append to session prediction history preview
    history = session_state.get("prediction_history", [])
    history.append({
        "session_id":    session_id,
        "timestamp":     datetime.now().strftime("%H:%M:%S"),
        "arousal":       arousal_res["predicted_class"].iloc[0],
        "cognitive_load":cog_res["predicted_class"].iloc[0],
        "confidence_pct":round(conf_avg, 1),
        "tier":          a_conf.tier,
    })
    session_state["prediction_history"] = history[-100:]

    return (
        arousal_res["predicted_class"].iloc[0], a_conf,
        cog_res["predicted_class"].iloc[0],     c_conf,
        arousal_proba, cog_proba, session_id,
    )


from contextlib import contextmanager
@contextmanager
def _null_ctx():
    yield


def render(df: pd.DataFrame, session_state: dict, rai, db_service, doctor_user) -> None:
    st.markdown(section_header("Cognitive Analysis Center", "Physiological Signal Inference", "◎"), unsafe_allow_html=True)

    # ── Select Patient Context ────────────────────────────────────────
    if "selected_patient_id" not in session_state:
        session_state["selected_patient_id"] = None

    active_patient_id = session_state["selected_patient_id"]

    if not active_patient_id:
        # Load clinician patients dropdown
        patients = db_service.get_patients_by_hospital(doctor_user["hospital_id"])
        if not patients:
            st.warning("No patients registered under this clinic. Please register a patient first in the Patient Directory.")
            if st.button("👥 Open Patient Directory"):
                session_state["nav_page"] = "Patient Directory"
                st.rerun()
            return
        
        patient_options = {p["id"]: f"{p['name']} ({p['id']})" for p in patients}
        selected_id = st.selectbox("Select Patient Profile for Assessment", list(patient_options.keys()), 
                                     format_func=lambda x: patient_options[x], key="analysis_pat_select")
        active_patient_id = selected_id
    else:
        pat_info = db_service.get_patient(active_patient_id)
        st.markdown(f"""
        <div class="glass-card card-cyan" style="margin-bottom:14px">
            <div class="section-label">Selected Patient Context</div>
            <div style="font-family:var(--font-display);font-size:0.92rem;color:var(--neon-cyan);margin-top:2px">
                {pat_info['name']}
            </div>
            <div style="font-size:0.7rem;color:var(--text-secondary);font-family:var(--font-data)">
                Patient ID: <b>{pat_info['id']}</b> &nbsp;|&nbsp; Age: <b>{pat_info['age']}</b> &nbsp;|&nbsp; 
                Work Context: <b>{pat_info['occupation']} ({pat_info['industry']})</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    tab_manual, tab_csv = st.tabs(["  Manual Input  ", "  CSV Upload  "])

    # ── Tab A: Manual Input ──────────────────────────────────────
    with tab_manual:
        info_html = (
            '<div class="section-label" style="margin-bottom:4px">Physiological Signal Input</div>'
            f'<div style="font-size:0.72rem;color:{C["secondary"]};font-family:var(--font-data);margin-top:2px">'
            'Enter raw clinical physiological measurements. The application will validate their clinical plausibility and scale them for ML inference.'
            '</div>'
        )
        st.markdown(glass_card(info_html, "cyan"), unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # EDA group
        st.markdown(f'<div class="section-label" style="margin-bottom:8px;color:{C["cyan"]}">⬡ EDA - Electrodermal Activity</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            eda_mean      = st.number_input("EDA Mean (µS)",      min_value=0.0, max_value=20.0, value=5.0, step=0.01, key="m_eda_mean")
        with c2:
            eda_std       = st.number_input("EDA Standard Deviation (µS)", min_value=0.0, max_value=10.0, value=0.1, step=0.01, key="m_eda_std")
        with c3:
            eda_peak_cnt  = st.number_input("EDA Peak Count",     min_value=0.0, max_value=100.0, value=15.0, step=1.0,  key="m_eda_pc")
        with c4:
            eda_peak_amp  = st.number_input("EDA Peak Amplitude (µS)", min_value=0.0, max_value=10.0, value=0.1, step=0.01, key="m_eda_pa")

        st.markdown(f'<div class="section-label" style="margin:8px 0;color:{C["purple"]}">♡ HRV - Heart Rate Variability</div>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        with h1:
            hr_bpm        = st.number_input("Heart Rate (BPM)",     min_value=40.0, max_value=180.0, value=75.0, step=1.0, key="m_hr")
        with h2:
            rmssd         = st.number_input("RMSSD (ms)",          min_value=5.0, max_value=250.0, value=40.0, step=1.0, key="m_rmssd")
        with h3:
            sdnn          = st.number_input("SDNN (ms)",           min_value=10.0, max_value=300.0, value=55.0, step=1.0, key="m_sdnn")

        st.markdown(f'<div class="section-label" style="margin:8px 0;color:{C["green"]}">≈ Respiration</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            resp_rate     = st.number_input("Respiration Rate (breaths/min)", min_value=5.0, max_value=40.0, value=14.0, step=0.1, key="m_rr")
        with r2:
            resp_var      = st.number_input("Respiration Variability", min_value=0.0, max_value=20.0, value=2.1, step=0.01, key="m_rv")

        sample = {
            "eda_mean": eda_mean, "eda_std": eda_std,
            "eda_peak_count": eda_peak_cnt, "eda_peak_amplitude": eda_peak_amp,
            "heart_rate_bpm": hr_bpm, "rmssd": rmssd, "sdnn": sdnn,
            "resp_rate_bpm": resp_rate, "resp_variability": resp_var,
        }

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚡  Run Cognitive Analysis", type="primary", key="run_manual")

        if run_btn:
            # Validate raw sample before inference
            val = _validator.validate_sample(sample)
            for w in val.warnings:
                st.warning(w)
            if not val.valid:
                for e in val.errors:
                    st.error(e)
            else:
                with st.spinner("Analysing physiological signals..."):
                    # Preprocess raw features to WESAD standard scaled z-scores for inference
                    scaled_sample = preprocess_raw_features(sample)
                    
                    a_class, a_conf, c_class, c_conf, a_proba, c_proba, sid = \
                        _run_prediction(scaled_sample, rai, session_state, db_service, active_patient_id, doctor_user["id"])

                st.markdown(neon_divider(), unsafe_allow_html=True)
                _render_results(a_class, a_conf, c_class, c_conf, a_proba, c_proba, sid, scaled_sample, rai)

    # ── Tab B: CSV Upload ────────────────────────────────────────
    with tab_csv:
        csv_info_html = (
            '<div class="section-label" style="margin-bottom:6px">Batch CSV Upload</div>'
            f'<div style="font-size:0.72rem;color:{C["secondary"]};font-family:var(--font-data)">'
            'Upload a CSV containing the 9 feature columns. Each row is logged separately to the patient record history. '
            'Column names must match exactly: '
            f'<code style="color:{C["cyan"]};font-size:0.68rem">'
            'eda_mean, eda_std, eda_peak_count, eda_peak_amplitude, '
            'heart_rate_bpm, rmssd, sdnn, resp_rate_bpm, resp_variability'
            '</code></div>'
        )
        st.markdown(glass_card(csv_info_html, "purple"), unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload feature CSV", type=["csv"], key="csv_upload")

        if uploaded is not None:
            try:
                upload_df = pd.read_csv(uploaded)
                missing   = [c for c in FEATURE_COLS if c not in upload_df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    # Validate the raw uploaded dataframe
                    val = _validator.validate_dataframe(upload_df)
                    for w in val.warnings:
                        st.warning(w)
                    if not val.valid:
                        for e in val.errors:
                            st.error(e)
                    else:
                        st.markdown(f"""
                        <div style="font-size:0.75rem;color:{C['secondary']};margin:8px 0;font-family:var(--font-data)">
                            Loaded <span style="color:{C['cyan']};font-weight:600">{len(upload_df)}</span> rows
                            &nbsp;|&nbsp;
                            <span style="color:{C['green']};font-weight:600">{len(FEATURE_COLS)}</span> features detected
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("⚡  Run Batch Analysis", type="primary", key="run_csv"):
                            with st.spinner(f"Processing and saving {len(upload_df)} samples to patient timeline..."):
                                # Preprocess the entire DataFrame to standard scaled features
                                upload_df_scaled = preprocess_raw_features(upload_df)
                                
                                # Iterate and run prediction/explanation for database
                                for idx, row in upload_df_scaled.iterrows():
                                    row_dict = row[FEATURE_COLS].to_dict()
                                    
                                    # Run inference
                                    arousal_res = rai.prediction.predict_emotional_arousal(row_dict)
                                    cog_res     = rai.prediction.predict_cognitive_load(row_dict)

                                    arousal_proba = {c: float(arousal_res[f"prob_{c}"].iloc[0]) for c in rai.prediction._artifacts["arousal"]["label_encoder"].classes_}
                                    cog_proba = {c: float(cog_res[f"prob_{c}"].iloc[0]) for c in rai.prediction._artifacts["cognitive_load"]["label_encoder"].classes_}

                                    a_c = rai._confidence["arousal"].evaluate(list(arousal_proba.values()))
                                    c_c = rai._confidence["cognitive_load"].evaluate(list(cog_proba.values()))

                                    # SHAP
                                    with rai.privacy.session() as sess_id:
                                        r_res = rai.explain_prediction(session_id=sess_id, target="cognitive_load", data=row_dict, top_n=3, sanitise=False)
                                    
                                    t_feats = [{"feature": fc.feature, "label": fc.label, "direction": fc.direction, "shap_value": fc.shap_value} for fc in r_res.top_features]

                                    # Log to SQLite (logs standardized z-score values)
                                    db_service.log_assessment(
                                        patient_id=active_patient_id,
                                        doctor_id=doctor_user["id"],
                                        session_id=sess_id,
                                        inputs=row_dict,
                                        a_pred=arousal_res["predicted_class"].iloc[0],
                                        a_conf=a_c.confidence_pct / 100,
                                        a_tier=a_c.tier,
                                        a_narrative=r_res.narrative,
                                        c_pred=cog_res["predicted_class"].iloc[0],
                                        c_conf=c_c.confidence_pct / 100,
                                        c_tier=c_c.tier,
                                        c_narrative=r_res.narrative,
                                        shap_list=t_feats,
                                        flag_review=1 if (a_c.flag_review or c_c.flag_review) else 0
                                    )

                            # Make results_df contain the scaled features, matching training z-score space
                            results_df = upload_df_scaled[FEATURE_COLS].copy()
                            
                            # Run a quick batch call for grid view display using scaled features
                            arousal_preds = rai.prediction.predict_emotional_arousal(upload_df_scaled)
                            cog_preds     = rai.prediction.predict_cognitive_load(upload_df_scaled)
                            
                            results_df["arousal_prediction"]       = arousal_preds["predicted_class"].values
                            results_df["arousal_confidence"]       = (arousal_preds["confidence"] * 100).round(1).values
                            results_df["cognitive_load_prediction"]= cog_preds["predicted_class"].values
                            results_df["cognitive_load_confidence"]= (cog_preds["confidence"] * 100).round(1).values

                            session_state["total_predictions"] = \
                                session_state.get("total_predictions", 0) + len(upload_df) * 2

                            st.markdown(neon_divider(), unsafe_allow_html=True)
                            st.markdown(section_header("Batch Results Logged", f"{len(results_df)} assessments saved", "▦"), unsafe_allow_html=True)

                            # Summary KPIs
                            b1, b2, b3, b4 = st.columns(4)
                            with b1:
                                st.markdown(kpi_card(str(len(results_df)), "Samples Saved", "▦", C["cyan"]), unsafe_allow_html=True)
                            with b2:
                                avg_a = results_df["arousal_confidence"].mean()
                                st.markdown(kpi_card(f"{avg_a:.1f}%", "Avg Arousal Conf.", "◎", C["purple"]), unsafe_allow_html=True)
                            with b3:
                                avg_c = results_df["cognitive_load_confidence"].mean()
                                st.markdown(kpi_card(f"{avg_c:.1f}%", "Avg Load Conf.", "◎", C["green"]), unsafe_allow_html=True)
                            with b4:
                                high_n = (results_df["arousal_prediction"] == "High").sum()
                                st.markdown(kpi_card(str(high_n), "High Arousal Rows", "⚠", C["pink"]), unsafe_allow_html=True)

                            st.dataframe(results_df, use_container_width=True, height=250)

                            csv_out = results_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇  Download Results CSV",
                                data=csv_out,
                                file_name="cogniarousal_batch_results.csv",
                                mime="text/csv",
                            )

            except Exception as exc:
                st.error(f"CSV processing error: {exc}. Check that all 9 feature columns are present and numeric.")


def _render_results(a_class, a_conf, c_class, c_conf, a_proba, c_proba, session_id,
                    input_data: dict = None, rai=None) -> None:
    st.markdown(section_header("Analysis Results", f"Session: {session_id[:16]}...", "⚡"), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(prediction_card(
            predicted_class=a_class,
            target_label="EMOTIONAL AROUSAL",
            confidence_pct=a_conf.confidence_pct,
            tier=a_conf.tier,
            flag_review=a_conf.flag_review,
        ), unsafe_allow_html=True)
        st.plotly_chart(probability_bar_chart(a_proba, "Arousal Probabilities"),
                        use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown(prediction_card(
            predicted_class=c_class,
            target_label="COGNITIVE LOAD",
            confidence_pct=c_conf.confidence_pct,
            tier=c_conf.tier,
            flag_review=c_conf.flag_review,
        ), unsafe_allow_html=True)
        st.plotly_chart(probability_bar_chart(c_proba, "Cognitive Load Probabilities"),
                        use_container_width=True, config={"displayModeBar": False})

    # Risk panel
    risk_level = _compute_risk(a_class, c_class)
    risk_color = {"Critical": C["red"], "High": C["pink"], "Moderate": C["orange"], "Low": C["green"]}[risk_level]

    risk_html = (
        '<div style="display:flex;align-items:center;justify-content:space-between">'
        '<div>'
        '<div class="section-label">Composite Risk Assessment</div>'
        f'<div style="font-family:var(--font-display);font-size:1.5rem;font-weight:700;'
        f'color:{risk_color};text-shadow:0 0 20px {risk_color}88;margin-top:4px">'
        f'{risk_level.upper()} RISK</div>'
        '</div>'
        '<div style="text-align:right">'
        '<div class="section-label">Arousal</div>'
        f'<div style="color:{C["pink"]};font-family:var(--font-display);font-size:0.9rem">{a_class}</div>'
        '<div class="section-label" style="margin-top:6px">Cognitive Load</div>'
        f'<div style="color:{C["orange"]};font-family:var(--font-display);font-size:0.9rem">{c_class}</div>'
        '</div></div>'
    )
    st.markdown(glass_card(risk_html, "pink"), unsafe_allow_html=True)

    # ── Report download ──────────────────────────────────────────
    if input_data and rai:
        st.markdown(neon_divider(), unsafe_allow_html=True)
        st.markdown(section_header("Download Report", "", "⬇"), unsafe_allow_html=True)
        try:
            rai_res = rai.explain_prediction(
                session_id=session_id, target="cognitive_load",
                data=input_data, top_n=9, sanitise=False,
            )
            top_feats = [
                {"feature": fc.feature, "label": fc.label,
                 "direction": fc.direction, "shap_value": fc.shap_value}
                for fc in rai_res.top_features
            ]
            rg = ReportGenerator(
                session_id=session_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                input_data=input_data,
                arousal_result={"predicted_class": a_class, "confidence_pct": a_conf.confidence_pct,
                                "tier": a_conf.tier, "class_probs": a_proba},
                cog_load_result={"predicted_class": c_class, "confidence_pct": c_conf.confidence_pct,
                                 "tier": c_conf.tier, "class_probs": c_proba},
                narrative=rai_res.narrative,
                top_features=top_feats,
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                st.download_button(
                    "⬇  Download PDF Report", data=rg.generate_pdf(),
                    file_name=f"cogniarousal_{session_id[:8]}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            with rc2:
                st.download_button(
                    "⬇  Download CSV Report", data=rg.generate_csv().encode("utf-8"),
                    file_name=f"cogniarousal_{session_id[:8]}.csv",
                    mime="text/csv", use_container_width=True,
                )
        except Exception:
            pass


def _compute_risk(arousal: str, cog_load: str) -> str:
    scores = {"Low": 0, "Medium": 1, "High": 2}
    total = scores.get(arousal, 0) + scores.get(cog_load, 0)
    return {0: "Low", 1: "Low", 2: "Moderate", 3: "High", 4: "Critical"}.get(total, "Moderate")
