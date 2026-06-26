"""
dashboard/pages/physio.py
Page 3 - Physiological Intelligence Center.
EDA, HRV, and Respiration signal monitoring and clinical patient trends tracking.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    C, glass_card, histogram_chart, kpi_card,
    neon_divider, scatter_chart, section_header, signal_line_chart,
)

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability",
]


def render(df: pd.DataFrame, session_state: dict, db_service=None, doctor_user=None) -> None:
    st.markdown(section_header("Physiological Intelligence Center", "Biosignal Monitoring & Analysis", "≋"), unsafe_allow_html=True)

    use_clinic_data = False
    fdf = pd.DataFrame()
    selected_id = None

    # Clinic Patient Monitoring selection
    if db_service and doctor_user:
        source_opt = st.selectbox(
            "Monitoring Data Source", 
            ["Clinic Patient Roster", "WESAD Research Dataset (Baseline Benchmarks)"], 
            key="physio_source_select"
        )
        
        if source_opt == "Clinic Patient Roster":
            use_clinic_data = True
            
            # Fetch hospital patients list
            patients = db_service.get_patients_by_hospital(doctor_user["hospital_id"])
            if not patients:
                st.warning("No patients registered under this clinic station. Please register a patient first in the Patient Directory.")
                return
            
            # Layout: Roster column on the left (list of patients), charts/signals on the right
            col_roster, col_chart_area = st.columns([1, 3])
            
            with col_roster:
                st.markdown('<div class="section-label" style="margin-bottom:8px">Clinic Patient Roster</div>', unsafe_allow_html=True)
                
                # Roster Search
                search_query = st.text_input("🔍 Search Roster", "", key="physio_roster_search")
                
                # Filter patients
                filtered_patients = patients
                if search_query:
                    q = search_query.lower().strip()
                    filtered_patients = [p for p in patients if q in p["name"].lower() or q in p["id"].lower()]
                
                if not filtered_patients:
                    st.info("No matching patients found.")
                    selected_id = None
                else:
                    # Track selected patient in session state
                    selected_id = session_state.get("selected_patient_id")
                    # If not in the filtered list or not set at all, pick the first one
                    valid_ids = [p["id"] for p in filtered_patients]
                    if not selected_id or selected_id not in valid_ids:
                        selected_id = filtered_patients[0]["id"]
                        session_state["selected_patient_id"] = selected_id
                    
                    for p in filtered_patients:
                        is_selected = (p["id"] == selected_id)
                        
                        # Render selection button
                        if st.button(f"👤 {p['name']} \n ({p['id']})", key=f"physio_roster_sel_{p['id']}", use_container_width=True, type="primary" if is_selected else "secondary"):
                            session_state["selected_patient_id"] = p["id"]
                            st.rerun()
                        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if not selected_id:
                return

            with col_chart_area:
                # Fetch patient profile details
                pat = db_service.get_patient(selected_id)
                if not pat:
                    st.error("Selected patient profile not found.")
                    return

                # Fetch assessments
                assessments = db_service.get_patient_assessments(selected_id)
                
                # Latest status and patient header card
                latest_status_html = ""
                if assessments:
                    latest = assessments[0]
                    arousal_color = C["pink"] if latest["arousal_pred"] == "High" else C["orange"] if latest["arousal_pred"] == "Medium" else C["green"]
                    load_color = C["pink"] if latest["load_pred"] == "High" else C["orange"] if latest["load_pred"] == "Medium" else C["green"]
                    
                    latest_status_html = f"""<div style="margin-top:8px; display:flex; gap:12px; font-family:var(--font-data); font-size:0.7rem">
<span style="color:var(--text-secondary)">Latest Assessment Status:</span>
<span style="color:{arousal_color};font-weight:bold">Arousal: {latest['arousal_pred']}</span>
<span style="color:var(--text-secondary)">|</span>
<span style="color:{load_color};font-weight:bold">Cognitive Load: {latest['load_pred']}</span>
<span style="color:var(--text-secondary)">|</span>
<span style="color:var(--text-secondary)">Logged: {latest['timestamp'][:19].replace('T', ' ')}</span>
</div>"""
                else:
                    latest_status_html = f"""<div style="margin-top:8px; font-family:var(--font-data); font-size:0.7rem; color:{C['orange']}">
⚠️ No assessments logged. Run an assessment under Cognitive Analysis.
</div>"""

                st.markdown(glass_card(f"""<div style="display:flex; justify-content:space-between; align-items:start;">
<div>
<span style="font-size:0.65rem;letter-spacing:0.2em;color:{C['cyan']};text-transform:uppercase;font-family:var(--font-data)">Active Patient Profile</span>
<h3 style="margin:2px 0 0 0;font-family:var(--font-display);color:{C['white']}">{pat['name']}</h3>
<div style="font-size:0.75rem;color:{C['secondary']};font-family:var(--font-data);margin-top:2px">
Patient ID: {pat['id']} &nbsp;|&nbsp; Age/Gender: {pat['age']} / {pat['gender']} &nbsp;|&nbsp; Occupation: {pat['occupation']}
</div>
{latest_status_html}
</div>
<div style="text-align:right">
<span style="font-size:0.65rem;letter-spacing:0.2em;color:{C['pink']};text-transform:uppercase;font-family:var(--font-data)">Clinical Care</span>
<div style="font-size:0.75rem;color:{C['white']};font-family:var(--font-data);font-weight:bold;margin-top:2px">
Dr. {pat['doctor_name']}
</div>
</div>
</div>""", "cyan"), unsafe_allow_html=True)

                if not assessments:
                    st.markdown(glass_card(f"""<div style="text-align:center;padding:40px 0">
<div style="font-size:2rem;margin-bottom:12px;color:{C['orange']}">⚠️</div>
<div style="font-family:var(--font-display);font-size:0.85rem;color:{C['secondary']};letter-spacing:0.15em">
NO PHYSIOLOGICAL DATA RECORDED
</div>
<div style="font-size:0.72rem;color:{C['secondary']};margin-top:8px;font-family:var(--font-data)">
Please run assessments under the Cognitive Analysis page for this patient to map bio-signal trends.
</div>
</div>""", "orange"), unsafe_allow_html=True)
                    return

                # Build DataFrame from patient assessment history (reverse for chronological order)
                assess_list = []
                for a in reversed(assessments):
                    assess_list.append({
                        "eda_mean":           a["eda_mean"],
                        "eda_std":            a["eda_std"],
                        "eda_peak_count":     a["eda_peak_count"],
                        "eda_peak_amplitude": a["eda_peak_amplitude"],
                        "heart_rate_bpm":     a["heart_rate_bpm"],
                        "rmssd":              a["rmssd"],
                        "sdnn":               a["sdnn"],
                        "resp_rate_bpm":      a["resp_rate_bpm"],
                        "resp_variability":   a["resp_variability"],
                        "label":              a["load_pred"],  # mapped category for graph legends
                        "arousal_pred":       a["arousal_pred"],
                        "load_pred":          a["load_pred"],
                        "timestamp":          a["timestamp"]
                    })
                fdf = pd.DataFrame(assess_list)
                render_physio_charts(fdf, use_clinic_data=True)

    if not use_clinic_data:
        # Fall back to WESAD Research Dataset
        if df.empty:
            st.error("No dataset loaded.")
            return

        # Subject selector
        subjects = sorted(df["subject_id"].unique()) if "subject_id" in df.columns else ["All"]
        col_sel, col_lbl = st.columns([2, 3])
        with col_sel:
            selected_subject = st.selectbox("Subject Benchmark", ["All"] + list(subjects), key="physio_subj")
        with col_lbl:
            lbl_html = (
                '<div style="padding-top:32px">'
                f'<span style="font-size:0.65rem;letter-spacing:0.2em;color:{C["secondary"]};'
                f'text-transform:uppercase;font-family:var(--font-data)">'
                'Showing window-level feature aggregates &nbsp;|&nbsp; 60s windows @ 700Hz WESAD chest sensor'
                '</span></div>'
            )
            st.markdown(lbl_html, unsafe_allow_html=True)

        fdf = df if selected_subject == "All" else df[df["subject_id"] == selected_subject]
        fdf = fdf.reset_index(drop=True)
        render_physio_charts(fdf, use_clinic_data=False)


def render_physio_charts(fdf: pd.DataFrame, use_clinic_data: bool) -> None:
    if use_clinic_data and "timestamp" in fdf.columns:
        x = pd.to_datetime(fdf["timestamp"])
    else:
        x = np.arange(len(fdf))

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── EDA Monitoring ───────────────────────────────────────────
    st.markdown(section_header("EDA Monitoring", "Electrodermal Activity Signal", "⬡"), unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown(kpi_card(f"{fdf['eda_mean'].mean():.3f}", "Mean EDA (z)", "⬡", C["cyan"]), unsafe_allow_html=True)
    with e2:
        st.markdown(kpi_card(f"{fdf['eda_std'].mean():.3f}", "EDA Std Dev (z)", "≈", C["purple"]), unsafe_allow_html=True)
    with e3:
        st.markdown(kpi_card(f"{fdf['eda_peak_count'].mean():.1f}", "Avg Peak Count", "▲", C["green"]), unsafe_allow_html=True)
    with e4:
        st.markdown(kpi_card(f"{fdf['eda_peak_amplitude'].mean():.3f}", "Avg Peak Amp (z)", "↑", C["pink"]), unsafe_allow_html=True)

    ea1, ea2 = st.columns(2)
    with ea1:
        st.plotly_chart(
            signal_line_chart(x, fdf["eda_mean"].values, "EDA Mean (z-score)", C["cyan"], "z"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with ea2:
        st.plotly_chart(
            signal_line_chart(x, fdf["eda_std"].values, "EDA Variability (z-score)", C["purple"], "z"),
            use_container_width=True, config={"displayModeBar": False},
        )

    ep1, ep2 = st.columns(2)
    with ep1:
        st.plotly_chart(
            signal_line_chart(x, fdf["eda_peak_count"].values, "EDA Peak Count", C["green"], "count"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with ep2:
        st.plotly_chart(
            signal_line_chart(x, fdf["eda_peak_amplitude"].values, "EDA Peak Amplitude (z-score)", C["pink"], "z"),
            use_container_width=True, config={"displayModeBar": False},
        )

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── HRV Monitoring ───────────────────────────────────────────
    st.markdown(section_header("HRV Monitoring", "Heart Rate Variability from ECG", "♡"), unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(kpi_card(f"{fdf['heart_rate_bpm'].mean():.3f}", "Mean HR (z)", "♡", C["pink"]), unsafe_allow_html=True)
    with h2:
        st.markdown(kpi_card(f"{fdf['rmssd'].mean():.4f}", "Mean RMSSD (z)", "≈", C["cyan"]), unsafe_allow_html=True)
    with h3:
        st.markdown(kpi_card(f"{fdf['sdnn'].mean():.4f}", "Mean SDNN (z)", "≈", C["purple"]), unsafe_allow_html=True)

    hv1, hv2, hv3 = st.columns(3)
    with hv1:
        st.plotly_chart(
            signal_line_chart(x, fdf["heart_rate_bpm"].values, "Heart Rate (z-score)", C["pink"], "z"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with hv2:
        st.plotly_chart(
            signal_line_chart(x, fdf["rmssd"].values, "RMSSD (z-score)", C["cyan"], "z"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with hv3:
        st.plotly_chart(
            signal_line_chart(x, fdf["sdnn"].values, "SDNN (z-score)", C["purple"], "z"),
            use_container_width=True, config={"displayModeBar": False},
        )

    # HR vs RMSSD scatter coloured by label
    st.plotly_chart(
        scatter_chart(fdf, "heart_rate_bpm", "rmssd", "label", "Heart Rate vs RMSSD by Label Category"),
        use_container_width=True, config={"displayModeBar": False},
    )

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Respiration Monitoring ───────────────────────────────────
    st.markdown(section_header("Respiration Monitoring", "Respiratory Rate & Variability", "≈"), unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(kpi_card(f"{fdf['resp_rate_bpm'].mean():.3f}", "Mean Resp Rate (z)", "≈", C["green"]), unsafe_allow_html=True)
    with r2:
        st.markdown(kpi_card(f"{fdf['resp_variability'].mean():.4f}", "Mean Resp Variability (z)", "≈", C["orange"]), unsafe_allow_html=True)

    rv1, rv2 = st.columns(2)
    with rv1:
        st.plotly_chart(
            signal_line_chart(x, fdf["resp_rate_bpm"].values, "Respiration Rate (z-score)", C["green"], "z"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with rv2:
        st.plotly_chart(
            signal_line_chart(x, fdf["resp_variability"].values, "Respiration Variability (z-score)", C["orange"], "z"),
            use_container_width=True, config={"displayModeBar": False},
        )

    # EDA vs Resp scatter
    st.plotly_chart(
        scatter_chart(fdf, "eda_mean", "resp_rate_bpm", "label", "EDA Mean vs Respiration Rate by Label Category"),
        use_container_width=True, config={"displayModeBar": False},
    )

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Distribution Histograms ──────────────────────────────────
    st.markdown(section_header("Feature Distributions", "Across All Windows / Session Logs", "▦"), unsafe_allow_html=True)
    hist_colors = [C["cyan"], C["purple"], C["pink"], C["green"],
                   C["orange"], C["pink"], C["cyan"], C["green"], C["purple"]]
    cols = st.columns(3)
    for i, col in enumerate(FEATURE_COLS):
        with cols[i % 3]:
            st.plotly_chart(
                histogram_chart(fdf[col], col, hist_colors[i]),
                use_container_width=True, config={"displayModeBar": False},
            )
