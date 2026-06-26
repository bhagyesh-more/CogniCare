"""
dashboard/pages/patients.py
Patient Management and Profile dashboards.
Provides UI to list, search, and register patient profiles, and view detailed
summaries of a selected patient's history, risk indices, and Plotly trends.
"""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from dashboard.components import (
    C, glass_card, kpi_card, prediction_card,
    shap_feature_row, narrative_box, neon_divider,
    section_header, status_badge
)
from src.report_generator import ReportGenerator

# ── Feature metadata map to print human-readable feature labels ───────
FEATURE_LBLS = {
    "eda_mean":           "EDA Mean",
    "eda_std":            "EDA Std Dev",
    "eda_peak_count":     "EDA Peak Count",
    "eda_peak_amplitude": "EDA Peak Amplitude",
    "heart_rate_bpm":     "Heart Rate",
    "rmssd":              "RMSSD",
    "sdnn":               "SDNN",
    "resp_rate_bpm":      "Respiration Rate",
    "resp_variability":   "Resp. Variability",
}


def render(db_service, doctor_user, session_state) -> None:
    # Handle patient selection reset
    if "selected_patient_id" not in session_state:
        session_state["selected_patient_id"] = None

    selected_pat_id = session_state["selected_patient_id"]

    if selected_pat_id:
        render_patient_profile(db_service, doctor_user, selected_pat_id, session_state)
    else:
        render_patient_explorer(db_service, doctor_user, session_state)


# ── 1. Patient Explorer / Directory Screen ────────────────────────────
def render_patient_explorer(db_service, doctor_user, session_state) -> None:
    st.markdown(section_header("Patient Directory", "Search and Manage Hospital Patient Profiles", "👥"), unsafe_allow_html=True)

    tab_explore, tab_register = st.tabs(["  Patient Registry  ", "  Register New Patient  "])

    # Directory View
    with tab_explore:
        search_q = st.text_input("🔍 Search patient by Name, ID, or Doctor", placeholder="Type to search...", key="pat_search").strip()
        
        if search_q:
            patients = db_service.search_patients(doctor_user["hospital_id"], search_q)
        else:
            patients = db_service.get_patients_by_hospital(doctor_user["hospital_id"])

        if not patients:
            st.markdown(glass_card(f"""
                <div style="text-align:center;padding:40px 0">
                    <div style="font-size:2rem;margin-bottom:12px">👤</div>
                    <div style="font-family:var(--font-display);font-size:0.85rem;color:{C['secondary']};letter-spacing:0.15em">
                        NO PATIENTS REGISTERED
                    </div>
                    <div style="font-size:0.72rem;color:{C['secondary']};margin-top:8px;font-family:var(--font-data)">
                        Register new patient profiles in the registry tab to begin tracking cognitive load.
                    </div>
                </div>
            """, "purple"), unsafe_allow_html=True)
        else:
            # Render list of patients
            st.markdown(f"""
            <div style="font-size:0.65rem;color:{C['secondary']};letter-spacing:0.15em;text-transform:uppercase;font-family:var(--font-data);margin-bottom:10px">
                Monitored Patients: <span style="color:{C['cyan']}">{len(patients)}</span>
            </div>
            """, unsafe_allow_html=True)

            for pat in patients:
                # Get count of assessments
                with db_service.get_connection() as conn:
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM assessments WHERE patient_id = ?", (pat["id"],))
                    a_count = c.fetchone()[0]

                col_p1, col_p2, col_p3 = st.columns([7, 3, 2])
                with col_p1:
                    pat_info = (
                        f'<div style="font-family:var(--font-display);font-size:0.9rem;color:{C["cyan"]};margin-bottom:4px">'
                        f'{pat["name"]} &nbsp;'
                        f'<span style="font-family:var(--font-data);font-size:0.62rem;color:{C["secondary"]};'
                        f'background:{C["border"]};padding:2px 8px;border-radius:4px">{pat["id"]}</span></div>'
                        f'<div style="font-size:0.72rem;color:{C["secondary"]};font-family:var(--font-data)">'
                        f'Age: <b>{pat["age"]}</b> &nbsp;|&nbsp; Gender: <b>{pat["gender"]}</b> &nbsp;|&nbsp; '
                        f'Occupation: <b>{pat["occupation"]}</b> ({pat["industry"]})</div>'
                    )
                    st.markdown(pat_info, unsafe_allow_html=True)
                with col_p2:
                    doc_info = (
                        f'<div style="font-size:0.7rem;color:{C["secondary"]};font-family:var(--font-data);line-height:1.6">'
                        f'Clinician: <b style="color:{C["white"]}">{pat["doctor_name"]}</b><br>'
                        f'Assessments: <b style="color:{C["purple"]}">{a_count}</b></div>'
                    )
                    st.markdown(doc_info, unsafe_allow_html=True)
                with col_p3:
                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                    if st.button("👁️ View Profile", key=f"sel_pat_{pat['id']}", use_container_width=True):
                        session_state["selected_patient_id"] = pat["id"]
                        st.rerun()
                
                st.markdown(f'<hr style="border:none;border-top:1px solid {C["border"]};margin:10px 0">', unsafe_allow_html=True)

    # Register Form
    with tab_register:
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <div class="section-label">Patient Registry Form</div>
            <div style="font-size:0.75rem;color:{C['secondary']};font-family:var(--font-data)">
                Register a new patient profile. All assessments, session logs, and predictions are attached to this record.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("register_patient_form"):
            # Personal
            st.markdown(f'<div class="section-label" style="color:{C["cyan"]};margin-bottom:8px">1. Personal Information</div>', unsafe_allow_html=True)
            p_c1, p_c2, p_c3 = st.columns(3)
            with p_c1:
                p_name = st.text_input("Patient Full Name", placeholder="Arthur Pendelton")
            with p_c2:
                p_age = st.number_input("Age", min_value=1, max_value=120, value=35)
            with p_c3:
                p_gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"])
            
            p_c4, p_c5 = st.columns(2)
            with p_c4:
                p_dob = st.text_input("Date of Birth", placeholder="YYYY-MM-DD")
            with p_c5:
                p_contact = st.text_input("Contact Number", placeholder="+1-555-0100")
            
            p_addr = st.text_area("Home Address", placeholder="Street name, City, Zip", height=60)
            
            # Professional
            st.markdown(f'<div class="section-label" style="color:{C["purple"]};margin:12px 0 8px">2. Professional Context</div>', unsafe_allow_html=True)
            pr_c1, pr_c2 = st.columns(2)
            with pr_c1:
                p_occ = st.text_input("Occupation", placeholder="Software Engineer")
                p_sched = st.selectbox("Work Schedule", ["Regular Day Shifts", "Fixed Night Shifts", "Rotating Shifts (24/7)", "Irregular / Freelance"])
            with pr_c2:
                p_ind = st.text_input("Industry Sector", placeholder="Information Technology")
                p_edu = st.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "Doctorate"])
            
            # Clinical
            st.markdown(f'<div class="section-label" style="color:{C["pink"]};margin:12px 0 8px">3. Clinical Intake Notes</div>', unsafe_allow_html=True)
            p_notes = st.text_area("Clinical Notes & Objectives", placeholder="Brief explanation of cognitive fatigue monitoring goals...", height=80)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            submit_pat = st.form_submit_button("⚡ Register Patient Profile", type="primary", use_container_width=True)

            if submit_pat:
                if not p_name or not p_dob:
                    st.error("Patient Name and Date of Birth are required.")
                else:
                    pat_id = db_service.create_patient(
                        doctor_id=doctor_user["id"],
                        hospital_id=doctor_user["hospital_id"],
                        name=p_name,
                        age=int(p_age),
                        gender=p_gender,
                        dob=p_dob,
                        contact=p_contact,
                        address=p_addr,
                        occupation=p_occ,
                        industry=p_ind,
                        schedule=p_sched,
                        education=p_edu,
                        notes=p_notes
                    )
                    session_state["selected_patient_id"] = pat_id
                    session_state["just_registered_patient_id"] = pat_id
                    st.rerun()


# ── 2. Patient Profile Dashboard Screen ───────────────────────────────
def render_patient_profile(db_service, doctor_user, patient_id, session_state) -> None:
    pat = db_service.get_patient(patient_id)
    if not pat:
        st.error("Patient profile not found.")
        session_state["selected_patient_id"] = None
        st.rerun()

    # Success feedback if just registered
    if session_state.get("just_registered_patient_id") == patient_id:
        st.success(f"🎉 Patient profile for {pat['name']} ({pat['id']}) created successfully!")
        session_state["just_registered_patient_id"] = None

    # Back button
    if st.button("⬅ Back to Patient Directory", key="profile_back_btn"):
        session_state["selected_patient_id"] = None
        st.rerun()

    # Section Header
    st.markdown(section_header(f"{pat['name']}", f"Patient Registry Profile Dashboard", "👤"), unsafe_allow_html=True)

    # ── Patient Summary Block ───────────────────────────────────────────
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        p_info = f"""
        <div style="line-height:2.0">
            <div class="section-label" style="color:var(--neon-cyan)">Personal Info</div>
            <div style="font-size:0.75rem;font-family:var(--font-data)">
                Patient ID: <b style="color:var(--text-white)">{pat['id']}</b><br>
                Age / Gender: <b>{pat['age']} / {pat['gender']}</b><br>
                Date of Birth: <b>{pat['dob']}</b><br>
                Contact: <b>{pat['contact_number']}</b><br>
                Address: <span style="color:var(--text-secondary)">{pat['address']}</span>
            </div>
        </div>
        """
        st.markdown(glass_card(p_info, "cyan"), unsafe_allow_html=True)

    with col_s2:
        pr_info = f"""
        <div style="line-height:2.0">
            <div class="section-label" style="color:var(--neon-purple)">Professional Context</div>
            <div style="font-size:0.75rem;font-family:var(--font-data)">
                Occupation: <b>{pat['occupation']}</b><br>
                Sector / Industry: <b>{pat['industry']}</b><br>
                Work Schedule: <b>{pat['work_schedule']}</b><br>
                Education: <b>{pat['education_level']}</b>
            </div>
        </div>
        """
        st.markdown(glass_card(pr_info, "purple"), unsafe_allow_html=True)

    with col_s3:
        cl_info = f"""
        <div style="line-height:2.0">
            <div class="section-label" style="color:var(--neon-pink)">Clinical Context</div>
            <div style="font-size:0.75rem;font-family:var(--font-data)">
                Assigned Doctor: <b style="color:var(--text-white)">{pat['doctor_name']}</b><br>
                Registered Date: <b>{pat['registration_date'][:10]}</b><br>
                Intake Objectives:<br>
                <span style="color:var(--text-secondary);font-size:0.7rem;font-style:italic">
                    "{pat['clinical_notes'] or 'No intakes notes'}"
                </span>
            </div>
        </div>
        """
        st.markdown(glass_card(cl_info, "pink"), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Fetch patient assessments ─────────────────────────────────────
    assessments = db_service.get_patient_assessments(patient_id)
    
    if not assessments:
        # Ask doctor to run assessment
        st.markdown(glass_card(f"""
            <div style="text-align:center;padding:40px 0">
                <div style="font-size:2rem;margin-bottom:12px">⚡</div>
                <div style="font-family:var(--font-display);font-size:0.85rem;color:{C['secondary']};letter-spacing:0.15em">
                    NO ASSESSMENT DATA AVAILABLE
                </div>
                <div style="font-size:0.72rem;color:{C['secondary']};margin-top:8px;font-family:var(--font-data);margin-bottom:20px">
                    Assessments are required to map trends and evaluate cognitive load configurations.
                </div>
            </div>
        """, "orange"), unsafe_allow_html=True)
        
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("⚡ Run Intake Assessment Now", type="primary", use_container_width=True):
            session_state["nav_page"] = "Cognitive Analysis"
            st.rerun()
        return

    # Latest Assessment Details
    latest = assessments[0]

    # Calculate Risk Score Indicator
    # Formed by prediction tags + confidence level
    arousal_class = latest["arousal_pred"]
    load_class = latest["load_pred"]
    risk_label = "NOMINAL / STABLE"
    risk_color = C["green"]
    risk_desc = "Patient exhibits stable physiological response thresholds."

    if arousal_class == "High" and load_class == "High":
        risk_label = "CRITICAL COGNITIVE SPUR"
        risk_color = C["pink"]
        risk_desc = "Severe stress indicators flagged in combination with high cognitive load. Intervention recommended."
    elif arousal_class == "High" or load_class == "High":
        risk_label = "ELEVATED FATIGUE WORKLOAD"
        risk_color = C["orange"]
        risk_desc = "Elevated indicators detected. Monitor patient schedule limits."
    elif arousal_class == "Medium" and load_class == "Medium":
        risk_label = "MODERATE RESPONSE ENGAGED"
        risk_color = C["cyan"]
        risk_desc = "Patient is engaged in focused mental workload."

    # ── Latest Assessment Section ─────────────────────────────────────
    st.markdown(section_header("Latest Assessment Insights", f"Assessment date: {latest['timestamp'][:19].replace('T', ' ')}", "⚡"), unsafe_allow_html=True)
    
    col_lat1, col_lat2 = st.columns([5, 7])
    
    with col_lat1:
        # Prediction Output Cards
        st.markdown(prediction_card(
            predicted_class=latest["load_pred"],
            target_label="COGNITIVE LOAD",
            confidence_pct=latest["load_conf"] * 100,
            tier=latest["load_tier"],
            flag_review=latest["flag_review"]
        ), unsafe_allow_html=True)

        st.markdown(prediction_card(
            predicted_class=latest["arousal_pred"],
            target_label="EMOTIONAL AROUSAL",
            confidence_pct=latest["arousal_conf"] * 100,
            tier=latest["arousal_tier"],
            flag_review=latest["flag_review"]
        ), unsafe_allow_html=True)
        
        # Risk Box
        risk_html = f"""
        <div class="glass-card" style="border-color:{risk_color}55;box-shadow:0 0 20px {risk_color}10">
            <div class="section-label" style="color:var(--text-secondary)">Clinical Risk Indicator</div>
            <div style="font-family:var(--font-display);font-size:1.1rem;font-weight:700;color:{risk_color};margin:6px 0">{risk_label}</div>
            <div style="font-size:0.72rem;font-family:var(--font-data);color:var(--text-secondary)">{risk_desc}</div>
        </div>
        """
        st.markdown(risk_html, unsafe_allow_html=True)

    with col_lat2:
        # SHAP Explanations
        st.markdown(f'<div class="section-label" style="margin-bottom:8px">SHAP Attributions</div>', unsafe_allow_html=True)
        try:
            shap_list = json.loads(latest["shap_values"])
            max_abs = max((abs(item["shap_value"]) for item in shap_list), default=1.0)
            
            shap_html = ""
            for item in shap_list:
                # Convert raw key label to nice label
                nice_lbl = FEATURE_LBLS.get(item["feature"], item["feature"])
                shap_html += shap_feature_row(
                    nice_lbl, item["shap_value"], max_abs, item["direction"]
                )
            st.markdown(glass_card(shap_html, "purple"), unsafe_allow_html=True)
        except Exception:
            st.warning("SHAP attribution arrays could not be deserialized.")

        # Transparency Narrative
        st.markdown(f'<div class="section-label" style="margin:12px 0 8px">AI Transparency Narrative</div>', unsafe_allow_html=True)
        st.markdown(narrative_box(latest["load_narrative"]), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Historical Trends (Plotly Charts) ──────────────────────────────
    st.markdown(section_header("Historical Analysis & Trends", "Chronological Load and Arousal Levels", "📈"), unsafe_allow_html=True)
    
    # Prep history df
    h_df = pd.DataFrame(assessments)
    h_df["date_parsed"] = pd.to_datetime(h_df["timestamp"])
    h_df = h_df.sort_values("date_parsed")

    # Map labels to numeric positions (Low=1, Medium=2, High=3)
    val_map = {"Low": 1, "Medium": 2, "High": 3}
    h_df["load_num"] = h_df["load_pred"].map(val_map)
    h_df["arousal_num"] = h_df["arousal_pred"].map(val_map)
    
    # 2 Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Trend chart for cognitive load and arousal levels
        fig_levels = go.Figure()
        fig_levels.add_trace(go.Scatter(
            x=h_df["date_parsed"], y=h_df["load_num"],
            mode="lines+markers", name="Cognitive Load",
            line=dict(color=C["cyan"], width=2.5),
            marker=dict(size=7, color=C["cyan"])
        ))
        fig_levels.add_trace(go.Scatter(
            x=h_df["date_parsed"], y=h_df["arousal_num"],
            mode="lines+markers", name="Arousal State",
            line=dict(color=C["pink"], width=2.5),
            marker=dict(size=7, color=C["pink"])
        ))
        fig_levels.update_layout(
            title=dict(text="Cognitive Load vs. Emotional Arousal Trend", font=dict(family="Space Grotesk", size=13, color=C["secondary"])),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk", color=C["secondary"]),
            margin=dict(l=10, r=10, t=40, b=10),
            height=250,
            xaxis=dict(gridcolor=C["grid"], tickfont=dict(size=9)),
            yaxis=dict(
                gridcolor=C["grid"], tickvals=[1, 2, 3],
                ticktext=["Low", "Medium", "High"], tickfont=dict(size=9), range=[0.5, 3.5]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_levels, use_container_width=True, config={"displayModeBar": False})

    with col_chart2:
        # Confidence Trend
        fig_conf = go.Figure()
        # Average the two confidences
        h_df["avg_conf"] = (h_df["arousal_conf"] + h_df["load_conf"]) / 2 * 100
        fig_conf.add_trace(go.Scatter(
            x=h_df["date_parsed"], y=h_df["avg_conf"],
            mode="lines+markers", fill="tozeroy",
            fillcolor="rgba(0,255,178,0.04)",
            line=dict(color=C["green"], width=2.5),
            marker=dict(size=7, color=C["green"])
        ))
        fig_conf.update_layout(
            title=dict(text="Average Model Prediction Confidence (%)", font=dict(family="Space Grotesk", size=13, color=C["secondary"])),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk", color=C["secondary"]),
            margin=dict(l=10, r=10, t=40, b=10),
            height=250,
            xaxis=dict(gridcolor=C["grid"], tickfont=dict(size=9)),
            yaxis=dict(gridcolor=C["grid"], range=[0, 110], tickfont=dict(size=9), ticksuffix="%")
        )
        st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})

    # Link button to detailed biosignal tracking
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("≋ View Detailed Physiological Signal Tracking", use_container_width=True, key="profile_to_physio_btn"):
        session_state["nav_page"] = "Physiological Intel"
        st.rerun()

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── Chronological Assessment Log ────────────────────────────────────
    st.markdown(section_header("Assessment History Log", "Exportable Clinical Logs", "◷"), unsafe_allow_html=True)
    
    for entry in assessments:
        c_time = entry["timestamp"][:19].replace("T", " ")
        c_id = entry["id"]
        
        col_ent1, col_ent2, col_ent3 = st.columns([7, 3, 2])
        
        with col_ent1:
            entry_html = (
                f'<div style="font-family:var(--font-data);font-size:0.75rem;line-height:1.8">'
                f'<span style="color:var(--text-secondary)">ID:</span> <b>ASS-{c_id}</b> &nbsp;|&nbsp; '
                f'<span style="color:var(--text-secondary)">Date:</span> <b>{c_time}</b><br>'
                f'<span style="color:{C["cyan"]}">Cognitive Load:</span> <b>{entry["load_pred"]}</b> &nbsp;|&nbsp; '
                f'<span style="color:{C["pink"]}">Emotional Arousal:</span> <b>{entry["arousal_pred"]}</b></div>'
            )
            st.markdown(entry_html, unsafe_allow_html=True)
            
        with col_ent2:
            avg_c = (entry["arousal_conf"] + entry["load_conf"]) / 2 * 100
            tier = entry["load_tier"]
            tier_color = C["green"] if tier == "High" else C["orange"] if tier == "Medium" else C["cyan"]
            badge = (
                f'<div style="font-size:0.7rem;color:var(--text-secondary);font-family:var(--font-data);line-height:1.6">'
                f'Conf: <b style="color:{C["white"]}">{avg_c:.1f}%</b><br>'
                f'<span class="status-badge" style="background:{tier_color}12;border:1px solid {tier_color}44;'
                f'color:{tier_color};font-size:0.55rem">{tier} Tier</span></div>'
            )
            st.markdown(badge, unsafe_allow_html=True)
            
        with col_ent3:
            # Generate Report PDF Trigger
            # We bundle parameters to ReportGenerator and render to bytes
            input_dict = {
                "eda_mean":           entry["eda_mean"],
                "eda_std":            entry["eda_std"],
                "eda_peak_count":     entry["eda_peak_count"],
                "eda_peak_amplitude": entry["eda_peak_amplitude"],
                "heart_rate_bpm":     entry["heart_rate_bpm"],
                "rmssd":              entry["rmssd"],
                "sdnn":               entry["sdnn"],
                "resp_rate_bpm":      entry["resp_rate_bpm"],
                "resp_variability":   entry["resp_variability"],
            }
            arousal_result = {
                "predicted_class": entry["arousal_pred"],
                "confidence_pct":  entry["arousal_conf"] * 100,
                "tier":            entry["arousal_tier"],
                "class_probs":     {} # dummy class probabilities for report
            }
            cog_result = {
                "predicted_class": entry["load_pred"],
                "confidence_pct":  entry["load_conf"] * 100,
                "tier":            entry["load_tier"],
                "class_probs":     {}
            }
            
            try:
                shap_list = json.loads(entry["shap_values"])
            except Exception:
                shap_list = []
                
            rep = ReportGenerator(
                session_id=entry["session_id"],
                timestamp=c_time,
                input_data=input_dict,
                arousal_result=arousal_result,
                cog_load_result=cog_result,
                narrative=entry["load_narrative"],
                top_features=shap_list
            )
            
            try:
                pdf_bytes = rep.generate_pdf()
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_bytes,
                    file_name=f"CogniCare_Report_ASS-{c_id}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{c_id}"
                )
            except Exception as e:
                st.error("Report PDF failed to compile.")

        st.markdown(f'<hr style="border:none;border-top:1px solid {C["border"]};margin:8px 0">', unsafe_allow_html=True)
