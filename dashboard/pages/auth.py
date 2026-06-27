"""
dashboard/pages/auth.py
Authentication forms for clinic registration and doctor login.
Manages database interaction for clinic setup and clinician session state updates.
"""

import streamlit as st
from dashboard.components import C, glass_card, section_header, neon_divider


def render(db_service, on_auth_success, on_cancel) -> None:
    st.markdown(section_header("Secure Access Center", "Hospital Portal & Clinician Authentication", "🔐"), unsafe_allow_html=True)

    tab_login, tab_register_doc, tab_register_hospital = st.tabs([
        "  Clinician Login  ", 
        "  Doctor Registration  ", 
        "  Register Hospital/Clinic  "
    ])

    # ── Tab 1: Doctor Login ───────────────────────────────────────────
    with tab_login:
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <div class="section-label">Clinician Sign-in</div>
            <div style="font-size:0.75rem;color:{C['secondary']};font-family:var(--font-data)">
                Enter your doctor credentials to access patient dashboards and analytical engines.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("doctor_login_form"):
            login_email = st.text_input("Clinician Email", placeholder="doctor@clinic.com", key="login_em").strip()
            login_pwd = st.text_input("Password", type="password", placeholder="••••••••", key="login_pw")
            
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            submit_login = st.form_submit_button("⚡ Secure Sign In", type="primary", use_container_width=True)

            if submit_login:
                if not login_email or not login_pwd:
                    st.error("Please fill in both Email and Password fields.")
                else:
                    doc = db_service.authenticate_doctor(login_email, login_pwd)
                    if doc:
                        on_auth_success(doc)
                        st.success(f"Welcome back, {doc['name']}! Loading portal...")
                        st.rerun()
                    else:
                        st.error("Invalid clinician credentials. Please check email/password or register your account.")



    # ── Tab 2: Doctor Registration ────────────────────────────────────
    with tab_register_doc:
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <div class="section-label">Clinician Registry</div>
            <div style="font-size:0.75rem;color:{C['secondary']};font-family:var(--font-data)">
                Create your clinician profile linked to an existing clinic registration code.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("doctor_register_form"):
            reg_clinic_code = st.text_input("Clinic Registration Code", placeholder="e.g. YOUR-CODE", key="reg_cc").strip().upper()
            reg_name = st.text_input("Full Name (with credentials)", placeholder="Dr. Evelyn Vance, MD", key="reg_nm")
            reg_email = st.text_input("Email Address", placeholder="doctor@clinic.com", key="reg_em").strip()
            reg_pwd = st.text_input("Password", type="password", placeholder="Minimum 6 characters", key="reg_pw")
            
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            submit_reg_doc = st.form_submit_button("⚡ Register Doctor Account", type="primary", use_container_width=True)

            if submit_reg_doc:
                if not reg_clinic_code or not reg_name or not reg_email or not reg_pwd:
                    st.error("All registration fields are required.")
                elif len(reg_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    # Verify hospital code exists
                    hospital = db_service.get_hospital_by_code(reg_clinic_code)
                    if not hospital:
                        st.error(f"Clinic registration code '{reg_clinic_code}' not found. Please register the clinic first.")
                    else:
                        success = db_service.register_doctor(
                            hospital_id=hospital["id"],
                            name=reg_name,
                            email=reg_email,
                            password=reg_pwd
                        )
                        if success:
                            st.success("Registration successful! You can now log in using the 'Clinician Login' tab.")
                        else:
                            st.error(f"Email '{reg_email}' is already registered on this platform.")

    # ── Tab 3: Hospital Registration ──────────────────────────────────
    with tab_register_hospital:
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <div class="section-label">Clinic Registration</div>
            <div style="font-size:0.75rem;color:{C['secondary']};font-family:var(--font-data)">
                Register your hospital, healthcare center, or clinic to create an isolated organization environment.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("hospital_register_form"):
            hosp_name = st.text_input("Hospital / Clinic Name", placeholder="Neo-General Hospital", key="hosp_nm")
            hosp_code = st.text_input("Custom Clinic Identifier Code", placeholder="e.g. NEO-GEN", key="hosp_cd").strip().upper()
            
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            submit_hosp = st.form_submit_button("⚡ Register Clinic", type="primary", use_container_width=True)

            if submit_hosp:
                if not hosp_name or not hosp_code:
                    st.error("Please fill in both Hospital Name and Identifier Code.")
                elif len(hosp_code) < 3:
                    st.error("Clinic code must be at least 3 characters.")
                else:
                    # Check if clinic code exists
                    exists = db_service.get_hospital_by_code(hosp_code)
                    if exists:
                        st.error(f"Clinic code '{hosp_code}' is already in use. Please select a unique identifier.")
                    else:
                        success = db_service.register_hospital(hosp_name, hosp_code)
                        if success:
                            st.success(f"Clinic '{hosp_name}' registered successfully with code '{hosp_code}'! "
                                       f"Doctors can now register using this code.")
                        else:
                            st.error("An error occurred during database insert.")

    # ── Cancel / Return Button ────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("⬅ Return to Landing Page", use_container_width=True):
        on_cancel()
