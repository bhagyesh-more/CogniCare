"""
app.py
CogniCare - Streamlit Clinical Cognitive Monitoring Dashboard
Entry point for clinicians and medical administrators.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="COGNICARE",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.styles import inject
from dashboard.components import C, gauge_chart, glass_card, kpi_card, neon_divider, section_header
import importlib
from dashboard.pages import (
    mission_control, analysis, physio,
    dataset, demo, landing, auth, patients
)

# Force reload of ML-dependent pages to prevent Streamlit from using stale cached signatures
importlib.reload(mission_control)
importlib.reload(analysis)
importlib.reload(physio)
importlib.reload(dataset)
importlib.reload(demo)
from src.monitoring import Monitor
from src.validator import Validator
from src.database import DatabaseService

logging.basicConfig(level=logging.WARNING)


def _load_secrets() -> None:
    """Bridge st.secrets → environment variables so non-Streamlit modules
    (e.g. src/database.py) can read credentials via os.environ.get().
    Safe to call multiple times; only sets vars that are not already present.
    Falls back silently when running locally without a secrets.toml.
    """
    MAPPING = {
        "secret_salt":          "COGNICARE_SECRET_SALT",
        "demo_hospital_name":   "DEMO_HOSPITAL_NAME",
        "demo_clinic_code":     "DEMO_CLINIC_CODE",
        "demo_doctor_name":     "DEMO_DOCTOR_NAME",
        "demo_doctor_email":    "DEMO_DOCTOR_EMAIL",
        "demo_doctor_password": "DEMO_DOCTOR_PASSWORD",
    }
    try:
        section = st.secrets.get("cognicare", {})
        for secret_key, env_key in MAPPING.items():
            if secret_key in section and env_key not in os.environ:
                os.environ[env_key] = str(section[secret_key])
    except Exception:
        pass  # No secrets file locally — database.py will use safe defaults


# Load secrets as early as possible (before DatabaseService is created)
_load_secrets()

FEATURE_CSV = Path("output/feature_dataset.csv")
MODELS_DIR  = Path("models")

# Sidebar clinical menu items
CLINICAL_NAV = [
    ("⬡", "Hospital Dashboard"),
    ("👥", "Patient Directory"),
    ("◎", "Cognitive Analysis"),
    ("≋", "Physiological Intel"),
    ("▦", "Dataset Intelligence"),
    ("◉", "Demo Mode"),
]


# ── Cached database and ML loaders ────────────────────────────────

@st.cache_resource(show_spinner="Loading feature dataset...")
def load_dataset() -> pd.DataFrame:
    if not FEATURE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_CSV)


@st.cache_resource(show_spinner="Initialising AI engines...")
def load_rai(dataset_hash: int):
    df = load_dataset()
    from src.responsible_ai import ResponsibleAI
    background = df if not df.empty else None
    return ResponsibleAI(models_dir=MODELS_DIR).load(background_df=background)


# ── Session state ─────────────────────────────────────────────────

def init_session_state() -> None:
    defaults = {
        "auth_state":         "landing",   # landing, login, logged_in
        "doctor_user":        None,        # dict holding active clinician
        "total_predictions":  0,
        "active_sessions":    0,
        "avg_confidence":     0.0,
        "last_analysis":      "-",
        "prediction_history": [],
        "rai_loaded":         False,
        "demo_result":        None,
        "monitor":            Monitor(),
        "app_start_time":     time.time(),
        "nav_page":           "Hospital Dashboard",
        "selected_patient_id":None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Hospital / Clinic Dashboard Page ───────────────────────────────

def render_hospital_dashboard(df: pd.DataFrame, db: DatabaseService) -> None:
    doc = st.session_state["doctor_user"]
    st.markdown(section_header(f"{doc['hospital_name']} Portal", "Hospital Cognitive Intelligence & Live Diagnostics", "🏥"), unsafe_allow_html=True)

    # Clinic summary stats
    stats = db.get_hospital_stats(doc["hospital_id"])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(doc["hospital_code"], "Clinic Station Code", "🏥", C["cyan"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(str(stats["doctors"]), "Active Doctors", "🥼", C["purple"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(str(stats["patients"]), "Registered Patients", "👤", C["orange"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(str(stats["assessments"]), "Clinical Assessments", "⚡", C["green"]), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # Show model accuracy gauges & datasets stats inside portal
    st.markdown(section_header("Model Stratification Indexes", " Stratified 5-Fold Cross-Validation Metrics", "◈"), unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(gauge_chart(83.7, "Arousal Accuracy", C["cyan"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(gauge_chart(79.0, "Arousal F1 Macro", C["purple"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(gauge_chart(83.7, "Cognitive Load Accuracy", C["green"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g4:
        st.plotly_chart(gauge_chart(79.0, "Cognitive Load F1 Macro", C["pink"]),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # Active Doctors List
    st.markdown(section_header("Registered Clinicians", "Medical Staff Account Registry", "🥼"), unsafe_allow_html=True)
    docs = db.get_doctors_in_hospital(doc["hospital_id"])
    doc_rows = ""
    for d in docs:
        is_self = " (Active Session)" if d["id"] == doc["id"] else ""
        doc_rows += (
            f'<div style="display:flex;justify-content:space-between;padding:6px 12px;margin-bottom:8px;'
            f'background:rgba(255,255,255,0.02);border:1px solid {C["border"]};border-radius:6px">'
            f'<span style="font-family:var(--font-body);font-size:0.78rem">'
            f'{d["name"]}<span style="color:{C["green"]}">{is_self}</span></span>'
            f'<span style="font-family:var(--font-data);font-size:0.75rem;color:{C["secondary"]}">{d["email"]}</span>'
            f'</div>'
        )
    st.markdown(glass_card(doc_rows, "purple"), unsafe_allow_html=True)


# ── Sidebar Navigation ─────────────────────────────────────────────

def render_sidebar(monitor: Monitor, db: DatabaseService, doctor_user: dict) -> str:
    with st.sidebar:
        # Brand header
        st.markdown(f"""
        <div style="padding:16px 0 10px">
            <div class="brand-title">COGNICARE</div>
            <div class="brand-subtitle">{doctor_user['hospital_name']} Station</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<hr style="border:none;border-top:1px solid {C["grid"]};margin:0 0 14px">', unsafe_allow_html=True)

        # Nav list
        st.markdown(f'<div class="section-label" style="margin-bottom:8px">Clinical Menu</div>', unsafe_allow_html=True)
        page_labels = [f"{icon}  {label}" for icon, label in CLINICAL_NAV]
        
        current_page_label = f"{CLINICAL_NAV[0][0]}  {st.session_state.get('nav_page', 'Hospital Dashboard')}"
        for icon, label in CLINICAL_NAV:
            if label == st.session_state.get('nav_page', 'Hospital Dashboard'):
                current_page_label = f"{icon}  {label}"
                break
        
        selected = st.radio("nav", page_labels, label_visibility="collapsed", key="nav_radio", index=page_labels.index(current_page_label))
        active_page = selected.split("  ", 1)[1]
        st.session_state["nav_page"] = active_page

        st.markdown(f'<hr style="border:none;border-top:1px solid {C["grid"]};margin:16px 0 14px">', unsafe_allow_html=True)

        # Hospital statistics summary
        stats = db.get_hospital_stats(doctor_user["hospital_id"])
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <div class="section-label" style="margin-bottom:8px">Clinic Summary</div>
            <div style="font-size:0.67rem;font-family:var(--font-data);line-height:2.0;color:{C['secondary']}">
                <div>Monitored Patients &nbsp;
                    <span style="color:{C['cyan']};float:right;font-weight:600">
                        {stats['patients']}
                    </span>
                </div>
                <div>Completed Audits &nbsp;&nbsp;
                    <span style="color:{C['green']};float:right;font-weight:600">
                        {stats['assessments']}
                    </span>
                </div>
                <div>Active Staff &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    <span style="color:{C['purple']};float:right;font-weight:600">
                        {stats['doctors']}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<hr style="border:none;border-top:1px solid {C["grid"]};margin:12px 0 12px">', unsafe_allow_html=True)

        # Logout button
        if st.button("🚪 Logout Clinician Session", use_container_width=True, key="sb_logout_btn"):
            st.session_state["auth_state"] = "landing"
            st.session_state["doctor_user"] = None
            st.session_state["selected_patient_id"] = None
            st.rerun()

        st.markdown(f"""
        <div style="font-size:0.55rem;color:{C['grid']};letter-spacing:0.08em;
                     text-align:center;font-family:var(--font-data);line-height:1.8;margin-top:20px">
            COGNICARE v1.1 CLINICAL<br>
            HOSPITAL PATIENT INTEL MODULE
        </div>
        """, unsafe_allow_html=True)

    return active_page


# ── Main Entrypoint ───────────────────────────────────────────────

def main() -> None:
    st.markdown(inject(), unsafe_allow_html=True)
    init_session_state()

    # Initialize SQLite database
    if "db" not in st.session_state:
        st.session_state["db"] = DatabaseService()
    db = st.session_state["db"]

    monitor: Monitor = st.session_state["monitor"]
    df  = load_dataset()
    rai = None

    # Verify models
    val = Validator()
    model_check = val.validate_models()

    if model_check.valid:
        try:
            dataset_hash = len(df)
            rai = load_rai(dataset_hash)
            st.session_state["rai_loaded"] = True
        except Exception as e:
            st.session_state["rai_loaded"] = False
            logging.exception("Failed to load RAI engines: %s", e)
            # Show a visible warning so the failure is diagnosable
            st.warning(
                f"⚠️ AI engines failed to load: `{type(e).__name__}: {e}`. "
                "Demo Mode and Cognitive Analysis will be unavailable. "
                "Check that models/arousal/ and models/cognitive_load/ exist."
            )
    else:
        st.session_state["rai_loaded"] = False

    # Route navigation according to Auth State
    auth_state = st.session_state["auth_state"]

    if auth_state == "landing":
        landing.render(
            db_service=db,
            on_register_click=lambda: st.session_state.update({"auth_state": "login"}),
            on_login_click=lambda: st.session_state.update({"auth_state": "login"}),
            on_explore_click=lambda: st.session_state.update({"auth_state": "explore"})
        )
        
    elif auth_state == "login":
        auth.render(
            db_service=db,
            on_auth_success=lambda doc: st.session_state.update({"auth_state": "logged_in", "doctor_user": doc}),
            on_cancel=lambda: st.session_state.update({"auth_state": "landing"})
        )
        
    elif auth_state == "explore":
        # Unauthenticated Demo exploration mode
        st.markdown("""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;margin-bottom:8px">
            <span style="font-family:var(--font-display);font-size:0.75rem;letter-spacing:0.15em;color:var(--neon-cyan)">
                PUBLIC PLATFORM EXPLORATION MODE
            </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⬅ Return to Landing Page", use_container_width=True, key="explore_back"):
            st.session_state["auth_state"] = "landing"
            st.rerun()
            
        st.markdown(neon_divider(), unsafe_allow_html=True)
        if rai is None:
            st.error("AI engines not loaded. Run train.py or verify models.")
        else:
            demo.render(df, st.session_state, rai)
            
    elif auth_state == "logged_in":
        # Render the full Doctor portal
        active_page = render_sidebar(monitor, db, st.session_state["doctor_user"])
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if active_page == "Hospital Dashboard":
            render_hospital_dashboard(df, db)
            
        elif active_page == "Patient Directory":
            patients.render(db, st.session_state["doctor_user"], st.session_state)
            
        elif active_page == "Cognitive Analysis":
            if rai is None:
                st.error("AI engines not loaded. Please train models or run train.py first.")
            else:
                analysis.render(df, st.session_state, rai, db, st.session_state["doctor_user"])
                
        elif active_page == "Physiological Intel":
            physio.render(df, st.session_state, db, st.session_state["doctor_user"])
            
        elif active_page == "Dataset Intelligence":
            dataset.render(df, st.session_state)
            
        elif active_page == "Demo Mode":
            if rai is None:
                st.error("AI engines not loaded. Please train models or run train.py first.")
            else:
                demo.render(df, st.session_state, rai)


if __name__ == "__main__":
    main()
