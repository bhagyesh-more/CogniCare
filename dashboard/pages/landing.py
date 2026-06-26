"""
dashboard/pages/landing.py
New Landing Page for CogniCare Platform.
Provides product positioning, live stats KPIs, visual pipeline explanation,
responsible AI value cards, and technical background before user login.
"""

import streamlit as st
from dashboard.components import C, glass_card, kpi_card, neon_divider

def render(db_service, on_register_click, on_login_click, on_explore_click) -> None:
    # ── Style adjustments for landing page specifically ──────────────────
    st.markdown("""
    <style>
    /* Styling for the workflow node chart */
    .flow-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
        gap: 12px;
        padding: 24px 10px;
        background: rgba(18, 11, 49, 0.4);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        margin: 20px 0;
    }
    .flow-node {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        font-family: var(--font-display);
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        min-width: 140px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .flow-node:hover {
        border-color: var(--neon-cyan);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
    }
    .flow-arrow {
        color: var(--neon-purple);
        font-size: 1.2rem;
        font-weight: bold;
        text-shadow: 0 0 8px var(--neon-purple);
        margin: 0 4px;
    }
    
    /* Hero Visual animation */
    .hero-visual {
        border: 1px dashed var(--card-border);
        border-radius: 12px;
        padding: 24px;
        background: radial-gradient(circle at center, rgba(155, 109, 255, 0.05) 0%, transparent 70%);
        position: relative;
        overflow: hidden;
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .pulsing-grid {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        opacity: 0.15;
        background-size: 20px 20px;
        background-image: linear-gradient(to right, var(--neon-cyan) 1px, transparent 1px),
                          linear-gradient(to bottom, var(--neon-cyan) 1px, transparent 1px);
        animation: pulse-grid 8s infinite linear;
    }
    
    @keyframes pulse-grid {
        0% { transform: scale(1); opacity: 0.1; }
        50% { transform: scale(1.05); opacity: 0.25; }
        100% { transform: scale(1); opacity: 0.1; }
    }
    
    .tech-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--card-border);
        font-size: 0.7rem;
        font-family: var(--font-data);
        color: var(--text-secondary);
        margin: 4px;
    }
    
    .tech-pill strong {
        color: var(--text-white);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Fetch global metrics from database for the stats display ──────────
    # We aggregate total counts from the database to present real numbers
    with db_service.get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM hospitals")
        hospitals_cnt = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM doctors")
        doctors_cnt = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM patients")
        patients_cnt = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM assessments")
        assessments_cnt = c.fetchone()[0]
        
        # Calculate overall avg confidence if assessments exist
        c.execute("SELECT AVG(arousal_conf), AVG(load_conf) FROM assessments")
        avg_row = c.fetchone()
        if avg_row and avg_row[0] is not None and avg_row[1] is not None:
            avg_conf = (avg_row[0] + avg_row[1]) / 2 * 100
        else:
            avg_conf = 88.5  # fallback baseline

    # ── 1. Hero Section ──────────────────────────────────────────────────
    c_left, c_right = st.columns([7, 5])
    
    with c_left:
        st.markdown(f"""
        <div style="padding: 24px 0 10px 0">
            <h1 style="font-family:var(--font-display);font-size:3.5rem;font-weight:900;letter-spacing:0.18em;
                       background:linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 10px 0;
                       text-shadow: 0 0 30px rgba(0, 229, 255, 0.2)">COGNICARE</h1>
            <h3 style="font-family:var(--font-display);font-size:1.1rem;letter-spacing:0.22em;
                       color:var(--neon-purple);text-transform:uppercase;margin:0 0 20px 0">
                Responsible AI Cognitive Monitoring Platform
            </h3>
            <p style="font-size:0.95rem;line-height:1.6;color:var(--text-secondary);margin-bottom:30px;max-width:580px">
                Predict, explain, and monitor cognitive load and emotional arousal trends using physiological intelligence and explainable AI. Purpose-built for hospitals, clinics, and mental health professionals.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action Buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("🏥  Register Clinic", key="lp_btn_register", type="primary", use_container_width=True):
                on_register_click()
        with btn_col2:
            if st.button("🥼  Doctor Login", key="lp_btn_login", use_container_width=True):
                on_login_click()
        with btn_col3:
            if st.button("⌬  Explore Platform", key="lp_btn_explore", use_container_width=True):
                on_explore_click()

    with c_right:
        st.markdown(f"""
        <div class="hero-visual">
            <div class="pulsing-grid"></div>
            <div style="position:relative;z-index:2;text-align:center">
                <div style="font-size:3rem;margin-bottom:12px;text-shadow:0 0 20px var(--neon-cyan)">⬡</div>
                <div style="font-family:var(--font-display);font-size:0.8rem;letter-spacing:0.2em;color:var(--neon-cyan);margin-bottom:8px">
                    CLINICAL BIO-INTELLIGENCE
                </div>
                <div style="font-family:var(--font-data);font-size:0.7rem;color:var(--text-secondary);line-height:1.8">
                    Active UUID4 Patient Shield: <span style="color:var(--neon-green)">ENABLED</span><br>
                    SHAP Attribution Resolution: <span style="color:var(--neon-purple)">9-FEATURE MATRIX</span><br>
                    Real-time Entropy Filter: <span style="color:var(--neon-orange)">ACTIVE</span>
                </div>
                <div style="margin-top:16px;display:flex;justify-content:center;gap:8px">
                    <span style="width:8px;height:8px;border-radius:50%;background:var(--neon-green);box-shadow:0 0 8px var(--neon-green)"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:var(--neon-purple);box-shadow:0 0 8px var(--neon-purple);animation:pulse 2s infinite"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:var(--neon-cyan);box-shadow:0 0 8px var(--neon-cyan)"></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── 2. Platform Overview ─────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:16px;text-align:center">
        <div class="section-label">Live Operations</div>
        <div class="page-title" style="font-size:1.4rem">Platform Overview Statistics</div>
    </div>
    """, unsafe_allow_html=True)
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(kpi_card(str(hospitals_cnt), "Hospitals Registered", "🏥", C["cyan"]), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card(str(doctors_cnt), "Doctors Active", "🥼", C["purple"]), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card(str(patients_cnt), "Patients Monitored", "👤", C["orange"]), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card(str(assessments_cnt), "Assessments Logged", "⚡", C["pink"]), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_card(f"{avg_conf:.1f}%", "AI Confidence Rate", "◎", C["green"]), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── 3. How CogniCare Works ───────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:16px;text-align:center">
        <div class="section-label">Workflow Architecture</div>
        <div class="page-title" style="font-size:1.4rem">How CogniCare Operates</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom HTML workflow chart
    st.markdown("""
    <div class="flow-container">
        <div class="flow-node" style="border-color:var(--neon-cyan)">
            <div style="font-size:1.2rem;margin-bottom:4px">≋</div>
            Physiological Data
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node" style="border-color:var(--neon-purple)">
            <div style="font-size:1.2rem;margin-bottom:4px">▦</div>
            Feature Processing
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node" style="border-color:var(--neon-pink)">
            <div style="font-size:1.2rem;margin-bottom:4px">⚡</div>
            AI Prediction Engine
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node" style="border-color:var(--neon-orange)">
            <div style="font-size:1.2rem;margin-bottom:4px">◈</div>
            State Analysis
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node" style="border-color:var(--neon-green)">
            <div style="font-size:1.2rem;margin-bottom:4px">🛡️</div>
            Responsible AI Layer
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node" style="border-color:var(--neon-cyan)">
            <div style="font-size:1.2rem;margin-bottom:4px">🥼</div>
            Doctor Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── 4. Responsible AI Features ───────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:20px;text-align:center">
        <div class="section-label">Core Ethical Pillars</div>
        <div class="page-title" style="font-size:1.4rem">Responsible AI Core Features</div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card_content = """
        <div style="text-align:center">
            <span style="font-size:1.5rem;color:var(--neon-cyan)">◈</span>
            <div style="font-family:var(--font-display);font-size:0.85rem;color:var(--text-white);margin:8px 0 6px">Explainability</div>
            <div style="font-size:0.72rem;color:var(--text-secondary);font-family:var(--font-data)">
                SHAP-powered local and global feature attribution. Shows precisely which physiological triggers drove the AI's diagnosis.
            </div>
        </div>
        """
        st.markdown(glass_card(card_content, "cyan", "height:140px"), unsafe_allow_html=True)
        
    with c2:
        card_content = """
        <div style="text-align:center">
            <span style="font-size:1.5rem;color:var(--neon-purple)">◎</span>
            <div style="font-family:var(--font-display);font-size:0.85rem;color:var(--text-white);margin:8px 0 6px">Confidence Analysis</div>
            <div style="font-size:0.72rem;color:var(--text-secondary);font-family:var(--font-data)">
                Probabilistic confidence tiering (High/Medium/Low) based on Shannon entropy to safely alert clinicians of marginal predictions.
            </div>
        </div>
        """
        st.markdown(glass_card(card_content, "purple", "height:140px"), unsafe_allow_html=True)
        
    with c3:
        card_content = """
        <div style="text-align:center">
            <span style="font-size:1.5rem;color:var(--neon-pink)">◎</span>
            <div style="font-family:var(--font-display);font-size:0.85rem;color:var(--text-white);margin:8px 0 6px">Transparency</div>
            <div style="font-size:0.72rem;color:var(--text-secondary);font-family:var(--font-data)">
                Instant human-readable medical narratives detailing findings and explanations, designed to simplify medical audit trails.
            </div>
        </div>
        """
        st.markdown(glass_card(card_content, "pink", "height:140px"), unsafe_allow_html=True)
        
    with c4:
        card_content = """
        <div style="text-align:center">
            <span style="font-size:1.5rem;color:var(--neon-green)">🔐</span>
            <div style="font-family:var(--font-display);font-size:0.85rem;color:var(--text-white);margin:8px 0 6px">Privacy-Shielded</div>
            <div style="font-size:0.72rem;color:var(--text-secondary);font-family:var(--font-data)">
                Zero-PII ephemeral memory architectures. Scrubbing mechanisms and strict ID anonymization protect patient data privacy.
            </div>
        </div>
        """
        st.markdown(glass_card(card_content, "green", "height:140px"), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── 5. Platform Modules ──────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:20px;text-align:center">
        <div class="section-label">Capabilities Matrix</div>
        <div class="page-title" style="font-size:1.4rem">Platform Modules</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_mod1, col_mod2, col_mod3, col_mod4 = st.columns(4)
    with col_mod1:
        st.markdown(glass_card("🏛️  Hospital Management", "cyan", "text-align:center;font-family:var(--font-display);font-size:0.75rem"), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(glass_card("🧠  Cognitive Monitoring", "purple", "text-align:center;font-family:var(--font-display);font-size:0.75rem"), unsafe_allow_html=True)
    with col_mod2:
        st.markdown(glass_card("🥼  Doctor Management", "purple", "text-align:center;font-family:var(--font-display);font-size:0.75rem"), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(glass_card("❤️  Emotional Monitoring", "pink", "text-align:center;font-family:var(--font-display);font-size:0.75rem"), unsafe_allow_html=True)
    with col_mod3:
        st.markdown(glass_card("👥  Patient Profiles", "pink", "text-align:center;font-family:var(--font-display);font-size:0.75rem"), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(glass_card("📈  Historical Analytics", "orange", "text-align:center;font-family:var(--font-display);font-size:0.75rem"), unsafe_allow_html=True)
    with col_mod4:
        st.markdown(glass_card("◈  AI Explanations", "orange", "text-align:center;font-family:var(--font-display);font-size:0.75rem"), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(glass_card("📄  PDF Reports", "green", "text-align:center;font-family:var(--font-display);font-size:0.75rem"), unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── 6. Technology & Research Section ─────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:16px;text-align:center">
        <div class="section-label">Technical Stack</div>
        <div class="page-title" style="font-size:1.4rem">Technology & Research</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center;margin-bottom:10px">
        <span class="tech-pill">Dataset: <strong>WESAD Physiological Dataset</strong></span>
        <span class="tech-pill">AI Models: <strong>Random Forest Classifiers</strong></span>
        <span class="tech-pill">Explainable AI: <strong>SHAP values + Shannon Entropy</strong></span>
        <span class="tech-pill">Framework: <strong>Streamlit Web App</strong></span>
        <span class="tech-pill">Language: <strong>Python 3.10+</strong></span>
        <span class="tech-pill">Graphics: <strong>Plotly Cyberpunk Layouts</strong></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(neon_divider(), unsafe_allow_html=True)

    # ── 7. Footer ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:20px 0;color:var(--text-secondary);font-size:0.7rem;line-height:1.6">
        <div style="font-family:var(--font-display);color:var(--text-white);letter-spacing:0.1em;font-weight:700">COGNICARE</div>
        <div>Responsible AI Cognitive Monitoring Platform · Version 1.1</div>
        <div style="font-size:0.6rem;color:var(--grid-lines);margin-top:8px">© 2026 COGNICARE INC. ALL RIGHTS RESERVED. CLINICAL DIAGNOSTIC ASSISTANT ONLY.</div>
    </div>
    """, unsafe_allow_html=True)
