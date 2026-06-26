"""
dashboard/styles.py
Complete CSS design system for CogniArousal cyberpunk dashboard.
All styles injected via st.markdown(get_css(), unsafe_allow_html=True).
"""

GOOGLE_FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
"""

CSS = """
<style>
/* ============================================================
   ROOT TOKENS
   ============================================================ */
:root {
    --bg-primary:    #070417;
    --bg-secondary:  #0D0828;
    --card-bg:       #120B31;
    --card-border:   #2A1B63;
    --neon-cyan:     #00E5FF;
    --neon-purple:   #9B6DFF;
    --neon-pink:     #FF4D7A;
    --neon-green:    #00FFB2;
    --neon-orange:   #FFB347;
    --neon-red:      #FF4D4D;
    --text-white:    #FFFFFF;
    --text-secondary:#8A82B5;
    --grid-lines:    #24174D;
    --font-display:  'Orbitron', monospace;
    --font-body:     'Space Grotesk', sans-serif;
    --font-data:     'Inter', sans-serif;
}

/* ============================================================
   GLOBAL RESET & BASE
   ============================================================ */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-white) !important;
}

.stApp {
    background: var(--bg-primary) !important;
    background-image:
        linear-gradient(var(--grid-lines) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-lines) 1px, transparent 1px);
    background-size: 40px 40px;
    background-attachment: fixed;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0620 0%, #080318 100%) !important;
    border-right: 1px solid var(--card-border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-white) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Keep toolbar visible for sidebar toggle button */
[data-testid="stToolbar"] { 
    visibility: visible !important;
    display: flex !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--neon-purple); border-radius: 2px; }

/* ============================================================
   TYPOGRAPHY
   ============================================================ */
.brand-title {
    font-family: var(--font-display) !important;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    margin: 0;
    padding: 0;
}

.brand-subtitle {
    font-family: var(--font-data) !important;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    margin-top: 2px;
}

.page-title {
    font-family: var(--font-display) !important;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--neon-cyan) !important;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.section-label {
    font-family: var(--font-data) !important;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
}

/* ============================================================
   GLASS CARD
   ============================================================ */
.glass-card {
    background: linear-gradient(135deg,
        rgba(18, 11, 49, 0.95) 0%,
        rgba(13, 8, 40, 0.90) 100%);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow:
        0 4px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255,255,255,0.04);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent, rgba(0,229,255,0.4), transparent);
}

.glass-card:hover {
    border-color: rgba(0,229,255,0.3);
    box-shadow:
        0 8px 40px rgba(0,0,0,0.5),
        0 0 20px rgba(0,229,255,0.08),
        inset 0 1px 0 rgba(255,255,255,0.06);
    transform: translateY(-1px);
}

/* Neon border variants */
.card-cyan  { border-color: rgba(0,229,255,0.35);   box-shadow: 0 0 20px rgba(0,229,255,0.08),   0 4px 32px rgba(0,0,0,0.4); }
.card-purple{ border-color: rgba(155,109,255,0.35);  box-shadow: 0 0 20px rgba(155,109,255,0.08), 0 4px 32px rgba(0,0,0,0.4); }
.card-pink  { border-color: rgba(255,77,122,0.35);   box-shadow: 0 0 20px rgba(255,77,122,0.08),  0 4px 32px rgba(0,0,0,0.4); }
.card-green { border-color: rgba(0,255,178,0.35);    box-shadow: 0 0 20px rgba(0,255,178,0.08),   0 4px 32px rgba(0,0,0,0.4); }
.card-orange{ border-color: rgba(255,179,71,0.35);   box-shadow: 0 0 20px rgba(255,179,71,0.08),  0 4px 32px rgba(0,0,0,0.4); }
.card-red   { border-color: rgba(255,77,77,0.35);    box-shadow: 0 0 20px rgba(255,77,77,0.08),   0 4px 32px rgba(0,0,0,0.4); }

/* ============================================================
   KPI CARD
   ============================================================ */
.kpi-card {
    background: linear-gradient(135deg, rgba(18,11,49,0.98), rgba(10,6,32,0.95));
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, var(--neon-cyan));
    opacity: 0.7;
}

.kpi-value {
    font-family: var(--font-display) !important;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
    color: var(--accent, var(--neon-cyan)) !important;
    text-shadow: 0 0 20px var(--accent, var(--neon-cyan));
    display: block;
    margin: 6px 0 4px;
}

.kpi-label {
    font-family: var(--font-data) !important;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
}

.kpi-icon {
    font-size: 1.2rem;
    margin-bottom: 4px;
    display: block;
}

/* ============================================================
   STATUS BADGE
   ============================================================ */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-family: var(--font-data) !important;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.status-online  { background: rgba(0,255,178,0.12); border: 1px solid rgba(0,255,178,0.4); color: #00FFB2 !important; }
.status-warning { background: rgba(255,179,71,0.12); border: 1px solid rgba(255,179,71,0.4); color: #FFB347 !important; }
.status-offline { background: rgba(255,77,77,0.12);  border: 1px solid rgba(255,77,77,0.4);  color: #FF4D4D !important; }
.status-idle    { background: rgba(155,109,255,0.12);border: 1px solid rgba(155,109,255,0.4);color: #9B6DFF !important; }

.pulse-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
    flex-shrink: 0;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

/* ============================================================
   PREDICTION RESULT CARD
   ============================================================ */
.prediction-card {
    background: linear-gradient(135deg, rgba(18,11,49,0.98), rgba(10,6,32,0.95));
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--card-border);
}

.prediction-class {
    font-family: var(--font-display) !important;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    display: block;
    margin: 8px 0;
}

.pred-high   { color: #FF4D7A !important; text-shadow: 0 0 30px rgba(255,77,122,0.6); }
.pred-medium { color: #FFB347 !important; text-shadow: 0 0 30px rgba(255,179,71,0.6); }
.pred-low    { color: #00FFB2 !important; text-shadow: 0 0 30px rgba(0,255,178,0.6);  }

.prediction-label {
    font-family: var(--font-data) !important;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
}

/* ============================================================
   CONFIDENCE BAR
   ============================================================ */
.confidence-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin: 8px 0 4px;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
    box-shadow: 0 0 8px var(--neon-cyan);
    transition: width 0.8s ease;
}

/* ============================================================
   SHAP FEATURE BAR
   ============================================================ */
.shap-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
}

.shap-label {
    font-family: var(--font-data) !important;
    font-size: 0.75rem;
    color: var(--text-white) !important;
    min-width: 160px;
}

.shap-bar-track {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    height: 8px;
    overflow: hidden;
}

.shap-bar-pos {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--neon-cyan), #00A8FF);
    box-shadow: 0 0 6px var(--neon-cyan);
}

.shap-bar-neg {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--neon-pink), #CC3366);
    box-shadow: 0 0 6px var(--neon-pink);
    margin-left: auto;
}

.shap-value {
    font-family: var(--font-data) !important;
    font-size: 0.72rem;
    font-weight: 600;
    min-width: 56px;
    text-align: right;
}

.shap-pos { color: var(--neon-cyan) !important; }
.shap-neg { color: var(--neon-pink) !important; }

/* ============================================================
   NARRATIVE BOX
   ============================================================ */
.narrative-box {
    background: linear-gradient(135deg,
        rgba(0,229,255,0.05) 0%,
        rgba(155,109,255,0.05) 100%);
    border: 1px solid rgba(0,229,255,0.2);
    border-left: 3px solid var(--neon-cyan);
    border-radius: 8px;
    padding: 16px 20px;
    font-family: var(--font-body) !important;
    font-size: 0.9rem;
    font-style: italic;
    color: rgba(255,255,255,0.9) !important;
    line-height: 1.6;
    margin: 12px 0;
}

/* ============================================================
   TIMELINE
   ============================================================ */
.timeline-row {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid var(--card-border);
    background: rgba(18,11,49,0.6);
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.timeline-row:hover {
    border-color: rgba(0,229,255,0.25);
    background: rgba(18,11,49,0.85);
}

.timeline-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--neon-cyan);
    box-shadow: 0 0 8px var(--neon-cyan);
    flex-shrink: 0;
}

.timeline-session {
    font-family: var(--font-data) !important;
    font-size: 0.65rem;
    color: var(--text-secondary) !important;
    min-width: 90px;
}

.timeline-time {
    font-family: var(--font-data) !important;
    font-size: 0.7rem;
    color: var(--text-secondary) !important;
    min-width: 80px;
}

.timeline-pred {
    font-family: var(--font-body) !important;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ============================================================
   DIVIDER
   ============================================================ */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent, var(--neon-purple), var(--neon-cyan), var(--neon-purple), transparent);
    border: none;
    margin: 20px 0;
    opacity: 0.4;
}

/* ============================================================
   STREAMLIT COMPONENT OVERRIDES
   ============================================================ */
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(18,11,49,0.6) !important;
    border-bottom: 1px solid var(--card-border) !important;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-data) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none !important;
    padding: 8px 16px !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.08) !important;
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div {
    background: var(--neon-cyan) !important;
}

/* Number inputs */
.stNumberInput input {
    background: rgba(18,11,49,0.8) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 6px !important;
    color: var(--text-white) !important;
    font-family: var(--font-data) !important;
}

.stNumberInput input:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 0 1px rgba(0,229,255,0.3) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(18,11,49,0.8) !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-white) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(155,109,255,0.12)) !important;
    border: 1px solid rgba(0,229,255,0.4) !important;
    color: var(--neon-cyan) !important;
    font-family: var(--font-data) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,229,255,0.22), rgba(155,109,255,0.22)) !important;
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.25) !important;
    transform: translateY(-1px) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.2), rgba(155,109,255,0.2)) !important;
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.15) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(18,11,49,0.6) !important;
    border: 1px dashed rgba(0,229,255,0.3) !important;
    border-radius: 8px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(18,11,49,0.6) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    color: var(--neon-cyan) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: rgba(18,11,49,0.6) !important;
}

/* Sidebar nav labels */
.nav-item {
    font-family: var(--font-data) !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 0;
}

/* Radio button nav override */
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--font-data) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    transition: color 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--neon-cyan) !important;
}

/* Info / warning / error boxes */
.stAlert {
    background: rgba(18,11,49,0.8) !important;
    border-radius: 8px !important;
}
</style>
"""


def inject(extra: str = "") -> str:
    """Return the full CSS injection block."""
    return GOOGLE_FONTS + CSS + extra
