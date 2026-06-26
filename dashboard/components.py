"""
dashboard/components.py
Reusable HTML/Plotly component renderers for the CogniArousal dashboard.

IMPORTANT: All inline style= attributes use CSS variables (var(--font-display) etc.)
instead of quoted font-family strings. This prevents Streamlit's HTML sanitizer
from escaping the single quotes and rendering raw HTML text on screen.
"""

from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Colour tokens (mirror CSS variables for Plotly charts) ────────
C = {
    "bg":        "#070417",
    "bg2":       "#0D0828",
    "card":      "#120B31",
    "border":    "#2A1B63",
    "cyan":      "#00E5FF",
    "purple":    "#9B6DFF",
    "pink":      "#FF4D7A",
    "green":     "#00FFB2",
    "orange":    "#FFB347",
    "red":       "#FF4D4D",
    "white":     "#FFFFFF",
    "secondary": "#8A82B5",
    "grid":      "#24174D",
}

# Font shortcuts — avoids quoting font names inside inline styles
_F_DISPLAY = "var(--font-display)"   # Orbitron
_F_BODY    = "var(--font-body)"      # Space Grotesk
_F_DATA    = "var(--font-data)"      # Inter

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, Inter, sans-serif", color=C["secondary"]),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"], tickfont=dict(size=10)),
    yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"], tickfont=dict(size=10)),
)


# ── HTML Components ───────────────────────────────────────────────

def kpi_card(value: str, label: str, icon: str,
             accent: str = "#00E5FF", delta: str = "") -> str:
    delta_html = (
        f'<span style="font-size:0.65rem;color:{accent};opacity:0.8">{delta}</span>'
        if delta else ""
    )
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<span class="kpi-icon">{icon}</span>'
        f'<span class="kpi-value">{value}</span>'
        f'<span class="kpi-label">{label}</span>'
        f'{delta_html}'
        f'</div>'
    )


def status_badge(label: str, state: str = "online") -> str:
    return (
        f'<span class="status-badge status-{state}">'
        f'<span class="pulse-dot"></span>{label}'
        f'</span>'
    )


def glass_card(content: str, accent: str = "cyan", extra_style: str = "") -> str:
    return (
        f'<div class="glass-card card-{accent}" style="{extra_style}">'
        f'{content}'
        f'</div>'
    )


def prediction_card(predicted_class: str, target_label: str,
                    confidence_pct: float, tier: str, flag_review: bool) -> str:
    pred_colors  = {"High": C["pink"],   "Medium": C["orange"], "Low": C["green"]}
    badge_colors = {"High": "#FF4D4D",   "Medium": C["orange"], "Low": C["green"]}
    tier_color       = pred_colors.get(predicted_class, C["cyan"])
    tier_badge_color = badge_colors.get(tier, C["cyan"])
    tier_class       = f"pred-{predicted_class.lower()}"
    flag_html = (
        f'<div style="margin-top:10px;font-size:0.65rem;color:{C["orange"]};'
        f'letter-spacing:0.1em">LOW CONFIDENCE - REVIEW RECOMMENDED</div>'
        if flag_review else ""
    )
    return (
        f'<div class="prediction-card" '
        f'style="border-color:{tier_color}40;box-shadow:0 0 24px {tier_color}12">'
        f'<span class="prediction-label">{target_label}</span>'
        f'<span class="prediction-class {tier_class}">{predicted_class}</span>'
        f'<div style="margin-top:6px">'
        f'<span class="section-label">Confidence</span>'
        f'<div class="confidence-bar-wrap" style="margin-top:5px">'
        f'<div class="confidence-bar-fill" '
        f'style="width:{confidence_pct:.1f}%;'
        f'background:linear-gradient(90deg,{tier_color},{tier_color}88)"></div>'
        f'</div>'
        f'<span style="font-family:{_F_DISPLAY};font-size:0.95rem;font-weight:700;'
        f'color:{tier_color}">{confidence_pct:.1f}%</span>'
        f'&nbsp;'
        f'<span class="status-badge" '
        f'style="background:{tier_badge_color}18;border:1px solid {tier_badge_color}55;'
        f'color:{tier_badge_color};font-size:0.55rem">{tier}</span>'
        f'</div>'
        f'{flag_html}'
        f'</div>'
    )


def shap_feature_row(label: str, shap_val: float, max_abs: float, direction: str) -> str:
    pct      = min(abs(shap_val) / max(max_abs, 1e-9) * 100, 100)
    is_pos   = shap_val >= 0
    color    = C["cyan"] if is_pos else C["pink"]
    val_cls  = "shap-pos" if is_pos else "shap-neg"
    sign     = "+" if is_pos else ""
    bar_cls  = "shap-bar-pos" if is_pos else "shap-bar-neg"
    dir_html = (
        f'<span style="font-size:0.6rem;color:{color};background:{color}15;'
        f'padding:1px 6px;border-radius:3px;margin-left:4px">{direction}</span>'
    )
    return (
        f'<div class="shap-row">'
        f'<span class="shap-label">{label}{dir_html}</span>'
        f'<div class="shap-bar-track">'
        f'<div class="{bar_cls}" style="width:{pct:.1f}%"></div>'
        f'</div>'
        f'<span class="shap-value {val_cls}">{sign}{shap_val:.4f}</span>'
        f'</div>'
    )


def narrative_box(text: str) -> str:
    return f'<div class="narrative-box">&#8220;{text}&#8221;</div>'


def neon_divider() -> str:
    return '<hr class="neon-divider">'


def section_header(title: str, subtitle: str = "", icon: str = "") -> str:
    sub = (
        f'<div class="section-label" style="margin-top:2px">{subtitle}</div>'
        if subtitle else ""
    )
    return (
        f'<div style="margin-bottom:16px">'
        f'<div class="page-title">{icon} {title}</div>'
        f'{sub}'
        f'</div>'
    )


def timeline_row(session_id: str, timestamp: str, arousal: str,
                 cog_load: str, confidence: float, tier: str) -> str:
    pred_colors  = {"High": C["pink"], "Medium": C["orange"], "Low": C["green"]}
    tier_color = pred_colors.get(tier,     C["cyan"])
    a_color    = pred_colors.get(arousal,  C["cyan"])
    c_color    = pred_colors.get(cog_load, C["cyan"])
    return (
        f'<div class="timeline-row">'
        f'<div class="timeline-dot"></div>'
        f'<span class="timeline-session" title="{session_id}">{session_id[:8]}...</span>'
        f'<span class="timeline-time">{timestamp}</span>'
        f'<span class="timeline-pred" style="color:{a_color};min-width:90px">'
        f'Arousal: {arousal}</span>'
        f'<span class="timeline-pred" style="color:{c_color};min-width:110px">'
        f'Load: {cog_load}</span>'
        f'<span style="font-family:{_F_DISPLAY};font-size:0.7rem;'
        f'color:{tier_color};min-width:55px">{confidence:.1f}%</span>'
        f'<span class="status-badge" '
        f'style="background:{tier_color}18;border:1px solid {tier_color}44;'
        f'color:{tier_color};font-size:0.55rem">{tier}</span>'
        f'</div>'
    )


# ── Plotly Charts ─────────────────────────────────────────────────

def gauge_chart(value: float, title: str, color: str = "#00E5FF",
                min_val: float = 0, max_val: float = 100) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(family="Space Grotesk", size=13, color=C["secondary"])),
        number=dict(font=dict(family="Orbitron", size=28, color=color), suffix="%"),
        gauge=dict(
            axis=dict(
                range=[min_val, max_val], tickwidth=1,
                tickcolor=C["grid"], tickfont=dict(size=9, color=C["secondary"]),
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[min_val, max_val * 0.6],       color="rgba(255,255,255,0.03)"),
                dict(range=[max_val * 0.6, max_val * 0.85],color="rgba(255,255,255,0.05)"),
                dict(range=[max_val * 0.85, max_val],      color="rgba(255,255,255,0.07)"),
            ],
            threshold=dict(line=dict(color=color, width=2), thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, height=200)
    return fig


def probability_bar_chart(class_probs: dict, title: str = "Class Probabilities") -> go.Figure:
    classes = list(class_probs.keys())
    probs   = [v * 100 for v in class_probs.values()]
    colors  = [C["pink"], C["green"], C["orange"]][:len(classes)]
    fig = go.Figure(go.Bar(
        x=classes, y=probs,
        marker=dict(color=colors, opacity=0.85, line=dict(color=colors, width=1)),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(family="Orbitron", size=11, color=C["white"]),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(family="Orbitron", size=11, color=C["secondary"]), x=0.5),
        height=220, showlegend=False, bargap=0.35,
    )
    fig.update_yaxes(range=[0, 115], ticksuffix="%")
    return fig


def signal_line_chart(x: np.ndarray, y: np.ndarray, title: str,
                      color: str = "#00E5FF", y_label: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=1.5, shape="spline", smoothing=0.4),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color)},0.06)",
    ))
    layout = {k: v for k, v in _PLOTLY_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(
        **layout,
        title=dict(text=title, font=dict(family="Orbitron", size=11, color=C["secondary"]), x=0),
        height=220,
        yaxis=dict(
            gridcolor=C["grid"], zerolinecolor=C["grid"], tickfont=dict(size=10),
            title=dict(text=y_label, font=dict(size=10)),
        ),
    )
    return fig


def scatter_chart(df: pd.DataFrame, x_col: str, y_col: str,
                  color_col: str, title: str) -> go.Figure:
    color_map = {"baseline": C["cyan"], "stress": C["pink"], "amusement": C["green"]}
    colors = [color_map.get(v, C["purple"]) for v in df[color_col]]
    fig = go.Figure(go.Scatter(
        x=df[x_col], y=df[y_col], mode="markers",
        marker=dict(color=colors, size=7, opacity=0.8,
                    line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
        text=df[color_col],
        hovertemplate=(
            f"<b>{x_col}</b>: %{{x:.3f}}<br>"
            f"<b>{y_col}</b>: %{{y:.3f}}<br>"
            f"<b>Label</b>: %{{text}}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(family="Orbitron", size=11, color=C["secondary"]), x=0),
        height=280,
    )
    return fig


def histogram_chart(series: pd.Series, title: str, color: str = "#9B6DFF") -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=series, nbinsx=25,
        marker=dict(color=color, opacity=0.8, line=dict(width=0.3, color=C["bg"])),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(family="Orbitron", size=10, color=C["secondary"]), x=0),
        height=200, bargap=0.05,
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Generate correlation matrix heatmap - completely rewritten to avoid layout issues"""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0.0, C["pink"]], [0.5, C["bg2"]], [1.0, C["cyan"]]],
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=9),
            hoverongaps=False,
            showscale=True,
            colorbar=dict(
                tickfont=dict(size=9, color=C["secondary"]),
                bgcolor="rgba(0,0,0,0)",
                bordercolor=C["grid"],
            ),
        ),
        layout=go.Layout(
            title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk, Inter, sans-serif", color=C["secondary"]),
            margin=dict(l=10, r=10, t=30, b=10),
            height=400,
            xaxis=dict(
                gridcolor=C["grid"],
                zerolinecolor=C["grid"],
                tickfont=dict(size=9),
                tickangle=-35
            ),
            yaxis=dict(
                gridcolor=C["grid"],
                zerolinecolor=C["grid"],
                tickfont=dict(size=9, color=C["secondary"])
            ),
        )
    )
    return fig


def global_shap_bar(df: pd.DataFrame, title: str) -> go.Figure:
    df_sorted = df.sort_values("mean_abs_shap")
    colors    = [C["cyan"]] * len(df_sorted)
    if len(colors):
        colors[-1] = C["purple"]
    fig = go.Figure(go.Bar(
        x=df_sorted["mean_abs_shap"], y=df_sorted["feature"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=[f"{v:.4f}" for v in df_sorted["mean_abs_shap"]],
        textposition="outside",
        textfont=dict(family="Inter", size=10, color=C["secondary"]),
    ))
    fig.update_layout(
    **_PLOTLY_LAYOUT,
    title=dict(
        text=title,
        font=dict(
            family="Orbitron",
            size=11,
            color=C["secondary"]
        ),
        x=0
    ),
    height=300,
    )
    fig.update_xaxes(title_text="Mean |SHAP|")
    return fig


def label_donut(value_counts: pd.Series) -> go.Figure:
    color_map = {
        "baseline": C["cyan"], "stress": C["pink"], "amusement": C["green"],
        "High": C["pink"], "Medium": C["orange"], "Low": C["green"],
    }
    colors = [color_map.get(l, C["purple"]) for l in value_counts.index]
    fig = go.Figure(go.Pie(
        labels=value_counts.index.tolist(),
        values=value_counts.values.tolist(),
        hole=0.6,
        marker=dict(colors=colors, line=dict(color=C["bg"], width=2)),
        textfont=dict(family="Space Grotesk", size=11),
        hovertemplate="<b>%{label}</b><br>%{value} windows (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT, showlegend=True,
        legend=dict(font=dict(size=10, color=C["secondary"]), bgcolor="rgba(0,0,0,0)"),
        height=260,
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    h    = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
