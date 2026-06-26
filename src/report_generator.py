"""
src/report_generator.py
Generates downloadable PDF and CSV analysis reports.

PDF structure:
    Page 1  - Header, session metadata, system info
    Page 2  - Input features table
    Page 3  - Predictions: arousal + cognitive load with confidence
    Page 4  - Responsible AI: SHAP top features + narrative
    Page 5  - Methodology & disclaimer

CSV structure:
    One row per report with all fields flattened.
"""

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fpdf import FPDF, XPos, YPos

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette (RGB)
_BG     = (7,   4,  23)
_CARD   = (18,  11,  49)
_CYAN   = (0,  229, 255)
_PURPLE = (155, 109, 255)
_PINK   = (255,  77, 122)
_GREEN  = (0,  255, 178)
_ORANGE = (255, 179,  71)
_WHITE  = (255, 255, 255)
_GREY   = (138, 130, 181)
_PRED_COLORS = {"High": _PINK, "Medium": _ORANGE, "Low": _GREEN}


class _ReportPDF(FPDF):
    """Custom FPDF subclass with CogniArousal branding."""

    def header(self) -> None:
        self.set_fill_color(*_BG)
        self.rect(0, 0, 210, 297, "F")

        # Top accent bar
        self.set_fill_color(*_CYAN)
        self.rect(0, 0, 210, 2, "F")

        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_GREY)
        self.set_xy(10, 6)
        self.cell(0, 5, "COGNIAROUSAL  |  RESPONSIBLE AI COGNITIVE ANALYSIS PLATFORM", align="L")
        self.set_xy(10, 6)
        self.cell(0, 5, f"Page {self.page_no()}", align="R")
        self.ln(8)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_GREY)
        self.cell(0, 5, "CONFIDENTIAL - FOR RESEARCH USE ONLY | IEEE EMBS CogniArousal v1.0", align="C")

    def neon_rule(self) -> None:
        self.set_draw_color(*_PURPLE)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def section_title(self, text: str, color: tuple = _CYAN) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.cell(0, 7, text.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.neon_rule()

    def kv_row(self, key: str, value: str, val_color: tuple = _WHITE) -> None:
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_GREY)
        self.cell(55, 5.5, key, border=0)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*val_color)
        self.cell(0, 5.5, str(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def prediction_block(self, target: str, pred_class: str,
                          confidence_pct: float, tier: str) -> None:
        color = _PRED_COLORS.get(pred_class, _CYAN)
        self.set_fill_color(*_CARD)
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        x, y = self.get_x(), self.get_y()
        self.rect(x, y, 90, 28, "DF")

        self.set_xy(x + 4, y + 3)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_GREY)
        self.cell(82, 4, target.upper().replace("_", " "), align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_xy(x + 4, y + 8)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*color)
        self.cell(82, 10, pred_class.upper(), align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_xy(x + 4, y + 19)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_GREY)
        self.cell(42, 5, f"Confidence: {confidence_pct:.1f}%", align="C")
        self.cell(40, 5, f"Tier: {tier}", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)


class ReportGenerator:
    """
    Generates PDF and CSV reports from a completed analysis result.

    Parameters
    ----------
    session_id   : anonymous session UUID
    timestamp    : ISO8601 timestamp string
    input_data   : dict of feature name → value
    arousal_result : dict with predicted_class, confidence_pct, tier, class_probs
    cog_load_result: dict with predicted_class, confidence_pct, tier, class_probs
    narrative    : human-readable AI explanation string
    top_features : list of dicts with feature, direction, shap_value
    """

    def __init__(
        self,
        session_id:      str,
        timestamp:       str,
        input_data:      dict,
        arousal_result:  dict,
        cog_load_result: dict,
        narrative:       str,
        top_features:    list[dict],
    ):
        self.session_id      = session_id
        self.timestamp       = timestamp
        self.input_data      = input_data
        self.arousal         = arousal_result
        self.cog_load        = cog_load_result
        self.narrative       = narrative
        self.top_features    = top_features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pdf(self) -> bytes:
        """Build and return PDF bytes (no file written unless save=True)."""
        pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(10, 15, 10)

        self._page_cover(pdf)
        self._page_features(pdf)
        self._page_predictions(pdf)
        self._page_responsible_ai(pdf)
        self._page_methodology(pdf)

        return bytes(pdf.output())

    def generate_csv(self) -> str:
        """Return report as a flat CSV string."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["field", "value"])

        # Metadata
        writer.writerow(["session_id",      self.session_id])
        writer.writerow(["timestamp",       self.timestamp])

        # Features
        for k, v in self.input_data.items():
            writer.writerow([f"feature_{k}", f"{v:.6f}"])

        # Predictions
        writer.writerow(["arousal_prediction",         self.arousal["predicted_class"]])
        writer.writerow(["arousal_confidence_pct",     f"{self.arousal['confidence_pct']:.2f}"])
        writer.writerow(["arousal_tier",               self.arousal["tier"]])
        writer.writerow(["cognitive_load_prediction",  self.cog_load["predicted_class"]])
        writer.writerow(["cognitive_load_confidence",  f"{self.cog_load['confidence_pct']:.2f}"])
        writer.writerow(["cognitive_load_tier",        self.cog_load["tier"]])

        # SHAP
        for i, fc in enumerate(self.top_features, 1):
            writer.writerow([f"shap_feature_{i}",    fc.get("feature", "")])
            writer.writerow([f"shap_direction_{i}",  fc.get("direction", "")])
            writer.writerow([f"shap_value_{i}",      f"{fc.get('shap_value', 0):.6f}"])

        # Narrative
        writer.writerow(["narrative", self.narrative])

        return buf.getvalue()

    def save_pdf(self) -> Path:
        """Save PDF to reports/ directory and return the path."""
        ts_safe = self.timestamp.replace(":", "-").replace(" ", "_")
        path = REPORTS_DIR / f"report_{self.session_id[:8]}_{ts_safe}.pdf"
        path.write_bytes(self.generate_pdf())
        return path

    # ------------------------------------------------------------------
    # PDF page builders
    # ------------------------------------------------------------------

    def _page_cover(self, pdf: _ReportPDF) -> None:
        pdf.add_page()

        # Large title block
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(*_CYAN)
        pdf.cell(0, 12, "COGNIAROUSAL", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_GREY)
        pdf.cell(0, 6, "Responsible AI Cognitive Analysis Report", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(6)
        pdf.neon_rule()
        pdf.ln(4)

        pdf.section_title("Session Metadata")
        pdf.kv_row("Session ID",  self.session_id)
        pdf.kv_row("Timestamp",   self.timestamp)
        pdf.kv_row("Report Type", "Single-Sample Responsible AI Analysis")

        pdf.ln(4)
        pdf.section_title("System Information", _PURPLE)
        pdf.kv_row("Platform",      "CogniArousal v1.0")
        pdf.kv_row("ML Model",      "Random Forest Classifier (scikit-learn 1.4)")
        pdf.kv_row("Explainability","SHAP TreeExplainer v0.51")
        pdf.kv_row("Dataset",       "WESAD Physiological Dataset (S2-S6)")
        pdf.kv_row("Privacy",       "UUID4 Anonymous Session - Zero PII Storage")

        pdf.ln(4)
        pdf.section_title("Executive Summary", _GREEN)
        a_cls  = self.arousal["predicted_class"]
        c_cls  = self.cog_load["predicted_class"]
        a_conf = self.arousal["confidence_pct"]
        c_conf = self.cog_load["confidence_pct"]

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_WHITE)
        pdf.multi_cell(
            0, 5.5,
            f"This report presents a fully transparent Responsible AI analysis of physiological "
            f"biosignals. The model predicts {a_cls} Emotional Arousal ({a_conf:.1f}% confidence) "
            f"and {c_cls} Cognitive Load ({c_conf:.1f}% confidence). "
            f"The following pages detail input features, prediction outputs, SHAP explanations, "
            f"and the AI-generated narrative summary.",
            align="L",
        )

    def _page_features(self, pdf: _ReportPDF) -> None:
        pdf.add_page()
        pdf.section_title("Input Features - Physiological Signal Vector")

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_CYAN)
        pdf.cell(70, 6, "Feature", border="B")
        pdf.cell(40, 6, "Value (z-score)", border="B")
        pdf.cell(0,  6, "Domain", border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        domain_map = {
            "eda_mean": "EDA", "eda_std": "EDA",
            "eda_peak_count": "EDA", "eda_peak_amplitude": "EDA",
            "heart_rate_bpm": "HRV", "rmssd": "HRV", "sdnn": "HRV",
            "resp_rate_bpm": "Respiration", "resp_variability": "Respiration",
        }
        alt = False
        for feat, val in self.input_data.items():
            pdf.set_fill_color(*((*_CARD,) if not alt else (12, 7, 40)))
            pdf.set_text_color(*_WHITE)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(70, 5.5, feat, fill=True)
            pdf.set_text_color(*_CYAN)
            pdf.cell(40, 5.5, f"{float(val):+.6f}", fill=True)
            pdf.set_text_color(*_GREY)
            pdf.cell(0, 5.5, domain_map.get(feat, ""), fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            alt = not alt

        pdf.ln(6)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*_GREY)
        pdf.multi_cell(
            0, 4,
            "Note: All feature values are StandardScaler z-scores computed relative to the WESAD "
            "training distribution. Positive values indicate above-mean physiological activation; "
            "negative values indicate below-mean (relaxed) state.",
        )

    def _page_predictions(self, pdf: _ReportPDF) -> None:
        pdf.add_page()
        pdf.section_title("Prediction Results")

        # Two prediction blocks side by side
        pdf.set_xy(10, pdf.get_y() + 2)
        self._prediction_block_draw(pdf, "Emotional Arousal", self.arousal, x=10)
        self._prediction_block_draw(pdf, "Cognitive Load", self.cog_load, x=108)

        pdf.set_y(pdf.get_y() + 35)
        pdf.section_title("Confidence Analysis", _PURPLE)

        for label, res in [("Emotional Arousal", self.arousal), ("Cognitive Load", self.cog_load)]:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*_GREY)
            pdf.cell(0, 5, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            probs = res.get("class_probs", {})
            for cls, prob in probs.items():
                bar_w = prob * 80
                c = _PRED_COLORS.get(cls, _CYAN)
                pdf.set_font("Helvetica", "", 7)
                pdf.set_text_color(*_GREY)
                pdf.cell(25, 4.5, cls)
                # Bar background
                pdf.set_fill_color(30, 20, 60)
                x, y = pdf.get_x(), pdf.get_y()
                pdf.rect(x, y + 0.5, 80, 3.5, "F")
                # Bar fill
                pdf.set_fill_color(*c)
                pdf.rect(x, y + 0.5, max(bar_w, 0.5), 3.5, "F")
                pdf.set_xy(x + 82, y)
                pdf.set_text_color(*c)
                pdf.cell(0, 4.5, f"{prob * 100:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

    def _prediction_block_draw(self, pdf: _ReportPDF, label: str,
                                res: dict, x: float) -> None:
        y = pdf.get_y()
        color = _PRED_COLORS.get(res["predicted_class"], _CYAN)
        pdf.set_fill_color(*_CARD)
        pdf.set_draw_color(*color)
        pdf.set_line_width(0.5)
        pdf.rect(x, y, 90, 30, "DF")

        pdf.set_xy(x + 3, y + 3)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*_GREY)
        pdf.cell(84, 4, label.upper(), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_xy(x + 3, y + 8)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*color)
        pdf.cell(84, 10, res["predicted_class"].upper(), align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_xy(x + 3, y + 20)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_GREY)
        pdf.cell(42, 5, f"Confidence: {res['confidence_pct']:.1f}%", align="C")
        pdf.cell(42, 5, f"Tier: {res['tier']}", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def _page_responsible_ai(self, pdf: _ReportPDF) -> None:
        pdf.add_page()
        pdf.section_title("Responsible AI - SHAP Explanations")

        if self.top_features:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*_CYAN)
            pdf.cell(65, 6, "Feature", border="B")
            pdf.cell(25, 6, "Direction", border="B")
            pdf.cell(0,  6, "SHAP Value", border="B",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            for fc in self.top_features:
                d = fc.get("direction", "")
                v = fc.get("shap_value", 0.0)
                color = _CYAN if v >= 0 else _PINK
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(*_WHITE)
                pdf.cell(65, 5.5, fc.get("label", fc.get("feature", "")))
                dir_c = _CYAN if d == "elevated" else _PINK
                pdf.set_text_color(*dir_c)
                pdf.cell(25, 5.5, d.upper())
                pdf.set_text_color(*color)
                pdf.cell(0, 5.5, f"{v:+.5f}",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*_GREY)
            pdf.cell(0, 6, "No SHAP explanation available.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(6)
        pdf.section_title("AI Transparency Narrative", _GREEN)
        pdf.set_fill_color(12, 7, 40)
        pdf.set_draw_color(*_CYAN)
        pdf.set_line_width(0.3)
        x, y = pdf.get_x(), pdf.get_y()
        h = max(20, len(self.narrative) // 6)
        pdf.rect(x, y, 190, h + 6, "DF")
        pdf.set_xy(x + 4, y + 3)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*_WHITE)
        pdf.multi_cell(182, 5, f'"{self.narrative}"')

    def _page_methodology(self, pdf: _ReportPDF) -> None:
        pdf.add_page()
        pdf.section_title("Methodology & Responsible AI Declaration")

        sections = [
            ("Dataset",
             "WESAD (Wearable Stress and Affect Detection) physiological dataset. "
             "Chest RespiBAN sensor @ 700 Hz. Subjects S2-S17 (S1 excluded). "
             "Labels: 1=baseline, 2=stress, 3=amusement."),
            ("Feature Engineering",
             "60-second non-overlapping windows. 9 features: EDA (mean, std, peak count, "
             "amplitude), HRV from ECG (heart rate, RMSSD, SDNN), Respiration (rate, variability). "
             "IQR outlier capping + median imputation + StandardScaler normalization."),
            ("ML Model",
             "RandomForestClassifier (300 trees, class_weight='balanced', random_state=42). "
             "Stratified 5-fold cross-validation. Two independent classifiers: "
             "Emotional Arousal and Cognitive Load."),
            ("Explainability",
             "SHAP (SHapley Additive exPlanations) TreeExplainer. Local per-sample "
             "SHAP values computed for the predicted class. Direction indicates whether "
             "each feature pushes the prediction higher (elevated) or lower (reduced)."),
            ("Privacy",
             "No personally identifiable information is stored at any stage. Session IDs "
             "are UUID4 (random, non-reversible). No subject mapping is maintained. "
             "All inference data is ephemeral."),
            ("Limitations & Disclaimer",
             "This platform is a research prototype for IEEE EMBS demonstration. "
             "Predictions should not be used for clinical decision-making without "
             "validation on independent cohorts. Results represent population-level "
             "physiological patterns from healthy university student participants."),
        ]

        for title, body in sections:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*_CYAN)
            pdf.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*_WHITE)
            pdf.multi_cell(0, 5, body)
            pdf.ln(2)
