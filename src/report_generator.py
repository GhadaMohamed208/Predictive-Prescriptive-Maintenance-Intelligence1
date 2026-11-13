"""Professional PDF engineering reports (ReportLab + matplotlib).

Only computed values/charts are included; sections without valid data are explicitly marked
unavailable. Colour scheme: black body text with blue headings on white (formal print style)."""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from . import config as C  # noqa: E402

NAVY = colors.HexColor("#0b2e59")
BLUE = colors.HexColor("#1f4e9c")
LIGHT = colors.HexColor("#e8eef8")
GRID = colors.HexColor("#b9c6dc")
STATE_PDF = {"Healthy": colors.HexColor("#1b7f4b"), "Warning": colors.HexColor("#b36b00"),
             "Critical": colors.HexColor("#b3261e")}
MPL_BLUE, MPL_DARK, MPL_GREY = "#1f4e9c", "#0b2e59", "#7a8699"


def _styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=ss["Normal"], fontName="Helvetica", fontSize=9, leading=12.5,
                          textColor=colors.black)
    return {
        "base": base,
        "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=colors.HexColor("#333333")),
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=NAVY),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                             textColor=BLUE, spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=9.5, textColor=NAVY),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8, leading=10.5),
        "cellb": ParagraphStyle("cellb", parent=base, fontSize=8, leading=10.5, fontName="Helvetica-Bold"),
        "center": ParagraphStyle("center", parent=base, alignment=TA_CENTER),
        "big": ParagraphStyle("big", parent=base, alignment=TA_CENTER, leading=27),
        "cellh": ParagraphStyle("cellh", parent=base, fontSize=8, leading=10.5, fontName="Helvetica-Bold", textColor=colors.white),
    }


def _esc(t) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fig_to_image(fig, width_mm=170) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    w, h = PILImage.open(buf).size
    buf.seek(0)
    return Image(buf, width=width_mm * mm, height=width_mm * mm * h / w)


def _style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=.25, lw=.6)
    ax.tick_params(labelsize=7)


def _table(data, col_widths, header=True, zebra=True, styles=None):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), .4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    if zebra:
        st += [("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT])]
    t.setStyle(TableStyle(st + (styles or [])))
    return t


def _on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 12 * mm, w, 12 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawString(15 * mm, h - 7.5 * mm, C.PRODUCT_SHORT.upper())
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(w - 15 * mm, h - 7.5 * mm, getattr(doc, "_report_title", "Engineering Report"))
    canvas.setStrokeColor(GRID)
    canvas.line(15 * mm, 14 * mm, w - 15 * mm, 14 * mm)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(15 * mm, 9.5 * mm, "Decision-support tool - not a certified safety system.")
    canvas.drawRightString(w - 15 * mm, 9.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _new_doc(buf, title):
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=19 * mm, bottomMargin=19 * mm, title=title, author=C.PRODUCT_SHORT)
    doc._report_title = title
    return doc


def _kv_table(rows, S, widths=(50 * mm, 130 * mm)):
    data = [[Paragraph(f"<b>{_esc(k)}</b>", S["cell"]), Paragraph(_esc(v), S["cell"])] for k, v in rows]
    return _table(data, list(widths), header=False)


def _unavailable(S, what, reason):
    return Paragraph(f"<i>{_esc(what)}: unavailable - {_esc(reason)}</i>", S["base"])


# ------------------------------------------------------------------ charts
def _chart_rul(info):
    h = info.get("rul_history")
    if h is None or len(h) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(h["cycle"], h["rul_pred"], color=MPL_BLUE, lw=1.6, label="Model RUL estimate")
    th = info["thresholds"]
    ax.axhline(th["warning"], color="#b36b00", ls="--", lw=1, label=f"Warning threshold ({th['warning']:.0f})")
    ax.axhline(th["critical"], color="#b3261e", ls="--", lw=1, label=f"Critical threshold ({th['critical']:.0f})")
    ax.set_xlabel("Operating cycle", fontsize=8); ax.set_ylabel("Estimated RUL (cycles)", fontsize=8)
    ax.set_title(f"Estimated RUL history - engine {info['engine_id']} (each point uses data up to that cycle)", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False); _style_ax(ax)
    return _fig_to_image(fig)


def _chart_sensors(info):
    d, sh = info.get("engine_df"), info.get("shifts")
    if d is None or not sh:
        return None
    top = [s["sensor"] for s in sh[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8), sharex=True)
    for ax, s in zip(axes.ravel(), top):
        ax.plot(d[C.CYCLE_COL], d[s], color=MPL_BLUE, lw=.9)
        ax.plot(d[C.CYCLE_COL], d[s].rolling(10, min_periods=1).mean(), color=MPL_DARK, lw=1.4)
        ax.set_title(C.sensor_label(s), fontsize=8); _style_ax(ax)
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    fig.supxlabel("Operating cycle", fontsize=8)
    fig.suptitle("Sensors with the largest shift vs. early baseline (thin: raw, thick: 10-cycle mean)", fontsize=8.5)
    fig.tight_layout()
    return _fig_to_image(fig)


def _chart_shap(info):
    loc = info.get("shap_local")
    if not loc:
        return None
    t = loc["by_sensor"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.barh(t["label"], t["shap"], color=[MPL_BLUE if v >= 0 else "#8aa4cf" for v in t["shap"]])
    ax.axvline(0, color="black", lw=.7)
    ax.set_xlabel("SHAP contribution to the RUL estimate (cycles)", fontsize=8)
    ax.set_title("Local SHAP: top sensors (grouped over each sensor's engineered features)", fontsize=8.5)
    _style_ax(ax)
    return _fig_to_image(fig, 150)


def _chart_anomaly(info):
    a = info.get("anomaly_scores")
    if a is None or len(a) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.plot(a[C.CYCLE_COL], a["anomaly_score"], color=MPL_BLUE, lw=1.2)
    ax.axhline(info["anomaly_threshold"], color="#b3261e", ls="--", lw=1, label="Anomaly threshold (99th pct. of reference)")
    fl = a[a["anomaly_flag"]]
    ax.scatter(fl[C.CYCLE_COL], fl["anomaly_score"], s=6, color="#b3261e", zorder=3)
    ax.set_xlabel("Operating cycle", fontsize=8); ax.set_ylabel("Isolation Forest score", fontsize=8)
    ax.set_title("Anomaly score (unsupervised; separate from the RUL model)", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False); _style_ax(ax)
    return _fig_to_image(fig)


# ------------------------------------------------------------------ engine report
def engine_report_pdf(info: dict) -> bytes:
    S = _styles()
    buf = io.BytesIO()
    title = f"Engine Report - {info['dataset_name']} / Engine {info['engine_id']}"
    doc = _new_doc(buf, title)
    el = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    state, rul = info["state"], info["rul"]
    sc = STATE_PDF[state]
    el += [Paragraph(C.PRODUCT_NAME, S["small"]), Spacer(1, 2 * mm),
           Paragraph("Engine Health &amp; Maintenance Decision-Support Report", S["h1"]), Spacer(1, 3 * mm)]
    el.append(_kv_table([
        ("Generated", now), ("Dataset", info["dataset_name"]), ("Engine identifier", info["engine_id"]),
        ("Last observed cycle", info["last_cycle"]), ("Input mode", info.get("input_mode", "Compatible engine trajectory")),
        ("Model", f"{info['model_meta']['model_name']} (version {info['model_meta']['model_version']})"),
    ], S))
    el.append(Spacer(1, 4 * mm))
    big = Table([[Paragraph(f"<font size=22 color='#0b2e59'><b>{rul:.1f}</b></font><br/><font size=8>estimated RUL (cycles)"
                            f"{' - at model cap' if info.get('at_cap') else ''}</font>", S["big"]),
                  Paragraph(f"<font size=18 color='{sc.hexval()}'><b>{_esc(state.upper())}</b></font><br/>"
                            f"<font size=8>derived state (Healthy &gt; {info['thresholds']['warning']:.0f}; "
                            f"Warning {info['thresholds']['critical']:.0f}-{info['thresholds']['warning']:.0f}; "
                            f"Critical &lt; {info['thresholds']['critical']:.0f})</font>", S["big"])]],
                colWidths=[85 * mm, 95 * mm])
    big.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .8, BLUE), ("INNERGRID", (0, 0), (-1, -1), .4, GRID),
                             ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("TOPPADDING", (0, 0), (-1, -1), 8),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    el.append(big)

    # 1 Executive summary (deterministic template)
    el.append(Paragraph("1. Executive summary", S["h2"]))
    parts = [f"At cycle {info['last_cycle']}, the model estimates a remaining useful life of {rul:.1f} cycles for engine "
             f"{info['engine_id']}, which maps to the derived state <b>{state}</b> under the configured thresholds."]
    if info.get("at_cap"):
        parts.append(f"The estimate sits at the {info['cap']:.0f}-cycle modelling cap, so the true remaining life may be longer.")
    if info.get("anomaly"):
        a = info["anomaly"]
        parts.append(f"The separate anomaly detector flagged {a['recent_flag_fraction']*100:.0f}% of the last {a['recent_n']} cycles.")
    if info.get("recs"):
        top = info["recs"][0]
        parts.append(f"Highest-priority recommendation: <b>{_esc(top['title'])}</b> ({top['priority']}).")
    if info.get("rul_true") is not None:
        parts.append(f"Official ground-truth RUL for this test engine is {info['rul_true']:.0f} cycles (raw); shown for evaluation only.")
    else:
        parts.append("No ground-truth RUL is available for this engine, so the estimate cannot be verified against a known value.")
    el.append(Paragraph(" ".join(parts), S["base"]))

    # 2 data quality
    el.append(Paragraph("2. Data quality and input compatibility", S["h2"]))
    dq = info.get("dq")
    if dq:
        el.append(Paragraph(f"Validation status: <b>{_esc(dq['status'])}</b>. {info.get('n_cycles', '')} cycles of history for this engine.", S["base"]))
        msgs = [("Error", m) for m in dq["errors"]] + [("Warning", m) for m in dq["warnings"]] + [("Info", m) for m in dq["info"]]
        if msgs:
            el.append(_table([[Paragraph("Severity", S["cellh"]), Paragraph("Message", S["cellh"])]] +
                             [[Paragraph(a, S["cell"]), Paragraph(_esc(b), S["cell"])] for a, b in msgs],
                             [22 * mm, 158 * mm]))
    else:
        el.append(_unavailable(S, "Validation report", "not provided"))
    if info.get("ood"):
        o = info["ood"]
        el.append(Paragraph(f"Out-of-distribution check: {o['n_outside']} of {len(info['model_meta']['feature_names'])} "
                            f"engineered features outside the training range ({o['fraction']*100:.0f}%)."
                            f"{' <b>Flag raised.</b>' if o.get('flag') else ''}", S["base"]))

    # 3 trajectory
    el.append(Paragraph("3. Estimated RUL history and sensor behaviour", S["h2"]))
    for fn, what in ((_chart_rul, "RUL history chart"), (_chart_sensors, "Sensor trend charts")):
        im = fn(info)
        el.append(im if im else _unavailable(S, what, "insufficient data"))
        el.append(Spacer(1, 2 * mm))
    if info.get("shifts"):
        rows = [[Paragraph(h, S["cellh"]) for h in ("Sensor", "Baseline mean", "Recent mean", "Shift (σ of early noise)", "Trend ρ")]]
        for s in info["shifts"][:8]:
            rows.append([Paragraph(s["label"], S["cell"]), f"{s['baseline_mean']:.4g}", f"{s['recent_mean']:.4g}",
                         f"{s['shift_sigma']:+.1f}", f"{s['trend_rho']:+.2f}"])
        el.append(_table(rows, [36 * mm, 34 * mm, 34 * mm, 44 * mm, 32 * mm]))

    # 4 explainability
    el.append(Paragraph("4. Explainability (SHAP)", S["h2"]))
    im = _chart_shap(info)
    if im:
        el.append(im)
        el.append(Paragraph(_esc(info["shap_text"]), S["base"]))
        el.append(Paragraph("Explanation computed for the latest cycle's engineered features. SHAP describes model attribution, "
                            "not physical causation.", S["small"]))
    else:
        el.append(_unavailable(S, "SHAP explanation", info.get("shap_reason", "explainer not available for this model")))

    # 5 anomaly
    el.append(Paragraph("5. Anomaly analysis (unsupervised, separate from the RUL model)", S["h2"]))
    im = _chart_anomaly(info)
    if im:
        el.append(im)
        el.append(Paragraph("An anomaly means the sensor pattern is unusual relative to early-life reference behaviour. It is not "
                            "automatically a failure; no fault labels exist to validate this detector.", S["small"]))
    else:
        el.append(_unavailable(S, "Anomaly scores", "detector unavailable"))

    # 6 recommendations
    el.append(Paragraph("6. Recommended inspection actions and supporting evidence", S["h2"]))
    if info.get("recs"):
        for r in info["recs"]:
            blk = [Paragraph(f"{_esc(r['id'])} - {_esc(r['title'])}", S["h3"]),
                   _kv_table([("Recommended action", r["action"]), ("Triggering condition", r["trigger"]),
                              ("Supporting evidence", r["evidence"]), ("Finding type", r["finding_type"]),
                              ("Priority / evidence strength", f"{r['priority']} / {r['evidence_strength']}"),
                              ("Rule source", r["source"]), ("Limitations", r["limitation"]),
                              ("Suggested verification", r["verification"]), ("Action status", r["status"])],
                             S, widths=(42 * mm, 138 * mm)), Spacer(1, 3 * mm)]
            el.append(KeepTogether(blk))
    else:
        el.append(_unavailable(S, "Recommendations", "no rule was triggered"))

    # 7 model & evaluation context
    el.append(Paragraph("7. Model and evaluation context (population metrics, not this engine)", S["h2"]))
    mm_ = info["model_meta"]
    met = info.get("metrics") or {}
    rows = [("Model", mm_["model_name"]), ("Feature set", mm_.get("feature_set", "n/a")),
            ("RUL cap (training/eval convention)", f"{mm_['rul_cap']} cycles"),
            ("Validation", "Engine-grouped 5-fold CV on the training file; official test set evaluated once at final cycles.")]
    cv, tc = met.get("internal_validation_cv_selected"), met.get("official_test_last_cycle_vs_capped_truth")
    if cv:
        rows.append(("Internal CV (all cycles)", f"MAE {cv['mae']:.2f}, RMSE {cv['rmse']:.2f} cycles, R² {cv['r2']:.3f}"))
    if tc:
        rows.append(("Official test (final cycles, capped truth)",
                     f"MAE {tc['mae']:.2f}, RMSE {tc['rmse']:.2f} cycles, R² {tc['r2']:.3f}, NASA score {tc['nasa_score']:.0f} (n={tc['n']})"))
    el.append(_kv_table(rows, S))

    # 8 limitations
    el.append(Paragraph("8. Limitations and required engineering verification", S["h2"]))
    for t in ("The data are simulated (NASA C-MAPSS FD001); results do not demonstrate performance on real equipment.",
              "RUL estimates are model outputs with measurable error and are capped; they are not failure dates.",
              "Derived states use configurable thresholds and are not NASA labels or industrial standards.",
              "Sensors are not mapped to physical components; no physical root cause is identified or implied.",
              "All recommendations require review and confirmation by an authorised maintenance engineer."):
        el.append(Paragraph("• " + t, S["base"]))
    el.append(Spacer(1, 3 * mm))
    el.append(Paragraph("<b>Disclaimer.</b> " + _esc(C.DISCLAIMER), S["small"]))
    doc.build(el, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


# ------------------------------------------------------------------ fleet report
def fleet_report_pdf(info: dict) -> bytes:
    S = _styles()
    buf = io.BytesIO()
    title = f"Fleet Summary - {info['dataset_name']}"
    doc = _new_doc(buf, title)
    t = info["table"]
    el = [Paragraph(C.PRODUCT_NAME, S["small"]), Spacer(1, 2 * mm),
          Paragraph("Fleet Summary Report", S["h1"]), Spacer(1, 3 * mm),
          _kv_table([("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")), ("Dataset", info["dataset_name"]),
                     ("Engines", len(t)), ("Model", f"{info['model_meta']['model_name']} (version {info['model_meta']['model_version']})"),
                     ("Thresholds", f"Healthy > {info['thresholds']['warning']:.0f}; Warning {info['thresholds']['critical']:.0f}-"
                                    f"{info['thresholds']['warning']:.0f}; Critical < {info['thresholds']['critical']:.0f} cycles"),
                     ("Scope", "Simulated engines of the supplied dataset; not a real operating fleet.")], S)]
    el.append(Paragraph("1. Executive summary", S["h2"]))
    cnt = info["counts"]
    el.append(Paragraph(f"Of {len(t)} engines, {cnt['Critical']} are Critical, {cnt['Warning']} Warning and {cnt['Healthy']} Healthy "
                        f"under the derived-state thresholds, based on predicted RUL at each engine's last observed cycle. "
                        f"Median predicted RUL is {t['rul_pred'].median():.1f} cycles.", S["base"]))
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.bar(C.STATES, [cnt[s] for s in C.STATES], color=[str(STATE_PDF[s].hexval()).replace("0x", "#") for s in C.STATES])
    ax.set_ylabel("Engines", fontsize=8); ax.set_title("Engines by derived state", fontsize=8.5); _style_ax(ax)
    el.append(_fig_to_image(fig, 120))
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.hist(t["rul_pred"], bins=20, color=MPL_BLUE, edgecolor="white")
    ax.set_xlabel("Predicted RUL (cycles)", fontsize=8); ax.set_ylabel("Engines", fontsize=8)
    ax.set_title("Distribution of predicted RUL", fontsize=8.5); _style_ax(ax)
    el.append(_fig_to_image(fig, 130))
    el.append(Paragraph("2. Engines requiring attention (documented ranking)", S["h2"]))
    el.append(Paragraph("Ranking: derived state (Critical first), then lower predicted RUL, then persistent anomaly flag. No cost or "
                        "downtime estimates are used.", S["small"]))
    rows = [[Paragraph(h, S["cellh"]) for h in ("Rank", "Engine", "Last cycle", "Pred. RUL", "State", "Anomaly (recent)")]]
    for _, r in t.head(20).iterrows():
        an = "n/a" if pd.isna(r.get("anomaly_recent_fraction")) else f"{r['anomaly_recent_fraction']*100:.0f}%"
        rows.append([int(r["attention_rank"]), int(r["unit"]), int(r["last_cycle"]), f"{r['rul_pred']:.1f}", r["state"], an])
    stl = [("TEXTCOLOR", (4, i), (4, i), STATE_PDF[rows[i][4]]) for i in range(1, len(rows))]
    el.append(_table(rows, [16 * mm, 22 * mm, 28 * mm, 28 * mm, 30 * mm, 36 * mm], styles=stl))
    met = info.get("metrics") or {}
    tc = met.get("official_test_last_cycle_vs_capped_truth")
    el.append(Paragraph("3. Model context", S["h2"]))
    if tc:
        el.append(Paragraph(f"Official test (final cycles, capped truth, n={tc['n']}): MAE {tc['mae']:.2f}, RMSE {tc['rmse']:.2f} cycles, "
                            f"R² {tc['r2']:.3f}. Fleet predictions above are model estimates and carry this typical error.", S["base"]))
    else:
        el.append(_unavailable(S, "Evaluation metrics", "not available"))
    el.append(Paragraph("4. Limitations", S["h2"]))
    el.append(Paragraph(_esc(C.DISCLAIMER), S["base"]))
    doc.build(el, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


# ================================================================== v2 reports (dataset-aware, plain language)
PRIO_PDF = {"Critical": "#b3261e", "High": "#b36b00", "Medium": "#8a7a00", "Low": "#1b7f4b", "Info": "#555555"}


def _stamp_rows(info: dict) -> list:
    return [("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Data source", info["source_badge"]), ("Dataset", info["dataset_name"]),
            ("Input mode", info["input_mode"]), ("Analysis level", info["capability_text"]),
            ("Dataset ID", info["dataset_token"]), ("Data type", "Historical data (not live telemetry)")]


def _chart_sensor_trends(info):
    d, sh = info.get("engine_df"), info.get("shifts")
    if d is None or not sh:
        return None
    labels = info.get("sensor_labels", {})
    top = [s["sensor"] for s in sh[:6]]
    n = len(top)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8), sharex=True)
    for ax, s in zip(axes.ravel(), top):
        ax.plot(d[C.CYCLE_COL], d[s], color=MPL_BLUE, lw=.9)
        ax.plot(d[C.CYCLE_COL], d[s].rolling(10, min_periods=1).mean(), color=MPL_DARK, lw=1.4)
        ax.set_title(labels.get(s, C.sensor_label(s)), fontsize=8); _style_ax(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.supxlabel("Cycle / record", fontsize=8)
    fig.suptitle("Sensors with the largest change from the asset's early readings (thin: raw, thick: 10-record average)", fontsize=8.5)
    fig.tight_layout()
    return _fig_to_image(fig)


def asset_report_pdf(info: dict) -> bytes:
    """Asset condition report. Uses only the active dataset; no model-development details."""
    S = _styles()
    buf = io.BytesIO()
    title = f"Asset Report - {info['asset_name']} - {info['dataset_name']}"
    doc = _new_doc(buf, title)
    ai = info["rul"] is not None
    el = [Paragraph(C.PRODUCT_NAME, S["small"]), Spacer(1, 2 * mm),
          Paragraph("Asset Condition &amp; Maintenance Report", S["h1"]), Spacer(1, 3 * mm)]
    el.append(_kv_table(_stamp_rows(info) + [("Asset", info["asset_name"]), ("Records analysed", f"{info['n_cycles']} (last cycle {info['last_cycle']})")], S))
    el.append(Spacer(1, 4 * mm))
    st_ = info["state"]
    scol = STATE_PDF.get(st_, colors.HexColor("#1f4e9c"))
    left = (f"<font size=22 color='#0b2e59'><b>{info['rul']:.0f}</b></font><br/><font size=8>estimated remaining life (cycles)"
            f"{' - at the model upper limit' if info.get('at_cap') else ''}</font>") if ai else \
        "<font size=11 color='#0b2e59'><b>Remaining life not available</b></font><br/><font size=8>no supported model for this data</font>"
    big = Table([[Paragraph(left, S["big"]),
                  Paragraph(f"<font size=16 color='{scol.hexval()}'><b>{_esc(str(st_).upper())}</b></font><br/><font size=8>condition at the last record</font>", S["big"])]],
                colWidths=[85 * mm, 95 * mm])
    big.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .8, BLUE), ("INNERGRID", (0, 0), (-1, -1), .4, GRID), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                             ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    el.append(big)

    el.append(Paragraph("1. Summary", S["h2"]))
    recs = info.get("recs", [])
    actionable = [r for r in recs if r["priority"] in ("Critical", "High", "Medium")]
    parts = [f"Asset {_esc(info['asset_name'])} was assessed on {info['n_cycles']} records from the dataset '{_esc(info['dataset_name'])}'."]
    if ai:
        parts.append(f"The model estimates about {info['rul']:.0f} cycles of remaining life; the resulting condition is <b>{_esc(st_)}</b>.")
    else:
        parts.append(f"Condition from sensor drift: <b>{_esc(st_)}</b>. Remaining life cannot be estimated for this data.")
    parts.append(f"{len(actionable)} finding(s) of medium or higher priority were raised." if actionable else "No finding of medium or higher priority was raised.")
    if info.get("rul_true") is not None:
        parts.append(f"A reference remaining-life value of {info['rul_true']:.0f} cycles was supplied with the data.")
    el.append(Paragraph(" ".join(parts), S["base"]))

    el.append(Paragraph("2. Data used", S["h2"]))
    dq = info.get("dq")
    if dq:
        el.append(Paragraph(f"Data validation: <b>{_esc(dq['status'])}</b>.", S["base"]))
        msgs = [("Warning", m) for m in dq["warnings"]] + [("Info", m) for m in dq["info"]]
        if msgs:
            el.append(_table([[Paragraph("Severity", S["cellh"]), Paragraph("Message", S["cellh"])]] +
                             [[Paragraph(a, S["cell"]), Paragraph(_esc(b), S["cell"])] for a, b in msgs], [22 * mm, 158 * mm]))
    else:
        el.append(Paragraph("Dataset-level validation was completed when the file was loaded.", S["base"]))

    el.append(Paragraph("3. Condition trend and sensor evidence", S["h2"]))
    if ai:
        h = info.get("rul_history")
        if h is not None and len(h):
            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.plot(h["cycle"], h["rul_pred"], color=MPL_BLUE, lw=1.6, label="Estimated remaining life")
            th = info["thresholds"]
            ax.axhline(th["warning"], color="#b36b00", ls="--", lw=1, label=f"Warning level ({th['warning']:.0f})")
            ax.axhline(th["critical"], color="#b3261e", ls="--", lw=1, label=f"Critical level ({th['critical']:.0f})")
            ax.set_xlabel("Cycle", fontsize=8); ax.set_ylabel("Cycles", fontsize=8)
            ax.set_title("Estimated remaining life over time (each point uses data up to that cycle)", fontsize=8.5)
            ax.legend(fontsize=7, frameon=False); _style_ax(ax)
            el.append(_fig_to_image(fig))
    else:
        el.append(_unavailable(S, "Remaining-life trend", "no supported remaining-life model applies to this dataset"))
    im = _chart_sensor_trends(info)
    el.append(im if im else _unavailable(S, "Sensor trend charts", "not enough records for a baseline comparison"))
    if info.get("shifts"):
        labels = info.get("sensor_labels", {})
        rows = [[Paragraph(h, S["cellh"]) for h in ("Sensor", "Early average", "Recent average", "Change (x normal noise)", "Direction")]]
        for s in info["shifts"][:8]:
            z = s["shift_sigma"]
            rows.append([Paragraph(_esc(labels.get(s["sensor"], s.get("label", s["sensor"]))), S["cell"]), f"{s['baseline_mean']:.4g}",
                         f"{s['recent_mean']:.4g}", "n/a" if pd.isna(z) else f"{z:+.1f}",
                         "n/a" if pd.isna(z) else ("increased" if z > 0 else "decreased")])
        el.append(_table(rows, [44 * mm, 32 * mm, 32 * mm, 40 * mm, 32 * mm]))
    a = info.get("anomaly")
    if a:
        el.append(Paragraph(f"Unusual-behaviour check: {a['recent_flag_fraction']*100:.0f}% of the last {a['recent_n']} records were flagged. "
                            "This is an observation relative to the asset type's early-life pattern, not a confirmed failure.", S["small"]))

    el.append(Paragraph("4. Findings and recommended actions", S["h2"]))
    if recs:
        for r in recs:
            ev = "; ".join(f"{x['label']} {x['direction']} by {abs(x['shift_sigma']):.1f}x normal noise" for x in r.get("sensor_evidence", [])) \
                or "No individual sensor singled out."
            blk = [Paragraph(f"{_esc(r['priority'])} priority - {_esc(r['condition'])}", S["h3"]),
                   _kv_table([("Finding type", r["finding_class"]), ("Why it was raised", r["evidence"]), ("Sensor evidence", ev),
                              ("Recommended action", r["action"]), ("How to verify", r["verification"]),
                              ("Confirmed fault?", r["confirmation"]), ("Action status", r["status"])], S, widths=(40 * mm, 140 * mm)),
                   Spacer(1, 3 * mm)]
            el.append(KeepTogether(blk))
    else:
        el.append(Paragraph("No finding was raised for this asset.", S["base"]))

    el.append(Paragraph("5. Maintenance action status", S["h2"]))
    if recs:
        rows = [[Paragraph(h, S["cellh"]) for h in ("Priority", "Action (short)", "Status")]]
        for r in recs:
            rows.append([r["priority"], Paragraph(_esc(r["title"]), S["cell"]), r["status"]])
        el.append(_table(rows, [24 * mm, 116 * mm, 40 * mm]))
    else:
        el.append(_unavailable(S, "Action list", "no recommendations"))

    el.append(Paragraph("6. Limitations", S["h2"]))
    for t in ("Findings are observations or model-based suspicions; none proves a physical fault. Only an inspection by an authorised engineer can confirm one.",
              "Condition and priority categories use configurable settings; they are not industry standards.",
              "The data are historical records of the selected dataset, not live telemetry." +
              (" The built-in dataset is simulated." if info["source_badge"].startswith("BUILT-IN") else ""),
              "Sensors are not mapped to physical components; no root cause is identified."):
        el.append(Paragraph("• " + _esc(t), S["base"]))
    el.append(Spacer(1, 3 * mm))
    el.append(Paragraph("<b>Disclaimer.</b> " + _esc(C.DISCLAIMER), S["small"]))
    doc.build(el, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


def summary_report_pdf(info: dict) -> bytes:
    """Fleet summary for the active dataset (same numbers as the Control Room / Fleet Overview)."""
    S = _styles()
    buf = io.BytesIO()
    doc = _new_doc(buf, f"Fleet Summary - {info['dataset_name']}")
    t, F = info["table"], info["findings"]
    order, colr = info["status_order"], info["status_colors"]
    cnt = info["counts"]
    ai = info["ai"]
    el = [Paragraph(C.PRODUCT_NAME, S["small"]), Spacer(1, 2 * mm), Paragraph("Fleet Summary Report", S["h1"]), Spacer(1, 3 * mm),
          _kv_table(_stamp_rows(info) + [("Assets", len(t))], S)]
    el.append(Paragraph("1. Summary", S["h2"]))
    txt = ", ".join(f"{cnt[s]} {s}" for s in order)
    msg = f"The dataset contains {len(t)} assets: {txt}. Condition is evaluated at each asset's last record."
    if ai:
        msg += f" Median estimated remaining life is {t['rul_pred'].median():.0f} cycles."
    msg += f" {info['kpis']['warnings']} finding(s) of medium or higher priority were raised; {info['kpis']['high_actions']} high-priority action(s) are open."
    el.append(Paragraph(_esc(msg), S["base"]))
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.bar(order, [cnt[s] for s in order], color=[colr[s] for s in order])
    ax.set_ylabel("Assets", fontsize=8); ax.set_title("Assets by condition", fontsize=8.5); _style_ax(ax)
    el.append(_fig_to_image(fig, 120))
    pc = t["priority"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 2.2))
    pr = ["Critical", "High", "Medium", "Low"]
    ax.bar(pr, [int(pc.get(x, 0)) for x in pr], color=[PRIO_PDF[x] for x in pr])
    ax.set_ylabel("Assets", fontsize=8); ax.set_title("Assets by maintenance priority", fontsize=8.5); _style_ax(ax)
    el.append(_fig_to_image(fig, 120))
    el.append(Paragraph("2. Assets requiring attention", S["h2"]))
    el.append(Paragraph("Order: condition (most severe first), then lower remaining life or larger sensor drift.", S["small"]))
    hdr = ("Rank", "Asset", "Last cycle", "Condition", "Priority") + (("Est. remaining life",) if ai else ("Largest drift",))
    rows = [[Paragraph(h, S["cellh"]) for h in hdr]]
    for _, r in t.head(20).iterrows():
        last = f"{r['rul_pred']:.0f}" if ai else (r["top_sensor"] if r["top_sensor"] != "—" else "n/a")
        rows.append([int(r["attention_rank"]), Paragraph(_esc(r["name"]), S["cell"]), int(r["last_cycle"]), r["state"], r["priority"], Paragraph(_esc(last), S["cell"])])
    el.append(_table(rows, [14 * mm, 38 * mm, 24 * mm, 36 * mm, 24 * mm, 44 * mm]))
    el.append(Paragraph("3. Main findings", S["h2"]))
    if len(F):
        rows = [[Paragraph(h, S["cellh"]) for h in ("Priority", "Asset", "Finding", "Type")]]
        for _, r in F[F["priority"] != "Low"].head(25).iterrows():
            rows.append([r["priority"], Paragraph(_esc(r["name"]), S["cell"]), Paragraph(_esc(r["condition"]), S["cell"]), Paragraph(_esc(r["finding_class"]), S["cell"])])
        el.append(_table(rows, [20 * mm, 32 * mm, 74 * mm, 54 * mm]) if len(rows) > 1 else Paragraph("No findings above low priority.", S["base"]))
    else:
        el.append(Paragraph("No findings were raised.", S["base"]))
    el.append(Paragraph("4. Limitations", S["h2"]))
    el.append(Paragraph(_esc(C.DISCLAIMER), S["base"]))
    doc.build(el, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()
