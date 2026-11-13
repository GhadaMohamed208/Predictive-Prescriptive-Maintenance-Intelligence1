"""Design system + small reusable UI components (CSS, charts theme, badges, KPI cards)."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from . import config as C

BG, PANEL, BORDER = "#0a101c", "#101a2b", "#1e2d46"
TXT, MUTED = "#e6edf7", "#8da0bd"
CYAN, BLUE = "#22d3ee", "#3987e5"
# Categorical series colours in fixed order (dark-surface steps of the reference palette)
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#9085e9", "#e66767", "#d55181", "#008300"]

CSS = f"""
<style>
:root {{ --bg:{BG}; --panel:{PANEL}; --border:{BORDER}; --txt:{TXT}; --muted:{MUTED}; --cyan:{CYAN};
        --green:#34d399; --amber:#f59e0b; --red:#ef4444; }}
.stApp {{ background:
   linear-gradient(rgba(34,211,238,.035) 1px, transparent 1px) 0 0/44px 44px,
   linear-gradient(90deg, rgba(34,211,238,.035) 1px, transparent 1px) 0 0/44px 44px,
   radial-gradient(1200px 600px at 85% -10%, rgba(57,135,229,.16), transparent 60%), {BG}; color:{TXT}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:1.6rem; max-width:1500px; }}
section[data-testid="stSidebar"] {{ background:#080d17; border-right:1px solid {BORDER}; }}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p {{ color:{MUTED}; }}
h1,h2,h3,h4 {{ letter-spacing:.2px; color:{TXT}; }}
.pg-head {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; padding:2px 0 14px;
            border-bottom:1px solid {BORDER}; margin-bottom:18px; }}
.pg-head .t {{ font-size:1.65rem; font-weight:650; line-height:1.15; }}
.pg-head .s {{ color:{MUTED}; font-size:.92rem; margin-top:3px; max-width:900px; }}
.pg-head .tag {{ font:600 .68rem ui-monospace,Menlo,monospace; letter-spacing:1.6px; color:{CYAN};
                 border:1px solid rgba(34,211,238,.35); padding:3px 8px; border-radius:2px; white-space:nowrap; }}
.panel {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:4px; padding:14px 16px; }}
.panel h4 {{ margin:0 0 6px; font-size:.78rem; letter-spacing:1.4px; color:{MUTED}; text-transform:uppercase; font-weight:600; }}
.kpi {{ background:linear-gradient(180deg,#121e33,{PANEL}); border:1px solid {BORDER}; border-top:2px solid {CYAN};
        border-radius:3px; padding:12px 14px; height:100%; }}
.kpi .l {{ font-size:.7rem; letter-spacing:1.4px; text-transform:uppercase; color:{MUTED}; font-weight:600; }}
.kpi .v {{ font:600 1.9rem ui-monospace,Menlo,monospace; color:{TXT}; margin-top:3px; }}
.kpi .u {{ font-size:.8rem; color:{MUTED}; margin-left:5px; }}
.kpi .n {{ font-size:.78rem; color:{MUTED}; margin-top:2px; }}
.kpi.Healthy {{ border-top-color:var(--green); }} .kpi.Warning {{ border-top-color:var(--amber); }}
.kpi.Critical {{ border-top-color:var(--red); }}
.badge {{ display:inline-flex; align-items:center; gap:7px; font:700 .82rem ui-monospace,Menlo,monospace; letter-spacing:1.2px;
          padding:5px 11px; border-radius:2px; border:1px solid; text-transform:uppercase; }}
.badge.Healthy {{ color:var(--green); border-color:rgba(52,211,153,.5); background:rgba(52,211,153,.08); }}
.badge.Warning {{ color:var(--amber); border-color:rgba(245,158,11,.55); background:rgba(245,158,11,.09); }}
.badge.Critical {{ color:var(--red); border-color:rgba(239,68,68,.6); background:rgba(239,68,68,.11); }}
.banner {{ border-left:4px solid; padding:12px 16px; border-radius:2px; margin:6px 0 12px; background:#0d1626; }}
.banner.Healthy {{ border-color:var(--green); }} .banner.Warning {{ border-color:var(--amber); }}
.banner.Critical {{ border-color:var(--red); background:rgba(239,68,68,.08); }}
.note {{ border:1px dashed #2a3c5e; color:{MUTED}; padding:9px 12px; font-size:.84rem; border-radius:3px; background:#0c1424; }}
.warnbox {{ border:1px solid rgba(245,158,11,.5); color:#f3c777; padding:9px 12px; font-size:.84rem; border-radius:3px;
            background:rgba(245,158,11,.06); }}
.stage {{ font:600 .72rem ui-monospace,Menlo,monospace; letter-spacing:1.5px; color:{CYAN}; text-transform:uppercase; margin:16px 0 6px; }}
.reveal {{ animation: rv .5s ease both; }}
@keyframes rv {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:none; }} }}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {{
   background:linear-gradient(180deg,#1aa7c4,#0e7f99); border:1px solid #22d3ee; color:#04121a; font-weight:700; letter-spacing:.6px; }}
.stButton>button, .stDownloadButton>button {{ border-radius:3px; }}
div[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:3px; background:#0d1627; }}
button[data-baseweb="tab"] {{ font-weight:600; letter-spacing:.4px; }}
.hero {{ position:relative; border:1px solid {BORDER}; border-radius:4px; overflow:hidden; min-height:520px;
         background: radial-gradient(900px 500px at 78% 45%, rgba(34,211,238,.16), transparent 62%),
                     radial-gradient(700px 400px at 10% 0%, rgba(57,135,229,.18), transparent 60%), #070c16; }}
.hero .grid {{ position:absolute; inset:0; background:
      linear-gradient(rgba(34,211,238,.06) 1px, transparent 1px) 0 0/40px 40px,
      linear-gradient(90deg, rgba(34,211,238,.06) 1px, transparent 1px) 0 0/40px 40px;
      mask-image: linear-gradient(90deg, transparent, #000 40%); animation: drift 24s linear infinite; }}
@keyframes drift {{ to {{ background-position: 40px 40px, 40px 40px; }} }}
.hero .copy {{ position:relative; z-index:2; padding:46px 48px; max-width:640px; }}
.hero .eyebrow {{ font:600 .74rem ui-monospace,Menlo,monospace; letter-spacing:2.4px; color:{CYAN}; }}
.hero h1 {{ font-size:2.45rem; line-height:1.1; margin:12px 0 14px; font-weight:700; }}
.hero p {{ color:#b6c4dc; font-size:1.02rem; line-height:1.55; }}
.hero .fan {{ position:absolute; right:-70px; top:50%; width:560px; height:560px; margin-top:-280px; opacity:.95; }}
.hero .spin {{ transform-origin:300px 300px; animation: spin 14s linear infinite; }}
.hero .spin2 {{ transform-origin:300px 300px; animation: spin 40s linear infinite reverse; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.hero .sim {{ position:absolute; right:18px; bottom:12px; font:.68rem ui-monospace,monospace; color:#5f7396; z-index:2; }}
@media (prefers-reduced-motion: reduce) {{
  .hero .grid, .hero .spin, .hero .spin2, .reveal {{ animation:none !important; }}
}}
@media (max-width: 900px) {{ .hero .fan {{ opacity:.25; right:-200px; }} .hero .copy {{ padding:28px 22px; }}
  .hero h1 {{ font-size:1.8rem; }} }}
.dscard {{ border:1px solid {BORDER}; border-left:3px solid {CYAN}; background:#0c1424; border-radius:3px; padding:9px 11px; }}
.dscard.ds-up {{ border-left-color:#c98500; }}
.dscard .b {{ font:700 .66rem ui-monospace,Menlo,monospace; letter-spacing:1.4px; color:{CYAN}; }}
.dscard.ds-up .b {{ color:#f3c777; }}
.dscard .n {{ font-size:.86rem; color:{TXT}; margin:3px 0 2px; word-break:break-all; }}
.dscard .m {{ font-size:.72rem; color:{MUTED}; }}
.chip {{ display:inline-block; font:600 .72rem ui-monospace,Menlo,monospace; letter-spacing:.8px; padding:2px 8px; border-radius:2px;
         border:1px solid; margin-right:6px; }}
.fc {{ border:1px solid {BORDER}; border-radius:3px; background:#0d1627; padding:10px 14px; margin:6px 0; }}
.fc .k {{ font-size:.68rem; letter-spacing:1.3px; text-transform:uppercase; color:{MUTED}; font-weight:600; margin-top:8px; }}
.fc .k:first-child {{ margin-top:0; }}
.fc .x {{ font-size:.92rem; color:{TXT}; line-height:1.45; }}
@media (max-width: 900px) {{ .kpi .v {{ font-size:1.5rem; }} .pg-head {{ flex-direction:column; align-items:flex-start; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", tag: str = "") -> None:
    st.markdown(f"<div class='pg-head'><div><div class='t'>{html.escape(title)}</div>"
                f"<div class='s'>{html.escape(subtitle)}</div></div>"
                f"{f'<div class=tag>{html.escape(tag)}</div>' if tag else ''}</div>", unsafe_allow_html=True)


def badge(state: str) -> str:
    if state in C.STATE_ICONS:
        return f"<span class='badge {state}'>{C.STATE_ICONS[state]} {html.escape(state)}</span>"
    col = {"Drift detected": C.STATE_COLORS["Warning"], "Stable": C.STATE_COLORS["Healthy"]}.get(state, MUTED)
    return f"<span class='badge' style='color:{col};border-color:{col}'>{html.escape(state)}</span>"


PRIORITY_COLORS = {"Critical": "#ef4444", "High": "#f59e0b", "Medium": "#e6c200", "Low": "#34d399", "Info": "#8da0bd"}


def chip(text: str, color: str = MUTED) -> str:
    return f"<span class='chip' style='color:{color};border-color:{color}'>{html.escape(text)}</span>"


def kpi(label: str, value: str, unit: str = "", note: str = "", cls: str = "") -> str:
    return (f"<div class='kpi {cls}'><div class='l'>{html.escape(label)}</div>"
            f"<div class='v'>{html.escape(str(value))}<span class='u'>{html.escape(unit)}</span></div>"
            f"<div class='n'>{html.escape(note)}</div></div>")


def note(text: str, warn: bool = False) -> None:
    st.markdown(f"<div class='{'warnbox' if warn else 'note'}'>{html.escape(text)}</div>", unsafe_allow_html=True)


def stage(label: str) -> None:
    st.markdown(f"<div class='stage'>{html.escape(label)}</div>", unsafe_allow_html=True)


def count_up(value: float, label: str, color: str = TXT, decimals: int = 1, animate: bool = True,
             height: int = 150) -> None:
    """Animated count-up to the *actual computed* value (static when animation is off or the
    viewer prefers reduced motion)."""
    anim = "true" if animate else "false"
    components.html(f"""
    <div style="font-family:ui-monospace,Menlo,monospace;color:{TXT};background:transparent">
      <div style="font-size:11px;letter-spacing:1.6px;color:{MUTED};text-transform:uppercase">{html.escape(label)}</div>
      <div id="v" style="font-size:64px;font-weight:700;color:{color};line-height:1.1">{value:.{decimals}f}</div>
      <div style="font-size:12px;color:{MUTED}">cycles · model estimate, not a failure date</div>
    </div>
    <script>
      const target={value:.6f}, dec={decimals}, el=document.getElementById('v');
      const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if({anim} && !reduce){{ const t0=performance.now(), dur=1100;
        function tick(t){{ const k=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-k,3);
          el.textContent=(target*e).toFixed(dec); if(k<1) requestAnimationFrame(tick); else el.textContent=target.toFixed(dec);}}
        requestAnimationFrame(tick); }}
    </script>""", height=height)


# ------------------------------------------------------------------ plotly theme
def style_fig(fig: go.Figure, title: str = "", height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TXT), x=0.01, xanchor="left"),
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1626",
        font=dict(color="#c4d1e6", size=12, family="system-ui, sans-serif"),
        margin=dict(l=55, r=20, t=55, b=45), showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0a101c", bordercolor="#2a3c5e", font=dict(color=TXT)),
        colorway=SERIES)
    fig.update_xaxes(gridcolor="#1a2740", zerolinecolor="#26385a", linecolor="#26385a", ticks="outside", tickcolor="#26385a")
    fig.update_yaxes(gridcolor="#1a2740", zerolinecolor="#26385a", linecolor="#26385a", ticks="outside", tickcolor="#26385a")
    return fig


def add_thresholds(fig: go.Figure, thr: dict, axis: str = "y") -> go.Figure:
    for name, val, col in (("Warning threshold", thr["warning"], C.STATE_COLORS["Warning"]),
                           ("Critical threshold", thr["critical"], C.STATE_COLORS["Critical"])):
        fig.add_hline(y=val, line=dict(color=col, width=1, dash="dash"),
                      annotation_text=f"{name} ({val:.0f})", annotation_font=dict(size=10, color=col),
                      annotation_position="top left")
    return fig


def hero_html(stats: list[tuple[str, str]]) -> str:
    blades = "".join(
        f"<path d='M300 300 L{300+20*__import__('math').cos(a)} {300+20*__import__('math').sin(a)} "
        f"L{300+235*__import__('math').cos(a+0.16)} {300+235*__import__('math').sin(a+0.16)} "
        f"A235 235 0 0 1 {300+235*__import__('math').cos(a+0.36)} {300+235*__import__('math').sin(a+0.36)} Z' "
        f"fill='rgba(57,135,229,.20)' stroke='rgba(34,211,238,.55)' stroke-width='1'/>"
        for a in [i * 2 * 3.14159265 / 22 for i in range(22)])
    rings = "".join(f"<circle cx='300' cy='300' r='{r}' fill='none' stroke='rgba(34,211,238,.{o})' stroke-width='1' stroke-dasharray='{d}'/>"
                    for r, o, d in ((262, 25, "2 8"), (280, 15, "1 5"), (125, 22, "4 6"), (60, 30, "none")))
    ticks = "".join(f"<line x1='300' y1='20' x2='300' y2='{30 if i % 5 else 38}' stroke='rgba(34,211,238,.45)' stroke-width='1' "
                    f"transform='rotate({i*6} 300 300)'/>" for i in range(60))
    svg = (f"<svg class='fan' viewBox='0 0 600 600' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>"
           f"<g class='spin2'>{ticks}</g>{rings}<g class='spin'>{blades}</g>"
           f"<circle cx='300' cy='300' r='34' fill='#0a101c' stroke='rgba(34,211,238,.8)' stroke-width='1.5'/>"
           f"<circle cx='300' cy='300' r='8' fill='rgba(34,211,238,.9)'/></svg>")
    chips = "".join(f"<div style='border:1px solid #1e2d46;background:rgba(10,16,28,.8);padding:8px 12px;border-radius:2px'>"
                    f"<div style='font:600 1.25rem ui-monospace,monospace;color:#e6edf7'>{html.escape(v)}</div>"
                    f"<div style='font-size:.68rem;letter-spacing:1.3px;color:#8da0bd;text-transform:uppercase'>{html.escape(k)}</div></div>"
                    for k, v in stats)
    return (f"<div class='hero'><div class='grid'></div>{svg}<div class='copy'>"
            f"<div class='eyebrow'>INDUSTRIAL AI · RELIABILITY ENGINEERING WORKSTATION</div>"
            f"<h1>Predictive &amp; Prescriptive Maintenance Intelligence</h1>"
            f"<p>Turn equipment sensor histories into condition assessments, degradation trends and evidence-based "
            f"inspection recommendations. Every number on screen is computed from the dataset you select; nothing is "
            f"loaded or filled in automatically.</p>"
            f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:22px'>{chips}</div></div>"
            f"<div class='sim'>Decorative illustration · not live telemetry</div></div>")
