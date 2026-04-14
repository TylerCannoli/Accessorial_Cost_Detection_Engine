"""
utils/styling.py
Dark glass theme — PACE Predictive Accessorial Cost Engine.
Purple-magenta background image, glass cards, glowing accents.
"""
import json
import os
import base64
import streamlit as st

from auth_utils import pace_role_is_admin

# Sidebar: hide flow-only pages (still routable via st.switch_page).
# Also hides the root "app" entry — app.py is the marketing landing page and
# immediately redirects authenticated users, so it does not belong in the nav.
_HIDE_FLOW_NAV_CSS = """
[data-testid="stSidebarNav"] a[href*="Login"],
[data-testid="stSidebarNav"] a[href*="login"],
[data-testid="stSidebarNav"] a[href*="_Login"],
[data-testid="stSidebarNav"] a[href*="loading"],
[data-testid="stSidebarNav"] a[href*="Loading"],
[data-testid="stSidebarNav"] a[href*="_loading"] {
    display: none !important;
}
[data-testid="stSidebarNav"] li:has(> a[href*="Login"]),
[data-testid="stSidebarNav"] li:has(> a[href*="login"]),
[data-testid="stSidebarNav"] li:has(> a[href*="_Login"]),
[data-testid="stSidebarNav"] li:has(> a[href*="loading"]),
[data-testid="stSidebarNav"] li:has(> a[href*="Loading"]),
[data-testid="stSidebarNav"] li:has(> a[href*="_loading"]) {
    display: none !important;
}

/* Hide the root app.py nav entry — authenticated users are immediately
   redirected away from it, so showing it as "app" in the sidebar is confusing. */
[data-testid="stSidebarNav"] li:first-child {
    display: none !important;
}
"""

# Non-admins: hide Admin in sidebar (8_Admin.py still enforces access).
# Match both casings — Streamlit hrefs may use 8_Admin or 8_admin depending on version.
_HIDE_ADMIN_NAV_CSS = """
[data-testid="stSidebarNav"] a[href*="Admin"],
[data-testid="stSidebarNav"] a[href*="admin"],
[data-testid="stSidebarNav"] li:has(> a[href*="Admin"]),
[data-testid="stSidebarNav"] li:has(> a[href*="admin"]) {
    display: none !important;
}
"""


def _bg_css() -> str:
    """Return background CSS props for the ::before blur layer."""
    _root = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(_root, "assets", "background.png")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            f"background-image: url('data:image/png;base64,{b64}');"
            "background-size: cover;"
            "background-position: center center;"
        )
    return (
        "background: "
        "radial-gradient(ellipse 65% 55% at 8% 62%, rgba(120,20,180,0.45) 0%, transparent 58%),"
        "radial-gradient(ellipse 55% 45% at 92% 18%, rgba(200,20,100,0.38) 0%, transparent 52%),"
        "linear-gradient(155deg, #060012 0%, #09021a 40%, #06010f 100%);"
    )


# ── Color tokens ──────────────────────────────────────────────────────────────
# Background / structure
DARK_BASE    = "#060012"
DARK_MID     = "#09021a"
GLASS_BG     = "rgba(12, 6, 30, 0.82)"
GLASS_BORDER = "rgba(180, 80, 220, 0.28)"
GLASS_GLOW   = "rgba(150, 50, 200, 0.18)"

# Accent colors
ACCENT_PURPLE = "#9333EA"
ACCENT_SOFT   = "#A78BFA"

# Text
TEXT_PRIMARY   = "#F1F5F9"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED     = "#64748B"

# Legacy color tokens — kept so existing pages don't break
NAVY_900 = "#0A0520"
NAVY_700 = "#1A0A40"
NAVY_500 = ACCENT_PURPLE    # purple (was blue)
NAVY_100 = "#2D1B4E"        # dark purple fill

# Risk tier colors — glowing on dark bg
RISK_HIGH_BG  = "rgba(220, 38, 38, 0.18)"
RISK_HIGH_FG  = "#F87171"
RISK_MED_BG   = "rgba(217, 119, 6, 0.18)"
RISK_MED_FG   = "#FCD34D"
RISK_LOW_BG   = "rgba(5, 150, 105, 0.18)"
RISK_LOW_FG   = "#34D399"

# Risk tier foreground colors (single hex — for text, borders, badges)
TIER_COLORS = {
    "Critical": "#F87171",
    "High":     "#FB923C",
    "Medium":   "#FCD34D",
    "Low":      "#34D399",
    "None":     "#94A3B8",
}

# Risk tier (bg, fg) pairs — for two-tone cell/card styling
TIER_BG_FG = {
    "Critical": ("#7F1D1D",   "#F87171"),
    "High":     (RISK_HIGH_BG, RISK_HIGH_FG),
    "Medium":   (RISK_MED_BG,  RISK_MED_FG),
    "Low":      (RISK_LOW_BG,  RISK_LOW_FG),
    "None":     ("#1E293B",   "#94A3B8"),
}

# Charge type foreground colors
CHARGE_COLORS = {
    "No Charge":            "#34D399",
    "Detention":            "#FCD34D",
    "Safety Surcharge":     "#FB923C",
    "Compliance Fee":       "#F87171",
    "Hazmat Fee":           "#C084FC",
    "High Risk / Multiple": "#EF4444",
}

# Chart theme defaults
CHART_BG      = "#0f0a1e"
CHART_GRID    = "rgba(150,50,200,0.18)"
CHART_AXIS    = "#A78BFA"

# Chart palette
CHART_PURPLE   = "#9333EA"
CHART_BLUE     = "#38BDF8"
CHART_RED      = "#EF4444"
CHART_BURGUNDY = "#9F1239"
CHART_LAVENDER = "#C4B5FD"


def chart_theme(**overrides) -> dict:
    """Dark-themed Plotly layout defaults. Merge with page-specific layout kwargs."""
    base = {
        "plot_bgcolor":  CHART_BG,
        "paper_bgcolor": "#0f0a1e",
        "font":   {"color": CHART_AXIS, "family": "Inter, Segoe UI, sans-serif"},
        "xaxis":  {"gridcolor": CHART_GRID, "color": CHART_AXIS,
                   "linecolor": "rgba(150,50,200,0.25)", "zerolinecolor": "rgba(150,50,200,0.2)"},
        "yaxis":  {"gridcolor": CHART_GRID, "color": CHART_AXIS,
                   "linecolor": "rgba(150,50,200,0.25)", "zerolinecolor": "rgba(150,50,200,0.2)"},
        "legend": {"bgcolor": "rgba(15,10,30,0.7)", "font": {"color": "#FFFFFF"},
                   "bordercolor": "rgba(150,50,200,0.3)", "borderwidth": 1},
    }
    base.update(overrides)
    return base


# ── Base page CSS (injected on every page) ────────────────────────────────────
_BASE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

@keyframes pace-in {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── App shell — no direct background ── */
.stApp {{
    background: none;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: {TEXT_PRIMARY};
    animation: pace-in 0.4s ease-out;
}}

/* ── Blurred background layer ── */
.stApp::before {{
    content: '';
    position: fixed;
    inset: -20px;
    z-index: -1;
    {_bg_css()}
    filter: blur(2px);
}}

/* ── Hide Streamlit chrome; keep sidebar expand/collapse control usable ── */
#MainMenu, footer {{ visibility: hidden !important; }}
/* Do not hide the whole header — it often contains [data-testid="collapsedControl"] */
[data-testid="stHeader"] {{
    background: transparent !important;
    visibility: visible !important;
}}
[data-testid="collapsedControl"] {{
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 1000002 !important;
}}
[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] [role="button"] {{
    color: {ACCENT_SOFT} !important;
    min-width: 2.5rem !important;
    min-height: 2.5rem !important;
}}
button[kind="headerNoPadding"] {{
    z-index: 1000003 !important;
    position: relative !important;
}}

/* ── Multipage sidebar — dark glass to match PACE theme ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(10, 5, 32, 0.98) 0%, rgba(6, 0, 18, 0.99) 100%) !important;
    border-right: 1px solid {GLASS_BORDER} !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35) !important;
}}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
    background: transparent !important;
}}
/* Inner scroll lives on stSidebarContent (Streamlit default). Tighten vertical rhythm so nav + footer fit without it. */
[data-testid="stSidebarContent"] {{
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    scrollbar-gutter: auto !important;
}}
[data-testid="stSidebarHeader"] {{
    margin-bottom: 0.35rem !important;
    height: auto !important;
    min-height: 0 !important;
}}
[data-testid="stSidebarUserContent"] {{
    padding-top: 0.35rem !important;
}}
[data-testid="stSidebarUserContent"] hr {{
    margin: 0.35rem 0 !important;
}}
[data-testid="stSidebarUserContent"] [data-testid="stCaptionContainer"] {{
    margin-top: 0 !important;
    margin-bottom: 0.15rem !important;
}}
[data-testid="stSidebarUserContent"] .stButton {{
    margin-top: 0 !important;
}}
[data-testid="stSidebarNav"] ul {{
    padding-top: 0.15rem !important;
    padding-bottom: 0.1rem !important;
}}
[data-testid="stSidebarNav"] li {{
    margin: 0 !important;
    padding: 0 !important;
}}
/* Border-top on li+li: separators between links only, not under the last visible row. */
[data-testid="stSidebarNav"] li + li {{
    border-top: 1px solid {GLASS_BORDER} !important;
    padding-top: 0.35rem !important;
    margin-top: 0.35rem !important;
}}
[data-testid="stSidebarNav"] a {{
    color: {TEXT_SECONDARY} !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    font-size: 0.8125rem !important;
    line-height: 1.2 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 0.28rem 0.45rem !important;
    gap: 0.35rem !important;
}}
[data-testid="stSidebarNav"] a:hover {{
    color: {TEXT_PRIMARY} !important;
    background: rgba(147, 51, 234, 0.12) !important;
}}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    color: #FFFFFF !important;
    background: rgba(147, 51, 234, 0.22) !important;
    border-left: 3px solid {ACCENT_PURPLE} !important;
}}
[data-testid="stSidebarNavSeparator"] {{
    display: none !important;
}}
/* Force-hide Streamlit's "View more / View less" nav toggle. */
[data-testid="stSidebarNavViewButton"] {{
    display: none !important;
}}
@media (max-height: 680px) {{
    [data-testid="stSidebarContent"] {{
        overflow-y: auto !important;
    }}
}}
/* ── Page content padding ── */
.block-container {{
    padding-top: 1rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1400px !important;
}}

/* ── Metric Cards ── */
[data-testid="stMetric"] {{
    background: rgba(12, 6, 30, 0.82) !important;
    border: 1px solid rgba(180, 80, 220, 0.28) !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    box-shadow: 0 0 18px rgba(150,50,200,0.12), 0 4px 16px rgba(0,0,0,0.35) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}}
[data-testid="stMetricLabel"] > div {{
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    color: {ACCENT_SOFT} !important;
    text-transform: uppercase !important;
}}
[data-testid="stMetricValue"] > div {{
    font-size: 26px !important;
    font-weight: 700 !important;
    color: {TEXT_PRIMARY} !important;
}}
[data-testid="stMetricDelta"] > div {{
    font-size: 12px !important;
    color: {TEXT_SECONDARY} !important;
}}

/* ── Buttons ── */
.stButton > button[kind="primary"] {{
    background-color: {NAVY_900} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 28px rgba(147,51,234,0.65) !important;
    background: linear-gradient(135deg, #A855F7, #E91E8C) !important;
}}
.stButton > button:not([kind="primary"]) {{
    background: rgba(30,10,60,0.7) !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid rgba(180,80,220,0.35) !important;
    border-radius: 8px !important;
}}
.stButton > button:not([kind="primary"]):hover {{
    background: rgba(60,20,100,0.7) !important;
    border-color: rgba(224,64,251,0.55) !important;
}}

/* ── Headings ── */
h1 {{ color: #FFFFFF !important; font-weight: 700 !important; text-shadow: 0 0 30px rgba(180,80,220,0.4); }}
h2 {{ color: #F1F5F9 !important; font-weight: 600 !important; }}
h3 {{ color: #E2E8F0 !important; font-weight: 600 !important; }}
h4, h5, h6 {{ color: #CBD5E1 !important; font-weight: 600 !important; }}
p, .stMarkdown p {{ color: #CBD5E1 !important; }}
strong, b {{ color: #F1F5F9 !important; }}
.stCaption, [data-testid="stCaptionContainer"] p {{ color: #94A3B8 !important; font-size: 13px !important; }}
.stDivider hr {{ border-color: rgba(180,80,220,0.25) !important; }}

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {{
    background: transparent !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}
[data-testid="stPlotlyChart"] > div {{
    background: transparent !important;
}}

/* ── Hide Plotly modebar ── */
.modebar-container {{ display: none !important; }}

/* ── Hide "Running..." status widget ── */
[data-testid="stStatusWidget"] {{ display: none !important; }}

/* ── Expand (⤢) buttons ── */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="baseButton-secondary"] {{
    background: rgba(20, 8, 50, 0.7) !important;
    border: 1px solid rgba(180,80,220,0.3) !important;
    border-radius: 10px !important;
    color: {ACCENT_SOFT} !important;
    font-size: 16px !important;
    padding: 6px 10px !important;
    line-height: 1 !important;
    box-shadow: 0 0 12px rgba(150,50,200,0.12) !important;
    backdrop-filter: blur(10px) !important;
    transition: box-shadow 0.2s, border-color 0.2s, color 0.2s !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="baseButton-secondary"]:hover {{
    background: rgba(30, 10, 70, 0.85) !important;
    border-color: rgba(180,80,220,0.6) !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 20px rgba(150,50,200,0.35) !important;
}}

/* ── Chart container hover elevation ── */
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stPlotlyChart"]) > div {{
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.3s ease !important;
    cursor: pointer;
}}
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stPlotlyChart"]) > div:hover {{
    transform: translateY(-6px) scale(1.012) !important;
    box-shadow: 0 20px 60px rgba(150,50,200,0.5),
                0 0 40px rgba(150,50,200,0.3),
                0 0 0 1px rgba(180,80,220,0.45) !important;
}}

/* ── Inputs & Forms ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-baseweb="input"],
[data-baseweb="select"] {{
    background: rgba(20,8,50,0.75) !important;
    border: 1px solid rgba(180,80,220,0.35) !important;
    border-radius: 8px !important;
    color: {TEXT_PRIMARY} !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {{
    border-color: {ACCENT_PURPLE} !important;
    box-shadow: 0 0 12px rgba(147,51,234,0.3) !important;
}}
[data-testid="stForm"] {{
    background: transparent !important;
    border: none !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid rgba(180,80,220,0.3) !important;
    border-radius: 0 !important;
    overflow: hidden !important;
}}
[data-testid="stDataFrame"] ::-webkit-scrollbar {{
    width: 6px !important;
    height: 6px !important;
}}
[data-testid="stDataFrame"] ::-webkit-scrollbar-track {{
    background: #0f0a1e !important;
}}
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {{
    background: rgba(147,51,234,0.5) !important;
    border-radius: 0 !important;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{ border-radius: 8px !important; }}

/* ══════════════════════════════════════════════════
   MOBILE RESPONSIVENESS — iOS & Android
   ══════════════════════════════════════════════════ */

/* ── Tablet (≤1024px): 2-column max for wide layouts ── */
@media (max-width: 1024px) {{
    .block-container {{
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }}
}}

/* ── Mobile (≤768px): single-column stacking ── */
@media (max-width: 768px) {{
    /* Reduce page-level padding */
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 0.75rem !important;
    }}

    /* Stack all st.columns() layouts vertically */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }}

    /* Compact metric cards */
    [data-testid="stMetric"] {{
        padding: 14px 16px !important;
    }}
    [data-testid="stMetricValue"] > div {{
        font-size: 20px !important;
    }}
    [data-testid="stMetricLabel"] > div {{
        font-size: 10px !important;
        letter-spacing: 0.5px !important;
    }}
    [data-testid="stMetricDelta"] > div {{
        font-size: 11px !important;
    }}

    /* Ensure minimum touch target size (44×44px) */
    .stButton > button {{
        min-height: 44px !important;
        padding: 10px 16px !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="baseButton-secondary"] {{
        min-width: 44px !important;
        min-height: 44px !important;
        padding: 10px 12px !important;
    }}

    /* Larger tap targets for nav links */
    [data-testid="stSidebarNav"] a {{
        padding: 0.55rem 0.65rem !important;
        font-size: 0.875rem !important;
    }}

    /* Dialogs — prevent overflow on narrow screens */
    [data-testid="modalDialog"],
    [data-testid="stDialog"] {{
        width: 95vw !important;
        max-width: 95vw !important;
    }}

    /* Sidebar: allow scroll on small screens */
    [data-testid="stSidebarContent"] {{
        overflow-y: auto !important;
    }}

    /* Inputs — minimum touch height */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {{
        min-height: 44px !important;
        font-size: 16px !important; /* prevents iOS auto-zoom */
    }}
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {{
        min-height: 44px !important;
    }}

    /* Reduce chart hover lift on touch to avoid stuck transforms */
    [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stPlotlyChart"]) > div:hover {{
        transform: none !important;
        box-shadow: 0 0 0 1px rgba(180,80,220,0.45) !important;
    }}

    /* Headings — clamp large text */
    h1 {{ font-size: clamp(1.4rem, 6vw, 2rem) !important; }}
    h2 {{ font-size: clamp(1.1rem, 5vw, 1.5rem) !important; }}
    h3 {{ font-size: clamp(1rem, 4.5vw, 1.25rem) !important; }}
}}

/* ── Small mobile (≤480px): tighten further ── */
@media (max-width: 480px) {{
    .block-container {{
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
    }}
    [data-testid="stMetricValue"] > div {{
        font-size: 18px !important;
    }}
    /* Caption text — keep readable */
    .stCaption, [data-testid="stCaptionContainer"] p {{
        font-size: 12px !important;
    }}
}}
</style>
"""


def inject_persistent_nav_hides() -> None:
    """
    Put sidebar hide rules in the parent <head> and watch the DOM.
    Survives Streamlit multipage transitions (Login/Loading always; Admin link hidden unless admin).
    Expands collapsed multipage nav (localStorage + programmatic click) and hides View more/less.
    """
    try:
        import streamlit.components.v1 as components
    except ImportError:
        return
    is_admin = pace_role_is_admin()
    css = _HIDE_FLOW_NAV_CSS + (_HIDE_ADMIN_NAV_CSS if not is_admin else "")
    css_literal = json.dumps(css)
    role_literal = json.dumps("admin" if is_admin else st.session_state.get("role"))
    components.html(
        f"""
<script>
(function () {{
  try {{
    var appWin = window.parent;
    var doc = appWin.document;
    if (!doc || !doc.head) return;
    /* Parent app window only — old iframes' MutationObservers must not read stale __paceUserRole. */
    appWin.__paceUserRole = {role_literal};
    var legacy = doc.getElementById("pace-hide-flow-nav");
    if (legacy) legacy.remove();

    var sid = "pace-sidebar-nav-filters";
    var stEl = doc.getElementById(sid);
    if (!stEl) {{
      stEl = doc.createElement("style");
      stEl.id = sid;
      doc.head.appendChild(stEl);
    }}
    stEl.textContent = {css_literal};

    function ensureSidebarNavExpanded() {{
      try {{
        if (window.localStorage)
          window.localStorage.setItem("sidebarNavState", "expanded");
      }} catch (e) {{}}
      var btn = doc.querySelector('[data-testid="stSidebarNavViewButton"]');
      if (!btn) return;
      var text = (btn.textContent || "").trim();
      if (text.indexOf("less") !== -1) return;
      if (text.indexOf("more") !== -1) btn.click();
    }}
    function hideSidebarViewToggle() {{
      var btn = doc.querySelector('[data-testid="stSidebarNavViewButton"]');
      if (btn) btn.style.setProperty("display", "none", "important");
    }}
    function hideFlowNavLinks() {{
      var nav = doc.querySelector('[data-testid="stSidebarNav"]');
      if (!nav) return;
      nav.querySelectorAll("a[href]").forEach(function (a) {{
        var h = a.getAttribute("href") || "";
        if (/Login|_Login|loading|_loading/i.test(h)) {{
          var li = a.closest("li");
          (li || a).style.setProperty("display", "none", "important");
        }}
      }});
    }}
    function paceIsAdminRole() {{
      return String(appWin.__paceUserRole || "").trim().toLowerCase() === "admin";
    }}
    function hideAdminNavLinks() {{
      if (paceIsAdminRole()) return;
      var nav = doc.querySelector('[data-testid="stSidebarNav"]');
      if (!nav) return;
      nav.querySelectorAll("a[href]").forEach(function (a) {{
        var h = a.getAttribute("href") || "";
        if (/Admin/i.test(h)) {{
          var li = a.closest("li");
          (li || a).style.setProperty("display", "none", "important");
        }}
      }});
    }}
    function applySidebarNavHiding() {{
      ensureSidebarNavExpanded();
      hideFlowNavLinks();
      hideAdminNavLinks();
      hideSidebarViewToggle();
    }}
    applySidebarNavHiding();
    if (!appWin.__paceSidebarNavObs) {{
      appWin.__paceSidebarNavObs = true;
      var root = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
      new MutationObserver(function () {{ applySidebarNavHiding(); }}).observe(root, {{
        subtree: true,
        childList: true,
      }});
    }}
  }} catch (e) {{}}
}})();
</script>
        """,
        height=0,
    )


def remove_nav_toggle_fallback() -> None:
    """Remove the ☰ fallback button (e.g. on landing/login where nav is intentionally hidden)."""
    try:
        import streamlit.components.v1 as components
    except ImportError:
        return
    components.html(
        """
<script>
try {
  var b = window.parent.document.getElementById("pace-nav-toggle-btn");
  if (b) b.remove();
} catch (e) {}
</script>
        """,
        height=0,
    )


def _inject_sidebar_open_fallback() -> None:
    """
    Streamlit's sidebar chevron is easy to lose behind custom CSS or new layouts.
    Inject a fixed ☰ into the parent document that forwards a click to the real toggle.
    Hidden on landing/login/loading via #pace-nav-toggle-btn {{ display: none }} there.
    """
    try:
        import streamlit.components.v1 as components
    except ImportError:
        return

    components.html(
        """
<script>
(function () {
  try {
    var doc = window.parent.document;
    if (!doc || !doc.body) return;
    if (doc.getElementById("pace-nav-toggle-btn")) return;

    var btn = doc.createElement("button");
    btn.id = "pace-nav-toggle-btn";
    btn.type = "button";
    btn.setAttribute("aria-label", "Open navigation menu");
    btn.innerHTML = "&#9776;";
    btn.style.cssText = [
      "position:fixed","left:10px","top:10px","z-index:2147483646",
      "width:44px","height:44px","border-radius:10px","cursor:pointer",
      "display:none","align-items:center","justify-content:center",
      "background:rgba(30,10,60,0.92)","color:#e9d5ff",
      "border:1px solid rgba(180,80,220,0.55)",
      "box-shadow:0 4px 20px rgba(0,0,0,0.45)",
      "font-size:20px","line-height:1","padding:0","font-family:system-ui,sans-serif"
    ].join(";");

    function clickNativeToggle() {
      var sels = [
        '[data-testid="collapsedControl"] button',
        '[data-testid="collapsedControl"] [role="button"]',
        '[data-testid="collapsedControl"]',
        'button[kind="headerNoPadding"]'
      ];
      for (var i = 0; i < sels.length; i++) {
        var el = doc.querySelector(sels[i]);
        if (el) { el.click(); return true; }
      }
      var hdr = doc.querySelector('[data-testid="stHeader"]');
      if (hdr) {
        var bs = hdr.querySelectorAll("button");
        for (var j = 0; j < bs.length; j++) {
          if (bs[j].offsetParent !== null) { bs[j].click(); return true; }
        }
      }
      return false;
    }

    function sidebarLooksCollapsed() {
      var side = doc.querySelector('[data-testid="stSidebar"]');
      if (!side) return true;
      var cs = window.getComputedStyle(side);
      if (cs.display === "none" || cs.visibility === "hidden") return true;
      var w = side.getBoundingClientRect().width;
      return w < 56;
    }

    function sync() {
      try {
        btn.style.display = sidebarLooksCollapsed() ? "flex" : "none";
      } catch (e) {}
    }

    btn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      clickNativeToggle();
      setTimeout(sync, 250);
    });

    doc.body.appendChild(btn);
    sync();
    var mo = new MutationObserver(function () { requestAnimationFrame(sync); });
    mo.observe(doc.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["style", "class"] });
    window.addEventListener("resize", sync);
  } catch (err) {}
})();
</script>
        """,
        height=0,
    )


def inject_css() -> None:
    """
    Inject PACE base CSS. Call at the top of every page.

    Uses Streamlit's default left sidebar for multipage navigation.
    Hides Login + Loading (flow-only pages; still work via st.switch_page).
    Hides the Admin page entry unless role is "admin" (same rule as pages/8_Admin.py).
    Keeps the full page list visible and hides Streamlit's "View more / View less" nav toggle.
    Compacts sidebar spacing and disables inner nav scrolling when the viewport is tall enough.
    """
    inject_persistent_nav_hides()
    admin_nav_css = _HIDE_ADMIN_NAV_CSS if not pace_role_is_admin() else ""
    st.markdown(
        _BASE_CSS.replace(
            "</style>",
            _HIDE_FLOW_NAV_CSS + admin_nav_css + "</style>",
        ),
        unsafe_allow_html=True,
    )
    _inject_sidebar_open_fallback()


def sidebar_account(username: str) -> None:
    """
    Sidebar footer below Streamlit's page list: account + sign-out.
    Call after inject_css() on authenticated pages.
    Admins get an explicit Admin link (multipage entry may be CSS-hidden before role sync).
    """
    with st.sidebar:
        st.caption(f"Signed in as **{username}**")
        if pace_role_is_admin():
            st.page_link(
                "pages/8_Admin.py",
                label="Admin panel",
                icon=":material/admin_panel_settings:",
                use_container_width=True,
            )
        if st.button("Sign out", key="pace_sidebar_signout", use_container_width=True):
            from auth_utils import logout

            logout()


def risk_badge_html(tier: str) -> str:
    """Return an HTML badge string for a risk tier label."""
    colors = {
        "High":   (RISK_HIGH_BG, RISK_HIGH_FG),
        "Medium": (RISK_MED_BG,  RISK_MED_FG),
        "Low":    (RISK_LOW_BG,  RISK_LOW_FG),
    }
    bg, fg = colors.get(tier, ("#F3F4F6", "#6B7280"))
    return (
        f'<span style="background:{bg}; color:{fg}; padding:3px 10px; '
        f'border-radius:4px; font-size:11px; font-weight:600;">{tier}</span>'
    )


# Keep for backwards compatibility — now a no-op
def sidebar_header(username: str) -> None:
    """Handle sidebar header."""
    pass
