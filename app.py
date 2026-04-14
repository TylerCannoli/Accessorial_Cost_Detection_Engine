"""PACE entrypoint — first sidebar item shows as PACE. Run: streamlit run app.py."""
import os
import base64
import streamlit as st
from auth_utils import check_auth, pace_role_is_admin
from utils.styling import remove_nav_toggle_fallback, inject_persistent_nav_hides


def _bg_css() -> str:
    """Handle bg css."""
    img_path = os.path.join(os.path.dirname(__file__), "assets", "background.png")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            f"background-image:url('data:image/png;base64,{b64}');"
            "background-size:cover;background-position:center;"
        )
    return "background:linear-gradient(155deg,#060012 0%,#09021a 40%,#06010f 100%);"


_bg_props = _bg_css()

st.set_page_config(
    page_title="PACE — Predictive Accessorial Cost Engine",
    page_icon="📦",
    layout="wide",
    # "expanded" pins session default so post-login pages show the nav (Streamlit uses first call per session).
    initial_sidebar_state="expanded",
)
remove_nav_toggle_fallback()
inject_persistent_nav_hides()

# — Inject Google Fonts via <link> — works on localhost where @import is blocked —
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown(f"""
<style>

@keyframes pace-in {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes ticker {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}
@keyframes pulse-glow {{
    0%, 100% {{ opacity: 0.45; }}
    50%       {{ opacity: 1; }}
}}
@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50%       {{ transform: translateY(-12px); }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}
@keyframes card-in {{
    from {{ opacity: 0; transform: translateY(30px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes eyebrow-in {{
    from {{ opacity: 0; transform: translateX(-16px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes stat-pop {{
    0%   {{ opacity: 0; transform: scale(0.85); }}
    60%  {{ transform: scale(1.04); }}
    100% {{ opacity: 1; transform: scale(1); }}
}}

*, *::before, *::after {{ box-sizing: border-box; }}

.stApp {{
    background: none;
    font-family: 'Outfit', sans-serif;
    animation: pace-in 0.7s ease-out;
}}
.stApp::before {{
    content: '';
    position: fixed; inset: -20px; z-index: -1;
    {_bg_props}
    filter: blur(3px);
}}
.stApp::after {{
    content: '';
    position: fixed; inset: 0;
    background-image: radial-gradient(rgba(147,51,234,0.045) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none; z-index: 0;
}}

#MainMenu, footer {{ visibility: hidden !important; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
#pace-nav-toggle-btn {{ display: none !important; }}
.block-container {{ position: relative; z-index: 1; padding: 0 !important; max-width: 100% !important; }}

/* Ambient glows */
.g1 {{
    position: fixed; top: -120px; left: -120px;
    width: 750px; height: 750px; border-radius: 50%;
    background: radial-gradient(circle, rgba(147,51,234,0.20) 0%, transparent 65%);
    pointer-events: none; z-index: 0; animation: pulse-glow 6s ease-in-out infinite;
}}
.g2 {{
    position: fixed; bottom: -120px; right: -120px;
    width: 650px; height: 650px; border-radius: 50%;
    background: radial-gradient(circle, rgba(194,24,91,0.17) 0%, transparent 65%);
    pointer-events: none; z-index: 0; animation: pulse-glow 8s ease-in-out infinite reverse;
}}
.g3 {{
    position: fixed; top: 40%; left: 40%;
    width: 450px; height: 450px; border-radius: 50%;
    background: radial-gradient(circle, rgba(147,51,234,0.07) 0%, transparent 65%);
    pointer-events: none; z-index: 0;
}}

/* — Nav — */
.pace-nav {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 22px clamp(1rem, 7vw, 56px);
    border-bottom: 1px solid rgba(147,51,234,0.22);
    position: relative; z-index: 10;
    background: rgba(6,0,18,0.45);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
}}
.pace-logo {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.75rem; letter-spacing: 0.28em; font-weight: 600;
    color: #ffffff;
    text-shadow: 0 0 24px rgba(255,255,255,0.25);
}}
.pace-nav-links {{ display: flex; align-items: center; gap: 36px; }}
.pace-nav-link {{
    font-family: 'Outfit', Arial, sans-serif;
    font-size: clamp(0.6rem, 2vw, 0.66rem); font-weight: 600; letter-spacing: 0.16em;
    text-transform: uppercase; color: #d8ccf0; text-decoration: none;
    min-height: 44px; display: flex; align-items: center;
    transition: color 0.2s;
}}
.pace-nav-link:hover {{ color: #ffffff; }}

/* — Ticker — */
.pace-ticker {{
    overflow: hidden; white-space: nowrap;
    padding: 11px 0;
    background: rgba(6,0,18,0.55);
    border-bottom: 1px solid rgba(147,51,234,0.14);
    position: relative; z-index: 5;
}}
.pace-ticker-inner {{ display: inline-block; animation: ticker 22s linear infinite; }}
.pace-ticker-item {{
    display: inline-flex; align-items: center; gap: 10px;
    font-family: 'Outfit', Arial, sans-serif;
    font-size: clamp(0.5rem, 1.8vw, 0.56rem); font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: #c9bce8; margin-right: 40px;
}}
.pace-ticker-item.lit {{ color: #e879f9; }}
.pace-ticker-dot {{ width: 3px; height: 3px; border-radius: 50%; background: #c084fc; flex-shrink: 0; }}

/* — Hero — */
.pace-hero {{
    display: grid; grid-template-columns: 1fr auto;
    align-items: start; gap: 36px;
    padding: clamp(32px, 7vw, 72px) clamp(1rem, 7vw, 56px) 0;
    position: relative; z-index: 5;
}}
.pace-eyebrow {{
    display: flex; align-items: center; gap: 14px;
    font-family: 'Outfit', Arial, sans-serif;
    font-size: clamp(0.5rem, 1.8vw, 0.56rem); font-weight: 700; letter-spacing: 0.26em;
    text-transform: uppercase; color: #e879f9; margin-bottom: 26px;
    animation: eyebrow-in 0.7s ease-out 0.2s both;
}}
.pace-eyebrow-line {{
    width: 34px; height: 2px;
    background: linear-gradient(90deg, #9333ea, #c2185b);
    box-shadow: 0 0 10px rgba(147,51,234,0.7);
    flex-shrink: 0;
}}
.pace-h1 {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(4.4rem, 9.5vw, 9rem);
    line-height: 0.9; letter-spacing: 0.02em; font-weight: 700;
    color: #f0eaff; margin: 0;
    text-shadow: 0 0 90px rgba(147,51,234,0.18);
    animation: pace-in 0.8s ease-out 0.35s both;
}}
.pace-h1-accent {{
    background: linear-gradient(120deg, #d8b4fe 0%, #9333ea 35%, #e879f9 65%, #c2185b 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: block;
    animation: shimmer 5s linear infinite;
    filter: drop-shadow(0 0 32px rgba(147,51,234,0.45));
}}
.pace-hero-desc {{
    font-family: 'Outfit', Arial, sans-serif;
    font-size: 16.5px; font-weight: 300; color: #c4b0e0;
    line-height: 1.85; max-width: 540px; margin-top: 30px;
    animation: pace-in 0.8s ease-out 0.5s both;
}}
.pace-hero-desc strong {{ color: #ede9fe; font-weight: 600; }}
.pace-hero-btns {{
    display: flex; align-items: center; gap: 14px; margin-top: 38px;
    animation: pace-in 0.8s ease-out 0.65s both;
}}
.pace-btn-outline {{
    font-family: 'Outfit', Arial, sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #e9d5ff;
    background: transparent;
    border: 1px solid rgba(192,132,252,0.5); border-radius: 3px;
    padding: 16px 38px; cursor: pointer; white-space: nowrap;
    transition: border-color 0.2s, background 0.2s, color 0.2s;
}}
.pace-btn-outline:hover {{
    border-color: #c084fc; background: rgba(147,51,234,0.1); color: #ffffff;
}}

/* — Hero Stats Card — */
.pace-stats-col {{
    display: flex; flex-direction: column; gap: 0;
    border: 1px solid rgba(147,51,234,0.2);
    border-radius: 8px; overflow: hidden;
    background: rgba(10,4,24,0.65);
    backdrop-filter: blur(16px);
    animation: float 6s ease-in-out infinite;
    min-width: 168px;
}}
.pace-stat:nth-child(1) {{ animation: stat-pop 0.6s ease-out 0.5s both; }}
.pace-stat:nth-child(2) {{ animation: stat-pop 0.6s ease-out 0.7s both; }}
.pace-stat:nth-child(3) {{ animation: stat-pop 0.6s ease-out 0.9s both; }}
.pace-stat {{
    padding: 24px 26px;
    border-bottom: 1px solid rgba(147,51,234,0.14);
    text-align: center;
}}
.pace-stat:last-child {{ border-bottom: none; }}
.pace-stat-num {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(2rem, 5vw, 3.4rem); letter-spacing: 0.04em; line-height: 1; font-weight: 600;
    background: linear-gradient(135deg, #f0e6ff, #e040fb, #c2185b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 14px rgba(147,51,234,0.45));
}}
.pace-stat-lbl {{
    font-family: 'Outfit', Arial, sans-serif;
    font-size: 8.5px; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #c9bce8; margin-top: 5px;
}}

/* — Divider — */
.pace-div {{
    height: 1px; margin: 52px clamp(1rem, 7vw, 56px) 0;
    background: linear-gradient(90deg, transparent, rgba(147,51,234,0.45), rgba(194,24,91,0.35), transparent);
    position: relative; z-index: 5;
}}

/* — Features — */
.pace-feat-label {{
    display: flex; align-items: center; gap: 14px;
    padding: 38px clamp(1rem, 7vw, 56px) 20px;
    font-family: 'Outfit', Arial, sans-serif;
    font-size: clamp(0.5rem, 1.8vw, 0.53rem); font-weight: 800; letter-spacing: 0.26em;
    text-transform: uppercase; color: #d8b4fe;
    position: relative; z-index: 5;
}}
.pace-feat-label::after {{
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(147,51,234,0.35), transparent);
}}
.pace-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1px; background: rgba(147,51,234,0.09);
    margin: 0 clamp(1rem, 7vw, 56px) 0;
    border: 1px solid rgba(147,51,234,0.15);
    border-radius: 8px; overflow: hidden;
    position: relative; z-index: 5;
}}
.pace-feat:nth-child(1) {{ animation: card-in 0.6s ease-out 0.2s both; }}
.pace-feat:nth-child(2) {{ animation: card-in 0.6s ease-out 0.35s both; }}
.pace-feat:nth-child(3) {{ animation: card-in 0.6s ease-out 0.5s both; }}
.pace-feat:nth-child(4) {{ animation: card-in 0.6s ease-out 0.65s both; }}
.pace-feat:nth-child(5) {{ animation: card-in 0.6s ease-out 0.8s both; }}
.pace-feat:nth-child(6) {{ animation: card-in 0.6s ease-out 0.95s both; }}
.pace-feat {{
    background: rgba(6,0,18,0.96); padding: 34px 30px;
    transition: background 0.25s, box-shadow 0.25s;
    position: relative; overflow: hidden;
}}
.pace-feat::before {{
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(147,51,234,0.08) 0%, transparent 60%);
    opacity: 0; transition: opacity 0.25s;
}}
.pace-feat:hover {{ background: rgba(14,6,34,0.99); box-shadow: inset 0 0 40px rgba(147,51,234,0.06); }}
.pace-feat:hover::before {{ opacity: 1; }}
.pace-feat-n {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 2.8rem; letter-spacing: 0.04em; line-height: 1; font-weight: 300;
    color: rgba(192,132,252,0.55); margin-bottom: 16px;
}}
.pace-feat-t {{
    font-family: 'Outfit', Arial, sans-serif;
    font-size: 15px; font-weight: 700; color: #f0eaff;
    margin-bottom: 10px; letter-spacing: 0.01em;
}}
.pace-feat-d {{
    font-family: 'Outfit', Arial, sans-serif;
    font-size: 13.5px; font-weight: 400; color: #ccc0e8; line-height: 1.75;
}}

/* Particle canvas */
#pace-particles {{
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
}}

/* — CTA button row — sits below hero, flush with hero padding — */
.pace-cta-row {{
    padding: 0 clamp(1rem, 7vw, 56px);
    margin-top: 36px;
    position: relative; z-index: 5;
    display: flex; align-items: center;
}}
/* Override Streamlit column gutters inside the CTA row */
.pace-cta-row [data-testid="stColumns"] {{
    gap: 0 !important;
    margin: 0 !important;
}}
.pace-cta-row [data-testid="column"]:nth-child(2) > div > div > div {{
    padding: 0 !important;
}}

/* — Footer — */
.pace-footer {{
    border-top: 1px solid rgba(147,51,234,0.12);
    padding: 24px clamp(1rem, 7vw, 56px); margin-top: 52px;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 12px;
    position: relative; z-index: 5;
    background: rgba(6,0,18,0.45);
}}
.pace-footer-l {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.25rem; letter-spacing: 0.22em; font-weight: 500; color: #ffffff;
}}
.pace-footer-r {{
    font-family: 'Outfit', Arial, sans-serif;
    font-size: 9px; color: rgba(210,195,240,0.65);
    letter-spacing: 0.1em; text-transform: uppercase;
}}

.stButton > button[kind="primary"] {{
    font-family: 'Outfit', Arial, sans-serif !important;
    background: linear-gradient(135deg, #9333EA, #C2185B) !important;
    color: #fff !important; border: none !important; border-radius: 3px !important;
    font-weight: 800 !important; font-size: 11px !important;
    letter-spacing: 0.16em !important; text-transform: uppercase !important;
    box-shadow: 0 0 34px rgba(147,51,234,0.55) !important;
    padding: 16px 38px !important; transition: all 0.2s !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 58px rgba(147,51,234,0.85) !important;
    transform: translateY(-2px) !important;
}}

/* ═══════════════════════════════════════════
   LANDING PAGE — MOBILE BREAKPOINTS
   ═══════════════════════════════════════════ */

/* Tablet (≤1024px): 2-col feature grid */
@media (max-width: 1024px) {{
    .pace-grid {{
        grid-template-columns: repeat(2, 1fr) !important;
    }}
}}

/* Mobile (≤768px): single-column everything */
@media (max-width: 768px) {{
    /* Scale down ambient glows so they don't overflow */
    .g1 {{ width: 400px !important; height: 400px !important; top: -100px !important; left: -100px !important; }}
    .g2 {{ width: 320px !important; height: 320px !important; bottom: -80px !important; right: -80px !important; }}
    .g3 {{ width: 260px !important; height: 260px !important; opacity: 0.6 !important; }}

    /* Hide desktop nav links — too small for touch */
    .pace-nav-links {{ display: none !important; }}

    /* Hero: stack text and stats vertically */
    .pace-hero {{
        grid-template-columns: 1fr !important;
        gap: 28px !important;
    }}
    .pace-stats-col {{
        flex-direction: row !important;
        min-width: unset !important;
        width: 100% !important;
        overflow-x: auto !important;
    }}
    .pace-stat {{
        flex: 1 !important;
        min-width: 90px !important;
        border-bottom: none !important;
        border-right: 1px solid rgba(147,51,234,0.14) !important;
        padding: 16px 14px !important;
    }}
    .pace-stat:last-child {{ border-right: none !important; }}

    /* Hero description: slightly smaller */
    .pace-hero-desc {{
        font-size: 15px !important;
        margin-top: 20px !important;
    }}
    .pace-hero-btns {{ margin-top: 24px !important; }}
    .pace-btn-outline {{
        min-height: 44px !important;
        padding: 12px 24px !important;
    }}

    /* Feature grid: single column */
    .pace-grid {{
        grid-template-columns: 1fr !important;
        margin: 0 clamp(1rem, 5vw, 2rem) !important;
    }}
    .pace-feat {{ padding: 24px 20px !important; }}

    /* Footer: center-align on small screens */
    .pace-footer {{
        justify-content: center !important;
        text-align: center !important;
    }}
}}

/* Small mobile (≤480px): hide glows entirely */
@media (max-width: 480px) {{
    .g1, .g2, .g3 {{ display: none !important; }}
    .pace-ticker {{ display: none !important; }} /* ticker unreadable at this size */
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="g1"></div>
<div class="g2"></div>
<div class="g3"></div>

<nav class="pace-nav">
  <span class="pace-logo">P · A · C · E</span>
  <div class="pace-nav-links">
    <a class="pace-nav-link" href="#">Platform</a>
    <a class="pace-nav-link" href="#">Features</a>
    <a class="pace-nav-link" href="#">About</a>
  </div>
</nav>

<div class="pace-ticker">
  <div class="pace-ticker-inner">
    <span class="pace-ticker-item lit"><span class="pace-ticker-dot"></span>Risk Scoring</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Detention Fees</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Lumper Charges</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Layover Alerts</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Carrier Analysis</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Route Intelligence</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Cost Prediction</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Batch Upload</span>
    <span class="pace-ticker-item lit"><span class="pace-ticker-dot"></span>Risk Scoring</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Detention Fees</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Lumper Charges</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Layover Alerts</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Carrier Analysis</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Route Intelligence</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Cost Prediction</span>
    <span class="pace-ticker-item"><span class="pace-ticker-dot"></span>Batch Upload</span>
  </div>
</div>

<section class="pace-hero">
  <div>
    <div class="pace-eyebrow"><span class="pace-eyebrow-line"></span>Freight Intelligence Platform · 2026</div>
    <h1 class="pace-h1">
      STOP PAYING FOR
      <span class="pace-h1-accent">CHARGES YOU</span>
      NEVER SAW COMING
    </h1>
    <p class="pace-hero-desc">
      PACE transforms historical shipment data into <strong>calibrated ML risk scores</strong> —
      surfaced before dispatch, not after the invoice. Accessorial charges aren't random.
      They're <strong>entirely predictable</strong> with the right system.
    </p>
    <div class="pace-hero-btns">
      <button class="pace-btn-outline">SEE THE PLATFORM</button>
    </div>
  </div>
  <div class="pace-stats-col">
    <div class="pace-stat">
      <div class="pace-stat-num">95%</div>
      <div class="pace-stat-lbl">Confidence Interval</div>
    </div>
    <div class="pace-stat">
      <div class="pace-stat-num">7+</div>
      <div class="pace-stat-lbl">Charge Types</div>
    </div>
    <div class="pace-stat">
      <div class="pace-stat-num">ML</div>
      <div class="pace-stat-lbl">Risk Engine</div>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)

# ── GET STARTED button — rendered here so it sits below the hero ──────────────
st.markdown('<div class="pace-cta-row">', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    if st.button("GET STARTED →", type="primary", use_container_width=True, key="signin"):
        st.switch_page("pages/_Login.py")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pace-div"></div>
<div class="pace-feat-label">Platform capabilities</div>

<div class="pace-grid">
  <div class="pace-feat"><div class="pace-feat-n">01</div><div class="pace-feat-t">DOT Number Lookup</div><div class="pace-feat-d">Score any carrier instantly using live FMCSA data, SMS violations, crash history, and economic signals.</div></div>
  <div class="pace-feat"><div class="pace-feat-n">02</div><div class="pace-feat-t">Batch Upload & Scoring</div><div class="pace-feat-d">Upload CSV or Excel files to validate, clean, and score thousands of shipments at once.</div></div>
  <div class="pace-feat"><div class="pace-feat-n">03</div><div class="pace-feat-t">Risk Analytics Dashboard</div><div class="pace-feat-d">Interactive dashboards showing carrier risk tiers, accessorial trends, and route analysis.</div></div>
  <div class="pace-feat"><div class="pace-feat-n">04</div><div class="pace-feat-t">ML Cost Estimator</div><div class="pace-feat-d">Random Forest predictor with 95% confidence intervals. Compare against carrier averages.</div></div>
  <div class="pace-feat"><div class="pace-feat-n">05</div><div class="pace-feat-t">Route & Lane Analysis</div><div class="pace-feat-d">Lane-level cost and risk metrics with mileage distributions and cost-per-mile breakdowns.</div></div>
  <div class="pace-feat"><div class="pace-feat-n">06</div><div class="pace-feat-t">Admin Control Panel</div><div class="pace-feat-d">Role-gated user management, model retraining, version rollback, and auto-update config.</div></div>
</div>

<canvas id="pace-particles"></canvas>

<footer class="pace-footer">
  <span class="pace-footer-l">P · A · C · E</span>
  <span class="pace-footer-r">University of Arkansas · © 2026</span>
</footer>

<script>
(function() {
  const canvas = document.getElementById('pace-particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const COLORS = ['147,51,234', '192,132,252', '232,121,249', '194,24,91'];
  for (let i = 0; i < 55; i++) {
    particles.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.6 + 0.3,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha: Math.random() * 0.5 + 0.15,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color},${p.alpha})`;
      ctx.fill();
    });
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 130) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(147,51,234,${(1 - dist/130) * 0.12})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", unsafe_allow_html=True)

if check_auth():
    if not st.session_state.get("_data_preloaded"):
        st.session_state["post_load_dest"] = (
            "pages/8_Admin.py" if pace_role_is_admin() else "pages/0_Home.py"
        )
        st.switch_page("pages/_loading.py")
