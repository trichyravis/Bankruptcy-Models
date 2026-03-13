
"""
═══════════════════════════════════════════════════════════════════════════════
  BANKRUPTCY PROBABILITY & COST MODEL
  The Mountain Path – World of Finance
  Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking Experience
  Models: Altman Z-Score · Ohlson O-Score · Zmijewski · Merton KMV
          Direct & Indirect Bankruptcy Cost Estimation
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.special import expit          # logistic / sigmoid
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Bankruptcy Model | The Mountain Path",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DESIGN CONSTANTS  (Mountain Path canonical palette)
# ============================================================================
COLORS = {
    'dark_blue':      '#003366',
    'medium_blue':    '#004d80',
    'light_blue':     '#ADD8E6',
    'accent_gold':    '#FFD700',
    'bg_dark':        '#0a1628',
    'card_bg':        '#112240',
    'text_primary':   '#e6f1ff',
    'text_secondary': '#8892b0',
    'text_dark':      '#1a1a2e',
    'success':        '#28a745',
    'danger':         '#dc3545',
}
BRANDING = {
    'name':        'The Mountain Path - World of Finance',
    'instructor':  'Prof. V. Ravichandran',
    'credentials': '28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence',
    'institutions':'Visiting Faculty: NMIMS Bangalore · BITS Pilani · RV University · Goa Institute of Management',
    'icon':        '🏔️',
    'linkedin':    'https://www.linkedin.com/in/trichyravis',
    'github':      'https://github.com/trichyravis',
}

C  = COLORS
DB = C['dark_blue']; MB = C['medium_blue']; LB = C['light_blue']
GD = C['accent_gold']; BG = C['bg_dark']; CB = C['card_bg']
TP = C['text_primary']; TS = C['text_secondary']
GR = C['success']; RD = C['danger']

# ============================================================================
# GLOBAL CSS
# ============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+Pro:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {{
    background: linear-gradient(135deg, #1a2332 0%, #243447 50%, #2a3f5f 100%);
}}
.main, .main *, .main p, .main span, .main div, .main li, .main label {{
    color: {TP} !important;
}}
.main h1,.main h2,.main h3,.main h4,.main h5,.main h6 {{
    color: {GD} !important;
    font-family: 'Playfair Display', serif;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BG} 0%, {DB} 100%);
    border-right: 1px solid rgba(255,215,0,0.25);
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div[class*="markdown"] {{
    color: {TP} !important; -webkit-text-fill-color: {TP} !important;
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: {LB} !important; -webkit-text-fill-color: {LB} !important;
    font-weight: 600 !important; opacity:1 !important;
}}
section[data-testid="stSidebar"] input {{
    color: {C['text_dark']} !important; background-color: #fff !important;
}}

/* Header */
.header-container {{
    background: linear-gradient(135deg, {DB}, {MB});
    border: 2px solid {GD}; border-radius: 12px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem; text-align: center;
}}
.header-container h1 {{
    font-family: 'Playfair Display', serif; color: {GD}; margin: 0; font-size: 2rem;
}}
.header-container p {{ color: {TP}; margin: 0.3rem 0 0; font-size: 0.9rem; }}

/* Metric card */
.metric-card {{
    background: {CB}; border: 1px solid rgba(255,215,0,0.3);
    border-radius: 10px; padding: 1.2rem; text-align: center; margin-bottom: 0.8rem;
}}
.metric-card .label {{
    color: {TS}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;
}}
.metric-card .value {{
    color: {GD}; font-size: 1.6rem; font-weight: 700;
    font-family: 'Playfair Display', serif; margin-top: 0.3rem;
}}
.metric-card .sub {{
    color: {TS}; font-size: 0.78rem; margin-top: 0.3rem;
}}

/* Section title */
.section-title {{
    font-family: 'Playfair Display', serif; color: {GD}; font-size: 1.3rem;
    border-bottom: 2px solid rgba(255,215,0,0.3);
    padding-bottom: 0.5rem; margin: 1.5rem 0 1rem;
}}

/* Info / formula box */
.info-box {{
    background: rgba(0,51,102,0.5); border: 1px solid {GD};
    border-radius: 8px; padding: 1rem 1.5rem; color: {TP}; margin: 0.8rem 0;
}}
.formula-box {{
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0,0,0,0.35); padding: 10px 14px;
    border-radius: 6px; border-left: 3px solid {GD};
    font-size: 0.85rem; color: {TP}; margin: 10px 0;
}}

/* Risk gauge bands */
.safe-band    {{ color: {GR};  font-weight:700; }}
.grey-band    {{ color: {GD};  font-weight:700; }}
.distress-band{{ color: {RD};  font-weight:700; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
    background: {CB}; border: 1px solid rgba(255,215,0,0.3);
    border-radius: 8px; color: {TP}; padding: 0.5rem 1rem;
}}
.stTabs [aria-selected="true"] {{
    background: {DB}; border: 2px solid {GD}; color: {GD};
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {MB}, {DB}) !important;
    color: {GD} !important; border: 2px solid {GD} !important;
    border-radius: 8px !important; font-weight: 600 !important;
    transition: all 0.3s ease !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {GD}, #d4af37) !important;
    color: {DB} !important; transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(255,215,0,0.4) !important;
}}

/* Alerts */
.stAlert {{ background-color: rgba(255,255,255,0.95) !important; }}
.stAlert p,.stAlert span,.stAlert div {{ color: {C['text_dark']} !important; }}

/* Expanders */
.streamlit-expanderHeader {{
    background: {CB} !important; border: 1px solid {GD} !important; border-radius:8px !important;
}}
.streamlit-expanderHeader p,.streamlit-expanderHeader span {{
    color: {GD} !important; font-weight:600 !important;
}}
details summary {{ color: {GD} !important; }}

footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER COMPONENTS
# ============================================================================
def header(title, subtitle=None, desc=None):
    s = f'<p style="font-size:1rem;color:{GD};font-weight:600;margin:0.5rem 0;">{subtitle}</p>' if subtitle else ""
    d = f'<p style="font-size:0.85rem;color:{TP};margin:0.3rem 0;">{desc}</p>' if desc else ""
    st.markdown(f"""
    <div class="header-container">
        <h1>{BRANDING['icon']} {title}</h1>{s}{d}
        <p>{BRANDING['name']}</p>
        <p style="font-size:0.8rem;color:{TS};">{BRANDING['instructor']} | {BRANDING['credentials']}</p>
    </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def mcard(label, value, sub=None, color=None):
    col = color or GD
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{col};-webkit-text-fill-color:{col};">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)

def ibox(content, title=None, border_color=None):
    bc = border_color or GD
    th = f"<h4 style='color:{GD};margin-top:0;'>{title}</h4>" if title else ""
    st.markdown(f'<div class="info-box" style="border-color:{bc};">{th}{content}</div>',
                unsafe_allow_html=True)

def formula(text):
    st.markdown(f'<div class="formula-box">{text}</div>', unsafe_allow_html=True)

def footer_bar():
    st.divider()
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem;">
        <p style="color:{GD};font-family:'Playfair Display',serif;font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">
            {BRANDING['icon']} {BRANDING['name']}</p>
        <p style="color:{TS};font-size:0.85rem;margin:0.3rem 0;">
            {BRANDING['instructor']} | {BRANDING['credentials']}</p>
        <p style="color:{TS};font-size:0.75rem;margin:0.2rem 0;">{BRANDING['institutions']}</p>
        <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,215,0,0.3);">
            <a href="{BRANDING['linkedin']}" target="_blank"
               style="color:{GD};text-decoration:none;margin:0 1rem;">🔗 LinkedIn</a>
            <a href="{BRANDING['github']}" target="_blank"
               style="color:{GD};text-decoration:none;margin:0 1rem;">💻 GitHub</a>
        </div>
    </div>""", unsafe_allow_html=True)

def plotly_layout_base(**kw):
    base = dict(paper_bgcolor=CB, plot_bgcolor=CB,
                font=dict(color=TP, family='Source Sans Pro'),
                margin=dict(l=60, r=40, t=50, b=50))
    base.update(kw)
    return base

def colorbar_cfg(title_text):
    return dict(title=dict(text=title_text, font=dict(color=GD)),
                tickfont=dict(color=TP), thickness=14, outlinewidth=0)

# ============================================================================
# BANKRUPTCY MODELS
# ============================================================================

# ── Altman Z-Score (Public Manufacturing) ──
def altman_z_public(wc_ta, re_ta, ebit_ta, mve_tl, sales_ta):
    z = 1.2*wc_ta + 1.4*re_ta + 3.3*ebit_ta + 0.6*mve_tl + 1.0*sales_ta
    if z > 2.99:   zone, prob, color = "Safe Zone",    max(0, min(5, (2.99-z)*(-10)))*0.01 + 0.02, GR
    elif z > 1.81: zone, prob, color = "Grey Zone",    0.30 + (2.99-z)/(2.99-1.81)*0.35, GD
    else:          zone, prob, color = "Distress Zone",min(0.95, 0.65 + (1.81-z)*0.15), RD
    return {"score": round(z,4), "zone": zone, "prob": round(prob,4), "color": color}

# ── Altman Z'-Score (Private firms) ──
def altman_z_private(wc_ta, re_ta, ebit_ta, bve_tl, sales_ta):
    z = 0.717*wc_ta + 0.847*re_ta + 3.107*ebit_ta + 0.420*bve_tl + 0.998*sales_ta
    if z > 2.9:    zone, prob, color = "Safe Zone",    0.02 + max(0,(2.9-z))*0.01, GR
    elif z > 1.23: zone, prob, color = "Grey Zone",    0.28 + (2.9-z)/(2.9-1.23)*0.38, GD
    else:          zone, prob, color = "Distress Zone",min(0.95, 0.66 + (1.23-z)*0.18), RD
    return {"score": round(z,4), "zone": zone, "prob": round(prob,4), "color": color}

# ── Ohlson O-Score → logit probability ──
def ohlson_o_score(total_assets, total_liabilities, working_capital,
                   current_liabilities, current_assets, net_income,
                   funds_from_ops, ebit, inta_deflator=1.0):
    # Normalise by GNP price-level index (use 1.0 for modern data)
    size   = np.log(max(total_assets / inta_deflator, 1e-6))
    tlta   = total_liabilities / max(total_assets, 1e-6)
    wcta   = working_capital   / max(total_assets, 1e-6)
    clca   = current_liabilities / max(current_assets, 1e-6)
    oeneg  = 1 if total_liabilities > total_assets else 0
    nita   = net_income / max(total_assets, 1e-6)
    futl   = funds_from_ops / max(total_liabilities, 1e-6)
    intwo  = 1 if net_income < 0 else 0          # simplified: 1 if NI < 0 in last 2 yrs
    chin   = 0.0   # change in NI — set 0 if prior year not available

    o = (-1.32 - 0.407*size + 6.03*tlta - 1.43*wcta + 0.076*clca
         - 1.72*oeneg - 2.37*nita - 1.83*futl + 0.285*intwo - 0.521*chin)
    prob = float(expit(o))
    if prob < 0.20:   zone, color = "Safe Zone",    GR
    elif prob < 0.50: zone, color = "Grey Zone",    GD
    else:             zone, color = "Distress Zone",RD
    return {"score": round(o,4), "prob": round(prob,4), "zone": zone, "color": color}

# ── Zmijewski Score (probit) ──
def zmijewski(roa, leverage, liquidity):
    # X = -4.336 - 4.513*ROA + 5.679*Leverage + 0.004*Liquidity
    x = -4.336 - 4.513*roa + 5.679*leverage + 0.004*liquidity
    prob = float(stats.norm.cdf(x))
    if prob < 0.20:   zone, color = "Safe Zone",    GR
    elif prob < 0.50: zone, color = "Grey Zone",    GD
    else:             zone, color = "Distress Zone",RD
    return {"score": round(x,4), "prob": round(prob,4), "zone": zone, "color": color}

# ── Merton KMV Distance-to-Default ──
def merton_kmv(equity_value, equity_vol, debt_face, risk_free, T=1.0,
               max_iter=100, tol=1e-6):
    """Iterative solution for asset value & asset volatility."""
    V = equity_value + debt_face   # initial guess
    sigma_V = equity_vol * equity_value / V

    for _ in range(max_iter):
        d1 = (np.log(V / debt_face) + (risk_free + 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
        d2 = d1 - sigma_V*np.sqrt(T)
        E_model  = V*stats.norm.cdf(d1) - debt_face*np.exp(-risk_free*T)*stats.norm.cdf(d2)
        sigma_E_model = (V/equity_value)*stats.norm.cdf(d1)*sigma_V
        V_new = equity_value + debt_face*np.exp(-risk_free*T)*stats.norm.cdf(-d2)
        sigma_V_new = equity_vol * equity_value / max(V_new, 1e-6)
        if abs(V_new - V) < tol and abs(sigma_V_new - sigma_V) < tol:
            break
        V, sigma_V = V_new, sigma_V_new

    d1 = (np.log(V / debt_face) + (risk_free + 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
    d2 = d1 - sigma_V*np.sqrt(T)
    dd = d2   # Distance-to-Default = d2 in Merton model
    prob_default = float(stats.norm.cdf(-dd))

    if prob_default < 0.10:  zone, color = "Safe Zone",    GR
    elif prob_default < 0.30: zone, color = "Grey Zone",   GD
    else:                     zone, color = "Distress Zone",RD

    return {
        "asset_value":   round(V, 2),
        "asset_vol":     round(sigma_V*100, 4),
        "d1": round(d1, 4), "d2": round(d2, 4),
        "dd":            round(dd, 4),
        "prob":          round(prob_default, 4),
        "zone": zone, "color": color,
    }

# ── Bankruptcy Cost Estimation ──
def bankruptcy_costs(total_assets, debt_ratio, revenue, ebitda,
                     industry_direct_pct, years_distress=2.0,
                     legal_pct=0.03, admin_pct=0.015):
    """
    Direct costs:   legal + administrative fees (% of assets)
    Indirect costs: lost sales, management distraction, customer/supplier flight
    """
    # Direct costs
    direct_legal  = total_assets * legal_pct
    direct_admin  = total_assets * admin_pct
    direct_other  = total_assets * (industry_direct_pct / 100)
    direct_total  = direct_legal + direct_admin + direct_other

    # Indirect costs
    lost_sales       = revenue * 0.08 * years_distress      # ~8% revenue loss per year
    mgmt_distraction = ebitda * 0.15 * years_distress       # 15% EBITDA productivity loss
    supplier_cost    = total_assets * 0.02                   # tighter trade credit
    customer_flight  = revenue * 0.05                        # customer defection
    indirect_total   = lost_sales + mgmt_distraction + supplier_cost + customer_flight

    total_cost  = direct_total + indirect_total
    pct_assets  = total_cost / max(total_assets, 1) * 100
    pct_revenue = total_cost / max(revenue, 1) * 100

    return {
        "direct_legal":   round(direct_legal, 2),
        "direct_admin":   round(direct_admin, 2),
        "direct_other":   round(direct_other, 2),
        "direct_total":   round(direct_total, 2),
        "lost_sales":     round(lost_sales, 2),
        "mgmt_dist":      round(mgmt_distraction, 2),
        "supplier_cost":  round(supplier_cost, 2),
        "customer_flight":round(customer_flight, 2),
        "indirect_total": round(indirect_total, 2),
        "total_cost":     round(total_cost, 2),
        "pct_assets":     round(pct_assets, 2),
        "pct_revenue":    round(pct_revenue, 2),
    }

# ── Probability gauge chart ──
def prob_gauge(prob, title="Default Probability", model=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 32, "color": GD,
                                         "family": "JetBrains Mono"}},
        title={"text": f"{title}<br><span style='font-size:0.8em;color:{TS}'>{model}</span>",
               "font": {"size": 14, "color": TP}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": TP, "tickfont": {"color": TP}},
            "bar":  {"color": RD if prob > 0.5 else GD if prob > 0.2 else GR,
                     "thickness": 0.35},
            "bgcolor": CB,
            "borderwidth": 1, "bordercolor": GD,
            "steps": [
                {"range": [0,  20],  "color": "rgba(40,167,69,0.15)"},
                {"range": [20, 50],  "color": "rgba(255,215,0,0.15)"},
                {"range": [50, 100], "color": "rgba(220,53,69,0.15)"},
            ],
            "threshold": {"line": {"color": RD, "width": 3},
                          "thickness": 0.75, "value": 50},
        }
    ))
    fig.update_layout(**plotly_layout_base(height=280,
                                           margin=dict(l=30,r=30,t=60,b=20)))
    return fig

# ============================================================================
# HEADER
# ============================================================================
header(
    "⚖️ Bankruptcy Probability & Cost Model",
    subtitle="Altman Z-Score · Ohlson O-Score · Zmijewski Probit · Merton KMV · Direct & Indirect Cost Estimation",
    desc="Enter financial statement data in the sidebar to run all four models simultaneously"
)

# ============================================================================
# SIDEBAR — INPUT PANEL
# ============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:0.8rem 0 1rem;
                border-bottom:1px solid rgba(255,215,0,0.3);margin-bottom:1rem;">
      <span style="font-family:'Playfair Display',serif;font-size:1.1rem;
                   color:{GD};font-weight:700;">
        {BRANDING['icon']} {BRANDING['name']}
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">🏭 Company Profile</div>', unsafe_allow_html=True)
    company_name = st.text_input("Company Name", value="ABC Industries Ltd.")
    firm_type    = st.selectbox("Firm Type", ["Public (Listed)", "Private (Unlisted)"])
    currency     = st.selectbox("Currency (₹ Cr / $ Mn)", ["₹ Crore", "$ Million", "€ Million"])

    st.markdown(f'<div class="section-title">📊 Balance Sheet Items</div>', unsafe_allow_html=True)
    total_assets       = st.number_input("Total Assets",             value=1000.0, min_value=1.0, step=10.0)
    total_liabilities  = st.number_input("Total Liabilities",        value=600.0,  min_value=0.0, step=10.0)
    current_assets     = st.number_input("Current Assets",           value=300.0,  min_value=0.0, step=5.0)
    current_liabilities= st.number_input("Current Liabilities",      value=200.0,  min_value=0.0, step=5.0)
    retained_earnings  = st.number_input("Retained Earnings",        value=120.0,  step=5.0)
    market_cap         = st.number_input("Market Cap / Book Equity",  value=500.0,  min_value=0.0, step=10.0)

    st.markdown(f'<div class="section-title">📈 Income & Cash Flow</div>', unsafe_allow_html=True)
    revenue       = st.number_input("Revenue / Net Sales",   value=800.0,  min_value=0.0, step=10.0)
    ebit          = st.number_input("EBIT",                  value=100.0,  step=5.0)
    ebitda        = st.number_input("EBITDA",                value=150.0,  step=5.0)
    net_income    = st.number_input("Net Income",            value=60.0,   step=5.0)
    funds_from_ops= st.number_input("Funds From Operations", value=80.0,   step=5.0)

    st.markdown(f'<div class="section-title">📉 Merton KMV Inputs</div>', unsafe_allow_html=True)
    equity_vol = st.slider("Equity Volatility (σ_E) %", 5.0, 120.0, 30.0, 0.5) / 100
    risk_free  = st.slider("Risk-Free Rate %",           2.0,  15.0,  6.5, 0.25) / 100
    debt_face  = st.number_input("Debt Face Value (Book)",value=600.0, min_value=0.0, step=10.0)

    st.markdown(f'<div class="section-title">💸 Bankruptcy Cost Inputs</div>', unsafe_allow_html=True)
    industry_direct = st.slider("Industry Direct Cost % of Assets", 0.5, 8.0, 2.5, 0.25)
    years_distress  = st.slider("Expected Years in Distress",        0.5, 5.0, 2.0, 0.25)

    st.divider()
    run_btn = st.button("▶  Run All Models", use_container_width=True)

# ============================================================================
# DERIVED RATIOS
# ============================================================================
working_capital  = current_assets - current_liabilities
equity_book      = total_assets - total_liabilities
roa              = net_income / max(total_assets, 1e-6)
leverage         = total_liabilities / max(total_assets, 1e-6)
liquidity        = current_assets / max(current_liabilities, 1e-6)

# Altman ratios
wc_ta   = working_capital   / max(total_assets, 1e-6)
re_ta   = retained_earnings / max(total_assets, 1e-6)
ebit_ta = ebit              / max(total_assets, 1e-6)
mve_tl  = market_cap        / max(total_liabilities, 1e-6)
bve_tl  = equity_book       / max(total_liabilities, 1e-6)
sal_ta  = revenue           / max(total_assets, 1e-6)

# ============================================================================
# RUN MODELS
# ============================================================================
if firm_type == "Public (Listed)":
    alt = altman_z_public(wc_ta, re_ta, ebit_ta, mve_tl, sal_ta)
    alt_label = "Altman Z-Score (Public)"
else:
    alt = altman_z_private(wc_ta, re_ta, ebit_ta, bve_tl, sal_ta)
    alt_label = "Altman Z'-Score (Private)"

ohl = ohlson_o_score(total_assets, total_liabilities, working_capital,
                     current_liabilities, current_assets, net_income, funds_from_ops, ebit)
zmi = zmijewski(roa, leverage, liquidity)
kmv = merton_kmv(market_cap, equity_vol, debt_face, risk_free)
bcc = bankruptcy_costs(total_assets, leverage, revenue, ebitda,
                       industry_direct, years_distress)

# Ensemble probability (simple average)
ensemble_prob = np.mean([alt["prob"], ohl["prob"], zmi["prob"], kmv["prob"]])

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Summary Dashboard",
    "🔢 Altman Z-Score",
    "📉 Ohlson O-Score",
    "📐 Zmijewski Probit",
    "🔬 Merton KMV",
    "💸 Bankruptcy Costs",
    "📖 Methodology",
])

# ══════════════════════════════════════════════════════════
#  TAB 1 — SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════
with tab1:
    sec("🎯 Ensemble Bankruptcy Assessment")

    # Top KPI strip
    cols = st.columns(5)
    kpis = [
        (alt_label,             f'{alt["score"]:.3f}',  alt["zone"],  alt["color"]),
        ("Ohlson O-Score",      f'{ohl["score"]:.3f}',  ohl["zone"],  ohl["color"]),
        ("Zmijewski Score",     f'{zmi["score"]:.3f}',  zmi["zone"],  zmi["color"]),
        ("Merton DD",           f'{kmv["dd"]:.3f}',     kmv["zone"],  kmv["color"]),
        ("Ensemble Prob",       f'{ensemble_prob*100:.1f}%',
         "HIGH RISK" if ensemble_prob>0.5 else "MODERATE" if ensemble_prob>0.2 else "LOW RISK",
         RD if ensemble_prob>0.5 else GD if ensemble_prob>0.2 else GR),
    ]
    for col, (lbl, val, zone, clr) in zip(cols, kpis):
        with col:
            mcard(lbl, val, sub=zone, color=clr)

    st.markdown("---")

    # Four gauges in one row
    sec("📡 Default Probability — All Models")
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(prob_gauge(alt["prob"], "Default Prob", alt_label),
                              use_container_width=True, key="gauge_alt_t1", key="pc_001")
    with g2: st.plotly_chart(prob_gauge(ohl["prob"], "Default Prob", "Ohlson O-Score"),
                              use_container_width=True, key="gauge_ohl_t1", key="pc_002")
    with g3: st.plotly_chart(prob_gauge(zmi["prob"], "Default Prob", "Zmijewski Probit"),
                              use_container_width=True, key="gauge_zmi_t1", key="pc_003")
    with g4: st.plotly_chart(prob_gauge(kmv["prob"], "Default Prob", "Merton KMV"),
                              use_container_width=True, key="gauge_kmv_t1", key="pc_004")

    st.markdown("---")
    sec("🌡 Ensemble Risk Assessment")

    # Ensemble gauge + comparison bar
    ec1, ec2 = st.columns([1, 2])
    with ec1:
        st.plotly_chart(prob_gauge(ensemble_prob, "Ensemble Probability", "Average of 4 Models"),
                        use_container_width=True, key="gauge_ens_t1", key="pc_005")

        risk_color = RD if ensemble_prob > 0.5 else GD if ensemble_prob > 0.2 else GR
        risk_label = ("🚨 HIGH RISK — Immediate attention required"
                      if ensemble_prob > 0.5 else
                      "⚠️ MODERATE RISK — Monitor closely"
                      if ensemble_prob > 0.2 else
                      "✅ LOW RISK — Financially sound")
        ibox(f"<p style='margin:0;font-weight:700;color:{risk_color};-webkit-text-fill-color:{risk_color};'>{risk_label}</p>",
             border_color=risk_color)

    with ec2:
        models = [alt_label, "Ohlson", "Zmijewski", "Merton KMV", "Ensemble"]
        probs  = [alt["prob"], ohl["prob"], zmi["prob"], kmv["prob"], ensemble_prob]
        bar_colors = [RD if p>0.5 else GD if p>0.2 else GR for p in probs]
        bar_colors[-1] = LB   # ensemble always light blue

        fig_cmp = go.Figure(go.Bar(
            x=models, y=[p*100 for p in probs],
            marker_color=bar_colors,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color=TP),
            width=0.55,
        ))
        fig_cmp.add_hline(y=50, line_dash="dash", line_color=RD,
                          annotation_text="50% threshold",
                          annotation_font_color=RD)
        fig_cmp.add_hline(y=20, line_dash="dot",  line_color=GD,
                          annotation_text="20% threshold",
                          annotation_font_color=GD)
        fig_cmp.update_layout(
            **plotly_layout_base(height=340),
            title=dict(text="Default Probability by Model (%)",
                       font=dict(color=GD, size=13), x=0.0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Probability (%)", ticksuffix="%",
                       gridcolor="rgba(255,255,255,0.05)", range=[0, 105]),
            showlegend=False,
        )
        st.plotly_chart(fig_cmp, use_container_width=True, key="bar_cmp_t1")

    st.markdown("---")
    sec(f"📋 Financial Ratio Summary — {company_name}")

    ratio_data = {
        "Ratio": ["Working Capital / Total Assets", "Retained Earnings / Total Assets",
                  "EBIT / Total Assets", "Market Value Equity / Total Liabilities",
                  "Revenue / Total Assets", "ROA (Net Income / Total Assets)",
                  "Leverage (Total Liabilities / Total Assets)",
                  "Current Ratio", "Debt Face / Market Cap"],
        "Value": [f"{wc_ta:.4f}", f"{re_ta:.4f}", f"{ebit_ta:.4f}",
                  f"{mve_tl:.4f}", f"{sal_ta:.4f}", f"{roa:.4f}",
                  f"{leverage:.4f}", f"{liquidity:.4f}",
                  f"{debt_face/max(market_cap,1):.4f}"],
        "Used In": ["All Z-Score models", "All Z-Score models", "All Z-Score models",
                    "Altman (Public)", "All Z-Score models", "Zmijewski",
                    "Zmijewski / Ohlson", "Zmijewski / Ohlson", "Merton KMV"],
    }
    st.dataframe(
        pd.DataFrame(ratio_data).style
        .set_properties(**{"font-family": "JetBrains Mono,monospace",
                           "font-size": "12px", "text-align": "center"})
        .set_table_styles([{"selector": "th",
                            "props": [("background-color", DB), ("color", GD),
                                      ("font-weight", "700"), ("font-size", "11px"),
                                      ("text-align", "center")]}]),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════
#  TAB 2 — ALTMAN Z-SCORE
# ══════════════════════════════════════════════════════════
with tab2:
    sec(f"🔢 {alt_label} — Detailed Breakdown")

    a1, a2 = st.columns([1, 2])
    with a1:
        mcard("Z-Score", f'{alt["score"]:.4f}', sub=alt["zone"], color=alt["color"])
        mcard("Default Probability", f'{alt["prob"]*100:.2f}%',
              sub="Mapped from score bands", color=alt["color"])
        st.plotly_chart(prob_gauge(alt["prob"], "Default Prob", alt_label),
                        use_container_width=True, key="gauge_alt_t2", key="pc_006")

    with a2:
        # Component waterfall
        if firm_type == "Public (Listed)":
            comp_names  = ["1.2 × WC/TA", "1.4 × RE/TA", "3.3 × EBIT/TA",
                           "0.6 × MVE/TL", "1.0 × Sales/TA"]
            comp_values = [1.2*wc_ta, 1.4*re_ta, 3.3*ebit_ta, 0.6*mve_tl, 1.0*sal_ta]
        else:
            comp_names  = ["0.717 × WC/TA", "0.847 × RE/TA", "3.107 × EBIT/TA",
                           "0.420 × BVE/TL", "0.998 × Sales/TA"]
            comp_values = [0.717*wc_ta, 0.847*re_ta, 3.107*ebit_ta, 0.420*bve_tl, 0.998*sal_ta]

        fig_wf = go.Figure(go.Bar(
            y=comp_names, x=comp_values, orientation="h",
            marker_color=[GR if v >= 0 else RD for v in comp_values],
            text=[f"{v:+.4f}" for v in comp_values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=10, color=TP),
            width=0.6,
        ))
        fig_wf.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_wf.update_layout(
            **plotly_layout_base(height=300, margin=dict(l=10, r=80, t=40, b=30)),
            title=dict(text="Z-Score Component Contributions",
                       font=dict(color=GD, size=13), x=0.0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_wf, use_container_width=True, key="wf_alt_t2")

        # Score bands visualisation
        score_val = alt["score"]
        fig_band = go.Figure()
        fig_band.add_shape(type="rect", x0=0,    x1=1.81, y0=0, y1=1,
                           fillcolor="rgba(220,53,69,0.2)", line_width=0)
        fig_band.add_shape(type="rect", x0=1.81, x1=2.99, y0=0, y1=1,
                           fillcolor="rgba(255,215,0,0.15)", line_width=0)
        fig_band.add_shape(type="rect", x0=2.99, x1=6.0,  y0=0, y1=1,
                           fillcolor="rgba(40,167,69,0.15)", line_width=0)
        fig_band.add_vline(x=score_val, line_color=GD, line_width=3,
                           annotation_text=f"  Z = {score_val:.3f}",
                           annotation_font_color=GD, annotation_font_size=13)
        for x, lbl in [(0.9, "DISTRESS"), (2.4, "GREY"), (4.5, "SAFE")]:
            fig_band.add_annotation(x=x, y=0.5, text=lbl,
                                    font=dict(color=TP, size=11), showarrow=False)
        fig_band.update_layout(
            **plotly_layout_base(height=120, margin=dict(l=20, r=20, t=30, b=30)),
            title=dict(text="Score Position on Zone Bands",
                       font=dict(color=GD, size=12), x=0.0),
            xaxis=dict(range=[0, 6], tickvals=[1.81, 2.99],
                       ticktext=["1.81", "2.99"], gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig_band, use_container_width=True, key="band_alt_t2")

    # Sensitivity table
    st.markdown("---")
    sec("🔍 Sensitivity: How Each Ratio Moves the Score")
    ibox(
        "<p>The table below shows the impact on the Z-Score of a ±10% change in each input ratio.</p>",
        title="Sensitivity Analysis"
    )
    if firm_type == "Public (Listed)":
        weights = {"WC/TA":1.2, "RE/TA":1.4, "EBIT/TA":3.3, "MVE/TL":0.6, "Sales/TA":1.0}
        base_r  = {"WC/TA":wc_ta,"RE/TA":re_ta,"EBIT/TA":ebit_ta,"MVE/TL":mve_tl,"Sales/TA":sal_ta}
    else:
        weights = {"WC/TA":0.717,"RE/TA":0.847,"EBIT/TA":3.107,"BVE/TL":0.420,"Sales/TA":0.998}
        base_r  = {"WC/TA":wc_ta,"RE/TA":re_ta,"EBIT/TA":ebit_ta,"BVE/TL":bve_tl,"Sales/TA":sal_ta}

    sens_rows = []
    for ratio, w in weights.items():
        base = base_r[ratio]
        z_up   = alt["score"] + w * base * 0.10
        z_down = alt["score"] - w * base * 0.10
        sens_rows.append({"Ratio": ratio, "Weight": w,
                          "Base Value": round(base, 4),
                          "Z (Base)":  round(alt["score"], 4),
                          "Z (+10%)":  round(z_up, 4),
                          "Z (−10%)":  round(z_down, 4),
                          "Impact per unit": round(w, 3)})
    st.dataframe(
        pd.DataFrame(sens_rows).style
        .set_properties(**{"font-family":"JetBrains Mono,monospace",
                           "font-size":"11px","text-align":"center"})
        .set_table_styles([{"selector":"th",
                            "props":[("background-color",DB),("color",GD),
                                     ("font-weight","700"),("text-align","center")]}]),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════
#  TAB 3 — OHLSON O-SCORE
# ══════════════════════════════════════════════════════════
with tab3:
    sec("📉 Ohlson O-Score — Logistic Regression Model")

    o1, o2 = st.columns([1, 2])
    with o1:
        mcard("O-Score", f'{ohl["score"]:.4f}', sub=ohl["zone"], color=ohl["color"])
        mcard("Default Probability", f'{ohl["prob"]*100:.2f}%',
              sub="P = 1/(1+e^−O)", color=ohl["color"])
        st.plotly_chart(prob_gauge(ohl["prob"], "Default Prob", "Ohlson O-Score"),
                        use_container_width=True, key="gauge_ohl_t3", key="pc_007")

    with o2:
        # Logistic curve with current position
        o_range = np.linspace(-6, 6, 300)
        p_range = expit(o_range)
        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(x=o_range, y=p_range*100, mode="lines",
                                     line=dict(color=LB, width=2.5),
                                     name="Logistic Curve"))
        fig_log.add_vline(x=ohl["score"], line_color=GD, line_width=2,
                          line_dash="dash")
        fig_log.add_hline(y=50, line_color=RD, line_width=1, line_dash="dot",
                          annotation_text="50% threshold",
                          annotation_font_color=RD)
        fig_log.add_trace(go.Scatter(
            x=[ohl["score"]], y=[ohl["prob"]*100], mode="markers",
            marker=dict(size=14, color=GD, line=dict(width=2, color=TP)),
            name=f"Current: {ohl['score']:.3f}"
        ))
        fig_log.update_layout(
            **plotly_layout_base(height=320),
            title=dict(text="Ohlson Logistic Curve — P(default) vs O-Score",
                       font=dict(color=GD, size=13), x=0.0),
            xaxis=dict(title="O-Score", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Default Probability (%)", ticksuffix="%",
                       gridcolor="rgba(255,255,255,0.05)", range=[0, 100]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TP)),
        )
        st.plotly_chart(fig_log, use_container_width=True, key="log_ohl_t3")

    ibox(
        "<p>The O-Score is a <b>logistic regression</b> model. Unlike Z-Score zone mapping, "
        "it directly outputs a calibrated probability via the logistic (sigmoid) function "
        "<code>P = 1/(1+e<sup>−O</sup>)</code>.</p>",
        title="Model Note"
    )


# ══════════════════════════════════════════════════════════
#  TAB 4 — ZMIJEWSKI PROBIT
# ══════════════════════════════════════════════════════════
with tab4:
    sec("📐 Zmijewski Score — Probit Regression Model")

    z1, z2 = st.columns([1, 2])
    with z1:
        mcard("Zmijewski Score", f'{zmi["score"]:.4f}', sub=zmi["zone"], color=zmi["color"])
        mcard("Default Probability", f'{zmi["prob"]*100:.2f}%',
              sub="P = Φ(X) — Normal CDF", color=zmi["color"])
        st.plotly_chart(prob_gauge(zmi["prob"], "Default Prob", "Zmijewski Probit"),
                        use_container_width=True, key="gauge_zmi_t4", key="pc_008")

    with z2:
        # Probit curve
        x_range = np.linspace(-5, 5, 300)
        p_probit = stats.norm.cdf(x_range)
        fig_probit = go.Figure()
        fig_probit.add_trace(go.Scatter(x=x_range, y=p_probit*100, mode="lines",
                                        line=dict(color=LB, width=2.5),
                                        name="Probit Curve"))
        fig_probit.add_vline(x=zmi["score"], line_color=GD, line_width=2,
                             line_dash="dash")
        fig_probit.add_hline(y=50, line_color=RD, line_width=1, line_dash="dot",
                             annotation_text="50% threshold",
                             annotation_font_color=RD)
        fig_probit.add_trace(go.Scatter(
            x=[zmi["score"]], y=[zmi["prob"]*100], mode="markers",
            marker=dict(size=14, color=GD, line=dict(width=2, color=TP)),
            name=f"Current: {zmi['score']:.3f}"
        ))
        fig_probit.update_layout(
            **plotly_layout_base(height=320),
            title=dict(text="Zmijewski Probit Curve — P(default) vs Score",
                       font=dict(color=GD, size=13), x=0.0),
            xaxis=dict(title="Zmijewski Score", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Default Probability (%)", ticksuffix="%",
                       gridcolor="rgba(255,255,255,0.05)", range=[0, 100]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TP)),
        )
        st.plotly_chart(fig_probit, use_container_width=True, key="probit_zmi_t4")

    # 3-factor sensitivity surface
    st.markdown("---")
    sec("🗺 Probability Surface: ROA vs Leverage")
    roa_range = np.linspace(-0.20, 0.20, 40)
    lev_range = np.linspace(0.10, 0.95, 40)
    ROA_grid, LEV_grid = np.meshgrid(roa_range, lev_range)
    X_grid = -4.336 - 4.513*ROA_grid + 5.679*LEV_grid + 0.004*liquidity
    P_grid = stats.norm.cdf(X_grid) * 100

    fig_surf = go.Figure(go.Heatmap(
        z=P_grid, x=np.round(roa_range, 3), y=np.round(lev_range, 3),
        colorscale=[[0,"#004d00"],[0.2,"#28a745"],[0.5,"#FFD700"],[0.8,"#dc3545"],[1,"#8b0000"]],
        colorbar=colorbar_cfg("P(default) %"),
        hoverongaps=False,
    ))
    fig_surf.add_trace(go.Scatter(
        x=[roa], y=[leverage], mode="markers",
        marker=dict(size=14, color=GD, symbol="star",
                    line=dict(width=2, color=TP)),
        name=f"{company_name}", showlegend=True,
    ))
    fig_surf.update_layout(
        **plotly_layout_base(height=400, margin=dict(l=60, r=80, t=50, b=60)),
        title=dict(text="Default Probability (%) — ROA vs Leverage Plane",
                   font=dict(color=GD, size=13), x=0.0),
        xaxis=dict(title="ROA", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Leverage (TL/TA)", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TP)),
    )
    st.plotly_chart(fig_surf, use_container_width=True, key="surf_zmi_t4")


# ══════════════════════════════════════════════════════════
#  TAB 5 — MERTON KMV
# ══════════════════════════════════════════════════════════
with tab5:
    sec("🔬 Merton KMV Structural Model — Distance to Default")

    m1, m2, m3 = st.columns(3)
    with m1:
        mcard("Asset Value (V)", f'{kmv["asset_value"]:,.2f}',
              sub=f"vs Debt Face {debt_face:,.0f}", color=LB)
        mcard("Asset Volatility σ_V", f'{kmv["asset_vol"]:.2f}%',
              sub=f"vs Equity Vol {equity_vol*100:.1f}%", color=LB)
    with m2:
        mcard("Distance to Default (DD)", f'{kmv["dd"]:.4f}',
              sub="σ units from default", color=GD if kmv["dd"]>2 else RD)
        mcard("d₁", f'{kmv["d1"]:.4f}', sub="N(d1) = Equity delta", color=TS)
    with m3:
        mcard("d₂", f'{kmv["d2"]:.4f}', sub="N(d2) = Risk-neutral PD", color=TS)
        mcard("Default Probability", f'{kmv["prob"]*100:.2f}%',
              sub=kmv["zone"], color=kmv["color"])

    st.plotly_chart(prob_gauge(kmv["prob"], "Default Prob", "Merton KMV"),
                    use_container_width=True, key="gauge_kmv_t5", key="pc_009")

    st.markdown("---")
    sec("📈 Asset Value Distribution vs Debt Threshold")

    # Show lognormal distribution of asset value at T=1
    V0      = kmv["asset_value"]
    sig_V   = kmv["asset_vol"] / 100
    mu_V    = risk_free - 0.5 * sig_V**2

    x_vals  = np.linspace(V0 * 0.2, V0 * 2.5, 400)
    y_vals  = stats.lognorm.pdf(x_vals, s=sig_V, scale=V0 * np.exp(mu_V))

    fig_dist = go.Figure()
    # Fill safe area
    mask_safe = x_vals >= debt_face
    fig_dist.add_trace(go.Scatter(
        x=x_vals[mask_safe], y=y_vals[mask_safe],
        fill="tozeroy", fillcolor="rgba(40,167,69,0.2)",
        line=dict(color=GR, width=0), name="V_T > Debt (Solvent)", showlegend=True
    ))
    # Fill default area
    mask_def  = x_vals <= debt_face
    fig_dist.add_trace(go.Scatter(
        x=x_vals[mask_def], y=y_vals[mask_def],
        fill="tozeroy", fillcolor="rgba(220,53,69,0.3)",
        line=dict(color=RD, width=0), name="V_T < Debt (Default)", showlegend=True
    ))
    fig_dist.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines",
        line=dict(color=LB, width=2.5), name="Asset Value Distribution"
    ))
    fig_dist.add_vline(x=debt_face, line_color=RD, line_width=2, line_dash="dash",
                       annotation_text=f"  Debt Face = {debt_face:,.0f}",
                       annotation_font_color=RD)
    fig_dist.add_vline(x=V0, line_color=GD, line_width=2, line_dash="dot",
                       annotation_text=f"  V₀ = {V0:,.0f}",
                       annotation_font_color=GD)
    fig_dist.update_layout(
        **plotly_layout_base(height=380),
        title=dict(text="Lognormal Asset Value Distribution at T = 1yr (risk-neutral)",
                   font=dict(color=GD, size=13), x=0.0),
        xaxis=dict(title="Asset Value V_T", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Probability Density", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TP)),
    )
    st.plotly_chart(fig_dist, use_container_width=True, key="dist_kmv_t5")


# ══════════════════════════════════════════════════════════
#  TAB 6 — BANKRUPTCY COSTS
# ══════════════════════════════════════════════════════════
with tab6:
    sec("💸 Bankruptcy Cost Estimation")
    curr = currency.split()[0]

    # KPI row
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1: mcard("Direct Costs", f'{curr} {bcc["direct_total"]:,.1f}',
                    sub=f"{bcc['direct_total']/max(total_assets,1)*100:.1f}% of assets", color=GD)
    with bc2: mcard("Indirect Costs", f'{curr} {bcc["indirect_total"]:,.1f}',
                    sub=f"{bcc['indirect_total']/max(total_assets,1)*100:.1f}% of assets", color=GD)
    with bc3: mcard("Total Bankruptcy Cost", f'{curr} {bcc["total_cost"]:,.1f}',
                    sub=f"{bcc['pct_assets']:.1f}% of Total Assets", color=RD)
    with bc4: mcard("Cost as % Revenue", f'{bcc["pct_revenue"]:.1f}%',
                    sub=f"{years_distress:.1f} years distress assumed", color=RD)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        sec("🔴 Direct Costs Breakdown")
        direct_labels = ["Legal & Advisory Fees", "Administrative Costs", "Industry-Specific Direct"]
        direct_vals   = [bcc["direct_legal"], bcc["direct_admin"], bcc["direct_other"]]
        fig_dir = go.Figure(go.Pie(
            labels=direct_labels, values=direct_vals,
            hole=0.45,
            marker=dict(colors=[RD, "#c0392b", "#e74c3c"],
                        line=dict(color=CB, width=2)),
            textfont=dict(color=TP, size=11),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>",
        ))
        fig_dir.update_layout(
            **plotly_layout_base(height=320, margin=dict(l=20,r=20,t=40,b=20)),
            title=dict(text=f"Direct Costs (Total: {curr} {bcc['direct_total']:,.1f})",
                       font=dict(color=GD, size=13), x=0.0),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TP, size=10)),
        )
        st.plotly_chart(fig_dir, use_container_width=True, key="pie_dir_t6")

    with c2:
        sec("🟡 Indirect Costs Breakdown")
        indir_labels = ["Lost Sales (Revenue Impact)", "Management Distraction",
                        "Supplier Tighter Credit", "Customer Defection"]
        indir_vals   = [bcc["lost_sales"], bcc["mgmt_dist"],
                        bcc["supplier_cost"], bcc["customer_flight"]]
        fig_ind = go.Figure(go.Pie(
            labels=indir_labels, values=indir_vals,
            hole=0.45,
            marker=dict(colors=[GD, "#d4af37", LB, "#85c1e9"],
                        line=dict(color=CB, width=2)),
            textfont=dict(color=CB, size=11),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>",
        ))
        fig_ind.update_layout(
            **plotly_layout_base(height=320, margin=dict(l=20,r=20,t=40,b=20)),
            title=dict(text=f"Indirect Costs (Total: {curr} {bcc['indirect_total']:,.1f})",
                       font=dict(color=GD, size=13), x=0.0),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TP, size=10)),
        )
        st.plotly_chart(fig_ind, use_container_width=True, key="pie_ind_t6")

    # Waterfall chart — full cost build-up
    st.markdown("---")
    sec("📊 Total Cost Build-Up — Waterfall Chart")
    wf_labels = ["Legal & Advisory", "Administrative", "Industry Direct",
                 "Lost Sales", "Mgmt Distraction", "Supplier Credit", "Customer Defection",
                 "TOTAL"]
    wf_vals   = [bcc["direct_legal"], bcc["direct_admin"], bcc["direct_other"],
                 bcc["lost_sales"],   bcc["mgmt_dist"],    bcc["supplier_cost"],
                 bcc["customer_flight"]]
    wf_measure= ["relative"]*7 + ["total"]
    wf_vals_wf= wf_vals + [bcc["total_cost"]]

    fig_wf2 = go.Figure(go.Waterfall(
        name="Cost Build-Up", orientation="v",
        measure=wf_measure,
        x=wf_labels, y=wf_vals_wf,
        connector=dict(line=dict(color=TS)),
        increasing=dict(marker_color=RD),
        totals=dict(marker_color=GD),
        text=[f"{curr} {v:,.1f}" for v in wf_vals_wf],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=9, color=TP),
    ))
    fig_wf2.update_layout(
        **plotly_layout_base(height=400),
        title=dict(text="Bankruptcy Cost Waterfall (Direct + Indirect)",
                   font=dict(color=GD, size=13), x=0.0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=f"Cost ({currency})", gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
    )
    st.plotly_chart(fig_wf2, use_container_width=True, key="wf2_cost_t6")

    # Expected cost = prob × total cost
    st.markdown("---")
    sec("🎯 Expected Bankruptcy Cost (Probability-Adjusted)")
    exp_cost = ensemble_prob * bcc["total_cost"]

    e1, e2, e3 = st.columns(3)
    with e1: mcard("Ensemble Default Probability", f'{ensemble_prob*100:.2f}%',
                   sub="Average of 4 models", color=RD if ensemble_prob>0.5 else GD)
    with e2: mcard("Total Bankruptcy Cost", f'{curr} {bcc["total_cost"]:,.1f}',
                   sub="Direct + Indirect", color=GD)
    with e3: mcard("Expected Bankruptcy Cost", f'{curr} {exp_cost:,.1f}',
                   sub="P(default) × Total Cost", color=RD if ensemble_prob>0.3 else GD)

    ibox(
        f"<p>The <b>Expected Bankruptcy Cost</b> represents the <i>probability-weighted</i> cost "
        f"of financial distress. It should be compared to the <b>tax shield benefit of debt</b> "
        f"(PV of interest tax savings) in a trade-off theory framework:</p>"
        f"<div class='formula-box' style='margin-top:8px;'>"
        f"Optimal Capital Structure: where PV(Tax Shield) = PV(Expected Bankruptcy Cost)</div>"
        f"<p style='margin-top:10px;'>At {ensemble_prob*100:.1f}% default probability, the expected cost "
        f"of {curr} {exp_cost:,.1f} represents <b>{exp_cost/max(total_assets,1)*100:.2f}% of Total Assets</b>.</p>",
        title="Trade-Off Theory Context"
    )

    # CSV download
    cost_df = pd.DataFrame({
        "Cost Component": ["Legal & Advisory", "Administrative", "Industry Direct",
                           "Total Direct", "Lost Sales", "Mgmt Distraction",
                           "Supplier Credit", "Customer Defection", "Total Indirect",
                           "TOTAL COST", "Expected Cost (Prob-Weighted)"],
        f"Amount ({currency})": [bcc["direct_legal"], bcc["direct_admin"], bcc["direct_other"],
                                  bcc["direct_total"], bcc["lost_sales"], bcc["mgmt_dist"],
                                  bcc["supplier_cost"], bcc["customer_flight"],
                                  bcc["indirect_total"], bcc["total_cost"], round(exp_cost,2)],
        "% of Total Assets": [f"{v/max(total_assets,1)*100:.2f}%"
                               for v in [bcc["direct_legal"], bcc["direct_admin"],
                                          bcc["direct_other"], bcc["direct_total"],
                                          bcc["lost_sales"], bcc["mgmt_dist"],
                                          bcc["supplier_cost"], bcc["customer_flight"],
                                          bcc["indirect_total"], bcc["total_cost"], exp_cost]],
    })
    st.download_button("⬇ Download Cost Report CSV",
                       cost_df.to_csv(index=False).encode(),
                       "bankruptcy_costs.csv", "text/csv")


# ══════════════════════════════════════════════════════════
#  TAB 7 — METHODOLOGY
# ══════════════════════════════════════════════════════════
with tab7:
    sec("📖 Methodology & Model Reference")

    # Model comparison table
    models_ref = [
        ("Altman Z-Score (1968)", "Public Mfg", "MDA", "5 ratios → linear score → zone",
         "Simple, interpretable, widely cited", "Assumes linearity; dated sample (1946-65)"),
        ("Altman Z'-Score (1983)", "Private Firms", "MDA", "Book equity replaces market value",
         "Applicable without market data", "Same linearity assumption"),
        ("Ohlson O-Score (1980)", "All firms", "Logit", "9 variables → logistic probability",
         "Directly calibrated probability", "Sensitive to accounting policy choices"),
        ("Zmijewski (1984)", "All firms", "Probit", "3 ratios → normal CDF probability",
         "Parsimonious; handles choice-based bias", "Fewer variables may miss nuance"),
        ("Merton KMV (1974)", "Public", "Structural", "Option-pricing; asset value & volatility",
         "Forward-looking; uses market data", "Requires equity vol; iterative solution"),
    ]
    ref_df = pd.DataFrame(models_ref, columns=["Model","Applicability","Method",
                                                 "Mechanism","Strengths","Limitations"])
    st.dataframe(
        ref_df.style
        .set_properties(**{"font-family":"JetBrains Mono,monospace",
                           "font-size":"11px","text-align":"left"})
        .set_table_styles([{"selector":"th",
                            "props":[("background-color",DB),("color",GD),
                                     ("font-weight","700"),("text-align","center")]}]),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    sec("📐 Key Formulae")

    col_f1, col_f2 = st.columns(2)
    formulas_left = [
        ("Altman Z-Score (Public)",
         "Z = 1.2·(WC/TA) + 1.4·(RE/TA) + 3.3·(EBIT/TA) + 0.6·(MVE/TL) + 1.0·(S/TA)<br>"
         "Safe: Z > 2.99 | Grey: 1.81–2.99 | Distress: Z < 1.81"),
        ("Altman Z'-Score (Private)",
         "Z' = 0.717·(WC/TA) + 0.847·(RE/TA) + 3.107·(EBIT/TA) + 0.420·(BVE/TL) + 0.998·(S/TA)<br>"
         "Safe: Z' > 2.9 | Grey: 1.23–2.9 | Distress: Z' < 1.23"),
        ("Ohlson O-Score",
         "O = −1.32 − 0.407·SIZE + 6.03·TLTA − 1.43·WCTA + 0.076·CLCA<br>"
         "     − 1.72·OENEG − 2.37·NITA − 1.83·FUTL + 0.285·INTWO − 0.521·CHIN<br>"
         "P(default) = 1 / (1 + e<sup>−O</sup>)"),
    ]
    formulas_right = [
        ("Zmijewski Score",
         "X = −4.336 − 4.513·ROA + 5.679·Leverage + 0.004·Liquidity<br>"
         "P(default) = Φ(X)   [standard normal CDF]"),
        ("Merton KMV — d₁, d₂",
         "d₁ = [ln(V/D) + (r + ½σ²ᵥ)·T] / (σᵥ·√T)<br>"
         "d₂ = d₁ − σᵥ·√T = Distance-to-Default<br>"
         "P(default) = N(−d₂)   [risk-neutral probability]"),
        ("Expected Bankruptcy Cost",
         "E[BC] = P(default) × [Direct Costs + Indirect Costs]<br>"
         "Trade-off optimum: PV(Tax Shield) = E[Bankruptcy Cost]"),
    ]

    with col_f1:
        for title_f, f_text in formulas_left:
            ibox(f'<div class="formula-box">{f_text}</div>', title=title_f)
    with col_f2:
        for title_f, f_text in formulas_right:
            ibox(f'<div class="formula-box">{f_text}</div>', title=title_f)

    st.markdown("---")
    ibox(
        "<p>• All models are based on <b>historical financial data</b> — they do not predict the future.</p>"
        "<p>• Accuracy varies by industry, country, and economic cycle.</p>"
        "<p>• The <b>ensemble probability</b> (simple average of 4 models) is indicative only.</p>"
        "<p>• Bankruptcy cost estimates use <b>literature-based averages</b> (Warner 1977, Altman 1984, "
        "Andrade & Kaplan 1998); actual costs vary significantly.</p>"
        "<p>• Merton KMV requires <b>publicly traded equity</b>; for private firms treat as approximate.</p>"
        "<p>• Risk-free rate: <b>user-specified</b>. Assumes continuous compounding in KMV.</p>",
        title="⚠️ Disclaimer & Limitations",
        border_color=RD
    )

# ============================================================================
# FOOTER
# ============================================================================
footer_bar()
