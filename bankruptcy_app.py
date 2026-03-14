
# ═══════════════════════════════════════════════════════════════════════════
#  BANKRUPTCY PROBABILITY & COST MODEL
#  The Mountain Path – World of Finance
#  Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking Experience
#  Models: Altman Z-Score · Ohlson O-Score · Zmijewski · Merton KMV
#          Direct & Indirect Bankruptcy Cost Estimation
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.special import expit
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bankruptcy Model | The Mountain Path",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ──────────────────────────────────────────────────────────────────
GD  = "#FFD700"   # gold
DB  = "#003366"   # dark blue
MB  = "#004d80"   # medium blue
LB  = "#ADD8E6"   # light blue
BG  = "#0a1628"   # bg dark
CB  = "#112240"   # card bg
TP  = "#e6f1ff"   # text primary
TS  = "#8892b0"   # text secondary
GR  = "#28a745"   # green
RD  = "#dc3545"   # red
TD  = "#1a1a2e"   # text dark

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+Pro:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp {{background:linear-gradient(135deg,#1a2332 0%,#243447 50%,#2a3f5f 100%);}}
.main,.main *,.main p,.main span,.main div,.main li,.main label{{color:{TP}!important;}}
.main h1,.main h2,.main h3,.main h4,.main h5,.main h6{{color:{GD}!important;font-family:'Playfair Display',serif;}}
section[data-testid="stSidebar"]{{background:linear-gradient(180deg,{BG} 0%,{DB} 100%);border-right:1px solid rgba(255,215,0,0.25);}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div[class*="markdown"]{{color:{TP}!important;-webkit-text-fill-color:{TP}!important;}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{{color:{LB}!important;-webkit-text-fill-color:{LB}!important;font-weight:600!important;opacity:1!important;}}
section[data-testid="stSidebar"] input{{color:{TD}!important;background-color:#fff!important;}}
.metric-card{{background:{CB};border:1px solid rgba(255,215,0,0.3);border-radius:10px;padding:1.2rem;text-align:center;margin-bottom:0.8rem;}}
.metric-card .label{{color:{TS};font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;}}
.metric-card .value{{color:{GD};font-size:1.6rem;font-weight:700;font-family:'Playfair Display',serif;margin-top:0.3rem;}}
.metric-card .sub{{color:{TS};font-size:0.78rem;margin-top:0.3rem;}}
.section-title{{font-family:'Playfair Display',serif;color:{GD};font-size:1.3rem;border-bottom:2px solid rgba(255,215,0,0.3);padding-bottom:0.5rem;margin:1.5rem 0 1rem;}}
.info-box{{background:rgba(0,51,102,0.5);border:1px solid {GD};border-radius:8px;padding:1rem 1.5rem;color:{TP};margin:0.8rem 0;}}
.formula-box{{font-family:'JetBrains Mono',monospace;background:rgba(0,0,0,0.35);padding:10px 14px;border-radius:6px;border-left:3px solid {GD};font-size:0.85rem;color:{TP};margin:10px 0;}}
.stTabs [data-baseweb="tab-list"]{{gap:8px;}}
.stTabs [data-baseweb="tab"]{{background:{CB};border:1px solid rgba(255,215,0,0.3);border-radius:8px;color:{TP};padding:0.5rem 1rem;}}
.stTabs [aria-selected="true"]{{background:{DB};border:2px solid {GD};color:{GD};}}
.stButton>button{{background:linear-gradient(135deg,{MB},{DB})!important;color:{GD}!important;border:2px solid {GD}!important;border-radius:8px!important;font-weight:600!important;}}
.stButton>button:hover{{background:linear-gradient(135deg,{GD},#d4af37)!important;color:{DB}!important;}}
.stAlert{{background-color:rgba(255,255,255,0.95)!important;}}
.stAlert p,.stAlert span,.stAlert div{{color:{TD}!important;}}
details summary{{color:{GD}!important;}}
footer{{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)

# ── UI helpers ────────────────────────────────────────────────────────────────
def header(title, subtitle=None, desc=None):
    s = f'<p style="font-size:1rem;color:{GD};font-weight:600;margin:0.5rem 0;">{subtitle}</p>' if subtitle else ""
    d = f'<p style="font-size:0.85rem;color:{TP};margin:0.3rem 0;">{desc}</p>' if desc else ""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{DB},{MB});border:2px solid {GD};border-radius:12px;
                padding:1.5rem 2rem;margin-bottom:1.5rem;text-align:center;">
        <h1 style="font-family:'Playfair Display',serif;color:{GD};margin:0;font-size:2rem;">{title}</h1>{s}{d}
        <p style="color:{TP};margin:0.3rem 0;">🏔️ The Mountain Path - World of Finance</p>
        <p style="color:{TS};font-size:0.8rem;margin:0;">Prof. V. Ravichandran | 28+ Years Corporate Finance &amp; Banking | 10+ Years Academic Excellence</p>
    </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def mcard(label, value, sub=None, color=None):
    clr = color or GD
    sh  = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value" style="color:{clr};-webkit-text-fill-color:{clr};">{value}</div>{sh}
    </div>""", unsafe_allow_html=True)

def ibox(content, title=None, bc=None):
    bclr = bc or GD
    th   = f"<h4 style='color:{GD};margin-top:0;'>{title}</h4>" if title else ""
    st.markdown(f'<div class="info-box" style="border-color:{bclr};">{th}{content}</div>',
                unsafe_allow_html=True)

def formula(text):
    st.markdown(f'<div class="formula-box">{text}</div>', unsafe_allow_html=True)

def playout(**kw):
    base = dict(paper_bgcolor=CB, plot_bgcolor=CB,
                font=dict(color=TP, family="Source Sans Pro"),
                margin=dict(l=60,r=40,t=50,b=50))
    base.update(kw)
    return base

def cbar(title_text):
    return dict(title=dict(text=title_text, font=dict(color=GD)),
                tickfont=dict(color=TP), thickness=14, outlinewidth=0)

def footer_bar():
    st.divider()
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem;">
      <p style="color:{GD};font-family:'Playfair Display',serif;font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">
        🏔️ The Mountain Path - World of Finance</p>
      <p style="color:{TS};font-size:0.85rem;margin:0.3rem 0;">Prof. V. Ravichandran | 28+ Years Corporate Finance &amp; Banking | 10+ Years Academic Excellence</p>
      <p style="color:{TS};font-size:0.75rem;">Visiting Faculty: NMIMS Bangalore · BITS Pilani · RV University · Goa Institute of Management</p>
      <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,215,0,0.3);">
        <a href="https://www.linkedin.com/in/trichyravis" target="_blank"
           style="color:{GD};text-decoration:none;margin:0 1rem;">🔗 LinkedIn</a>
        <a href="https://github.com/trichyravis" target="_blank"
           style="color:{GD};text-decoration:none;margin:0 1rem;">💻 GitHub</a>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Bankruptcy models ─────────────────────────────────────────────────────────
def altman_z_public(wc_ta,re_ta,ebit_ta,mve_tl,sal_ta):
    z = 1.2*wc_ta+1.4*re_ta+3.3*ebit_ta+0.6*mve_tl+1.0*sal_ta
    if z>2.99:   zone,prob,clr="Safe Zone",   max(0.01,0.02+(2.99-z)*0.01),GR
    elif z>1.81: zone,prob,clr="Grey Zone",   0.30+(2.99-z)/(2.99-1.81)*0.35,GD
    else:        zone,prob,clr="Distress Zone",min(0.95,0.65+(1.81-z)*0.15),RD
    return dict(score=round(z,4),zone=zone,prob=round(max(0.01,prob),4),color=clr)

def altman_z_private(wc_ta,re_ta,ebit_ta,bve_tl,sal_ta):
    z = 0.717*wc_ta+0.847*re_ta+3.107*ebit_ta+0.420*bve_tl+0.998*sal_ta
    if z>2.9:    zone,prob,clr="Safe Zone",   max(0.01,0.02+(2.9-z)*0.01),GR
    elif z>1.23: zone,prob,clr="Grey Zone",   0.28+(2.9-z)/(2.9-1.23)*0.38,GD
    else:        zone,prob,clr="Distress Zone",min(0.95,0.66+(1.23-z)*0.18),RD
    return dict(score=round(z,4),zone=zone,prob=round(max(0.01,prob),4),color=clr)

def ohlson_o_score(total_assets,total_liabilities,working_capital,
                   current_liabilities,current_assets,net_income,funds_from_ops,ebit):
    size  = np.log(max(total_assets,1e-6))
    tlta  = total_liabilities/max(total_assets,1e-6)
    wcta  = working_capital/max(total_assets,1e-6)
    clca  = current_liabilities/max(current_assets,1e-6)
    oeneg = 1 if total_liabilities>total_assets else 0
    nita  = net_income/max(total_assets,1e-6)
    futl  = funds_from_ops/max(total_liabilities,1e-6)
    intwo = 1 if net_income<0 else 0
    o = -1.32-0.407*size+6.03*tlta-1.43*wcta+0.076*clca-1.72*oeneg-2.37*nita-1.83*futl+0.285*intwo
    prob = float(expit(o))
    if prob<0.20:   zone,clr="Safe Zone",GR
    elif prob<0.50: zone,clr="Grey Zone",GD
    else:           zone,clr="Distress Zone",RD
    return dict(score=round(o,4),prob=round(prob,4),zone=zone,color=clr)

def zmijewski(roa,leverage,liquidity):
    x = -4.336-4.513*roa+5.679*leverage+0.004*liquidity
    prob = float(stats.norm.cdf(x))
    if prob<0.20:   zone,clr="Safe Zone",GR
    elif prob<0.50: zone,clr="Grey Zone",GD
    else:           zone,clr="Distress Zone",RD
    return dict(score=round(x,4),prob=round(prob,4),zone=zone,color=clr)

def merton_kmv(equity_value,equity_vol,debt_face,risk_free,T=1.0):
    V,sigma_V = equity_value+debt_face, equity_vol*equity_value/(equity_value+debt_face)
    for _ in range(200):
        d1 = (np.log(max(V,1e-6)/max(debt_face,1e-6))+(risk_free+0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
        d2 = d1-sigma_V*np.sqrt(T)
        V_new     = equity_value+debt_face*np.exp(-risk_free*T)*stats.norm.cdf(-d2)
        sV_new    = equity_vol*equity_value/max(V_new,1e-6)
        if abs(V_new-V)<1e-6 and abs(sV_new-sigma_V)<1e-8: break
        V,sigma_V = V_new,sV_new
    d1 = (np.log(max(V,1e-6)/max(debt_face,1e-6))+(risk_free+0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
    d2 = d1-sigma_V*np.sqrt(T)
    prob = float(stats.norm.cdf(-d2))
    if prob<0.10:   zone,clr="Safe Zone",GR
    elif prob<0.30: zone,clr="Grey Zone",GD
    else:           zone,clr="Distress Zone",RD
    return dict(asset_value=round(V,2),asset_vol=round(sigma_V*100,4),
                d1=round(d1,4),d2=round(d2,4),dd=round(d2,4),
                prob=round(prob,4),zone=zone,color=clr)

def bankruptcy_costs(total_assets,leverage,revenue,ebitda,
                     industry_direct_pct,years_distress=2.0):
    dl  = total_assets*0.030
    da  = total_assets*0.015
    do  = total_assets*(industry_direct_pct/100)
    dt  = dl+da+do
    ls  = revenue*0.08*years_distress
    md  = ebitda*0.15*years_distress
    sc  = total_assets*0.02
    cf  = revenue*0.05
    it  = ls+md+sc+cf
    tc  = dt+it
    return dict(direct_legal=round(dl,2),direct_admin=round(da,2),
                direct_other=round(do,2),direct_total=round(dt,2),
                lost_sales=round(ls,2),mgmt_dist=round(md,2),
                supplier_cost=round(sc,2),customer_flight=round(cf,2),
                indirect_total=round(it,2),total_cost=round(tc,2),
                pct_assets=round(tc/max(total_assets,1)*100,2),
                pct_revenue=round(tc/max(revenue,1)*100,2))

def prob_gauge(prob, title="Default Probability", model="", key="gauge"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob*100,
        number={"suffix":"%","font":{"size":32,"color":GD,"family":"JetBrains Mono"}},
        title={"text":f"{title}<br><span style='font-size:0.8em;color:{TS}'>{model}</span>",
               "font":{"size":14,"color":TP}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":TP,"tickfont":{"color":TP}},
            "bar":{"color":RD if prob>0.5 else GD if prob>0.2 else GR,"thickness":0.35},
            "bgcolor":CB,"borderwidth":1,"bordercolor":GD,
            "steps":[{"range":[0,20],"color":"rgba(40,167,69,0.15)"},
                     {"range":[20,50],"color":"rgba(255,215,0,0.15)"},
                     {"range":[50,100],"color":"rgba(220,53,69,0.15)"}],
            "threshold":{"line":{"color":RD,"width":3},"thickness":0.75,"value":50},
        }
    ))
    fig.update_layout(**playout(height=280,margin=dict(l=30,r=30,t=60,b=20)))
    st.plotly_chart(fig, use_container_width=True, key=key)

# ── Header ────────────────────────────────────────────────────────────────────
header("⚖️ Bankruptcy Probability & Cost Model",
       subtitle="Altman Z-Score · Ohlson O-Score · Zmijewski Probit · Merton KMV · Direct & Indirect Cost Estimation",
       desc="Enter financial statement data in the sidebar to run all four models simultaneously")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:0.8rem 0 1rem;
                border-bottom:1px solid rgba(255,215,0,0.3);margin-bottom:1rem;">
      <span style="font-family:'Playfair Display',serif;font-size:1.1rem;
                   color:{GD};font-weight:700;">🏔️ The Mountain Path</span>
    </div>""", unsafe_allow_html=True)

    sec("🏭 Company Profile")
    company_name = st.text_input("Company Name", value="ABC Industries Ltd.")
    firm_type    = st.selectbox("Firm Type", ["Public (Listed)","Private (Unlisted)"])
    currency     = st.selectbox("Currency", ["₹ Crore","$ Million","€ Million"])

    sec("📊 Balance Sheet")
    total_assets        = st.number_input("Total Assets",            value=1000.0,min_value=1.0,step=10.0)
    total_liabilities   = st.number_input("Total Liabilities",       value=600.0, min_value=0.0,step=10.0)
    current_assets      = st.number_input("Current Assets",          value=300.0, min_value=0.0,step=5.0)
    current_liabilities = st.number_input("Current Liabilities",     value=200.0, min_value=0.0,step=5.0)
    retained_earnings   = st.number_input("Retained Earnings",       value=120.0, step=5.0)
    market_cap          = st.number_input("Market Cap / Book Equity", value=500.0, min_value=0.0,step=10.0)

    sec("📈 Income & Cash Flow")
    revenue        = st.number_input("Revenue / Net Sales",   value=800.0,min_value=0.0,step=10.0)
    ebit           = st.number_input("EBIT",                  value=100.0,step=5.0)
    ebitda         = st.number_input("EBITDA",                value=150.0,step=5.0)
    net_income     = st.number_input("Net Income",            value=60.0, step=5.0)
    funds_from_ops = st.number_input("Funds From Operations", value=80.0, step=5.0)

    sec("📉 Merton KMV")
    equity_vol = st.slider("Equity Volatility σ_E %", 5.0,120.0,30.0,0.5)/100
    risk_free  = st.slider("Risk-Free Rate %",         2.0, 15.0, 6.5,0.25)/100
    debt_face  = st.number_input("Debt Face Value",    value=600.0,min_value=0.0,step=10.0)

    sec("💸 Cost Inputs")
    industry_direct = st.slider("Industry Direct Cost % of Assets",0.5,8.0,2.5,0.25)
    years_distress  = st.slider("Expected Years in Distress",       0.5,5.0,2.0,0.25)

    st.divider()
    run_btn = st.button("▶  Run All Models", use_container_width=True)

# ── Derived ratios ────────────────────────────────────────────────────────────
working_capital = current_assets - current_liabilities
equity_book     = total_assets - total_liabilities
roa             = net_income   / max(total_assets,1e-6)
leverage        = total_liabilities / max(total_assets,1e-6)
liquidity       = current_assets    / max(current_liabilities,1e-6)
wc_ta           = working_capital   / max(total_assets,1e-6)
re_ta           = retained_earnings / max(total_assets,1e-6)
ebit_ta         = ebit              / max(total_assets,1e-6)
mve_tl          = market_cap        / max(total_liabilities,1e-6)
bve_tl          = equity_book       / max(total_liabilities,1e-6)
sal_ta          = revenue           / max(total_assets,1e-6)

# ── Run models ────────────────────────────────────────────────────────────────
if firm_type=="Public (Listed)":
    alt      = altman_z_public(wc_ta,re_ta,ebit_ta,mve_tl,sal_ta)
    alt_label= "Altman Z-Score (Public)"
else:
    alt      = altman_z_private(wc_ta,re_ta,ebit_ta,bve_tl,sal_ta)
    alt_label= "Altman Z'-Score (Private)"

ohl = ohlson_o_score(total_assets,total_liabilities,working_capital,
                     current_liabilities,current_assets,net_income,funds_from_ops,ebit)
zmi = zmijewski(roa,leverage,liquidity)
kmv = merton_kmv(market_cap,equity_vol,debt_face,risk_free)
bcc = bankruptcy_costs(total_assets,leverage,revenue,ebitda,industry_direct,years_distress)

ensemble_prob = np.mean([alt["prob"],ohl["prob"],zmi["prob"],kmv["prob"]])
curr          = currency.split()[0]

# ══════════════════════════════════════════════════════════════════════════════
#  TWO-ROW NAVIGATION  (Radio selector → nested st.tabs)
# ══════════════════════════════════════════════════════════════════════════════

# ── Styled top-row radio as pill tabs ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Radio pill container ── */
div.row-widget.stRadio > div[role="radiogroup"] {
    display: flex !important;
    gap: 12px !important;
    flex-direction: row !important;
    background: #112240 !important;
    padding: 8px 12px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,215,0,0.25) !important;
}
/* ── Each pill ── */
div.row-widget.stRadio > div[role="radiogroup"] > label {
    background: transparent !important;
    border: 1px solid rgba(255,215,0,0.4) !important;
    border-radius: 8px !important;
    padding: 8px 22px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    white-space: nowrap !important;
}
/* ── Text inside pill — target every possible child ── */
div.row-widget.stRadio > div[role="radiogroup"] > label p,
div.row-widget.stRadio > div[role="radiogroup"] > label span,
div.row-widget.stRadio > div[role="radiogroup"] > label div:not(:first-child),
div.row-widget.stRadio > div[role="radiogroup"] > label div[data-testid],
div.row-widget.stRadio > div[role="radiogroup"] > label > div > div {
    color: #e6f1ff !important;
    -webkit-text-fill-color: #e6f1ff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    opacity: 1 !important;
    visibility: visible !important;
}
/* ── Hover state ── */
div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
    background: #004d80 !important;
    border-color: #FFD700 !important;
}
div.row-widget.stRadio > div[role="radiogroup"] > label:hover p,
div.row-widget.stRadio > div[role="radiogroup"] > label:hover span,
div.row-widget.stRadio > div[role="radiogroup"] > label:hover > div > div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
/* ── Active / selected pill ── */
div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked),
div.row-widget.stRadio > div[role="radiogroup"] > label[aria-checked="true"] {
    background: linear-gradient(135deg, #003366, #004d80) !important;
    border: 2px solid #FFD700 !important;
}
div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) p,
div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) span,
div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) > div > div,
div.row-widget.stRadio > div[role="radiogroup"] > label[aria-checked="true"] p,
div.row-widget.stRadio > div[role="radiogroup"] > label[aria-checked="true"] span,
div.row-widget.stRadio > div[role="radiogroup"] > label[aria-checked="true"] > div > div {
    color: #FFD700 !important;
    -webkit-text-fill-color: #FFD700 !important;
    font-weight: 700 !important;
}
/* ── Hide the radio dot ── */
div.row-widget.stRadio > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

top_nav = st.radio(
    label="Navigation",
    options=["📊 Models & Analysis", "📚 Reference & Education"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  ROW 1 — MODELS & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if top_nav == "📊 Models & Analysis":
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "📊 Summary Dashboard",
        "🔢 Altman Z-Score",
        "📉 Ohlson O-Score",
        "📐 Zmijewski Probit",
        "🔬 Merton KMV",
        "💸 Bankruptcy Costs",
    ])

    with tab1:
        sec("🎯 Ensemble Bankruptcy Assessment")
        cols = st.columns(5)
        kpis = [
            (alt_label,        f'{alt["score"]:.3f}',        alt["zone"],  alt["color"]),
            ("Ohlson O-Score",  f'{ohl["score"]:.3f}',        ohl["zone"],  ohl["color"]),
            ("Zmijewski Score", f'{zmi["score"]:.3f}',        zmi["zone"],  zmi["color"]),
            ("Merton DD",       f'{kmv["dd"]:.3f}',           kmv["zone"],  kmv["color"]),
            ("Ensemble Prob",   f'{ensemble_prob*100:.1f}%',
             "HIGH RISK" if ensemble_prob>0.5 else "MODERATE" if ensemble_prob>0.2 else "LOW RISK",
             RD if ensemble_prob>0.5 else GD if ensemble_prob>0.2 else GR),
        ]
        for col,(lbl,val,zone,clr) in zip(cols,kpis):
            with col: mcard(lbl,val,sub=zone,color=clr)

        st.markdown("---")
        sec("📡 Default Probability — All Models")
        g1,g2,g3,g4 = st.columns(4)
        with g1: prob_gauge(alt["prob"],"Default Prob",alt_label,   key="gauge_alt_t1")
        with g2: prob_gauge(ohl["prob"],"Default Prob","Ohlson",    key="gauge_ohl_t1")
        with g3: prob_gauge(zmi["prob"],"Default Prob","Zmijewski", key="gauge_zmi_t1")
        with g4: prob_gauge(kmv["prob"],"Default Prob","Merton KMV",key="gauge_kmv_t1")

        st.markdown("---")
        sec("🌡 Ensemble Risk Assessment")
        ec1,ec2 = st.columns([1,2])
        with ec1:
            prob_gauge(ensemble_prob,"Ensemble Probability","Average of 4 Models",key="gauge_ens_t1")
            risk_color = RD if ensemble_prob>0.5 else GD if ensemble_prob>0.2 else GR
            risk_label = ("🚨 HIGH RISK — Immediate attention required" if ensemble_prob>0.5 else
                          "⚠️ MODERATE RISK — Monitor closely"          if ensemble_prob>0.2 else
                          "✅ LOW RISK — Financially sound")
            ibox(f"<p style='margin:0;font-weight:700;color:{risk_color};-webkit-text-fill-color:{risk_color};'>{risk_label}</p>",
                 bc=risk_color)

        with ec2:
            models   = [alt_label,"Ohlson","Zmijewski","Merton KMV","Ensemble"]
            probs    = [alt["prob"],ohl["prob"],zmi["prob"],kmv["prob"],ensemble_prob]
            bcolors  = [RD if p>0.5 else GD if p>0.2 else GR for p in probs]; bcolors[-1]=LB
            fig_cmp  = go.Figure(go.Bar(
                x=models, y=[p*100 for p in probs], marker_color=bcolors,
                text=[f"{p*100:.1f}%" for p in probs], textposition="outside",
                textfont=dict(family="JetBrains Mono",size=11,color=TP), width=0.55))
            fig_cmp.add_hline(y=50,line_dash="dash",line_color=RD,
                              annotation_text="50% threshold",annotation_font_color=RD)
            fig_cmp.add_hline(y=20,line_dash="dot", line_color=GD,
                              annotation_text="20% threshold",annotation_font_color=GD)
            fig_cmp.update_layout(**playout(height=340),
                title=dict(text="Default Probability by Model (%)",font=dict(color=GD,size=13),x=0.0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Probability (%)",ticksuffix="%",
                           gridcolor="rgba(255,255,255,0.05)",range=[0,105]),showlegend=False)
            st.plotly_chart(fig_cmp,use_container_width=True,key="bar_cmp_t1")

        st.markdown("---")
        sec(f"📋 Financial Ratio Summary — {company_name}")
        rd_df = pd.DataFrame({
            "Ratio":  ["WC / TA","RE / TA","EBIT / TA","MVE / TL","Sales / TA",
                       "ROA","Leverage","Current Ratio","Debt / Market Cap"],
            "Value":  [f"{wc_ta:.4f}",f"{re_ta:.4f}",f"{ebit_ta:.4f}",f"{mve_tl:.4f}",
                       f"{sal_ta:.4f}",f"{roa:.4f}",f"{leverage:.4f}",f"{liquidity:.4f}",
                       f"{debt_face/max(market_cap,1):.4f}"],
            "Used In":["All Z-Score","All Z-Score","All Z-Score","Altman Public",
                       "All Z-Score","Zmijewski","Zmijewski/Ohlson","Zmijewski/Ohlson","Merton KMV"],
        })
        st.dataframe(rd_df.style
            .set_properties(**{"font-family":"JetBrains Mono,monospace","font-size":"12px","text-align":"center"})
            .set_table_styles([{"selector":"th","props":[("background-color",DB),("color",GD),
                                                          ("font-weight","700"),("text-align","center")]}]),
            use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────────
    #  TAB 2 — ALTMAN Z-SCORE
    # ──────────────────────────────────────────────────────────────────────────────

    with tab2:
        sec(f"🔢 {alt_label} — Detailed Breakdown")
        a1,a2 = st.columns([1,2])
        with a1:
            mcard("Z-Score",f'{alt["score"]:.4f}',sub=alt["zone"],color=alt["color"])
            mcard("Default Probability",f'{alt["prob"]*100:.2f}%',sub="Mapped from score bands",color=alt["color"])
            prob_gauge(alt["prob"],"Default Prob",alt_label,key="gauge_alt_t2")

        with a2:
            if firm_type=="Public (Listed)":
                cn = ["1.2 × WC/TA","1.4 × RE/TA","3.3 × EBIT/TA","0.6 × MVE/TL","1.0 × Sales/TA"]
                cv = [1.2*wc_ta,1.4*re_ta,3.3*ebit_ta,0.6*mve_tl,1.0*sal_ta]
            else:
                cn = ["0.717 × WC/TA","0.847 × RE/TA","3.107 × EBIT/TA","0.420 × BVE/TL","0.998 × Sales/TA"]
                cv = [0.717*wc_ta,0.847*re_ta,3.107*ebit_ta,0.420*bve_tl,0.998*sal_ta]
            fig_wf = go.Figure(go.Bar(
                y=cn, x=cv, orientation="h",
                marker_color=[GR if v>=0 else RD for v in cv],
                text=[f"{v:+.4f}" for v in cv], textposition="outside",
                textfont=dict(family="JetBrains Mono",size=10,color=TP), width=0.6))
            fig_wf.add_vline(x=0,line_color="rgba(255,255,255,0.2)",line_width=1)
            fig_wf.update_layout(**playout(height=300,margin=dict(l=10,r=80,t=40,b=30)),
                title=dict(text="Z-Score Component Contributions",font=dict(color=GD,size=13),x=0.0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=False),showlegend=False)
            st.plotly_chart(fig_wf,use_container_width=True,key="wf_alt_t2")

            score_val = alt["score"]
            fig_band = go.Figure()
            for x0,x1,clr in [(0,1.81,"rgba(220,53,69,0.2)"),(1.81,2.99,"rgba(255,215,0,0.15)"),(2.99,6,"rgba(40,167,69,0.15)")]:
                fig_band.add_shape(type="rect",x0=x0,x1=x1,y0=0,y1=1,fillcolor=clr,line_width=0)
            fig_band.add_vline(x=score_val,line_color=GD,line_width=3,
                               annotation_text=f"  Z={score_val:.3f}",annotation_font_color=GD)
            for x,lbl in [(0.9,"DISTRESS"),(2.4,"GREY"),(4.5,"SAFE")]:
                fig_band.add_annotation(x=x,y=0.5,text=lbl,font=dict(color=TP,size=11),showarrow=False)
            fig_band.update_layout(**playout(height=120,margin=dict(l=20,r=20,t=30,b=30)),
                title=dict(text="Score Position on Zone Bands",font=dict(color=GD,size=12),x=0.0),
                xaxis=dict(range=[0,6],tickvals=[1.81,2.99],ticktext=["1.81","2.99"],
                           gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(visible=False))
            st.plotly_chart(fig_band,use_container_width=True,key="band_alt_t2")

        st.markdown("---")
        sec("🔍 Sensitivity: ±10% Change in Each Ratio")
        if firm_type=="Public (Listed)":
            wts  = {"WC/TA":1.2,"RE/TA":1.4,"EBIT/TA":3.3,"MVE/TL":0.6,"Sales/TA":1.0}
            base = {"WC/TA":wc_ta,"RE/TA":re_ta,"EBIT/TA":ebit_ta,"MVE/TL":mve_tl,"Sales/TA":sal_ta}
        else:
            wts  = {"WC/TA":0.717,"RE/TA":0.847,"EBIT/TA":3.107,"BVE/TL":0.420,"Sales/TA":0.998}
            base = {"WC/TA":wc_ta,"RE/TA":re_ta,"EBIT/TA":ebit_ta,"BVE/TL":bve_tl,"Sales/TA":sal_ta}
        sens = [{"Ratio":r,"Weight":w,"Base Value":round(base[r],4),
                 "Z (Base)":round(alt["score"],4),
                 "Z (+10%)":round(alt["score"]+w*base[r]*0.10,4),
                 "Z (−10%)":round(alt["score"]-w*base[r]*0.10,4),
                 "Unit Impact":round(w,3)} for r,w in wts.items()]
        st.dataframe(pd.DataFrame(sens).style
            .set_properties(**{"font-family":"JetBrains Mono,monospace","font-size":"11px","text-align":"center"})
            .set_table_styles([{"selector":"th","props":[("background-color",DB),("color",GD),
                                                          ("font-weight","700"),("text-align","center")]}]),
            use_container_width=True,hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────────
    #  TAB 3 — OHLSON
    # ──────────────────────────────────────────────────────────────────────────────

    with tab3:
        sec("📉 Ohlson O-Score — Logistic Regression Model")
        o1,o2 = st.columns([1,2])
        with o1:
            mcard("O-Score",f'{ohl["score"]:.4f}',sub=ohl["zone"],color=ohl["color"])
            mcard("Default Probability",f'{ohl["prob"]*100:.2f}%',sub="P=1/(1+e^−O)",color=ohl["color"])
            prob_gauge(ohl["prob"],"Default Prob","Ohlson O-Score",key="gauge_ohl_t3")
        with o2:
            o_range = np.linspace(-6,6,300)
            p_range = expit(o_range)
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(x=o_range,y=p_range*100,mode="lines",
                                         line=dict(color=LB,width=2.5),name="Logistic Curve"))
            fig_log.add_vline(x=ohl["score"],line_color=GD,line_width=2,line_dash="dash")
            fig_log.add_hline(y=50,line_color=RD,line_width=1,line_dash="dot",
                              annotation_text="50% threshold",annotation_font_color=RD)
            fig_log.add_trace(go.Scatter(x=[ohl["score"]],y=[ohl["prob"]*100],mode="markers",
                                         marker=dict(size=14,color=GD,line=dict(width=2,color=TP)),
                                         name=f"Current: {ohl['score']:.3f}"))
            fig_log.update_layout(**playout(height=320),
                title=dict(text="Ohlson Logistic Curve — P(default) vs O-Score",
                           font=dict(color=GD,size=13),x=0.0),
                xaxis=dict(title="O-Score",gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Default Probability (%)",ticksuffix="%",
                           gridcolor="rgba(255,255,255,0.05)",range=[0,100]),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TP)))
            st.plotly_chart(fig_log,use_container_width=True,key="log_ohl_t3")

    # ──────────────────────────────────────────────────────────────────────────────
    #  TAB 4 — ZMIJEWSKI
    # ──────────────────────────────────────────────────────────────────────────────

    with tab4:
        sec("📐 Zmijewski Score — Probit Regression Model")
        z1,z2 = st.columns([1,2])
        with z1:
            mcard("Zmijewski Score",f'{zmi["score"]:.4f}',sub=zmi["zone"],color=zmi["color"])
            mcard("Default Probability",f'{zmi["prob"]*100:.2f}%',sub="P=Φ(X)–Normal CDF",color=zmi["color"])
            prob_gauge(zmi["prob"],"Default Prob","Zmijewski Probit",key="gauge_zmi_t4")
        with z2:
            x_range  = np.linspace(-5,5,300)
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(x=x_range,y=stats.norm.cdf(x_range)*100,mode="lines",
                                          line=dict(color=LB,width=2.5),name="Probit Curve"))
            fig_prob.add_vline(x=zmi["score"],line_color=GD,line_width=2,line_dash="dash")
            fig_prob.add_hline(y=50,line_color=RD,line_width=1,line_dash="dot",
                               annotation_text="50% threshold",annotation_font_color=RD)
            fig_prob.add_trace(go.Scatter(x=[zmi["score"]],y=[zmi["prob"]*100],mode="markers",
                                          marker=dict(size=14,color=GD,line=dict(width=2,color=TP)),
                                          name=f"Current: {zmi['score']:.3f}"))
            fig_prob.update_layout(**playout(height=320),
                title=dict(text="Zmijewski Probit Curve — P(default) vs Score",
                           font=dict(color=GD,size=13),x=0.0),
                xaxis=dict(title="Zmijewski Score",gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Default Probability (%)",ticksuffix="%",
                           gridcolor="rgba(255,255,255,0.05)",range=[0,100]),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TP)))
            st.plotly_chart(fig_prob,use_container_width=True,key="probit_zmi_t4")

        st.markdown("---")
        sec("🗺 Probability Surface: ROA vs Leverage")
        roa_r  = np.linspace(-0.20,0.20,40)
        lev_r  = np.linspace(0.10,0.95,40)
        ROA_g,LEV_g = np.meshgrid(roa_r,lev_r)
        X_g    = -4.336-4.513*ROA_g+5.679*LEV_g+0.004*liquidity
        P_g    = stats.norm.cdf(X_g)*100
        fig_sf = go.Figure(go.Heatmap(
            z=P_g,x=np.round(roa_r,3),y=np.round(lev_r,3),
            colorscale=[[0,"#004d00"],[0.2,"#28a745"],[0.5,"#FFD700"],[0.8,"#dc3545"],[1,"#8b0000"]],
            colorbar=cbar("P(default) %"),hoverongaps=False))
        fig_sf.add_trace(go.Scatter(x=[roa],y=[leverage],mode="markers",
            marker=dict(size=14,color=GD,symbol="star",line=dict(width=2,color=TP)),
            name=company_name,showlegend=True))
        fig_sf.update_layout(**playout(height=400,margin=dict(l=60,r=80,t=50,b=60)),
            title=dict(text="Default Probability (%) — ROA vs Leverage Plane",
                       font=dict(color=GD,size=13),x=0.0),
            xaxis=dict(title="ROA",gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Leverage (TL/TA)",gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TP)))
        st.plotly_chart(fig_sf,use_container_width=True,key="surf_zmi_t4")

    # ──────────────────────────────────────────────────────────────────────────────
    #  TAB 5 — MERTON KMV
    # ──────────────────────────────────────────────────────────────────────────────

    with tab5:
        sec("🔬 Merton KMV Structural Model — Distance to Default")
        m1,m2,m3 = st.columns(3)
        with m1:
            mcard("Asset Value (V)",f'{kmv["asset_value"]:,.2f}',sub=f"vs Debt {debt_face:,.0f}",color=LB)
            mcard("Asset Volatility σ_V",f'{kmv["asset_vol"]:.2f}%',sub=f"vs Equity {equity_vol*100:.1f}%",color=LB)
        with m2:
            mcard("Distance to Default",f'{kmv["dd"]:.4f}',sub="σ units from default",
                  color=GD if kmv["dd"]>2 else RD)
            mcard("d₁",f'{kmv["d1"]:.4f}',sub="N(d1)=Equity delta",color=TS)
        with m3:
            mcard("d₂",f'{kmv["d2"]:.4f}',sub="N(d2)=Risk-neutral PD",color=TS)
            mcard("Default Probability",f'{kmv["prob"]*100:.2f}%',sub=kmv["zone"],color=kmv["color"])

        prob_gauge(kmv["prob"],"Default Prob","Merton KMV",key="gauge_kmv_t5")

        st.markdown("---")
        sec("📈 Asset Value Distribution vs Debt Threshold")
        V0     = kmv["asset_value"]
        sig_V  = kmv["asset_vol"]/100
        mu_V   = risk_free-0.5*sig_V**2
        x_v    = np.linspace(V0*0.2,V0*2.5,400)
        y_v    = stats.lognorm.pdf(x_v,s=sig_V,scale=V0*np.exp(mu_V))
        fig_d  = go.Figure()
        ms     = x_v>=debt_face
        md     = x_v<=debt_face
        fig_d.add_trace(go.Scatter(x=x_v[ms],y=y_v[ms],fill="tozeroy",
                                   fillcolor="rgba(40,167,69,0.2)",line=dict(color=GR,width=0),
                                   name="V_T > Debt (Solvent)"))
        fig_d.add_trace(go.Scatter(x=x_v[md],y=y_v[md],fill="tozeroy",
                                   fillcolor="rgba(220,53,69,0.3)",line=dict(color=RD,width=0),
                                   name="V_T < Debt (Default)"))
        fig_d.add_trace(go.Scatter(x=x_v,y=y_v,mode="lines",
                                   line=dict(color=LB,width=2.5),name="Asset Distribution"))
        fig_d.add_vline(x=debt_face,line_color=RD,line_width=2,line_dash="dash",
                        annotation_text=f"  Debt={debt_face:,.0f}",annotation_font_color=RD)
        fig_d.add_vline(x=V0,line_color=GD,line_width=2,line_dash="dot",
                        annotation_text=f"  V₀={V0:,.0f}",annotation_font_color=GD)
        fig_d.update_layout(**playout(height=380),
            title=dict(text="Lognormal Asset Value Distribution at T=1yr",
                       font=dict(color=GD,size=13),x=0.0),
            xaxis=dict(title="Asset Value V_T",gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Probability Density",gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TP)))
        st.plotly_chart(fig_d,use_container_width=True,key="dist_kmv_t5")

    # ──────────────────────────────────────────────────────────────────────────────
    #  TAB 6 — BANKRUPTCY COSTS
    # ──────────────────────────────────────────────────────────────────────────────

    with tab6:
        sec("💸 Bankruptcy Cost Estimation")
        bc1,bc2,bc3,bc4 = st.columns(4)
        with bc1: mcard("Direct Costs",f'{curr} {bcc["direct_total"]:,.1f}',
                        sub=f'{bcc["direct_total"]/max(total_assets,1)*100:.1f}% of assets',color=GD)
        with bc2: mcard("Indirect Costs",f'{curr} {bcc["indirect_total"]:,.1f}',
                        sub=f'{bcc["indirect_total"]/max(total_assets,1)*100:.1f}% of assets',color=GD)
        with bc3: mcard("Total Bankruptcy Cost",f'{curr} {bcc["total_cost"]:,.1f}',
                        sub=f'{bcc["pct_assets"]:.1f}% of Total Assets',color=RD)
        with bc4: mcard("Cost as % Revenue",f'{bcc["pct_revenue"]:.1f}%',
                        sub=f'{years_distress:.1f} years distress',color=RD)

        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            sec("🔴 Direct Costs Breakdown")
            fig_dir = go.Figure(go.Pie(
                labels=["Legal & Advisory","Administrative","Industry-Specific"],
                values=[bcc["direct_legal"],bcc["direct_admin"],bcc["direct_other"]],
                hole=0.45, marker=dict(colors=[RD,"#c0392b","#e74c3c"],line=dict(color=CB,width=2)),
                textfont=dict(color=TP,size=11), textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>"))
            fig_dir.update_layout(**playout(height=320,margin=dict(l=20,r=20,t=40,b=20)),
                title=dict(text=f"Direct Costs (Total: {curr} {bcc['direct_total']:,.1f})",
                           font=dict(color=GD,size=13),x=0.0),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TP,size=10)))
            st.plotly_chart(fig_dir,use_container_width=True,key="pie_dir_t6")
        with c2:
            sec("🟡 Indirect Costs Breakdown")
            fig_ind = go.Figure(go.Pie(
                labels=["Lost Sales","Mgmt Distraction","Supplier Credit","Customer Defection"],
                values=[bcc["lost_sales"],bcc["mgmt_dist"],bcc["supplier_cost"],bcc["customer_flight"]],
                hole=0.45, marker=dict(colors=[GD,"#d4af37",LB,"#85c1e9"],line=dict(color=CB,width=2)),
                textfont=dict(color=CB,size=11), textinfo="label+percent"))
            fig_ind.update_layout(**playout(height=320,margin=dict(l=20,r=20,t=40,b=20)),
                title=dict(text=f"Indirect Costs (Total: {curr} {bcc['indirect_total']:,.1f})",
                           font=dict(color=GD,size=13),x=0.0),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TP,size=10)))
            st.plotly_chart(fig_ind,use_container_width=True,key="pie_ind_t6")

        st.markdown("---")
        sec("📊 Total Cost Build-Up — Waterfall")
        wf_labs = ["Legal","Admin","Industry Direct","Lost Sales","Mgmt Distract","Supplier","Customer","TOTAL"]
        wf_vals = [bcc["direct_legal"],bcc["direct_admin"],bcc["direct_other"],
                   bcc["lost_sales"],bcc["mgmt_dist"],bcc["supplier_cost"],
                   bcc["customer_flight"],bcc["total_cost"]]
        fig_wf2 = go.Figure(go.Waterfall(
            orientation="v", measure=["relative"]*7+["total"],
            x=wf_labs, y=wf_vals,
            connector=dict(line=dict(color=TS)),
            increasing=dict(marker_color=RD), totals=dict(marker_color=GD),
            text=[f"{curr} {v:,.1f}" for v in wf_vals], textposition="outside",
            textfont=dict(family="JetBrains Mono",size=9,color=TP)))
        fig_wf2.update_layout(**playout(height=400),
            title=dict(text="Bankruptcy Cost Waterfall",font=dict(color=GD,size=13),x=0.0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=f"Cost ({currency})",gridcolor="rgba(255,255,255,0.05)"),
            showlegend=False)
        st.plotly_chart(fig_wf2,use_container_width=True,key="wf2_cost_t6")

        st.markdown("---")
        sec("🎯 Expected Bankruptcy Cost (Probability-Adjusted)")
        exp_cost = ensemble_prob*bcc["total_cost"]
        e1,e2,e3 = st.columns(3)
        with e1: mcard("Ensemble Prob",f'{ensemble_prob*100:.2f}%',sub="Avg of 4 models",
                       color=RD if ensemble_prob>0.5 else GD)
        with e2: mcard("Total Bankruptcy Cost",f'{curr} {bcc["total_cost"]:,.1f}',
                       sub="Direct + Indirect",color=GD)
        with e3: mcard("Expected Cost",f'{curr} {exp_cost:,.1f}',
                       sub="P(default) × Total Cost",color=RD if ensemble_prob>0.3 else GD)

        cost_df = pd.DataFrame({
            "Cost Component":["Legal","Admin","Industry Direct","Total Direct",
                              "Lost Sales","Mgmt Distraction","Supplier","Customer",
                              "Total Indirect","TOTAL","Expected (Prob-Weighted)"],
            f"Amount ({currency})":[bcc["direct_legal"],bcc["direct_admin"],bcc["direct_other"],
                                     bcc["direct_total"],bcc["lost_sales"],bcc["mgmt_dist"],
                                     bcc["supplier_cost"],bcc["customer_flight"],
                                     bcc["indirect_total"],bcc["total_cost"],round(exp_cost,2)],
            "% of Assets":[f"{v/max(total_assets,1)*100:.2f}%"
                           for v in [bcc["direct_legal"],bcc["direct_admin"],bcc["direct_other"],
                                      bcc["direct_total"],bcc["lost_sales"],bcc["mgmt_dist"],
                                      bcc["supplier_cost"],bcc["customer_flight"],
                                      bcc["indirect_total"],bcc["total_cost"],exp_cost]],
        })
        st.download_button("⬇ Download Cost Report CSV",
                           cost_df.to_csv(index=False).encode(),
                           "bankruptcy_costs.csv","text/csv")



# ══════════════════════════════════════════════════════════════════════════════
#  ROW 2 — REFERENCE & EDUCATION
# ══════════════════════════════════════════════════════════════════════════════
else:  # 📚 Reference & Education
    ref_tab1, ref_tab2 = st.tabs([
        "📖 Methodology",
        "🎓 Model Education",
    ])

    with ref_tab1:
        # ──────────────────────────────────────────────────────────────────────────────
        #  TAB 7 — METHODOLOGY
        # ──────────────────────────────────────────────────────────────────────────────
        sec("📖 Methodology & Technical Reference")
        ibox("<p>This dashboard tracks four complementary bankruptcy prediction models and a cost estimation "
             "framework. All models are computed in real-time from sidebar inputs. Risk-free rate assumed: "
             f"<b>user-specified ({risk_free*100:.1f}% p.a.)</b>. Ensemble probability = simple average of 4 models.</p>",
             title="About This Dashboard")
        
        models_ref = [
            ("Altman Z-Score (1968)","Public Mfg","MDA","5 ratios → linear score → zone",
             "Simple, interpretable","Sample age; no direct probability"),
            ("Altman Z'-Score (1983)","Private","MDA","Book equity replaces market value",
             "No market data needed","Same linearity limits"),
            ("Ohlson O-Score (1980)","All firms","Logit","9 vars → sigmoid → probability",
             "Calibrated probability","GNP deflator; CHIN needs prior year"),
            ("Zmijewski (1984)","All firms","Probit","3 ratios → normal CDF",
             "Parsimonious; bias-corrected","Only 3 variables"),
            ("Merton KMV (1974)","Public","Structural","Option pricing → asset value → DD",
             "Forward-looking; uses market data","GBM assumption; needs equity vol"),
        ]
        st.dataframe(pd.DataFrame(models_ref,
            columns=["Model","Applicability","Method","Mechanism","Strengths","Limitations"]).style
            .set_properties(**{"font-family":"JetBrains Mono,monospace","font-size":"11px","text-align":"left"})
            .set_table_styles([{"selector":"th","props":[("background-color",DB),("color",GD),
                                                          ("font-weight","700"),("text-align","center")]}]),
            use_container_width=True,hide_index=True)
        
        st.markdown("---")
        sec("📐 Key Formulae")
        f1,f2 = st.columns(2)
        with f1:
            ibox("<div class='formula-box'>Z = 1.2·(WC/TA)+1.4·(RE/TA)+3.3·(EBIT/TA)+0.6·(MVE/TL)+1.0·(S/TA)<br>"
                 "Safe: Z&gt;2.99 | Grey: 1.81–2.99 | Distress: Z&lt;1.81</div>",
                 title="Altman Z-Score (Public)")
            ibox("<div class='formula-box'>O = −1.32 − 0.407·SIZE + 6.03·TLTA − 1.43·WCTA + 0.076·CLCA<br>"
                 "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− 1.72·OENEG − 2.37·NITA − 1.83·FUTL + 0.285·INTWO − 0.521·CHIN<br>"
                 "P(default) = 1/(1+e<sup>−O</sup>)</div>",
                 title="Ohlson O-Score")
            ibox("<div class='formula-box'>E[BC] = P(default) × [Direct + Indirect Costs]<br>"
                 "Optimum: PV(Tax Shield) = E[Bankruptcy Cost]</div>",
                 title="Expected Bankruptcy Cost")
        with f2:
            ibox("<div class='formula-box'>X = −4.336 − 4.513·ROA + 5.679·Leverage + 0.004·Liquidity<br>"
                 "P(default) = Φ(X)</div>",
                 title="Zmijewski Score")
            ibox("<div class='formula-box'>d₁ = [ln(V/D)+(r+½σ²ᵥ)T]/(σᵥ√T)<br>"
                 "d₂ = d₁ − σᵥ√T = Distance-to-Default<br>"
                 "P(default) = N(−d₂)</div>",
                 title="Merton KMV")
        
        st.markdown("---")
        ibox("<p>• All models based on historical data — not future predictions.<br>"
             "• Ensemble probability is indicative only — equal model weights may not suit all contexts.<br>"
             "• Bankruptcy cost estimates use Warner (1977), Altman (1984), Andrade &amp; Kaplan (1998) benchmarks.<br>"
             "• Merton KMV requires publicly traded equity; use accounting models for private firms.<br>"
             "• Under IBC 2016 (India), direct CIRP costs typically 2–6% of resolution value.</p>",
             title="⚠️ Limitations & Disclaimers", bc=RD)
        
        
        

    with ref_tab2:
        # ──────────────────────────────────────────────────────────────────────────────
        #  TAB 8 — MODEL EDUCATION
        # ──────────────────────────────────────────────────────────────────────────────
        sec("🎓 Bankruptcy Prediction Models — Detailed Educational Notes")
        
        st.html(f"""
        <div class="info-box" style="margin-bottom:1.5rem;">
          <p style="margin:0;font-size:0.92rem;line-height:1.8;">
          This tab provides rigorous academic treatment of every model —
          covering <b style="color:{GD};-webkit-text-fill-color:{GD};">theory, derivation, assumptions,
          empirical evidence, limitations</b> and
          <b style="color:{GD};-webkit-text-fill-color:{GD};">practical guidance</b>
          for FRM, CFA, and MBA students.
          </p>
        </div>
        """)
        
        edu_tabs = st.tabs([
            "1️⃣ Altman Z-Score",
            "2️⃣ Ohlson O-Score",
            "3️⃣ Zmijewski Probit",
            "4️⃣ Merton KMV",
            "5️⃣ Bankruptcy Costs",
            "📊 Model Comparison",
        ])
        
        # ── EDU 1: Altman ─────────────────────────────────────────────────────────
        with edu_tabs[0]:
            sec("🔢 Altman Z-Score (1968) & Z'-Score (1983)")
            c1, c2 = st.columns(2)
            with c1:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">📜 Historical Background</h4>
                  <p style="font-size:0.85rem;line-height:1.9;">
                  Edward I. Altman published the Z-Score in 1968 in the <i>Journal of Finance</i>,
                  using 66 manufacturing firms (33 bankrupt, 33 non-bankrupt) from 1946–1965.
                  It was among the first applications of <b>Multiple Discriminant Analysis (MDA)</b>
                  to corporate bankruptcy — a landmark in quantitative finance.<br><br>
                  Extended in 1983 (Z'-Score) for <b>private firms</b> by substituting book equity
                  for market value. A further variant (Z''-Score) targets non-manufacturing and
                  emerging-market firms.
                  </p>
                </div>
                <div class="info-box" style="margin-top:12px;">
                  <h4 style="color:{GD};margin-top:0;">⚙️ Multiple Discriminant Analysis</h4>
                  <p style="font-size:0.85rem;line-height:1.9;">
                  MDA finds a <b>linear combination</b> of predictors that maximises the ratio of
                  between-group variance to within-group variance — equivalent to maximising the
                  <b>Mahalanobis distance</b> between bankrupt and non-bankrupt group centroids.
                  </p>
                  <div class="formula-box">Z = w&#8321;X&#8321; + w&#8322;X&#8322; + ... + w&#8345;X&#8345;</div>
                  <p style="font-size:0.82rem;line-height:1.8;">
                  Weights wᵢ chosen so the discriminant score best separates the two groups.
                  Classification zones are derived empirically from the estimation sample.
                  </p>
                </div>
                """)
        
            with c2:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">📐 Five Variables Explained</h4>
                  <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
                    <tr style="border-bottom:1px solid rgba(255,215,0,0.3);">
                      <th style="color:{GD};padding:6px;text-align:left;">Var</th>
                      <th style="color:{GD};padding:6px;text-align:left;">Ratio</th>
                      <th style="color:{GD};padding:6px;text-align:left;">Economic Meaning</th>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:5px;color:{LB};">X&#8321;</td>
                      <td style="padding:5px;">WC/TA</td>
                      <td style="padding:5px;">Short-term liquidity buffer</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:5px;color:{LB};">X&#8322;</td>
                      <td style="padding:5px;">RE/TA</td>
                      <td style="padding:5px;">Cumulative profitability; proxy for firm age</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:5px;color:{LB};">X&#8323;</td>
                      <td style="padding:5px;">EBIT/TA</td>
                      <td style="padding:5px;">Core operating return, pre-leverage</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:5px;color:{LB};">X&#8324;</td>
                      <td style="padding:5px;">MVE/TL</td>
                      <td style="padding:5px;">Market solvency cushion over obligations</td>
                    </tr>
                    <tr>
                      <td style="padding:5px;color:{LB};">X&#8325;</td>
                      <td style="padding:5px;">Sales/TA</td>
                      <td style="padding:5px;">Asset turnover; competitive efficiency</td>
                    </tr>
                  </table>
                </div>
                <div class="info-box" style="margin-top:12px;">
                  <h4 style="color:{GD};margin-top:0;">🎯 Zone Cutoffs</h4>
                  <div style="font-size:0.85rem;line-height:2.2;">
                    <span style="color:{GR};font-weight:700;">✅ Safe Zone Z &gt; 2.99</span> — Healthy; bankruptcy unlikely<br>
                    <span style="color:{GD};font-weight:700;">⚠️ Grey Zone 1.81–2.99</span> — Elevated risk; monitor closely<br>
                    <span style="color:{RD};font-weight:700;">🚨 Distress Zone Z &lt; 1.81</span> — High risk; ~95% accuracy in original sample<br><br>
                    <b>Z'-Score (Private) cutoffs:</b><br>
                    Safe: Z' &gt; 2.9 &nbsp;|&nbsp; Grey: 1.23–2.9 &nbsp;|&nbsp; Distress: Z' &lt; 1.23
                  </div>
                </div>
                """)
        
            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">✅ Key Assumptions</h4>
                  <ul style="line-height:2.0;font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    <li><b>Multivariate normality</b> of predictors within each group</li>
                    <li><b>Equal covariance matrices</b> across groups (homoscedasticity)</li>
                    <li><b>Linear separability</b> — linear combination sufficiently discriminates</li>
                    <li><b>Independent observations</b> — no serial correlation</li>
                    <li><b>Stationary relationships</b> — 1946–65 coefficients remain valid today</li>
                    <li><b>Accounting comparability</b> across firms and GAAP systems</li>
                  </ul>
                </div>
                """)
            with cb:
                st.html(f"""
                <div class="info-box" style="border-color:{RD};">
                  <h4 style="color:{RD};margin-top:0;">⚠️ Limitations &amp; Criticisms</h4>
                  <ul style="line-height:2.0;font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    <li><b>Sample bias:</b> Only 66 US manufacturing firms — not representative of services or tech</li>
                    <li><b>Temporal instability:</b> Coefficients dated; economy and standards have changed</li>
                    <li><b>No direct probability:</b> Score maps to zones, not a calibrated default probability</li>
                    <li><b>Linearity constraint:</b> Cannot capture non-linear relationships</li>
                    <li><b>MDA assumptions violated:</b> Financial ratios are rarely normally distributed</li>
                    <li><b>Window dressing:</b> Susceptible to earnings management</li>
                  </ul>
                </div>
                """)
        
            st.html(f"""
            <div class="info-box" style="margin-top:12px;">
              <h4 style="color:{GD};margin-top:0;">📚 Empirical Evidence</h4>
              <p style="font-size:0.85rem;line-height:1.9;">
              Altman (1968) reported <b>95% accuracy one year prior</b> to bankruptcy on the original sample,
              72% two years prior. Out-of-sample studies show 70–80% accuracy. Begley, Ming &amp; Watts (1996)
              re-estimated on 1980s data and found the original coefficients performed poorly out-of-sample,
              suggesting coefficient instability. Despite criticisms, the Z-Score remains the most cited model
              in credit analysis and bond covenant design. It is part of the <b>CFA Level 1 curriculum</b>
              and referenced in Basel II internal ratings guidance.
              </p>
            </div>
            """)
        
        # ── EDU 2: Ohlson ─────────────────────────────────────────────────────────
        with edu_tabs[1]:
            sec("📉 Ohlson O-Score (1980) — Logistic Regression")
            c1, c2 = st.columns(2)
            with c1:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">📜 Historical Background</h4>
                  <p style="font-size:0.85rem;line-height:1.9;">
                  James A. Ohlson (1980) published in the <i>Journal of Accounting Research</i>
                  using <b>2,163 firms (105 bankrupt)</b> from SEC filings 1970–1976 — far larger
                  and more representative than Altman's sample.<br><br>
                  Key innovation: <b>conditional logit</b> instead of MDA, directly producing a
                  <b>calibrated probability</b> of bankruptcy. Also introduced lag structures and
                  addressed choice-based sampling bias.
                  </p>
                </div>
                <div class="info-box" style="margin-top:12px;">
                  <h4 style="color:{GD};margin-top:0;">⚙️ Logistic Regression Mechanics</h4>
                  <div class="formula-box">
                    P(Y=1) = 1 / (1 + e<sup>−(β&#8320; + β&#8321;X&#8321; + ... + βₙXₙ)</sup>)<br><br>
                    ln[P/(1−P)] = β&#8320; + β&#8321;X&#8321; + ... = O-Score
                  </div>
                  <p style="font-size:0.82rem;line-height:1.8;">
                  Unlike MDA, logistic regression does <b>NOT</b> require multivariate normality
                  or equal covariance matrices — more robust to financial ratio distributions.
                  </p>
                </div>
                """)
            with c2:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">📐 Nine Variables with Coefficients</h4>
                  <table style="width:100%;border-collapse:collapse;font-size:0.80rem;">
                    <tr style="border-bottom:1px solid rgba(255,215,0,0.3);">
                      <th style="color:{GD};padding:5px;">Var</th>
                      <th style="color:{GD};padding:5px;">Definition</th>
                      <th style="color:{GD};padding:5px;">Coeff</th>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">SIZE</td><td style="padding:4px;">log(TA)</td>
                      <td style="padding:4px;color:{GR};">−0.407</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">TLTA</td><td style="padding:4px;">TL / TA</td>
                      <td style="padding:4px;color:{RD};">+6.03</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">WCTA</td><td style="padding:4px;">WC / TA</td>
                      <td style="padding:4px;color:{GR};">−1.43</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">CLCA</td><td style="padding:4px;">CL / CA</td>
                      <td style="padding:4px;color:{RD};">+0.076</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">OENEG</td><td style="padding:4px;">1 if TL &gt; TA</td>
                      <td style="padding:4px;color:{GR};">−1.72</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">NITA</td><td style="padding:4px;">NI / TA</td>
                      <td style="padding:4px;color:{GR};">−2.37</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">FUTL</td><td style="padding:4px;">FFO / TL</td>
                      <td style="padding:4px;color:{GR};">−1.83</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">INTWO</td><td style="padding:4px;">1 if NI &lt; 0</td>
                      <td style="padding:4px;color:{RD};">+0.285</td>
                    </tr>
                    <tr>
                      <td style="padding:4px;color:{LB};">CHIN</td><td style="padding:4px;">Change in NI</td>
                      <td style="padding:4px;color:{GR};">−0.521</td>
                    </tr>
                  </table>
                </div>
                """)
        
            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">✅ Key Assumptions</h4>
                  <ul style="line-height:2.0;font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    <li><b>Binary outcome:</b> Y=1 (bankrupt within 1 year), Y=0 (survived)</li>
                    <li><b>Linear log-odds:</b> Logit is linear in predictors</li>
                    <li><b>Independent observations</b> — no firm appears twice</li>
                    <li><b>No perfect multicollinearity</b> among predictors</li>
                    <li><b>Sufficient sample size:</b> ~10–15 events per predictor for reliable MLE</li>
                  </ul>
                </div>
                """)
            with cb:
                st.html(f"""
                <div class="info-box" style="border-color:{RD};">
                  <h4 style="color:{RD};margin-top:0;">⚠️ Limitations</h4>
                  <ul style="line-height:2.0;font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    <li><b>CHIN requires prior-year NI:</b> Not computable for single-year reports</li>
                    <li><b>Industry-agnostic:</b> Single coefficient set across all sectors</li>
                    <li><b>Static model:</b> Shumway (2001) — time-varying hazard rates improve accuracy</li>
                    <li><b>50% threshold arbitrary:</b> Optimal cutoff depends on misclassification costs</li>
                    <li><b>Accounting manipulation:</b> Susceptible to earnings quality issues</li>
                  </ul>
                </div>
                """)
        
        # ── EDU 3: Zmijewski ──────────────────────────────────────────────────────
        with edu_tabs[2]:
            sec("📐 Zmijewski Score (1984) — Probit Regression")
            c1, c2 = st.columns(2)
            with c1:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">📜 Historical Background</h4>
                  <p style="font-size:0.85rem;line-height:1.9;">
                  Mark E. Zmijewski (1984) in the <i>Journal of Accounting Research</i> provided primarily
                  a <b>methodological critique</b>. He identified two critical biases in prior studies:<br><br>
                  <b>1. Choice-based sampling bias</b> — oversampling bankrupt firms inflates apparent
                  accuracy by <b>10–15 percentage points</b>.<br>
                  <b>2. Complete data bias</b> — excluding firms with missing data correlates with distress.<br><br>
                  Used probit on <b>840 firms (40 bankrupt)</b> from 1972–1978, preserving the true
                  population bankruptcy ratio.
                  </p>
                </div>
                """)
            with c2:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">⚙️ Probit vs Logit</h4>
                  <table style="width:100%;font-size:0.82rem;border-collapse:collapse;">
                    <tr style="border-bottom:1px solid rgba(255,215,0,0.3);">
                      <th style="color:{GD};padding:6px;">Feature</th>
                      <th style="color:{GD};padding:6px;">Logit (Ohlson)</th>
                      <th style="color:{GD};padding:6px;">Probit (Zmijewski)</th>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:5px;color:{LB};">Link function</td>
                      <td style="padding:5px;">Sigmoid 1/(1+e⁻ˣ)</td>
                      <td style="padding:5px;">Normal CDF Φ(x)</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:5px;color:{LB};">Error distribution</td>
                      <td style="padding:5px;">Logistic</td>
                      <td style="padding:5px;">Standard Normal</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:5px;color:{LB};">Tail behaviour</td>
                      <td style="padding:5px;">Heavier tails</td>
                      <td style="padding:5px;">Lighter tails</td>
                    </tr>
                    <tr>
                      <td style="padding:5px;color:{LB};">Practical difference</td>
                      <td style="padding:5px;" colspan="2">Very similar — choice rarely matters</td>
                    </tr>
                  </table>
                  <div class="formula-box" style="margin-top:10px;">
                    X = −4.336 − 4.513·ROA + 5.679·Leverage + 0.004·Liquidity<br>
                    P(default) = Φ(X)
                  </div>
                </div>
                """)
        
            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">✅ Key Assumptions</h4>
                  <ul style="line-height:2.0;font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    <li><b>Normal latent variable:</b> Distress propensity normally distributed</li>
                    <li><b>Representative sampling:</b> True bankruptcy frequency preserved (WESML)</li>
                    <li><b>Three ratios sufficient:</b> ROA, leverage, liquidity capture key health dimensions</li>
                    <li><b>Linearity in probit index</b></li>
                  </ul>
                </div>
                """)
            with cb:
                st.html(f"""
                <div class="info-box" style="border-color:{RD};">
                  <h4 style="color:{RD};margin-top:0;">⚠️ Limitations</h4>
                  <ul style="line-height:2.0;font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    <li><b>Only 3 variables</b> — misses important information in complex distress situations</li>
                    <li><b>No market data</b> — entirely backward-looking accounting model</li>
                    <li><b>US-centric:</b> Calibrated on US firms 1972–78; recalibration needed for India</li>
                    <li><b>Static model:</b> Point-in-time; doesn't capture deteriorating trends</li>
                  </ul>
                </div>
                """)
        
        # ── EDU 4: Merton KMV ─────────────────────────────────────────────────────
        with edu_tabs[3]:
            sec("🔬 Merton KMV Structural Model (1974) — Distance to Default")
            c1, c2 = st.columns(2)
            with c1:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">📜 Historical Background</h4>
                  <p style="font-size:0.85rem;line-height:1.9;">
                  Robert C. Merton (1974) in the <i>Journal of Finance</i> extended Black-Scholes (1973)
                  to corporate liabilities, showing that <b>equity is a call option on firm assets</b>
                  with debt face value as the strike price.<br><br>
                  KMV Corporation commercialised this as <b>Expected Default Frequency (EDF™)</b> in the
                  1990s, mapping Distance-to-Default to actual default rates using a large empirical database.
                  Acquired by Moody's in 2002 — now the basis of <b>Moody's Analytics CreditEdge™</b>.
                  </p>
                </div>
                <div class="info-box" style="margin-top:12px;">
                  <h4 style="color:{GD};margin-top:0;">💡 Core Intuition</h4>
                  <p style="font-size:0.85rem;line-height:1.8;">
                  Equity holders have a <b>residual claim</b> on firm assets after debt:
                  </p>
                  <div class="formula-box">E = max(V − D, 0)</div>
                  <p style="font-size:0.82rem;line-height:1.8;">
                  This is a <b>European call option</b> on V with strike D. By Black-Scholes, we can
                  back out the unobservable asset value V and asset volatility σ_V from observable
                  equity price and equity volatility σ_E.
                  </p>
                </div>
                """)
            with c2:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">📐 Full Derivation</h4>
                  <p style="font-size:0.82rem;margin-bottom:6px;"><b>Step 1 — Equity Pricing (Black-Scholes):</b></p>
                  <div class="formula-box">
                    E = V·N(d&#8321;) − D·e<sup>−rT</sup>·N(d&#8322;)<br><br>
                    d&#8321; = [ln(V/D) + (r + ½σ²_V)T] / (σ_V√T)<br>
                    d&#8322; = d&#8321; − σ_V√T
                  </div>
                  <p style="font-size:0.82rem;margin:8px 0 4px;"><b>Step 2 — Volatility Linkage (Itô's Lemma):</b></p>
                  <div class="formula-box">σ_E · E = N(d&#8321;) · σ_V · V</div>
                  <p style="font-size:0.82rem;margin:8px 0 4px;"><b>Step 3 — Iterative Solution:</b><br>
                  Solve simultaneously for V and σ_V (200 iterations, tol=10⁻⁶).</p>
                  <p style="font-size:0.82rem;margin:8px 0 4px;"><b>Step 4 — Distance to Default:</b></p>
                  <div class="formula-box">
                    DD = d&#8322; = [ln(V/D) + (r − ½σ²_V)T] / (σ_V√T)<br>
                    P(default) = N(−DD)
                  </div>
                  <p style="font-size:0.78rem;color:{TS};margin-top:6px;">
                  DD = how many σ units above the default threshold the firm currently sits.
                  Higher DD = safer. P(default) = N(−d₂) is the <i>risk-neutral</i> probability.
                  </p>
                </div>
                """)
        
            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">✅ Key Assumptions</h4>
                  <ul style="line-height:2.0;font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    <li><b>GBM:</b> dV = μV dt + σ_V V dW (lognormal asset returns)</li>
                    <li><b>Constant volatility</b> σ_V over [0, T]</li>
                    <li><b>Constant risk-free rate</b> r</li>
                    <li><b>Single debt maturity</b> at T (simplified balance sheet)</li>
                    <li><b>Default only at maturity</b> (European, not American, option)</li>
                    <li><b>Frictionless markets</b> — no taxes, transaction costs</li>
                    <li><b>Publicly traded equity</b> required</li>
                  </ul>
                </div>
                """)
            with cb:
                st.html(f"""
                <div class="info-box" style="border-color:{RD};">
                  <h4 style="color:{RD};margin-top:0;">⚠️ Limitations &amp; Extensions</h4>
                  <ul style="line-height:2.0;font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    <li><b>GBM violated:</b> Fat tails, jumps, stochastic volatility in practice</li>
                    <li><b>Single debt class:</b> Ignores seniority, covenants, convertibles</li>
                    <li><b>European barrier:</b> Black-Cox (1976) allows continuous default</li>
                    <li><b>Risk-neutral PD:</b> N(−d₂) ≠ real-world PD; needs drift adjustment</li>
                    <li><b>No private firms:</b> Requires observable equity price and volatility</li>
                    <li><b>KMV refinement:</b> Uses STD + 0.5·LTD as default point, not total debt</li>
                  </ul>
                  <p style="font-size:0.78rem;margin-top:8px;color:{TS};">
                  Extensions: Leland (1994) endogenous default, CreditGrades™, CDS-implied PDs.
                  </p>
                </div>
                """)
        
            st.html(f"""
            <div class="info-box" style="margin-top:12px;">
              <h4 style="color:{GD};margin-top:0;">📚 Empirical Performance</h4>
              <p style="font-size:0.85rem;line-height:1.9;">
              Hillegeist et al. (2004, <i>Review of Accounting Studies</i>) showed the Merton model contains
              significantly more information about bankruptcy than Altman or Ohlson after controlling for
              market-to-book and firm size. Bharath &amp; Shumway (2008, <i>Review of Financial Studies</i>)
              found that a <b>naïve Merton approximation</b> performs nearly as well as the full iterative
              solution — suggesting structural form drives predictive power, not precise calibration.
              Industry usage: <b>Moody's CreditEdge™, JP Morgan CreditMetrics™, Bloomberg Default Risk</b>
              all build on Merton's framework. BIS endorses structural models for Basel II/III IRBA.
              </p>
            </div>
            """)
        
        # ── EDU 5: Bankruptcy Costs ───────────────────────────────────────────────
        with edu_tabs[4]:
            sec("💸 Bankruptcy Cost Theory &amp; Estimation")
            c1, c2 = st.columns(2)
            with c1:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">📜 Trade-Off Theory of Capital Structure</h4>
                  <p style="font-size:0.85rem;line-height:1.9;">
                  Modigliani &amp; Miller (1963) showed debt creates value through the <b>interest tax shield</b>.
                  But higher debt raises the probability and cost of financial distress. The
                  <b>Static Trade-Off Theory</b> posits an optimal capital structure where:
                  </p>
                  <div class="formula-box">
                    V_L = V_U + PV(Tax Shield) − PV(Distress Costs)<br><br>
                    Optimum: ∂[PV(Tax Shield)]/∂D = ∂[PV(Distress Cost)]/∂D
                  </div>
                </div>
                <div class="info-box" style="margin-top:12px;">
                  <h4 style="color:{GD};margin-top:0;">🔴 Direct Costs — Evidence</h4>
                  <ul style="font-size:0.85rem;line-height:1.9;margin:0;padding-left:1.2rem;">
                    <li><b>Legal fees:</b> Bankruptcy attorneys, restructuring counsel (1–4% of assets)</li>
                    <li><b>Administrative:</b> Court fees, trustee fees, accountant fees</li>
                    <li><b>Advisory:</b> Investment banking / restructuring advisor (0.5–1.5%)</li>
                  </ul>
                  <p style="font-size:0.82rem;margin-top:8px;">
                  <b>Warner (1977):</b> 5.3% of pre-bankruptcy value (railroads)<br>
                  <b>Weiss (1990):</b> 3.1% for 37 NYSE/AMEX firms<br>
                  <b>LoPucki &amp; Doherty (2004):</b> 1.4–3.9% for large Chapter 11
                  </p>
                </div>
                """)
            with c2:
                st.html(f"""
                <div class="info-box">
                  <h4 style="color:{GD};margin-top:0;">🟡 Indirect Costs — Evidence</h4>
                  <ul style="font-size:0.85rem;line-height:1.9;margin:0;padding-left:1.2rem;">
                    <li><b>Lost sales:</b> Opler &amp; Titman (1994) — distressed firms lose <b>26% more market share</b>
                        in industry downturns than healthy peers</li>
                    <li><b>Management distraction:</b> Executives spend 10–20% of time on restructuring
                        vs. operations (Gilson et al. 1990)</li>
                    <li><b>Supplier credit tightening:</b> Trade creditors demand prepayment</li>
                    <li><b>Employee flight:</b> Replacement costs 50–200% of annual salary</li>
                    <li><b>Debt overhang (Myers 1977):</b> Shareholders reject positive-NPV projects
                        when gains accrue to debtholders</li>
                    <li><b>Asset fire sales:</b> 20–40% discount (Shleifer &amp; Vishny 1992)</li>
                  </ul>
                </div>
                <div class="info-box" style="margin-top:12px;">
                  <h4 style="color:{GD};margin-top:0;">📊 Total Cost Literature Summary</h4>
                  <table style="width:100%;font-size:0.82rem;border-collapse:collapse;">
                    <tr style="border-bottom:1px solid rgba(255,215,0,0.3);">
                      <th style="color:{GD};padding:5px;">Study</th>
                      <th style="color:{GD};padding:5px;">Sample</th>
                      <th style="color:{GD};padding:5px;">Total Cost</th>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">Altman (1984)</td>
                      <td style="padding:4px;">Retail + Industrials</td>
                      <td style="padding:4px;">11–17% of assets</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                      <td style="padding:4px;color:{LB};">Andrade &amp; Kaplan (1998)</td>
                      <td style="padding:4px;">LBO distress</td>
                      <td style="padding:4px;">10–23% of value</td>
                    </tr>
                    <tr>
                      <td style="padding:4px;color:{LB};">Bris et al. (2006)</td>
                      <td style="padding:4px;">Ch.7 &amp; Ch.11</td>
                      <td style="padding:4px;">8.1% median (Ch.7)</td>
                    </tr>
                  </table>
                </div>
                """)
        
        # ── EDU 6: Model Comparison ───────────────────────────────────────────────
        with edu_tabs[5]:
            sec("📊 Comparative Analysis of All Models")
        
            categories = ["Accuracy","Probability Output","No Market Data",
                          "Private Firms OK","Interpretability","Theoretical Rigour","Ease of Use"]
            scores_map = {
                "Altman":  [78,20,100,70,95,55,95],
                "Ohlson":  [82,95,100,90,75,70,80],
                "Zmijewski":[76,95,100,90,85,72,90],
                "Merton":  [88,90,10,20,60,98,45],
            }
            radar_colors = [GD, LB, GR, RD]
            cats_c = categories + [categories[0]]
        
            fig_radar = go.Figure()
            for (model, vals), clr in zip(scores_map.items(), radar_colors):
                vc = vals + [vals[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vc, theta=cats_c, fill="toself", name=model,
                    line=dict(color=clr, width=2),
                    opacity=0.85))
            fig_radar.update_layout(
                **playout(height=480),
                polar=dict(
                    bgcolor=CB,
                    radialaxis=dict(visible=True, range=[0,100],
                                    gridcolor="rgba(255,255,255,0.1)",
                                    tickfont=dict(color=TS,size=9)),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.15)",
                                     tickfont=dict(color=TP,size=10))),
                title=dict(text="Model Capability Radar (0–100 scale)",
                           font=dict(color=GD,size=13),x=0.0),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TP)))
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_edu_t8")
        
            st.markdown("---")
            comp_data = {
                "Criterion":["Year","Methodology","Sample","Variables","Output",
                             "Calibrated Probability","Market Data Required",
                             "Private Firms","Best Use Case","Key Weakness"],
                "Altman Z-Score":["1968","MDA","66 firms","5 ratios","Score + Zone",
                                   "❌ Zone only","✅ Market cap (public)","✅ Z'-Score",
                                   "Bond analysis, quick screen","No probability; dated sample"],
                "Ohlson O-Score":["1980","Logistic Regression","2,163 firms","9 variables",
                                   "Calibrated probability","✅ Direct sigmoid","❌ Accounting only",
                                   "✅ Yes","Credit decisions, provisioning","Needs prior-year NI"],
                "Zmijewski":["1984","Probit Regression","840 firms","3 ratios",
                              "Calibrated probability","✅ Normal CDF","❌ Accounting only",
                              "✅ Yes","Quick probabilistic check","Only 3 variables"],
                "Merton KMV":["1974/1990s","Structural / Options","Market data","Equity + debt",
                               "Distance-to-Default + PD","✅ Risk-neutral","✅ Required",
                               "❌ Needs equity market","Bond pricing, Basel IRBA","GBM assumption; no private firms"],
            }
            st.dataframe(pd.DataFrame(comp_data).style
                .set_properties(**{"font-family":"JetBrains Mono,monospace",
                                   "font-size":"11px","text-align":"left","white-space":"normal"})
                .set_table_styles([{"selector":"th","props":[("background-color",DB),("color",GD),
                                                              ("font-weight","700"),("text-align","center")]}]),
                use_container_width=True,hide_index=True)
        
            st.markdown("---")
            st.html(f"""
            <div class="info-box">
              <h4 style="color:{GD};margin-top:0;">🧭 Practitioner's Guide — Which Model to Use?</h4>
              <table style="width:100%;font-size:0.84rem;border-collapse:collapse;line-height:1.8;">
                <tr style="border-bottom:1px solid rgba(255,215,0,0.4);">
                  <th style="color:{GD};padding:7px;text-align:left;">Situation</th>
                  <th style="color:{GD};padding:7px;text-align:left;">Recommended</th>
                  <th style="color:{GD};padding:7px;text-align:left;">Reason</th>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                  <td style="padding:6px;color:{LB};">Large listed manufacturing firm</td>
                  <td style="padding:6px;">Altman Z + Merton KMV</td>
                  <td style="padding:6px;">Cross-validate market and accounting signals</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                  <td style="padding:6px;color:{LB};">Private SME / family business</td>
                  <td style="padding:6px;">Altman Z' + Ohlson + Zmijewski</td>
                  <td style="padding:6px;">No market data; accounting models only</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                  <td style="padding:6px;color:{LB};">Bank loan underwriting</td>
                  <td style="padding:6px;">Ohlson or Zmijewski</td>
                  <td style="padding:6px;">Calibrated probability for pricing and provisioning</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                  <td style="padding:6px;color:{LB};">Bond valuation / credit spreads</td>
                  <td style="padding:6px;">Merton KMV</td>
                  <td style="padding:6px;">Structural model links directly to credit spread via N(−d₂)</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                  <td style="padding:6px;color:{LB};">Regulatory stress testing (Basel III)</td>
                  <td style="padding:6px;">Merton KMV + internal PD model</td>
                  <td style="padding:6px;">IRBA requires PD, LGD, EAD; structural models preferred</td>
                </tr>
                <tr>
                  <td style="padding:6px;color:{LB};">Academic research / benchmarking</td>
                  <td style="padding:6px;">All four + ensemble</td>
                  <td style="padding:6px;">Triangulation reduces model risk</td>
                </tr>
              </table>
            </div>
            """)
        

footer_bar()
