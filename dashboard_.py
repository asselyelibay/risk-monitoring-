# -*- coding: utf-8 -*-
"""
Anti-AI Bubble Portfolio Dashboard - Professional Version
Author: Assel Yelibay
"""
#PART 1: IMPORTS & SETUP
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime


#PART 2: PAGE CONFIGURATION

st.set_page_config(
    page_title="Anti-AI Portfolio Dashboard",
    page_icon="📊",
    layout="wide"
)

#PART 3: CUSTOM CSS STYLING
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1f2937; margin-bottom: 0;}
    .sub-header {font-size: 1rem; color: #6b7280; margin-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 600;}
    .stTabs [data-baseweb="tab-list"] {gap: 1.5rem;}
    .stTabs [data-baseweb="tab"] {padding: 0.5rem 1.5rem; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

#PART 4: SIDEBAR (LEFT PANEL)
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")
    
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    target_vol_input = st.slider("Target Volatility (%)", 10.0, 25.0, 16.0, 0.5) / 100
    risk_free = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0, 0.1) / 100
    
    st.markdown("---")
    st.markdown("### Asset Universe")
    st.markdown("**25 Defensive Stocks**")
    st.caption("Consumer Staples • Healthcare • Utilities • REITs • Energy")
    
    st.markdown("---")
    st.markdown("**Updated:** " + datetime.now().strftime("%H:%M"))

# HEADER

st.markdown('<p class="main-header">Anti-AI Bubble Portfolio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Defensive Multi-Asset Strategy | Benchmark: MSCI World (URTH)</p>', unsafe_allow_html=True)

#PART 5: DATA LOADING
tickers = [
    "KO","PEP","PG","CL","KMB",
    "JNJ","PFE","ABT","LLY","UNH",
    "SO","DUK","NEE","AEP","EXC",
    "PLD","AMT","CCI","EQIX","PSA",
    "XOM","CVX","COP","SLB","EOG"
]
benchmark = "URTH"

@st.cache_data(ttl=3600)
def load_data(tickers, start):
    data = yf.download(tickers, start=start, progress=False)["Close"]
    return data.ffill().dropna(how='all')

with st.spinner("Loading data..."):
    data = load_data(tickers + [benchmark], start_date)

if data.empty:
    st.error("Failed to load data")
    st.stop()
    
#PART 6: CALCULATE RETURNS
returns = data[tickers].pct_change().dropna()
bench_returns = data[benchmark].pct_change().dropna()

# Align dates
common_dates = returns.index.intersection(bench_returns.index)
returns = returns.loc[common_dates]
bench_returns = bench_returns.loc[common_dates]

#PART 7: ANNUALIZED STATISTICS
mu = returns.mean() * 252
cov = returns.cov() * 252
n = len(tickers)


# PART 8: PORTFOLIO OPTIMIZATION


target_vol = target_vol_input
w0 = np.ones(n)/n
bounds = [(0.015, 0.15)]*n

def weight_constraint(w):
    return np.sum(w) - 1

def portfolio_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(cov, w)))

def objective(w):
    port_ret = np.dot(w, mu)
    port_vol = portfolio_vol(w)
    return -(port_ret - risk_free) / port_vol

# Minimum volatility portfolio
min_vol_result = minimize(
    portfolio_vol,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=[{"type":"eq","fun":weight_constraint}],
    options={'maxiter': 1000}
)

min_vol = portfolio_vol(min_vol_result.x)

if min_vol > target_vol:
    st.warning(f"""
     Target volatility ({target_vol*100:.1f}%) is below the minimum achievable volatility 
    ({min_vol*100:.2f}%). Adjusted automatically.
    """)
    target_vol = min_vol * 1.05

constraints = [
    {"type": "eq", "fun": weight_constraint},
    {"type": "ineq", "fun": lambda w: target_vol - portfolio_vol(w)}
]

result = minimize(
    objective,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

if not result.success:
    st.error("Portfolio optimization failed due to constraints.")
    st.stop()

w_opt = result.x
ret_opt = np.dot(w_opt, mu)
vol_opt = portfolio_vol(w_opt)
sharpe_opt = (ret_opt - risk_free)/vol_opt


#PART 9: CALCULATION OF RISK METRICS
if not result.success:
    st.error(" Portfolio optimization failed.")
    st.stop()

w_opt = result.x
ret_opt = np.dot(w_opt, mu)
vol_opt = portfolio_vol(w_opt)
sharpe_opt = (ret_opt - risk_free)/vol_opt

# --- Portfolio Returns ---
port_returns = (returns * w_opt).sum(axis=1)
cumulative = (1 + port_returns).cumprod()
bench_cumulative = (1 + bench_returns).cumprod()

# --- Risk Metrics ---
var_95 = np.percentile(port_returns, 5)
cvar_95 = port_returns[port_returns <= var_95].mean()
max_dd = (cumulative / cumulative.cummax() - 1).min()

excess_returns = port_returns - bench_returns
tracking_error = excess_returns.std() * np.sqrt(252)
information_ratio = (
    excess_returns.mean() * 252 / tracking_error
    if tracking_error > 0 else 0
)

beta = np.cov(port_returns, bench_returns)[0,1] / np.var(bench_returns)
correlation = np.corrcoef(port_returns, bench_returns)[0,1]


#PART 10: DISPLAY METRICS (TOP ROW)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Return (Annual)", f"{ret_opt*100:.2f}%")
with col2:
    st.metric("Volatility", f"{vol_opt*100:.2f}%")
with col3:
    st.metric("Sharpe Ratio", f"{sharpe_opt:.2f}")
with col4:
    st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
with col5:
    st.metric("Info Ratio", f"{information_ratio:.2f}")

st.markdown("---")

#PART 11: TABS
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Allocation", "Risk Analysis", "Performance"])

# TAB 1: OVERVIEW

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Annual Return', 'Volatility', 'Sharpe Ratio', 'VaR 95%', 'CVaR 95%', 
                      'Max Drawdown', 'Beta', 'Correlation', 'Tracking Error'],
            'Value': [
                f"{ret_opt*100:.2f}%",
                f"{vol_opt*100:.2f}%",
                f"{sharpe_opt:.2f}",
                f"{var_95*100:.2f}%",
                f"{cvar_95*100:.2f}%",
                f"{max_dd*100:.2f}%",
                f"{beta:.2f}",
                f"{correlation:.2f}",
                f"{tracking_error*100:.2f}%"
            ]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("Portfolio vs Benchmark")
        comparison_df = pd.DataFrame({
            'Metric': ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
            'Portfolio': [
                f"{ret_opt*100:.2f}%",
                f"{vol_opt*100:.2f}%",
                f"{sharpe_opt:.2f}",
                f"{max_dd*100:.2f}%"
            ],
            'MSCI World': [
                f"{bench_returns.mean()*252*100:.2f}%",
                f"{bench_returns.std()*np.sqrt(252)*100:.2f}%",
                f"{(bench_returns.mean()*252 - risk_free)/(bench_returns.std()*np.sqrt(252)):.2f}",
                f"{((1+bench_returns).cumprod()/((1+bench_returns).cumprod().cummax())-1).min()*100:.2f}%"
            ]
        })
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Cumulative Performance
    st.subheader("Cumulative Performance")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative.index, 
        y=(cumulative - 1) * 100,
        name="Portfolio",
        line=dict(color='#10b981', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=bench_cumulative.index,
        y=(bench_cumulative - 1) * 100,
        name="MSCI World",
        line=dict(color='#3b82f6', width=2, dash='dash')
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# TAB 2: ALLOCATION

with tab2:
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Top 15 Holdings")
        weights_df = pd.DataFrame({
            'Ticker': tickers,
            'Weight': w_opt * 100
        }).sort_values('Weight', ascending=False).head(15)
        
        fig = px.bar(
            weights_df, 
            x='Ticker', 
            y='Weight',
            text='Weight',
            color='Weight',
            color_continuous_scale='viridis'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Weight (%)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sector Breakdown")
        sector_map = {
            "Consumer Staples": ["KO","PEP","PG","CL","KMB"],
            "Healthcare": ["JNJ","PFE","ABT","LLY","UNH"],
            "Utilities": ["SO","DUK","NEE","AEP","EXC"],
            "REITs": ["PLD","AMT","CCI","EQIX","PSA"],
            "Energy": ["XOM","CVX","COP","SLB","EOG"]
        }
        
        sector_weights = {
            s: sum([w_opt[tickers.index(t)] for t in stocks if t in tickers]) * 100
            for s, stocks in sector_map.items()
        }
        
        fig = px.pie(
            names=list(sector_weights.keys()),
            values=list(sector_weights.values()),
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textinfo='label+percent', textfont_size=12)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector table
        sector_df = pd.DataFrame({
            'Sector': list(sector_weights.keys()),
            'Weight': [f"{v:.2f}%" for v in sector_weights.values()]
        }).sort_values('Weight', ascending=False)
        st.dataframe(sector_df, hide_index=True, use_container_width=True)


# TAB 3: RISK ANALYSIS
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Contribution")
        port_var = portfolio_vol(w_opt)**2
        marginal_contrib = cov.dot(w_opt)
        risk_contrib = w_opt * marginal_contrib
        risk_contrib_pct = (risk_contrib / risk_contrib.sum()) * 100
        
        risk_df = pd.DataFrame({
            'Ticker': tickers,
            'Risk Contribution (%)': risk_contrib_pct
        }).sort_values('Risk Contribution (%)', ascending=False).head(10)
        
        fig = px.bar(
            risk_df,
            x='Ticker',
            y='Risk Contribution (%)',
            text='Risk Contribution (%)',
            color='Risk Contribution (%)',
            color_continuous_scale='Reds'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Drawdown Analysis")
        drawdown = cumulative / cumulative.cummax() - 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            line=dict(color='#ef4444', width=2),
            name='Drawdown'
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Stress Testing
    st.subheader("Stress Testing Scenarios")
    
    scenarios = {
        "Market Correction (-10%)": -0.10,
        "Bear Market (-25%)": -0.25,
        "AI Bubble Burst (-40%)": -0.40,
        "Severe Crisis (-50%)": -0.50
    }
    
    stress_results = []
    for name, shock in scenarios.items():
        port_shocked = cumulative.iloc[-1] * (1 + shock) - 1
        bench_shocked = bench_cumulative.iloc[-1] * (1 + shock) - 1
        stress_results.append({
            'Scenario': name,
            'Portfolio Impact': f"{port_shocked*100:.2f}%",
            'Benchmark Impact': f"{bench_shocked*100:.2f}%",
            'Relative Outperformance': f"{(port_shocked - bench_shocked)*100:.2f}%"
        })
    
    stress_df = pd.DataFrame(stress_results)
    st.dataframe(stress_df, hide_index=True, use_container_width=True)


# TAB 4: PERFORMANCE
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rolling Returns (1-Year)")
        rolling_ret = port_returns.rolling(252).apply(lambda x: (1+x).prod()-1) * 100
        bench_rolling_ret = bench_returns.rolling(252).apply(lambda x: (1+x).prod()-1) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_ret.index, y=rolling_ret, name="Portfolio", line=dict(color='#10b981', width=2)))
        fig.add_trace(go.Scatter(x=bench_rolling_ret.index, y=bench_rolling_ret, name="Benchmark", line=dict(color='#3b82f6', width=2)))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Rolling 1Y Return (%)",
            hovermode='x unified',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Rolling Volatility (1-Year)")
        rolling_vol = port_returns.rolling(252).std() * np.sqrt(252) * 100
        bench_rolling_vol = bench_returns.rolling(252).std() * np.sqrt(252) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, name="Portfolio", line=dict(color='#10b981', width=2)))
        fig.add_trace(go.Scatter(x=bench_rolling_vol.index, y=bench_rolling_vol, name="Benchmark", line=dict(color='#3b82f6', width=2)))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Rolling 1Y Volatility (%)",
            hovermode='x unified',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")
    monthly_returns = port_returns.resample('M').apply(lambda x: (1+x).prod()-1) * 100
    monthly_returns_table = monthly_returns.to_frame()
    monthly_returns_table['Year'] = monthly_returns_table.index.year
    monthly_returns_table['Month'] = monthly_returns_table.index.month
    pivot_table = monthly_returns_table.pivot(index='Year', columns='Month', values=0)
    
    fig = px.imshow(
        pivot_table,
        labels=dict(x="Month", y="Year", color="Return (%)"),
        x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# FOOTER

st.markdown("---")
st.markdown("**Developed by Assel Yelibay**")