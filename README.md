# risk-monitoring-
Anti-AI Bubble Portfolio Dashboard A professional Streamlit-based portfolio analytics dashboard focused on defensive asset allocation and risk management under high-volatility scenarios. Overview This project implements a volatility-constrained portfolio optimization strategy using a defensive equity universe across:
- Consumer Staples 
- Healthcare
- Utilities 
- REITs
- Energy
  
The objective is to construct a risk-efficient portfolio resilient to potential market corrections or AI-driven valuation bubbles.

## Key Features
- Volatility-targeted portfolio optimization
- Sharpe ratio maximization under constraints
- Risk contribution decomposition
- Drawdown and CVaR analysis
- Benchmark comparison (MSCI World – URTH)
- Rolling performance metrics
- Scenario-based stress testing
- Sector allocation visualization
- Monthly return heatmap
  
### Methodology
- Daily returns computed from Yahoo Finance data
- Annualized mean and covariance estimation
- Constrained optimization via Sequential Least Squares Programming (SLSQP)
- Long-only allocation
- Weight bounds applied
- Volatility cap enforced
- Sharpe ratio maximization objective

### Risk metrics include:
- Annualized return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Value-at-Risk (95%)
- Conditional Value-at-Risk (95%)
- Beta and correlation vs benchmark
- Tracking error
- Information ratio 

### Tech Stack
Python 
Streamlit 
NumPy 
Pandas
SciPy 
Plotly 
yfinance 

Run Locally pip install -r requirements.txt streamlit run app.py

Live Application https://portfolio-risk-dashboard-hcxx2prqqlwdcegf3dhyyx.streamlit.app/

Author Assel Yelibay 
