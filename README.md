# Multi-Scale Decomposition of Log-Volatility: A Rough/Persistent Model for Forecasting and Risk Management

**Master's Thesis — Riccardo Centonze**  
Master's Degree in Financial Risk and Data Analysis, Sapienza University of Rome  
Supervisor: Prof. Sergio Bianchi  
Academic Year 2025/2026

---

## Overview

This repository contains the full Python implementation of the empirical analysis presented in the thesis. The central contribution is the **RP_Model**: an extension of the Heterogeneous Autoregressive (HAR) model of Corsi (2009) that explicitly incorporates two latent components of log-volatility — a **rough** (short-memory) component $X_R$ and a **persistent** (long-memory) component $X_P$ — identified via Detrended Fluctuation Analysis (DFA).

The model is evaluated on five international equity indices (FTSE 100, S&P 500, Nikkei 225, Hang Seng, DAX) over the period 2000–2026 on point-forecast accuracy (RMSE, QLIKE), distributional calibration (PIT, CRPS), and Value-at-Risk backtesting (Christoffersen 1998).

---

## Repository Structure

```
.
├── Code_implementation.ipynb  # Main notebook (41 cells, fully documented)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

The notebook is self-contained and runs end-to-end on Google Colab with Google Drive storage. All data are downloaded automatically from Yahoo Finance via `yfinance`.

---

## Notebook Structure

| Cell | Content |
|------|---------|
| 1    | Environment setup, imports, global parameters |
| 2    | Download daily prices from Yahoo Finance (5 indices, 2000–2026) |
| 3    | Data cleaning: alignment on common trading dates, duplicate removal |
| 4    | Normalised price series plot (Figure 3.1) |
| 5    | Log-returns and volatility proxies ($r_t$, $r_t^2$, $\log r_t^2$) |
| 6    | Market regime identification (growth vs. crisis periods) |
| 7    | Extreme return spikes visualisation |
| 8    | Empirical return distribution vs. Gaussian benchmark (Figure 3.2) |
| 9    | Descriptive statistics of log-returns (Table 3.1) |
| 10   | ACF and Ljung–Box serial dependence diagnostics (Figure 3.3) |
| 11   | DFA on $\log(r_t^2)$: Hurst exponent estimation |
| 12   | DFA on $X_{\text{obs}}$, $X_R$, $X_P$: component-level Hurst exponents (Figure 4.2) |
| 13   | ACF of the latent components $X_R$ and $X_P$ (Figure 4.3) |
| 14   | Multi-scale Monte Carlo simulation of the RP_Model |
| 15   | Rough-component weight $\omega(h)$ vs. forecast horizon (Figure 5.1) |
| 16   | Observed vs. simulated returns (visual diagnostics) |
| 17   | Train/test split (70/30, April 2018 cutoff) |
| 18   | Walk-forward cross-validation (overfitting check) |
| 19   | HAR and RP_Model estimation + benchmark comparison (RMSE, QLIKE) |
| 20   | Diebold–Mariano test: RP_Model vs. HAR (Figure 6.2) |
| 21   | Mincer–Zarnowitz efficiency test |
| 22   | RP_Model with expanding window (monthly re-estimation) |
| 23   | Stability of $\gamma_R$ and $\gamma_P$ over time (Figure 5.2) |
| 24   | DM test: RP_Model (expanding window) vs. HAR |
| 25   | Residual diagnostics |
| 26–27 | VaR backtesting: Christoffersen (1998) UC, IND, CC tests |
| 28   | PIT test — Normal and Student-$t$ innovations (Figure 7.1–7.2) |
| 29   | Final analytical figures (log-volatility decomposition) |
| 30   | CRPS computation with closed-form Normal and Student-$t$ expressions |
| 31   | CRPS skill score and DM test on CRPS (Figure 7.3) |
| 32   | Kurtosis of standardised residuals (Figure 7.4) |
| 33   | CRPS decomposition: Reliability, Resolution, Uncertainty (Hersbach 2000) |

---

## Key Model Specifications

### EWMA Log-Sigma Proxy

$$X_{\text{obs},t} = \tfrac{1}{2} \log\!\left(\hat{\sigma}_t^2 + \varepsilon\right), \qquad \hat{\sigma}_t^2 = (1-\lambda)\,r_t^2 + \lambda\,\hat{\sigma}_{t-1}^2$$

with $\lambda = 0.94$ (RiskMetrics standard) and $\varepsilon = 10^{-12}$.

### Rough/Persistent Decomposition

$$X_P = \text{EWMA}(X_{\text{obs}},\; \text{span} = \max(120,\; 40h)), \qquad X_R = X_{\text{obs}} - X_P$$

### RP_Model

$$X_{t+h} = c + b_d X_{t-1} + b_w \bar{X}_{t-5} + b_m \bar{X}_{t-22} + \gamma_R X_{R,t-1} + \gamma_P X_{P,t-1} + \varepsilon_t$$

Key empirical finding: $\gamma_R < 0$ at $h=1$, acting as an **anti-noise filter** that lowers forecasts during short-lived volatility spikes and reduces MSE.

---

## How to Run

### On Google Colab 

1. Upload the notebook to Google Colab.
2. Mount Google Drive when prompted (Cell 1).
3. Run cells sequentially from Cell 1 to Cell 33.
4. All outputs (CSV files, figures) are saved to `MyDrive/tesi_volatilita/`.

### Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Remove or comment out the `drive.mount(...)` line in Cell 1.
3. Set `BASE_DIR` to a local folder, e.g.:
   ```python
   BASE_DIR = "./output"
   ```
4. Run the notebook with Jupyter:
   ```bash
   jupyter notebook Code_implementation.ipynb
   ```

---

## Dependencies

See `requirements.txt`. Main libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Numerical computation |
| `pandas` | ≥2.0 | Data manipulation |
| `yfinance` | ≥0.2 | Yahoo Finance data download |
| `statsmodels` | ≥0.14 | ACF, Ljung–Box, OLS |
| `scipy` | ≥1.10 | Statistical tests, distributions |
| `matplotlib` | ≥3.7 | Plotting |
| `seaborn` | ≥0.12 | Heatmaps |

---

## Data

Daily closing prices are downloaded automatically from Yahoo Finance for the following indices:

| Index | Ticker | Exchange | Period |
|-------|--------|----------|--------|
| FTSE 100 | ^FTSE | London SE | 06/03/2000 – 06/03/2026 |
| S&P 500 | ^GSPC | NYSE | 06/03/2000 – 06/03/2026 |
| Nikkei 225 | ^N225 | Tokyo SE | 06/03/2000 – 06/03/2026 |
| Hang Seng | ^HSI | Hong Kong SE | 06/03/2000 – 06/03/2026 |
| DAX | ^GDAXI | Frankfurt SE | 06/03/2000 – 06/03/2026 |

After alignment on common trading dates: **5,795 daily observations** per index.

---

## References

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Diebold, F. X., Gunther, T. A., & Tay, A. S. (1998). Evaluating density forecasts with applications to financial risk management. *International Economic Review*, 39(4), 863–883.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Eliazar, I. (2024). Power Brownian Motion. *Journal of Physics A*, 57, 03LT01.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. *Weather and Forecasting*, 15(5), 559–570.
- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, 4th ed.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.

---

## Contact

Riccardo Centonze — centonze.2198299@studenti.uniroma1.it
