import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, cdist
from statsmodels.tsa.stattools import acf

TICKER = 'MU'
ALPHA = 1.0
MIN_SIZE = 30
P_VALUE_CUTOFF = 0.05
R_PERMUTATIONS = 199

df_raw = yf.download(TICKER, start="2024-03-04", end="2026-03-12", interval='1d')['Close']
if isinstance(df_raw, pd.DataFrame):
    df_raw = df_raw.squeeze()

log_returns = np.log(df_raw / df_raw.shift(1)).dropna()
data = log_returns.values.reshape(-1, 1)
dates = log_returns.index

def calculate_q(segment_data, tau, alpha=1.0):
    n, m = tau, len(segment_data) - tau
    x, y = segment_data[:tau], segment_data[tau:]
    term1 = 2 * np.mean(cdist(x, y, metric='euclidean')**alpha)
    term2 = np.mean(pdist(x, metric='euclidean')**alpha) if n > 1 else 0
    term3 = np.mean(pdist(y, metric='euclidean')**alpha) if m > 1 else 0
    return (n * m / (n + m)) * (term1 - term2 - term3)

def get_best_tau(segment_data, min_size):
    T = len(segment_data)
    if T < 2 * min_size: return None, -1
    best_q, best_tau = -1, None
    for tau in range(min_size, T - min_size + 1):
        q = calculate_q(segment_data, tau, ALPHA)
        if q > best_q:
            best_q, best_tau = q, tau
    return best_tau, best_q

def permutation_test(segment_data, observed_q, best_tau, R=99):
    count_exceed = 0
    temp_data = segment_data.copy()
    for _ in range(R):
        np.random.shuffle(temp_data)
        if calculate_q(temp_data, best_tau, ALPHA) >= observed_q:
            count_exceed += 1
    return (count_exceed + 1) / (R + 1)

change_points = [0, len(data)]
found_new = True
while found_new:
    found_new = False
    best_overall_q, best_overall_tau, best_cluster_idx = -1, None, -1
    change_points.sort()
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i+1]
        tau_rel, q = get_best_tau(data[start:end], MIN_SIZE)
        if q > best_overall_q:
            best_overall_q, best_overall_tau, best_cluster_idx = q, start + tau_rel, i
    if best_overall_tau is not None:
        start, end = change_points[best_cluster_idx], change_points[best_cluster_idx+1]
        p_val = permutation_test(data[start:end], best_overall_q, best_overall_tau - start, R_PERMUTATIONS)
        if p_val < P_VALUE_CUTOFF:
            change_points.append(best_overall_tau)
            found_new = True

final_cp = sorted(change_points)

segments = [(final_cp[i], final_cp[i+1]) for i in range(len(final_cp)-1)]
recent_segments = segments[-3:]

sub_titles = [f"<b>{TICKER} 全時期</b>"]
for start, end in reversed(recent_segments):
    seg_data = log_returns.iloc[start:end].values
    avg_ret = np.mean(seg_data)*100
    std_ret = np.std(seg_data)*100
    sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    title = (f"ACF: {dates[start].date()} ~ {dates[end-1].date()}<br>"
             f"Mean: {avg_ret:.4f} | Std: {std_ret:.4f} | <b>Sharpe: {sharpe:.2f}</b>")
    sub_titles.append(title)

fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
    vertical_spacing=0.2,
    subplot_titles=sub_titles
)

fig.add_trace(go.Scatter(
    x=dates, y=log_returns.values, name='Log Return',
    line=dict(color='#1f77b4', width=1), opacity=0.6
), row=1, col=1)

for cp_idx in final_cp:
    if cp_idx != 0 and cp_idx != len(data):
        fig.add_vline(x=dates[cp_idx], line_width=1.5, line_dash="dash", line_color="#ff7f0e", row=1, col=1)

for i, (start, end) in enumerate(reversed(recent_segments)):
    seg_data = log_returns.iloc[start:end].values
    sq_rets = seg_data ** 2
    lags = 20
    acf_values = acf(sq_rets, nlags=lags)

    fig.add_trace(go.Bar(
        x=list(range(1, lags + 1)),
        y=acf_values[1:],
        marker_color='#2ca02c', showlegend=False
    ), row=2, col=i+1)

    conf_level = 1.96 / np.sqrt(len(sq_rets))
    fig.add_hline(y=conf_level, line_dash="dash", line_color="rgba(255,255,255,0.5)", row=2, col=i+1)
    fig.add_hline(y=-conf_level, line_dash="dash", line_color="rgba(255,255,255,0.5)", row=2, col=i+1)

fig.update_layout(
    height=900, template="plotly_dark",
    title_text=f"E-Divisive 分析報告: {TICKER} (結構轉折)",
    showlegend=False
)
fig.show()