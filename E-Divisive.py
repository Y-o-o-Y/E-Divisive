import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf

# ==========================================
# 參數配置
# ==========================================
TICKER = 'MU'
START_DATE = "2024-03-04"
END_DATE = "2026-03-12"
MIN_SIZE = 30           # 每個 Regime 的最小交易日天數
P_VALUE_CUTOFF = 0.05   # 置換檢定顯著水準
R_PERMUTATIONS = 199    # 蒙地卡羅置換次數

# ==========================================
# 1. 資料獲取與預處理
# ==========================================
# 依規範抓取 Close，計算 Log Returns
df_raw = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)['Close']
if isinstance(df_raw, pd.DataFrame):
    df_raw = df_raw.squeeze()

# 資產回報率採用 Log Return
log_returns = np.log(df_raw / df_raw.shift(1)).dropna()
data = log_returns.values
dates = log_returns.index

# ==========================================
# 2. 核心算法優化 (The Sorting Trick & V-Statistic)
# ==========================================
def sad_1d(sorted_arr):
    """
    一維排序序列的兩兩絕對差之和 (Sum of Absolute Differences, SAD)
    時間複雜度: O(N)
    公式: \sum |x_i - x_j| = 2 \sum_{i=1}^n (2i - n - 1) x_{(i)}
    """
    n = len(sorted_arr)
    if n <= 1:
        return 0.0
    weights = 2 * np.arange(1, n + 1) - n - 1
    return 2.0 * float(np.dot(weights, sorted_arr))

def calculate_q_fast(x_arr, y_arr):
    """
    嚴格遵循標準 V-統計量定義的一維能量距離 Q 統計量
    分母統一採用 n*m, n^2, m^2，嚴禁使用 pdist 的 n(n-1)/2
    利用交叉距離恆等式: S(X U Y) = S(X) + S(Y) + 2*S(X, Y)
    時間複雜度: O(N log N)
    """
    n = len(x_arr)
    m = len(y_arr)
    if n == 0 or m == 0:
        return 0.0
    
    x_sort = np.sort(x_arr)
    y_sort = np.sort(y_arr)
    xy_sort = np.sort(np.concatenate([x_arr, y_arr]))
    
    # 計算樣本內距離和
    s_x = sad_1d(x_sort)
    s_y = sad_1d(y_sort)
    s_xy_all = sad_1d(xy_sort)
    
    # 交叉距離和
    s_xy = (s_xy_all - s_x - s_y) / 2.0
    
    # V-統計量期望值定義
    e_xy = s_xy / (n * m)
    e_xx = s_x / (n * n)
    e_yy = s_y / (m * m)
    
    # 能量距離 E(X, Y)
    energy_distance = 2.0 * e_xy - e_xx - e_yy
    return (n * m / (n + m)) * energy_distance

def get_best_tau(segment_data, min_size):
    """搜尋當前區間內能量距離最大的分割點 tau"""
    T = len(segment_data)
    if T < 2 * min_size:
        return None, -1.0
    
    best_q = -1.0
    best_tau = None
    
    for tau in range(min_size, T - min_size + 1):
        x = segment_data[:tau]
        y = segment_data[tau:]
        q = calculate_q_fast(x, y)
        if q > best_q:
            best_q = q
            best_tau = tau
            
    return best_tau, best_q

def permutation_test(segment_data, observed_q, best_tau, r_permutations=199):
    """置換檢定: 打亂時序結構以驗證統計顯著性"""
    count_exceed = 0
    temp_data = segment_data.copy()
    
    for _ in range(r_permutations):
        np.random.shuffle(temp_data)
        x_perm = temp_data[:best_tau]
        y_perm = temp_data[best_tau:]
        q_perm = calculate_q_fast(x_perm, y_perm)
        if q_perm >= observed_q:
            count_exceed += 1
            
    return (count_exceed + 1) / (r_permutations + 1)

# ==========================================
# 3. 貪婪階層式二元分割主迴圈 (E-Divisive)
# ==========================================
print(f"[INFO] 開始對 {TICKER} 執行 E-Divisive 結構轉折偵測...")
change_points = [0, len(data)]
found_new = True

while found_new:
    found_new = False
    best_overall_q = -1.0
    best_overall_tau = None
    best_cluster_idx = -1
    
    change_points.sort()
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i + 1]
        
        tau_rel, q = get_best_tau(data[start:end], MIN_SIZE)
        if tau_rel is not None and q > best_overall_q:
            best_overall_q = q
            best_overall_tau = start + tau_rel
            best_cluster_idx = i
            
    if best_overall_tau is not None and best_overall_q > 0:
        start = change_points[best_cluster_idx]
        end = change_points[best_cluster_idx + 1]
        tau_rel = best_overall_tau - start
        
        p_val = permutation_test(data[start:end], best_overall_q, tau_rel, R_PERMUTATIONS)
        print(f"  -> 候選斷裂點: {dates[best_overall_tau].strftime('%Y-%m-%d')} | Q: {best_overall_q:.4f} | p-value: {p_val:.4f}")
        
        if p_val < P_VALUE_CUTOFF:
            change_points.append(best_overall_tau)
            found_new = True
            print(f"     [已採納] 成功新增變動點: {dates[best_overall_tau].strftime('%Y-%m-%d')}")
        else:
            print("     [已終止] p-value 未達顯著水準，停止該分支分割。")

final_cp = sorted(change_points)
print(f"[INFO] 偵測完成，最終斷裂點數量: {len(final_cp) - 2}")

# ==========================================
# 4. Plotly 互動式圖表可視化
# ==========================================
segments = [(final_cp[i], final_cp[i + 1]) for i in range(len(final_cp) - 1)]
recent_segments = segments[-3:] if len(segments) >= 3 else segments

sub_titles = [f"<b>{TICKER} Log Returns 全時期與轉折點</b>"]
for start, end in reversed(recent_segments):
    seg_data = data[start:end]
    avg_ret = float(np.mean(seg_data) * 100)
    std_ret = float(np.std(seg_data) * 100)
    sharpe = float((avg_ret / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0
    
    title = (f"ACF: {dates[start].date()} ~ {dates[end-1].date()}<br>"
             f"Mean: {avg_ret:.4f}% | Std: {std_ret:.4f}% | <b>Sharpe: {sharpe:.2f}</b>")
    sub_titles.append(title)

# 動態設定 subplot 欄數
cols_count = len(recent_segments)
specs = [[{"colspan": cols_count} if cols_count > 1 else {}] + [None] * (cols_count - 1)]
specs.append([{} for _ in range(cols_count)])

fig = make_subplots(
    rows=2, cols=cols_count,
    specs=specs,
    vertical_spacing=0.18,
    subplot_titles=sub_titles
)

# Panel 1: Log Returns 與轉折點垂直虛線
fig.add_trace(go.Scatter(
    x=dates, y=log_returns.values, name='Log Return',
    line=dict(color='#00d4ff', width=1.2), opacity=0.7
), row=1, col=1)

for cp_idx in final_cp:
    if cp_idx != 0 and cp_idx != len(data):
        fig.add_vline(
            x=dates[cp_idx], line_width=1.5, line_dash="dash",
            line_color="#ff7f0e", row=1, col=1
        )

# Panel 2: 最近三個區間的 Squared Returns ACF
for i, (start, end) in enumerate(reversed(recent_segments)):
    seg_data = data[start:end]
    sq_rets = seg_data ** 2
    lags = 20
    acf_values = acf(sq_rets, nlags=lags)
    
    fig.add_trace(go.Bar(
        x=list(range(1, lags + 1)),
        y=acf_values[1:],
        marker_color='#2ecc71', showlegend=False
    ), row=2, col=i + 1)
    
    # 95% 白噪音雙尾信賴區間 (+/- 1.96 / sqrt(N))
    conf_level = 1.96 / np.sqrt(len(sq_rets))
    fig.add_hline(y=conf_level, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=2, col=i + 1)
    fig.add_hline(y=-conf_level, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=2, col=i + 1)
    fig.update_xaxes(title_text="Lag", row=2, col=i + 1)
    fig.update_yaxes(title_text="ACF", row=2, col=i + 1)

fig.update_layout(
    height=850,
    template="plotly_dark",
    title_text=f"E-Divisive 結構轉折偵測分析: {TICKER}",
    showlegend=False
)

fig.show()
