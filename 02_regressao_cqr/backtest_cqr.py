"""
Backtest de intervalos de previsão para retornos diários do Ibovespa.

Compara 4 métodos para construir intervalos 90% sobre retornos t+1:
  (1) Paramétrico: μ ± 1.645·σ rolling de 252 dias.
  (2) Bootstrap dos resíduos sobre regressão linear (1000 reps, percentis 5/95).
  (3) Conformalized Quantile Regression (CQR) com LightGBM como base.
  (4) CV+ (variante jackknife+ com K=10 folds) com LightGBM.

Dataset: IBOV diário 2004-01-05 a 2024-12-30 (5198 obs, ^BVSP via yfinance).
Split:
  - Treino: 2004-2018
  - Calibração: 2019 (para CQR; CV+ usa todo treino+calib internamente)
  - Teste: 2020-2024 (atravessa COVID, eleições e regime alta de juros)

Output: tabela de métricas (cobertura, largura) e cobertura condicional por decil de volatilidade realizada.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

np.random.seed(42)

ALPHA = 0.10            # alvo de cobertura: 90%
HALF = ALPHA / 2        # 5% / 95%

# 1. Carregar e preparar dados
df = pd.read_csv('ibov_returns.csv', index_col=0, parse_dates=True)
df.columns = ['ret']

# Features: lagged returns + realized volatility
df['ret_lag1'] = df['ret'].shift(1)
df['ret_lag5'] = df['ret'].shift(5)
df['ret_lag22'] = df['ret'].shift(22)
df['rv_5'] = df['ret'].rolling(5).std().shift(1)
df['rv_22'] = df['ret'].rolling(22).std().shift(1)
df['absret_lag1'] = df['ret'].shift(1).abs()
df['absret_lag5'] = df['ret'].shift(5).abs()

df = df.dropna()
print(f'After feature build: {df.shape[0]} rows from {df.index.min().date()} to {df.index.max().date()}')

FEATURES = ['ret_lag1','ret_lag5','ret_lag22','rv_5','rv_22','absret_lag1','absret_lag5']
TARGET = 'ret'

# Splits
train = df[df.index < '2019-01-01']
cal   = df[(df.index >= '2019-01-01') & (df.index < '2020-01-01')]
test  = df[df.index >= '2020-01-01']

print(f'Train: {len(train)} | Cal: {len(cal)} | Test: {len(test)}')

X_train, y_train = train[FEATURES].values, train[TARGET].values
X_cal,   y_cal   = cal[FEATURES].values,   cal[TARGET].values
X_test,  y_test  = test[FEATURES].values,  test[TARGET].values
X_full = np.vstack([X_train, X_cal])
y_full = np.concatenate([y_train, y_cal])

# Realized volatility on test for conditional coverage
rv_test = test['rv_22'].values

# ----------------------------------------------------------------------
# Method 1: Parametric (rolling normal)
# ----------------------------------------------------------------------
def method_parametric():
    full = pd.concat([train, cal, test])
    mu  = full['ret'].rolling(252).mean()
    sig = full['ret'].rolling(252).std()
    z = 1.6448536269514722  # 95% one-sided
    lo = (mu - z*sig).reindex(test.index).values
    hi = (mu + z*sig).reindex(test.index).values
    return lo, hi

# ----------------------------------------------------------------------
# Method 2: Residual bootstrap on linear regression
# ----------------------------------------------------------------------
def method_bootstrap(B=1000):
    lr = LinearRegression().fit(X_train, y_train)
    resid = y_train - lr.predict(X_train)
    pred_test = lr.predict(X_test)
    rng = np.random.default_rng(42)
    boots = rng.choice(resid, size=(B, len(y_test)), replace=True)
    samples = pred_test + boots               # shape (B, n_test)
    lo = np.quantile(samples, HALF, axis=0)
    hi = np.quantile(samples, 1-HALF, axis=0)
    return lo, hi

# ----------------------------------------------------------------------
# Method 3: CQR (Conformalized Quantile Regression) com LightGBM
# ----------------------------------------------------------------------
def fit_lgb_quantile(X, y, q):
    return lgb.LGBMRegressor(
        objective='quantile', alpha=q,
        num_leaves=31, learning_rate=0.05,
        n_estimators=300, min_data_in_leaf=20,
        verbose=-1, random_state=42
    ).fit(X, y)

def method_cqr():
    q_lo_model = fit_lgb_quantile(X_train, y_train, HALF)
    q_hi_model = fit_lgb_quantile(X_train, y_train, 1-HALF)

    q_lo_cal = q_lo_model.predict(X_cal)
    q_hi_cal = q_hi_model.predict(X_cal)
    E = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

    n = len(y_cal)
    level = np.ceil((1-ALPHA)*(n+1))/n
    Q = np.quantile(E, min(level, 1.0))

    q_lo_test = q_lo_model.predict(X_test)
    q_hi_test = q_hi_model.predict(X_test)
    return q_lo_test - Q, q_hi_test + Q

# ----------------------------------------------------------------------
# Method 4: CV+ (K=10) com LightGBM ponto + leave-one-out residuals via folds
# ----------------------------------------------------------------------
def method_cvplus(K=10):
    # Apenas retornos médios; CV+ adapta uniformemente.
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    n_train = len(X_full)

    # Para cada fold, treina-se f̂^{-fold}; residuais |y_i - f̂^{-fold}(x_i)| por ponto
    # E armazena-se f̂^{-fold}(x_test) para todo x_test, posteriormente combinado.
    R = np.zeros(n_train)                       # residuais leave-one-fold-out
    fold_id = np.zeros(n_train, dtype=int)
    pred_test_per_fold = np.zeros((K, len(X_test)))

    for k, (tr_idx, val_idx) in enumerate(kf.split(X_full)):
        m = lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.05,
            n_estimators=300, min_data_in_leaf=20,
            verbose=-1, random_state=42
        ).fit(X_full[tr_idx], y_full[tr_idx])
        R[val_idx] = np.abs(y_full[val_idx] - m.predict(X_full[val_idx]))
        fold_id[val_idx] = k
        pred_test_per_fold[k] = m.predict(X_test)

    # Para cada x_test, monta-se distribuição empírica de {f̂^{-fold(i)}(x_test) ± R_i}
    # Conforme Barber et al. 2021, jackknife+/CV+ usam estes valores.
    pred_at_x_for_each_i = pred_test_per_fold[fold_id]   # (n_train, n_test)
    L = pred_at_x_for_each_i - R[:, None]                # lower candidates
    H = pred_at_x_for_each_i + R[:, None]                # upper candidates

    lo = np.quantile(L, HALF, axis=0)
    hi = np.quantile(H, 1-HALF, axis=0)
    return lo, hi

# ----------------------------------------------------------------------
# Executar e medir
# ----------------------------------------------------------------------
print('Fitting Parametric ...');  lo1, hi1 = method_parametric()
print('Fitting Bootstrap ...');   lo2, hi2 = method_bootstrap()
print('Fitting CQR ...');         lo3, hi3 = method_cqr()
print('Fitting CV+ ...');         lo4, hi4 = method_cvplus(K=10)

methods = {
    'Paramétrico (rolling 252d)': (lo1, hi1),
    'Bootstrap (LR + 1000 reps)': (lo2, hi2),
    'CQR (LightGBM-quantile)':    (lo3, hi3),
    'CV+ (K=10, LightGBM)':       (lo4, hi4),
}

# Drop NaNs from parametric (initial 252-day warmup)
def metrics(lo, hi, y, rv):
    mask = ~(np.isnan(lo) | np.isnan(hi))
    y_, lo_, hi_, rv_ = y[mask], lo[mask], hi[mask], rv[mask]
    cov = ((y_ >= lo_) & (y_ <= hi_)).mean()
    width = (hi_ - lo_).mean()
    # Conditional coverage by RV decile
    deciles = pd.qcut(rv_, 10, labels=False, duplicates='drop')
    cond = pd.Series((y_ >= lo_) & (y_ <= hi_)).groupby(deciles).mean().values
    return cov, width, cond, mask.sum()

results = {}
for name, (lo, hi) in methods.items():
    cov, w, cond, n_eff = metrics(lo, hi, y_test, rv_test)
    results[name] = dict(coverage=cov, width=w, cond=cond, n=n_eff)
    print(f'{name:30s}  cov={cov:.3f}  width={w:.4f}  n={n_eff}')

# Save
out = pd.DataFrame({
    name: {'coverage': r['coverage'], 'width': r['width'], 'n': r['n']}
    for name, r in results.items()
}).T
out.to_csv('results_summary.csv')

cond_df = pd.DataFrame(
    {name: r['cond'] for name, r in results.items()},
    index=[f'D{i+1}' for i in range(10)]
)
cond_df.to_csv('results_conditional.csv')

print('\nConditional coverage by RV decile (D1=lowest vol, D10=highest):')
print(cond_df.round(3))
print('\nSaved results_summary.csv and results_conditional.csv')

# ----------------------------------------------------------------------
# Figura: banda paramétrica vs banda CQR em fev-abr 2020 (estresse COVID)
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

window_start = pd.Timestamp('2020-02-03')
window_end   = pd.Timestamp('2020-04-30')

test_idx = test.index
mask_win = (test_idx >= window_start) & (test_idx <= window_end)

dates_win = test_idx[mask_win]
y_win     = y_test[mask_win]
lo_par_w  = lo1[mask_win]; hi_par_w = hi1[mask_win]
lo_cqr_w  = lo3[mask_win]; hi_cqr_w = hi3[mask_win]
lo_boot_w = lo2[mask_win]; hi_boot_w = hi2[mask_win]

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)

# Bandas
ax.fill_between(dates_win, 100*lo_par_w, 100*hi_par_w,
                color='#888888', alpha=0.30, label='Paramétrico (rolling 252d)')
ax.plot(dates_win, 100*lo_par_w, color='#555555', linewidth=0.9)
ax.plot(dates_win, 100*hi_par_w, color='#555555', linewidth=0.9)

ax.fill_between(dates_win, 100*lo_cqr_w, 100*hi_cqr_w,
                color='#FF6719', alpha=0.25, label='CQR (LightGBM-quantile)')
ax.plot(dates_win, 100*lo_cqr_w, color='#FF6719', linewidth=1.2)
ax.plot(dates_win, 100*hi_cqr_w, color='#FF6719', linewidth=1.2)

# Retorno realizado
ax.plot(dates_win, 100*y_win, color='#0a1420', linewidth=1.3,
        marker='o', markersize=2.5, label='Retorno realizado IBOV')

# Marcador 18-mar-2020
target = pd.Timestamp('2020-03-18')
if target in pd.DatetimeIndex(dates_win):
    pos = list(pd.DatetimeIndex(dates_win)).index(target)
    ax.axvline(target, color='#c0392b', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.annotate('18/03/2020', xy=(target, 100*y_win[pos]),
                xytext=(8, -25), textcoords='offset points',
                fontsize=9, color='#c0392b')

ax.axhline(0, color='#999', linewidth=0.6)
ax.set_ylabel('Retorno diário (%)', fontsize=10)
ax.set_xlabel('')
ax.legend(loc='upper right', frameon=False, fontsize=9)
ax.grid(True, alpha=0.25)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('band_march_2020.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('\nSaved band_march_2020.png')

# ----------------------------------------------------------------------
# Valores específicos de 18-mar-2020 (para uso no artigo)
# ----------------------------------------------------------------------
target = pd.Timestamp('2020-03-18')
ti = list(test.index)
if target in ti:
    j = ti.index(target)
    print(f'\n=== 18/03/2020 ===')
    print(f'  Retorno realizado:  {100*y_test[j]:+.2f}%')
    print(f'  Paramétrico:        [{100*lo1[j]:+.2f}%, {100*hi1[j]:+.2f}%]  largura={100*(hi1[j]-lo1[j]):.2f}%')
    print(f'  Bootstrap residual: [{100*lo2[j]:+.2f}%, {100*hi2[j]:+.2f}%]  largura={100*(hi2[j]-lo2[j]):.2f}%')
    print(f'  CQR:                [{100*lo3[j]:+.2f}%, {100*hi3[j]:+.2f}%]  largura={100*(hi3[j]-lo3[j]):.2f}%')
    print(f'  CV+:                [{100*lo4[j]:+.2f}%, {100*hi4[j]:+.2f}%]  largura={100*(hi4[j]-lo4[j]):.2f}%')
