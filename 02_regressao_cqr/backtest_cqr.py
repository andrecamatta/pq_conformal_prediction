"""
Backtest de intervalos de previsão para retornos diários do Ibovespa.

Compara métodos para construir intervalos 90% sobre retornos t+1:
  (1) Paramétrico: μ ± 1.645·σ rolling de 252 dias.
  (2) Bootstrap em blocos (Künsch, 1989), L=22 dias, 1000 reamostragens.
  (3) CQR com LightGBM-quantile como base (Romano, Patterson, Candès, 2019;
      LightGBM de Ke et al., 2017).
  (4) CQR com regressão quantílica linear como base (Koenker e Bassett, 1978).
  (5) CV+ (variante jackknife+ com K=10 folds) com LightGBM como regressor.

Dataset: IBOV diário 2004-01-05 a 2025-12-30 (^BVSP via yfinance).
Splits:
  - Treino: 2004-2013 (~2449 dias)
  - Calibração: 2014 (~248 dias, usada pelas CQR; CV+ une treino+cal)
  - Teste: 2015-2025 (~2729 dias). Cobre crise fiscal 2015-16,
                     Joesley 2017, eleição 2018, COVID, ciclo de
                     juros e fiscal 2022-2025.

Saída:
  - results_summary.csv: cobertura global e largura média por método.
  - results_conditional.csv: cobertura por decil de volatilidade realizada.
  - band_march_2020.png: bandas paramétrica e CQR em fev-abr/2020.

Reprodutibilidade: seed 42 em todo o pipeline.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

np.random.seed(42)

ALPHA = 0.10
HALF = ALPHA / 2
Z90 = 1.6448536269514722

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# 1. Carregar dados e construir features
# ----------------------------------------------------------------------
df = pd.read_csv(os.path.join(HERE, "ibov_returns.csv"), parse_dates=["Date"])
df = df.rename(columns={"Date": "date"}).sort_values("date").reset_index(drop=True)
r = df["ret"].values
n = len(r)


def shift_lag(x, k):
    out = np.full(n, np.nan)
    out[k:] = x[:-k]
    return out


def rolling_std(x, w):
    out = np.full(n, np.nan)
    for i in range(w, n + 1):
        out[i - 1] = np.std(x[i - w:i], ddof=1)
    return out


df["ret_lag1"]   = shift_lag(r, 1)
df["ret_lag5"]   = shift_lag(r, 5)
df["ret_lag22"]  = shift_lag(r, 22)
df["rv_5"]       = shift_lag(rolling_std(r, 5), 1)
df["rv_22"]      = shift_lag(rolling_std(r, 22), 1)
df["absret_lag1"] = np.abs(df["ret_lag1"])
df["absret_lag5"] = np.abs(df["ret_lag5"])
df = df.dropna().reset_index(drop=True)
print(f"After feature build: {len(df)} rows from {df.date.iloc[0].date()} to {df.date.iloc[-1].date()}")

FEATURES = ["ret_lag1", "ret_lag5", "ret_lag22", "rv_5", "rv_22", "absret_lag1", "absret_lag5"]

train = df[df.date < pd.Timestamp(2014, 1, 1)].copy()
cal   = df[(df.date >= pd.Timestamp(2014, 1, 1)) & (df.date < pd.Timestamp(2015, 1, 1))].copy()
test  = df[df.date >= pd.Timestamp(2015, 1, 1)].copy()
print(f"Train: {len(train)} | Cal: {len(cal)} | Test: {len(test)}")

X_train = train[FEATURES].values
y_train = train["ret"].values
X_cal   = cal[FEATURES].values
y_cal   = cal["ret"].values
X_test  = test[FEATURES].values
y_test  = test["ret"].values
X_full  = np.vstack([X_train, X_cal])
y_full  = np.concatenate([y_train, y_cal])
rv_test = test["rv_22"].values
test_dates = test.date.values


# ----------------------------------------------------------------------
# Método 1: Paramétrico (rolling normal 252d)
# ----------------------------------------------------------------------
def method_parametric():
    r_all = df["ret"].values
    n_all = len(r_all)
    mu = np.full(n_all, np.nan)
    sig = np.full(n_all, np.nan)
    for i in range(252, n_all + 1):
        win = r_all[i - 252:i]
        mu[i - 1] = np.mean(win)
        sig[i - 1] = np.std(win, ddof=1)
    full_dates = df.date.values
    lo = np.full(len(test_dates), np.nan)
    hi = np.full(len(test_dates), np.nan)
    full_dates_idx = pd.Series(np.arange(n_all), index=full_dates)
    for k, d in enumerate(test_dates):
        if d in full_dates_idx.index:
            j = full_dates_idx[d]
            lo[k] = mu[j] - Z90 * sig[j]
            hi[k] = mu[j] + Z90 * sig[j]
    return lo, hi


# ----------------------------------------------------------------------
# Método 2: Bootstrap em blocos (Künsch, 1989) sobre regressão linear
# ----------------------------------------------------------------------
def ols_fit(X, y):
    Xi = np.hstack([np.ones((X.shape[0], 1)), X])
    return np.linalg.lstsq(Xi, y, rcond=None)[0]


def ols_predict(beta, X):
    return np.hstack([np.ones((X.shape[0], 1)), X]) @ beta


def method_block_bootstrap(B=1000, L=22):
    beta = ols_fit(X_train, y_train)
    resid = y_train - ols_predict(beta, X_train)
    pred_test = ols_predict(beta, X_test)
    rng = np.random.RandomState(42)
    n_test = len(y_test)
    n_resid = len(resid)
    n_blocks = int(np.ceil(n_test / L))
    samples = np.zeros((B, n_test))
    for b in range(B):
        starts = rng.randint(0, n_resid - L + 1, size=n_blocks)
        bs = np.concatenate([resid[s:s + L] for s in starts])[:n_test]
        samples[b, :] = pred_test + bs
    lo = np.quantile(samples, HALF, axis=0)
    hi = np.quantile(samples, 1 - HALF, axis=0)
    return lo, hi


# ----------------------------------------------------------------------
# Métodos 3 e 4: CQR com base LightGBM e CQR com base linear (Koenker-Bassett)
# ----------------------------------------------------------------------
def fit_lgb_quantile(X, y, q):
    m = lgb.LGBMRegressor(
        objective="quantile", alpha=q,
        num_leaves=31, learning_rate=0.05,
        n_estimators=300, min_data_in_leaf=20,
        verbose=-1, random_state=42,
    )
    m.fit(X, y)
    return m


def _conformal_adjust(q_lo_cal, q_hi_cal, y_cal_):
    """Calcula Q̂ conformal a partir dos scores E_i = max(qlo - y, y - qhi)."""
    E = np.maximum(q_lo_cal - y_cal_, y_cal_ - q_hi_cal)
    n_cal = len(y_cal_)
    level = min(np.ceil((1 - ALPHA) * (n_cal + 1)) / n_cal, 1.0)
    return float(np.quantile(E, level, method="higher"))


def method_cqr_lgb():
    q_lo_model = fit_lgb_quantile(X_train, y_train, HALF)
    q_hi_model = fit_lgb_quantile(X_train, y_train, 1 - HALF)
    q_lo_cal = q_lo_model.predict(X_cal)
    q_hi_cal = q_hi_model.predict(X_cal)
    Q = _conformal_adjust(q_lo_cal, q_hi_cal, y_cal)
    q_lo_test = q_lo_model.predict(X_test)
    q_hi_test = q_hi_model.predict(X_test)
    return q_lo_test - Q, q_hi_test + Q, Q


def method_cqr_linear():
    """CQR usando regressão quantílica linear (Koenker e Bassett, 1978) como base."""
    X_train_c = sm.add_constant(X_train)
    X_cal_c   = sm.add_constant(X_cal)
    X_test_c  = sm.add_constant(X_test)
    beta_lo = sm.QuantReg(y_train, X_train_c).fit(q=HALF,     max_iter=5000).params
    beta_hi = sm.QuantReg(y_train, X_train_c).fit(q=1 - HALF, max_iter=5000).params
    q_lo_cal = X_cal_c @ beta_lo
    q_hi_cal = X_cal_c @ beta_hi
    Q = _conformal_adjust(q_lo_cal, q_hi_cal, y_cal)
    q_lo_test = X_test_c @ beta_lo
    q_hi_test = X_test_c @ beta_hi
    return q_lo_test - Q, q_hi_test + Q, Q


# ----------------------------------------------------------------------
# Método 5: CV+ (K=10) com LightGBM (usado pelo artigo 02b)
# ----------------------------------------------------------------------
def method_cvplus(K=10):
    rng = np.random.RandomState(42)
    n_train = X_full.shape[0]
    n_test  = X_test.shape[0]
    perm = rng.permutation(n_train)
    fold_size = n_train // K
    folds = []
    for k in range(K):
        s = k * fold_size
        e = n_train if k == K - 1 else (k + 1) * fold_size
        folds.append(perm[s:e])
    R = np.zeros(n_train)
    fold_id = np.zeros(n_train, dtype=int)
    pred_test_per_fold = np.zeros((K, n_test))
    for k, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
        m = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=31, learning_rate=0.05,
            n_estimators=300, min_data_in_leaf=20,
            verbose=-1, random_state=42,
        )
        m.fit(X_full[tr_idx], y_full[tr_idx])
        R[val_idx] = np.abs(y_full[val_idx] - m.predict(X_full[val_idx]))
        fold_id[val_idx] = k
        pred_test_per_fold[k] = m.predict(X_test)
    pred_at_x = pred_test_per_fold[fold_id, :]
    L_mat = pred_at_x - R[:, None]
    H_mat = pred_at_x + R[:, None]
    lo = np.quantile(L_mat, HALF, axis=0)
    hi = np.quantile(H_mat, 1 - HALF, axis=0)
    return lo, hi


# ----------------------------------------------------------------------
# Executar
# ----------------------------------------------------------------------
print("Fitting Parametric ...");        lo1, hi1 = method_parametric()
print("Fitting Block bootstrap ...");   lo2, hi2 = method_block_bootstrap()
print("Fitting CQR-LightGBM ...");      lo3, hi3, Q_lgb    = method_cqr_lgb()
print("Fitting CQR-Linear (KB) ...");   lo4, hi4, Q_lin    = method_cqr_linear()
print("Fitting CV+ ...");               lo5, hi5 = method_cvplus()

methods = [
    ("Paramétrico (rolling 252d)",            lo1, hi1),
    ("Bootstrap blocos (LR, L=22, 1000 reps)", lo2, hi2),
    ("CQR-LightGBM (quantile)",               lo3, hi3),
    ("CQR-Linear (Koenker-Bassett)",          lo4, hi4),
    ("CV+ (K=10, LightGBM)",                  lo5, hi5),
]


def metrics(lo, hi, y, rv):
    mask = ~(np.isnan(lo) | np.isnan(hi))
    y_, lo_, hi_, rv_ = y[mask], lo[mask], hi[mask], rv[mask]
    cov = float(np.mean((y_ >= lo_) & (y_ <= hi_)))
    width = float(np.mean(hi_ - lo_))
    cuts = np.quantile(rv_, np.linspace(0, 1, 11))
    decile = np.clip(np.searchsorted(cuts[1:], rv_, side="left") + 1, 1, 10)
    cond = [float(np.mean((y_[decile == d] >= lo_[decile == d]) & (y_[decile == d] <= hi_[decile == d])))
            for d in range(1, 11)]
    return cov, width, cond, int(mask.sum())


results = {}
for name, lo, hi in methods:
    cov, w, cond, n_eff = metrics(lo, hi, y_test, rv_test)
    results[name] = dict(coverage=cov, width=w, cond=cond, n=n_eff)
    print(f"{name:42s} cov={cov:.4f}  width={w:.4f}  n={n_eff}")

# CSVs canônicos
summary_df = pd.DataFrame({
    "method":   [m[0] for m in methods],
    "coverage": [results[m[0]]["coverage"] for m in methods],
    "width":    [results[m[0]]["width"]    for m in methods],
    "n":        [results[m[0]]["n"]        for m in methods],
})
summary_df.to_csv(os.path.join(HERE, "results_summary.csv"), index=False)

cond_df = pd.DataFrame({"decile": [f"D{i}" for i in range(1, 11)]})
for name, _, _ in methods:
    cond_df[name] = results[name]["cond"]
cond_df.to_csv(os.path.join(HERE, "results_conditional.csv"), index=False)

print("\nresults_summary.csv:")
print(summary_df.to_string(index=False))
print("\nresults_conditional.csv:")
print(cond_df.to_string(index=False))

# Figura: banda paramétrica vs CQR-LightGBM em fev-abr 2020
win_start = pd.Timestamp(2020, 2, 3)
win_end   = pd.Timestamp(2020, 4, 30)
mask_win = (test.date >= win_start) & (test.date <= win_end)
dates_win = pd.to_datetime(test.date[mask_win])
y_win = y_test[mask_win.values]
lo_par_w = lo1[mask_win.values]; hi_par_w = hi1[mask_win.values]
lo_cqr_w = lo3[mask_win.values]; hi_cqr_w = hi3[mask_win.values]

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
ax.fill_between(dates_win, 100 * lo_par_w, 100 * hi_par_w, color="#888888", alpha=0.30,
                edgecolor="#555555", linewidth=0.9, label="Paramétrico (rolling 252d)")
ax.fill_between(dates_win, 100 * lo_cqr_w, 100 * hi_cqr_w, color="#FF6719", alpha=0.25,
                edgecolor="#FF6719", linewidth=1.2, label="CQR-LightGBM")
ax.plot(dates_win, 100 * y_win, color="#0a1420", linewidth=1.3,
        marker="o", markersize=2.8, label="Retorno realizado IBOV")
ax.axhline(0, color="#999", linewidth=0.6)
ax.axvline(pd.Timestamp(2020, 3, 18), color="#c0392b", linewidth=0.8, linestyle="--")
ax.set_ylabel("Retorno diário (%)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.25)
ax.xaxis.set_major_formatter(DateFormatter("%d-%b"))
plt.tight_layout()
plt.savefig(os.path.join(HERE, "band_march_2020.png"))
plt.close(fig)
print(f"\nSaved band_march_2020.png")

# Detalhe em 18-mar-2020
target = pd.Timestamp(2020, 3, 18)
idx_arr = np.where(test_dates == np.datetime64(target.date()))[0]
if len(idx_arr) > 0:
    j = idx_arr[0]
    print(f"\n=== 18/03/2020 ===")
    print(f"  Retorno realizado:    {100*y_test[j]:+.2f}%")
    print(f"  Paramétrico:          [{100*lo1[j]:+.2f}%, {100*hi1[j]:+.2f}%]  largura={100*(hi1[j]-lo1[j]):.2f}%")
    print(f"  Bootstrap blocos:     [{100*lo2[j]:+.2f}%, {100*hi2[j]:+.2f}%]  largura={100*(hi2[j]-lo2[j]):.2f}%")
    print(f"  CQR-LightGBM:         [{100*lo3[j]:+.2f}%, {100*hi3[j]:+.2f}%]  largura={100*(hi3[j]-lo3[j]):.2f}%")
    print(f"  CQR-Linear (KB):      [{100*lo4[j]:+.2f}%, {100*hi4[j]:+.2f}%]  largura={100*(hi4[j]-lo4[j]):.2f}%")
    print(f"  CV+:                  [{100*lo5[j]:+.2f}%, {100*hi5[j]:+.2f}%]  largura={100*(hi5[j]-lo5[j]):.2f}%")

print(f"\nQ̂ conformal: CQR-LightGBM = {Q_lgb:+.4f}, CQR-Linear = {Q_lin:+.4f}")
