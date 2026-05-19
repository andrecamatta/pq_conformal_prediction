"""
Backtest de intervalos de previsão para retornos diários do Ibovespa.

Compara 4 métodos para construir intervalos 90% sobre retornos t+1:
  (1) Paramétrico: μ ± 1.645·σ rolling de 252 dias.
  (2) Bootstrap dos resíduos sobre regressão linear (1000 reps, percentis 5/95).
  (3) Conformalized Quantile Regression (CQR) com LightGBM como base.
  (4) CV+ (variante jackknife+ com K=10 folds) com LightGBM.

Dataset: IBOV diário 2004-01-05 a 2024-12-30 (^BVSP via yfinance).
Split:
  - Treino: 2004-2018
  - Calibração: 2019 (para CQR; CV+ usa treino+calib internamente)
  - Teste: 2020-2024 (atravessa COVID, eleições e regime alta de juros)

Output: tabela de métricas (cobertura, largura) e cobertura condicional por decil
de volatilidade realizada, figura de bandas em fev-abr 2020.
"""

using CSV, DataFrames, Dates, Random, Statistics, Printf
using LightGBM
using Plots

Random.seed!(42)

const ALPHA = 0.10           # alvo de cobertura: 90%
const HALF  = ALPHA / 2       # 5% / 95%
const Z90   = 1.6448536269514722

# ----------------------------------------------------------------------
# 1. Carregar dados e construir features
# ----------------------------------------------------------------------
df = CSV.read("ibov_returns.csv", DataFrame)
rename!(df, :Date => :date)
sort!(df, :date)

# Features: lags de retorno + volatilidade realizada
function rolling_std(x::Vector{Float64}, w::Int)
    n = length(x)
    out = fill(NaN, n)
    @inbounds for i in w:n
        out[i] = std(view(x, (i-w+1):i))
    end
    return out
end

r = df.ret
df.ret_lag1   = [NaN; r[1:end-1]]
df.ret_lag5   = [fill(NaN, 5); r[1:end-5]]
df.ret_lag22  = [fill(NaN, 22); r[1:end-22]]
df.rv_5       = [NaN; rolling_std(r, 5)[1:end-1]]
df.rv_22      = [NaN; rolling_std(r, 22)[1:end-1]]
df.absret_lag1 = abs.(df.ret_lag1)
df.absret_lag5 = abs.(df.ret_lag5)

dropmissing!(df)
df = df[.!any.(eachrow(isnan.(df[:, Not(:date)]))), :]

println("After feature build: $(nrow(df)) rows from $(df.date[1]) to $(df.date[end])")

const FEATURES = [:ret_lag1, :ret_lag5, :ret_lag22, :rv_5, :rv_22, :absret_lag1, :absret_lag5]

# Splits temporais
train = df[df.date .< Date(2019,1,1), :]
cal   = df[(df.date .>= Date(2019,1,1)) .& (df.date .< Date(2020,1,1)), :]
test  = df[df.date .>= Date(2020,1,1), :]
println("Train: $(nrow(train)) | Cal: $(nrow(cal)) | Test: $(nrow(test))")

X_train = Matrix(train[:, FEATURES])
y_train = train.ret
X_cal   = Matrix(cal[:, FEATURES])
y_cal   = cal.ret
X_test  = Matrix(test[:, FEATURES])
y_test  = test.ret
X_full  = vcat(X_train, X_cal)
y_full  = vcat(y_train, y_cal)
rv_test = test.rv_22

# ----------------------------------------------------------------------
# Método 1: Paramétrico (rolling normal 252d)
# ----------------------------------------------------------------------
function method_parametric(df_all::DataFrame, test_idx::Vector{Date})
    r = df_all.ret
    n = length(r)
    μ = fill(NaN, n); σ = fill(NaN, n)
    @inbounds for i in 252:n
        win = view(r, (i-251):i)
        μ[i] = mean(win); σ[i] = std(win)
    end
    full_idx = df_all.date
    lo = fill(NaN, length(test_idx)); hi = fill(NaN, length(test_idx))
    for (k, d) in enumerate(test_idx)
        j = searchsortedfirst(full_idx, d)
        if j <= n && full_idx[j] == d
            lo[k] = μ[j] - Z90*σ[j]
            hi[k] = μ[j] + Z90*σ[j]
        end
    end
    return lo, hi
end

# ----------------------------------------------------------------------
# Método 2: Bootstrap dos resíduos sobre regressão linear
# ----------------------------------------------------------------------
function ols_fit(X::Matrix{Float64}, y::Vector{Float64})
    Xi = hcat(ones(size(X,1)), X)
    β = Xi \ y
    return β
end
ols_predict(β::Vector{Float64}, X::Matrix{Float64}) = hcat(ones(size(X,1)), X) * β

function method_bootstrap(B::Int=1000)
    β = ols_fit(X_train, y_train)
    resid = y_train .- ols_predict(β, X_train)
    pred_test = ols_predict(β, X_test)
    rng = MersenneTwister(42)
    n_test = length(y_test)
    samples = Matrix{Float64}(undef, B, n_test)
    for b in 1:B
        bs = resid[rand(rng, 1:length(resid), n_test)]
        samples[b, :] = pred_test .+ bs
    end
    lo = [quantile(view(samples, :, j), HALF)     for j in 1:n_test]
    hi = [quantile(view(samples, :, j), 1 - HALF) for j in 1:n_test]
    return lo, hi
end

# ----------------------------------------------------------------------
# Método 3: CQR (Conformalized Quantile Regression) com LightGBM
# ----------------------------------------------------------------------
function fit_lgb_quantile(X, y, q)
    m = LGBMRegression(
        objective = "quantile", alpha = q,
        num_leaves = 31, learning_rate = 0.05,
        num_iterations = 300, min_data_in_leaf = 20,
        verbosity = -1,
    )
    LightGBM.fit!(m, X, Float64.(y))
    return m
end

function method_cqr()
    q_lo_model = fit_lgb_quantile(X_train, y_train, HALF)
    q_hi_model = fit_lgb_quantile(X_train, y_train, 1 - HALF)

    q_lo_cal = vec(LightGBM.predict(q_lo_model, X_cal))
    q_hi_cal = vec(LightGBM.predict(q_hi_model, X_cal))
    E = max.(q_lo_cal .- y_cal, y_cal .- q_hi_cal)

    n = length(y_cal)
    level = min(ceil((1 - ALPHA)*(n + 1))/n, 1.0)
    Q = quantile(E, level)

    q_lo_test = vec(LightGBM.predict(q_lo_model, X_test))
    q_hi_test = vec(LightGBM.predict(q_hi_model, X_test))
    return q_lo_test .- Q, q_hi_test .+ Q
end

# ----------------------------------------------------------------------
# Método 4: CV+ (K=10) com LightGBM regressão padrão
# ----------------------------------------------------------------------
function kfold_indices(n::Int, K::Int, rng::AbstractRNG)
    perm = randperm(rng, n)
    fold_size = div(n, K)
    folds = Vector{Vector{Int}}(undef, K)
    for k in 1:K
        s = (k - 1)*fold_size + 1
        e = k == K ? n : k*fold_size
        folds[k] = perm[s:e]
    end
    return folds
end

function method_cvplus(K::Int=10)
    rng = MersenneTwister(42)
    n_train = size(X_full, 1)
    n_test  = size(X_test, 1)

    folds = kfold_indices(n_train, K, rng)
    R        = zeros(n_train)
    fold_id  = zeros(Int, n_train)
    pred_test_per_fold = zeros(K, n_test)

    for (k, val_idx) in enumerate(folds)
        tr_idx = setdiff(1:n_train, val_idx)
        m = LGBMRegression(
            objective = "regression",
            num_leaves = 31, learning_rate = 0.05,
            num_iterations = 300, min_data_in_leaf = 20,
            verbosity = -1,
        )
        LightGBM.fit!(m, X_full[tr_idx, :], Float64.(y_full[tr_idx]))
        ŷ_val = vec(LightGBM.predict(m, X_full[val_idx, :]))
        R[val_idx] .= abs.(y_full[val_idx] .- ŷ_val)
        fold_id[val_idx] .= k
        pred_test_per_fold[k, :] = vec(LightGBM.predict(m, X_test))
    end

    # Para cada x_test, distribuição empírica de {f̂^{-k(i)}(x_test) ± R_i}
    pred_at_x_for_each_i = pred_test_per_fold[fold_id, :]   # (n_train, n_test)
    L = pred_at_x_for_each_i .- R
    H = pred_at_x_for_each_i .+ R

    lo = [quantile(view(L, :, j), HALF)     for j in 1:n_test]
    hi = [quantile(view(H, :, j), 1 - HALF) for j in 1:n_test]
    return lo, hi
end

# ----------------------------------------------------------------------
# Executar e medir
# ----------------------------------------------------------------------
println("Fitting Parametric ..."); lo1, hi1 = method_parametric(df, test.date)
println("Fitting Bootstrap ...");  lo2, hi2 = method_bootstrap()
println("Fitting CQR ...");        lo3, hi3 = method_cqr()
println("Fitting CV+ ...");        lo4, hi4 = method_cvplus(10)

methods = [
    ("Paramétrico (rolling 252d)", lo1, hi1),
    ("Bootstrap (LR + 1000 reps)", lo2, hi2),
    ("CQR (LightGBM-quantile)",    lo3, hi3),
    ("CV+ (K=10, LightGBM)",       lo4, hi4),
]

function metrics(lo, hi, y, rv)
    mask = .!(isnan.(lo) .| isnan.(hi))
    y_, lo_, hi_, rv_ = y[mask], lo[mask], hi[mask], rv[mask]
    cov   = mean((y_ .>= lo_) .& (y_ .<= hi_))
    width = mean(hi_ .- lo_)
    # Cobertura condicional por decil de RV
    cuts = quantile(rv_, range(0, 1; length=11))
    decile = clamp.(searchsortedfirst.(Ref(cuts[2:end]), rv_), 1, 10)
    cond = [mean(((y_ .>= lo_) .& (y_ .<= hi_))[decile .== d]) for d in 1:10]
    return cov, width, cond, sum(mask)
end

results = Dict{String, NamedTuple}()
for (name, lo, hi) in methods
    cov, w, cond, n_eff = metrics(lo, hi, y_test, rv_test)
    results[name] = (coverage=cov, width=w, cond=cond, n=n_eff)
    @printf("%-30s cov=%.3f  width=%.4f  n=%d\n", name, cov, w, n_eff)
end

# Salvar tabelas
summary_df = DataFrame(
    method   = [name for (name, _, _) in methods],
    coverage = [results[name].coverage for (name, _, _) in methods],
    width    = [results[name].width    for (name, _, _) in methods],
    n        = [results[name].n        for (name, _, _) in methods],
)
CSV.write("results_summary.csv", summary_df)

cond_df = DataFrame(decile=["D$i" for i in 1:10])
for (name, _, _) in methods
    cond_df[!, name] = results[name].cond
end
CSV.write("results_conditional.csv", cond_df)

println("\nConditional coverage by RV decile (D1=lowest vol, D10=highest):")
show(stdout, "text/plain", cond_df); println()
println("\nSaved results_summary.csv and results_conditional.csv")

# ----------------------------------------------------------------------
# Figura: banda paramétrica vs banda CQR em fev-abr 2020
# ----------------------------------------------------------------------
win_start = Date(2020,2,3)
win_end   = Date(2020,4,30)
mask_win  = (test.date .>= win_start) .& (test.date .<= win_end)
dates_win = test.date[mask_win]
y_win     = y_test[mask_win]
lo_par_w  = lo1[mask_win]; hi_par_w = hi1[mask_win]
lo_cqr_w  = lo3[mask_win]; hi_cqr_w = hi3[mask_win]

p = plot(size=(1000, 520), dpi=150, legend=:topright, grid=true, gridalpha=0.25,
         ylabel="Retorno diário (%)", xlabel="", framestyle=:box)
plot!(p, dates_win, 100 .* hi_par_w, fillrange=100 .* lo_par_w,
      fillcolor="#888888", fillalpha=0.30, linecolor="#555555", linewidth=0.9,
      label="Paramétrico (rolling 252d)")
plot!(p, dates_win, 100 .* lo_par_w, linecolor="#555555", linewidth=0.9, label="")
plot!(p, dates_win, 100 .* hi_cqr_w, fillrange=100 .* lo_cqr_w,
      fillcolor="#FF6719", fillalpha=0.25, linecolor="#FF6719", linewidth=1.2,
      label="CQR (LightGBM-quantile)")
plot!(p, dates_win, 100 .* lo_cqr_w, linecolor="#FF6719", linewidth=1.2, label="")
plot!(p, dates_win, 100 .* y_win, linecolor="#0a1420", linewidth=1.3,
      marker=:circle, markersize=2.5, label="Retorno realizado IBOV")
hline!(p, [0], linecolor="#999", linewidth=0.6, label="")
target = Date(2020,3,18)
vline!(p, [target], linecolor="#c0392b", linewidth=0.8, linestyle=:dash, label="")

savefig(p, "band_march_2020.png")
println("Saved band_march_2020.png")

# Valores específicos de 18-mar-2020 (para uso no artigo)
ti = test.date
if target in ti
    j = findfirst(==(target), ti)
    println("\n=== 18/03/2020 ===")
    @printf("  Retorno realizado:  %+.2f%%\n", 100*y_test[j])
    @printf("  Paramétrico:        [%+.2f%%, %+.2f%%]  largura=%.2f%%\n",
            100*lo1[j], 100*hi1[j], 100*(hi1[j]-lo1[j]))
    @printf("  Bootstrap residual: [%+.2f%%, %+.2f%%]  largura=%.2f%%\n",
            100*lo2[j], 100*hi2[j], 100*(hi2[j]-lo2[j]))
    @printf("  CQR:                [%+.2f%%, %+.2f%%]  largura=%.2f%%\n",
            100*lo3[j], 100*hi3[j], 100*(hi3[j]-lo3[j]))
    @printf("  CV+:                [%+.2f%%, %+.2f%%]  largura=%.2f%%\n",
            100*lo4[j], 100*hi4[j], 100*(hi4[j]-lo4[j]))
end
