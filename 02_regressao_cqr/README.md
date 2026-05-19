# 02 — Intervalos de previsão em regressão: CQR e CV+ aplicados a retornos do Ibovespa

*Backtest* comparando intervalos paramétricos, *bootstrap*, CQR (Conformalized Quantile Regression) e CV+ em seis anos de retornos diários do Ibovespa, com diagnóstico de cobertura por decil de volatilidade realizada.

Artigo completo: [pilulasdequant.com.br](https://pilulasdequant.com.br)

## Conteúdo

- `backtest_cqr.jl` — implementação dos quatro métodos e geração das métricas e da figura usada no artigo.
- `ibov_returns.csv` — retornos logarítmicos diários do Ibovespa (^BVSP via Yahoo Finance), de 2004-01-05 a 2024-12-30.
- `Project.toml` / `Manifest.toml` — manifesto do ambiente Julia.

## Como reproduzir

Requer Julia 1.10+ instalado.

```bash
cd 02_regressao_cqr
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. backtest_cqr.jl
```

O *script* gera:

- `results_summary.csv` — cobertura global e largura média por método.
- `results_conditional.csv` — cobertura empírica por decil de volatilidade realizada.
- `band_march_2020.png` — bandas paramétrica e CQR durante o choque de COVID (fev-abr 2020).

## Pacotes principais

- `LightGBM.jl` — *gradient boosting* com objetivo *quantile* para CQR e regressão padrão para CV+.
- `DataFrames.jl` / `CSV.jl` — I/O e manipulação de dados.
- `Statistics`, `Random`, `Dates`, `Printf` — *stdlib*.
- `Plots.jl` — geração da figura.

## Splits temporais

| Conjunto | Período | Uso |
|----------|---------|-----|
| Treino | 2004-2018 | Ajuste dos modelos *quantile* (CQR) e do estimador base (CV+) |
| Calibração | 2019 | Cálculo do *score* de CQR para o ajuste conformal |
| Teste | 2020-2024 | Avaliação out-of-sample (COVID, ciclo eleitoral 2022, alta de juros) |
