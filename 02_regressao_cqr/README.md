# 02 — Intervalos de previsão em regressão: CQR e CV+ aplicados a retornos do Ibovespa

*Backtest* comparando intervalos paramétricos, *bootstrap* em blocos, CQR (Conformalized Quantile Regression) com bases LightGBM e linear, e CV+, em onze anos de retornos diários do Ibovespa, com diagnóstico de cobertura por decil de volatilidade realizada.

Artigo completo: [pilulasdequant.com.br](https://pilulasdequant.com.br)

## Conteúdo

- `backtest_cqr.py` — implementação dos cinco métodos e geração das métricas e da figura usada no artigo.
- `ibov_returns.csv` — retornos logarítmicos diários do Ibovespa (^BVSP via Yahoo Finance), de 2004-01-05 a 2025-12-30.

## Como reproduzir

Requer Python 3.10+ com `lightgbm`, `statsmodels`, `pandas`, `numpy`, `matplotlib`.

```bash
cd 02_regressao_cqr
pip install lightgbm statsmodels pandas numpy matplotlib
python backtest_cqr.py
```

O *script* gera:

- `results_summary.csv` — cobertura global e largura média por método.
- `results_conditional.csv` — cobertura empírica por decil de volatilidade realizada.
- `band_march_2020.png` — bandas paramétrica e CQR durante o choque de COVID (fev-abr 2020).

## Métodos

1. **Paramétrico** — janela móvel de 252 dias, μ ± 1,645σ.
2. **Bootstrap em blocos** (Künsch, 1989) — L = 22 dias, 1000 reamostragens sobre resíduos da regressão linear.
3. **CQR-LightGBM** (Romano, Patterson, Candès, 2019) — *gradient boosting* com objetivo *quantile*.
4. **CQR-Linear** (Koenker, Bassett, 1978) — regressão quantílica linear como base, sanity check da escolha do estimador.
5. **CV+** (jackknife+, K=10) — *gradient boosting* com regressão padrão.

## Splits temporais

| Conjunto | Período | Uso |
|----------|---------|-----|
| Treino | 2004-2013 (2449 dias) | Ajuste dos modelos *quantile* (CQR) e do estimador base (CV+) |
| Calibração | 2014 (248 dias) | Cálculo do *score* CQR para o ajuste conformal |
| Teste | 2015-2025 (2729 dias) | Avaliação out-of-sample |

O conjunto de teste cobre crise fiscal brasileira de 2015-16, Joesley Day em 2017, eleição de 2018, pandemia em 2020, ciclo de juros e disputa eleitoral em 2022, e ciclo fiscal subsequente.

## Reprodutibilidade

Seed 42 em todo o pipeline (NumPy, LightGBM, bootstrap, CV+).

## Referências centrais

- Romano, Y.; Patterson, E.; Candès, E. J. *Conformalized Quantile Regression*. NeurIPS 2019.
- Barber, R. F.; Candès, E. J.; Ramdas, A.; Tibshirani, R. J. *Conformal prediction beyond exchangeability*. Annals of Statistics, 2023.
- Koenker, R.; Bassett, G. *Regression Quantiles*. Econometrica, 1978.
- Künsch, H. R. *The jackknife and the bootstrap for general stationary observations*. Annals of Statistics, 1989.
- Ke, G.; et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NIPS 2017.
