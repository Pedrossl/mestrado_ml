# scripts/sad — analises especificas de SAD

Espelha a estrutura de `scripts/gad/`.

## Status atual — CONCLUIDO (selecao de features + tuning de hiperparametros)

Ver `docs/sad/FEATURE_SELECTION.md` para a analise completa. Resumo:

- **Nenhuma feature removida** (`FEATURE_DROP_COLUMNS_BY_TARGET["SAD"]` fica
  vazia — decisao validada por 18 testes individuais de remocao, nao um
  placeholder). Diferente de GAD, aqui a CV/Permutation Importance nao
  concordou com o Monte Carlo v1 em nenhum caso.
- **`max_depth=4` tunado** (`XGB_MAX_DEPTH_BY_TARGET["SAD"]` em
  `scripts/config.py`) — arvores mais RASAS ajudaram, o oposto de GAD
  (que precisou de arvores mais profundas, `max_depth=8`).
- Kappa final: **0.8415** (17 features, depth=4), vs 0.7788 no baseline
  original — melhora de 8.1%, inteiramente por hiperparametro.

Scripts:

- `analysis/spearman_correlation.py` — correlacao de Spearman. `output/sad/correlacao/`
- `analysis/permutation_importance.py` — importancia por CV 10-fold. `output/sad/permutation_importance_sad.txt`
- `analysis/mc_v1_candidatos.py` — testa as 5 candidatas prioritarias isoladamente
- `analysis/mc_v1_leave_one_out.py` — testa as 17 features uma a uma (mais abrangente)
- `analysis/mc_v1_tuning.py` — tuning de hiperparametros (max_depth, n_estimators, etc.)
- `experimento_hard_samples/` — pipeline oficial (Monte Carlo v1, passo1/2/3)

**Cuidado ao reusar `mc_v1_candidatos.py`/`mc_v1_leave_one_out.py` como
modelo**: eles usam sempre as mesmas linhas do baseline (`df_base.drop(columns=drops)`)
em vez de chamar `preparar_dados()` de novo por cenario — isso evita
reincorporar linhas via `dropna()` quando a feature removida tem valores
faltantes (bug real que apareceu ao testar `Poverty Status`, ver
`docs/sad/FEATURE_SELECTION.md`, Etapa 3.2).

Scripts genericos (`scripts/models/`, `scripts/evaluation/`,
`scripts/analysis/eda.py`, `feature_ablation.py`, etc.) ja aceitam
`target='SAD'` e nao precisam ser duplicados — so chamar com o alvo certo.

Ver `scripts/gad/` como referencia de organizacao.
