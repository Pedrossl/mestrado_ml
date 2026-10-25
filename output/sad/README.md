# output/sad — resultados especificos de SAD

## Conteudo atual — CONCLUIDO (ver docs/sad/FEATURE_SELECTION.md)

- `correlacao/` — Spearman com o alvo + pares redundantes (etapa 1)
- `permutation_importance_sad.txt` — etapa 2
- `experimento_hard_samples/` — pipeline oficial Monte Carlo v1 (17 features, max_depth=4, Kappa 0.8134)
- `feature_removal_runs/01_candidatos/` — 5 candidatas prioritarias testadas isoladamente
- `feature_removal_runs/02_leave_one_out/` — as 17 features testadas uma a uma (nenhuma melhora o Kappa)

Resultado oficial consolidado em `resultados/sad/melhor_resultado_monte_carlo.txt`
(Kappa 0.8415, metodologia com depth tunado usado tambem na identificacao dos
hard samples — ver nota metodologica la).

Ainda falta: ablation, sensibilidade — vao espelhar `output/gad/` se algum dia
forem julgados uteis para SAD.

Nota: `output/plots/<Algoritmo>/SAD/` ja existe automaticamente quando os
scripts genericos (`scripts/models/`, `scripts/evaluation/`,
`scripts/hyperparameters/`) rodam com `target='SAD'` — essa parte nao precisa
de mudanca, ja e organizada por alvo internamente. Ex:
`output/plots/XGBoost/SAD/feature_importance_sad.txt`.
