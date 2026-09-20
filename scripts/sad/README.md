# scripts/sad — analises especificas de SAD

Espelha a estrutura de `scripts/gad/`. Ainda vazio porque o trabalho de
selecao de features e tuning do Monte Carlo v1 so foi feito para GAD ate agora.

Quando o SAD entrar em pauta:

- Scripts de analise especifica de SAD (Spearman, Permutation Importance,
  Monte Carlo v1 tuning, etc.) vao em `scripts/sad/analysis/`, seguindo o
  mesmo padrao usado em `scripts/gad/analysis/`.
- O experimento de hard samples do SAD (se for feito) vai em
  `scripts/sad/experimento_hard_samples/`.
- A lista de features a remover para SAD entra em
  `FEATURE_DROP_COLUMNS_BY_TARGET["SAD"]` em `scripts/config.py` — hoje esta
  vazia de proposito, pois a normalizacao e as variaveis a remover para SAD
  provavelmente serao diferentes das de GAD.
- Scripts genericos (`scripts/models/`, `scripts/evaluation/`,
  `scripts/analysis/eda.py`, `correlation.py`, etc.) ja aceitam
  `target='SAD'` e nao precisam ser duplicados — so chamar com o alvo certo.

Ver `scripts/gad/` como referencia de organizacao.
