# slide2 — visuais para os slides 28 a 33

Uma pasta por slide, **exatamente 2 arquivos em cada**: a imagem pronta para inserir
no PowerPoint e o arquivo com os números (para recriar o gráfico nativo no deck).

Todos os dados são do conjunto **GAD** (287 crianças: 243 sem GAD, 44 com GAD),
10-Fold Stratified CV, `random_state=42`.

| Pasta | Slide | Imagem | Números |
|---|---|---|---|
| `29_comparativo/` | 29 — Comparativo entre algoritmos | `comparativo_algoritmos_gad.png` | `comparativo_algoritmos_gad.csv` |
| `31_roc/` | 31 — Curvas ROC | `roc_comparativo_gad_ic95.png` | `roc_pontos_gad.csv` |
| `28_30_matriz_confusao/` | 28 ou 30 — Matriz de confusão / trade-off | `matriz_confusao_xgboost_smote_gad.png` | `matriz_confusao_numeros.txt` |
| `32_monte_carlo/` | 32 — Monte Carlo e a correção do vazamento | `monte_carlo_antes_depois_gad.png` | `monte_carlo_antes_depois.csv` |
| `33_feature_importance/` | 33 — Feature importance | `feature_importance_gad.png` | `feature_importance_gad.csv` |

## O que cada um mostra

**29 — Comparativo.** Barras agrupadas com Accuracy, Sensibilidade, Especificidade,
PPV e F1 (± IC 95%) para XGBoost, SVM e ADTree, todos com SMOTE. O CSV traz também
o Kappa. Mensagem: XGBoost vence em quase tudo; o ADTree só ganha em sensibilidade
porque desmonta a especificidade (23,5%) — sensibilidade alta sozinha não vale nada.

**31 — ROC.** Curva média dos 10 folds com banda de IC 95% e as curvas de cada fold
ao fundo. AUC XGBoost = 0,751 ± 0,103; SVM = 0,678 ± 0,069. O CSV tem os pares
FPR/TPR (100 pontos) para redesenhar nativo — **leia o rodapé do CSV**, há uma nota
sobre uma diferença de ~0,01 no AUC ao redesenhar a partir dos pontos.

**28/30 — Matriz de confusão.** XGBoost + SMOTE, threshold padrão 0,50, agregada
dos 10 folds. O TXT traz também a matriz no threshold de Youden (0,22) e o
trade-off clínico em número de crianças: baixar o threshold detecta **+8 crianças
com GAD** ao custo de **+18 falsos alarmes** (2,2 alarmes por caso a mais).

**32 — Monte Carlo.** Antes × depois da correção do vazamento (200 simulações).
O ponto do slide é a queda: a sensibilidade cai de 84,3% para 9,1% quando o modelo
deixa de ser avaliado nos próprios hard samples usados no treino.

**33 — Feature importance.** Top 17 variáveis (ganho médio dos 10 folds, ± IC 95%).
O CSV tem os nomes em inglês e em português.

## Fontes

Os arquivos de números citam, no rodapé, o arquivo de origem em `output/`.
A pasta `SLIDE/` (a antiga) continua existindo com o material completo, incluindo
as versões do conjunto SAD — esta aqui é só o recorte pedido.
