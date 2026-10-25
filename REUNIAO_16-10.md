# Reunião 16/10 — SAD: seleção de features, tuning e comparação com GAD

## ⭐ RESULTADO FINAL (Monte Carlo v1, média ± σ entre 200 simulações)

| Métrica | **GAD** (15 features, depth 8, SMOTE 1.0) | **SAD** (17 features, depth 3, SMOTE 0.2) |
|---|---|---|
| **Kappa** | **0.883 ± 0.114** | **0.899 ± 0.160** |
| Sensibilidade | 93.3% ± 11.4 | 87.3% ± 21.9 |
| Especificidade | 96.8% ± 4.0 | 99.8% ± 1.3 |
| F1 | 90.7% ± 9.1 | 90.7% ± 14.9 |
| Acurácia | 96.1% ± 3.8 | 98.5% ± 2.4 |
| Kappa com smoothing | 0.887 ± 0.113 | 0.891 ± 0.162 |
| Positivos nos 20 hard samples | 4 | 2 |

**SAD chegou ao nível do GAD em Kappa.** O GAD detecta mais casos (sensibilidade maior) e é
mais estável (σ menor). O SAD erra menos nos negativos (especificidade quase 100%).

Os dois alvos precisaram de ajustes opostos:
- **GAD:** remover features + árvores profundas + balanceamento completo
- **SAD:** manter as features + árvores rasas + balanceamento leve

---

## Evolução (todos os resultados, Kappa ± σ)

| Etapa | Kappa | Sens | Spec | F1 |
|---|---|---|---|---|
| **GAD** — original (17 features, depth 6) | 0.813 ± 0.128 | 84.3% ± 15.3 | 96.2% ± 4.1 | 85.7% ± 10.1 |
| GAD — sem CD e Family History (15 features) | 0.867 ± 0.117 | 93.0% ± 11.5 | 96.2% ± 4.0 | 89.4% ± 9.3 |
| **GAD — final (+ depth 8)** | **0.883 ± 0.114** | 93.3% ± 11.4 | 96.8% ± 4.0 | 90.7% ± 9.1 |
| **SAD** — original (17 features, depth 6, SMOTE 1.0) | 0.779 ± 0.142 | 83.0% ± 17.7 | 95.5% ± 4.7 | 82.1% ± 11.8 |
| SAD — depth 4 | 0.842 ± 0.152 | 82.4% ± 18.0 | 98.6% ± 2.7 | 86.8% ± 12.8 |
| **SAD — final (depth 3 + SMOTE 0.2)** | **0.899 ± 0.160** | 87.3% ± 21.9 | 99.8% ± 1.3 | 90.7% ± 14.9 |

Ganho total: GAD +8.6%, SAD +15.5%.

---

## Metodologia (igual para os dois alvos)

- **Modelo:** XGBoost + SMOTE
- **Avaliação oficial:** Monte Carlo v1. Split 80/20, os 20 casos de teste mais incertos
  ("hard samples"), 200 simulações sorteando 15 deles para o retreino.
- **Seleção de features em 3 etapas:** Spearman (redundância), Permutation Importance
  (ruído) e validação no Monte Carlo. Só se remove o que melhora no Monte Carlo.
- **Comparação justa:** quando duas configs geram hard samples diferentes, elas também são
  avaliadas nos **mesmos casos** e no **teste inteiro** (58 casos, 9 com SAD).

## O que foi testado no SAD

| Frente | Testes | Resultado |
|---|---|---|
| Remover features (com SMOTE 1.0) | 20 | Todas pioraram |
| Combinar features correlacionadas em scores | 4 | Pioraram |
| Criar interações (ex: sono × fobia social) | 7 | Pioraram |
| Hiperparâmetros (depth, n_estimators, regularização, threshold) | ~30 | depth 3–4 ajudou |
| Outros algoritmos (LightGBM, CatBoost, RandomForest, HistGB) | 9 | XGBoost segue o melhor |
| **Balanceamento** (ratio do SMOTE, under/oversampling) | 35 | **Ajuste decisivo** (0.84 → 0.90) |
| Grid amplo (SMOTE × depth × n_estimators × learning_rate) | 240 | Sem ganho real (prova mais fácil) |
| **Remover features na config final** | 18 + 35 combinações | **Achado novo, ver abaixo** |

As 4 variáveis removidas no pré-processamento (Depression, Type A Stressors, Physical Symptoms,
Family History – Substance Abuse) ficaram fora por decisão do projeto. Fica registrado que
Physical Symptoms tem a 3ª maior correlação com SAD no dataset bruto (ρ = 0.32).

## Achados principais

1. **O balanceamento era o problema do SAD.** O SAD já tem 19% de positivos no treino, então
   o SMOTE completo criava sintéticos demais.
2. **Cuidado com Kappa muito alto no Monte Carlo.** O grid achou uma config com Kappa 0.99, mas
   os hard samples dela tinham só 1 criança positiva. Avaliadas na mesma prova, as configs
   empataram, e por isso essa não foi adotada.
3. **Bugs corrigidos:** remover Poverty Status alterava o número de amostras (287 → 306). Além
   disso, o pipeline oficial passou a escolher os hard samples com o mesmo `max_depth` do retreino.

---

## 🆕 Nova rodada: remoção de variáveis na config final (decisão pendente)

Refeita a importância (Permutation, gain do XGBoost, Spearman) e a remoção de variáveis já com
SMOTE 0.2 + depth 3. Melhor combinação: **remover Sex + Number of Bio. Parents + Number of
Impairments (14 features)**. As duas configs foram avaliadas nos mesmos casos:

| Prova (mesmos casos para as duas) | 17 features (atual) | **14 features** |
|---|---|---|
| A: hard samples da atual (2 positivos) | 0.899 ± 0.160 | **0.955 ± 0.123** |
| B: hard samples da 14f (4 positivos) | 0.839 ± 0.160 | **0.873 ± 0.132** |
| C: teste inteiro (58 casos, 9 positivos) | 0.848 ± 0.086 | **0.879 ± 0.082** |

A versão de 14 features ganha nas três provas, em Kappa, sensibilidade e σ.

**Ponto a decidir:** no Monte Carlo oficial, cada config usa os próprios hard samples. A versão de
14 features gera uma prova mais difícil (4 positivos, igual ao GAD), então o número oficial **cai**
de 0.899 para **0.873 ± 0.132**, mesmo o modelo sendo melhor. Por outro lado, a comparação com ox
GAD (0.883, também com 4 positivos) fica de igual para igual.

## Onde está cada coisa

- Configuração por alvo: `scripts/config.py`
- Resultado oficial: `resultados/gad/` e `resultados/sad/melhor_resultado_monte_carlo.txt`
- Detalhamento completo: `docs/gad/FEATURE_SELECTION.md`, `docs/sad/FEATURE_SELECTION.md`
- Scripts dos testes: `scripts/sad/analysis/`
- Saídas dos testes: `output/sad/feature_removal_runs/` (rodada nova em `07_remocao_config_final/`)
