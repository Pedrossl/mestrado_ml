# Feature Selection — SAD

Espelha [docs/gad/FEATURE_SELECTION.md](../gad/FEATURE_SELECTION.md), mas para SAD.

## Conclusao (selecao de features + tuning completo)

Configuracao final: **17 features (nenhuma removida), max_depth=3, SMOTE
sampling_strategy=0.2**.
Kappa **0.8992**, Sensibilidade 87.25%, F1 90.65%, Especificidade 99.75%
(Monte Carlo v1, 200 simulacoes, sorteio 15/20). Ver
`resultados/sad/melhor_resultado_monte_carlo.txt`.

Com isso **SAD alcancou o GAD** (Kappa 0.8992 vs 0.8832) pela mesma regra
(Monte Carlo v1). Tres achados:

1. **Nenhuma feature pode ser removida** — testadas as 5 candidatas prioritarias
   e depois as 17 uma a uma (leave-one-out); toda remocao piora. Em SAD cada
   feature carrega sinal proprio. `FEATURE_DROP_COLUMNS_BY_TARGET["SAD"]` vazia.
2. **Arvores rasas** (`max_depth=3`) — oposto do GAD (depth=8).
3. **Balanceamento leve** (`SMOTE sampling_strategy=0.2`) — o ajuste decisivo.
   SAD e pouco desbalanceado (16% positivos) e o SMOTE completo (1.0) distorcia
   a distribuicao com sinteticos demais; reduzir para 0.2 subiu o Kappa de
   0.8415 para 0.8992 e a sensibilidade de 82% para 87%. Ver Etapa 8.

## Por que nao decidir so com Spearman + Permutation Importance

A experiencia com GAD mostrou que CV e Monte Carlo podem discordar (ex: "Number
of Bio. Parents" melhorava a CV mas piorava o Monte Carlo v1). Os dois metodos
abaixo (Spearman e Permutation Importance) tambem usam CV por baixo dos panos —
entao qualquer candidata identificada aqui **precisa** ser validada num
Monte Carlo v1 de SAD antes de entrar em `FEATURE_DROP_COLUMNS_BY_TARGET["SAD"]`.
Isso se confirmou na pratica: ver Etapa 3.

## Etapa 1 — Correlacao de Spearman

Script: `scripts/sad/analysis/spearman_correlation.py`
Resultado: `output/sad/correlacao/correlacao_com_sad.txt` e `matriz_spearman_sad.png`

### Correlacao com SAD (17 features, ordenado por |rho|)

```
Number of Sleep Disturbances            rho=+0.388
Social Phobia                           rho=+0.315
Number of Sensory Sensitivities         rho=+0.259
Number of Type B Stressors              rho=+0.235
Number of Impairments                   rho=+0.233
CD                                      rho=+0.195
Frequency Irritable Mood                rho=+0.171
ODD                                     rho=+0.163
Frequency Temper Tantrums               rho=+0.155
Number of Siblings                      rho=-0.092  <- fraca
Race                                    rho=-0.089  <- fraca
Poverty Status                          rho=+0.059  <- fraca
Number of Bio. Parents                  rho=-0.049  <- fraca
Family History - Psychiatric Diagnosis  rho=+0.046  <- fraca
Sex                                     rho=-0.029  <- fraca
ADHD                                    rho=-0.019  <- fraca
Age                                     rho=+0.002  <- fraca
```

**Diferenca notavel em relacao a GAD**: em GAD, as variaveis demograficas/
socioeconomicas (Sex, Race, Poverty Status, Bio Parents, Age) tinham sinal
moderado. Em SAD, praticamente todas elas (Sex, Race, Poverty Status, Bio
Parents, Age, ADHD, Family History, Number of Siblings) tem |rho| < 0.10 — SAD
parece muito mais associado a variaveis clinicas (sono, sensibilidade
sensorial, fobia social, impairments) do que a fatores demograficos.

### Pares redundantes (|rho| >= 0.40)

```
ODD <-> Number of Impairments:            rho=+0.528  (candidata: ODD)
CD <-> ODD:                                rho=+0.502  (candidata: ODD)
ODD <-> Frequency Irritable Mood:          rho=+0.501  (candidata: ODD)
Number of Impairments <-> Freq. Irritable: rho=+0.473  (candidata: Freq. Irritable Mood)
Freq. Temper Tantrums <-> Freq. Irritable: rho=+0.468  (candidata: Freq. Temper Tantrums)
Race <-> Number of Bio. Parents:           rho=-0.414  (candidata: Number of Bio. Parents)
Social Phobia <-> Sensory Sensitivities:   rho=+0.409  (candidata: Sensory Sensitivities)
Social Phobia <-> Number of Impairments:   rho=+0.408  (candidata: Number of Impairments)
```

**Achado central**: ao contrario de GAD (onde ODD era a feature *mais forte*
do bloco comportamental e CD foi removida por ser mais fraca), em SAD **ODD e
a mais fraca** em todos os 3 pares em que aparece (perde para CD, para Number
of Impairments e para Frequency Irritable Mood). Isso e o oposto do que
aconteceu em GAD — reforca que a selecao de features realmente precisa ser
feita por alvo, nao herdada de GAD.

## Etapa 2 — Permutation Importance

Script: `scripts/sad/analysis/permutation_importance.py`
Resultado: `output/sad/permutation_importance_sad.txt`

```
 1  Sex                                     -0.0548  <- candidata
 2  Family History - Psychiatric Diagnosis  -0.0527  <- candidata
 3  Number of Impairments                   -0.0468  <- candidata
 4  Number of Siblings                      -0.0436  <- candidata
 5  Number of Bio. Parents                  -0.0305  <- candidata
 6  Frequency Temper Tantrums               -0.0297  <- candidata
 7  Number of Sensory Sensitivities         -0.0296  <- candidata
 8  Poverty Status                          -0.0253  <- candidata
 9  ADHD                                    -0.0173  <- candidata
10  Race                                    -0.0139  <- candidata
11  Age                                     -0.0137  <- candidata
12  Social Phobia                           -0.0116  <- candidata
13  ODD                                     +0.0000  <- candidata
14  Frequency Irritable Mood                +0.0212
15  CD                                      +0.0219
16  Number of Type B Stressors              +0.0406
17  Number of Sleep Disturbances            +0.1018
```

**Atencao**: 12 das 17 features tem importancia **negativa** — um sinal muito
mais ruidoso do que GAD teve (onde so 1-2 features eram negativas). Isso sugere
que o modelo de SAD com todas as 17 features esta mais proximo de overfitting
generalizado do que o de GAD, e nao necessariamente que 12 features devam ser
removidas de uma vez — remover todas ao mesmo tempo mudaria o problema demais
para validar com confianca (mesmo erro que causou o confundimento de amostras
em GAD na primeira tentativa, ver `docs/gad/FEATURE_SELECTION.md`). O caminho
correto e testar candidatas poucas por vez, validando cada uma no Monte Carlo.

`ODD` deu importancia exatamente 0.0000 — nao e erro do script: `ODD` tem
variancia normal (34/287 positivos), mas correlaciona fortemente com CD,
Number of Impairments e Frequency Irritable Mood (ver Etapa 1), entao o
XGBoost aparentemente nunca depende de ODD nas seus splits de decisao em
nenhum dos 10 folds — bate exatamente com o achado do Spearman.

## Feature importance do XGBoost (complementar, nao substitui Permutation Importance)

Gerado via `scripts.models.modelo_xgboost.analisar_importancia_features('SAD')`.
Resultado: `output/plots/XGBoost/SAD/feature_importance_sad.txt`

```
 1  Number of Sleep Disturbances            0.1998
 2  Number of Sensory Sensitivities         0.1569
 3  Number of Type B Stressors              0.0697
 4  Social Phobia                           0.0641
 5  ODD                                     0.0480
 ...
17  Race                                    0.0249
```

**Nota metodologica**: esta e a importancia por "gain" (reducao de loss durante
o treino) — mede se a feature foi usada em algum split, nao se afeta a decisao
final no conjunto de teste. Por isso ODD aparece aqui com importancia moderada
(rank 5) mas teve Permutation Importance zero: o XGBoost usa ODD nos splits
durante o treino, mas ela nao muda a classificacao final porque outras
features correlacionadas (CD, Impairments) "cobrem" a mesma informacao. Para
decidir remocao, Permutation Importance (que mede o efeito real na metrica de
avaliacao) e mais confiavel que gain — mesma logica usada para GAD.

## Candidatas identificadas por Spearman + Permutation Importance

Cruzando os 2 sinais, as candidatas mais fortes a remocao eram (em ordem de
prioridade, mais evidencia combinada primeiro):

1. **ODD** — mais fraca em 3 pares redundantes (Spearman) + importancia zero (Permutation)
2. **Family History - Psychiatric Diagnosis** — correlacao fraca com SAD (rho=0.046) + Permutation Importance bem negativa (-0.053), igual ao papel que teve em GAD
3. **Number of Bio. Parents** — redundante com Race (Spearman) + Permutation Importance negativa (-0.031). Atencao: em GAD, remover Bio Parents piorou o Monte Carlo apesar de melhorar a CV — motivo extra para nao remover sem validar.
4. **Sex** e **Number of Impairments** — maior importancia negativa (Permutation), mas Number of Impairments tem correlacao razoavel com SAD (rho=0.233) e gain importancia moderada — sinal contraditorio.

## Etapa 3 — Validacao Monte Carlo v1

### 3.1 — As 5 candidatas prioritarias (hiperparametros default)

Script: `scripts/sad/analysis/mc_v1_candidatos.py`

```
Cenario                          Feat      Kappa    Delta     Sens       F1     Spec
------------------------------------------------------------------------------------
baseline (17 features)             17     0.7788        -   83.00%   82.11%   95.47%
sem ODD                            16     0.7439  -0.0348   80.25%   79.35%   94.72%
sem Family History                 16     0.7518  -0.0270   81.12%   79.92%   94.88%
sem Number of Bio. Parents         16     0.7193  -0.0594   78.12%   77.42%   94.22%
sem Sex                            16     0.7269  -0.0518   79.42%   80.52%   92.71%
sem Number of Impairments          16     0.7664  -0.0124   82.88%   81.23%   94.88%
```

Todas as 5 pioraram. Isso levou a testar mais a fundo (3.2).

### 3.2 — Todas as 17 features, uma a uma (leave-one-out)

Como o sinal da Permutation Importance era muito mais ruidoso em SAD do que em
GAD (12/17 features negativas), fizemos a varredura completa em vez de confiar
só nas 5 candidatas prioritarias. Script: `scripts/sad/analysis/mc_v1_leave_one_out.py`.

**⚠️ Armadilha encontrada e corrigida**: a primeira rodada (hiperparametros
default) apontou `sem Poverty Status` como a unica remocao que melhorava o
Kappa (+0.0097). Mas `Poverty Status` tem 19 valores faltantes — removê-la faz
`preparar_dados()` reincorporar essas 19 linhas via `dropna()`, mudando a
amostra de 287 para 306 (exatamente o mesmo bug de confundimento de amostra
que ja tinha sido identificado e corrigido em GAD, ver
`docs/gad/FEATURE_SELECTION.md`, secao 1). Testando com as **mesmas 287
linhas** do baseline, remover Poverty Status na verdade **piora** o Kappa
(0.8415 -> 0.6666 com max_depth=4 — a pior remocao entre as 17). Os scripts
`mc_v1_leave_one_out.py` e `mc_v1_candidatos.py` foram corrigidos para nunca
mais chamar `preparar_dados()` de novo por cenario — sempre usam as mesmas
linhas do baseline (`df_base.drop(columns=drops)`), evitando esse erro se
alguem reexecutar essa analise no futuro.

Resultado final (com `max_depth=4`, ja apos o tuning da Etapa 4 — ordenado por
delta, melhor primeiro):

```
Cenario                                       Feat      Kappa     Delta     Sens     Spec
--------------------------------------------------------------------------------------------
baseline (17 features)                          17     0.8415         —   82.38%   98.62%
sem ODD                                         16     0.8269   -0.0147   81.12%   98.34%
sem ADHD                                        16     0.8190   -0.0226   79.12%   98.66%
sem Frequency Irritable Mood                    16     0.8041   -0.0374   77.00%   98.73%
sem CD                                          16     0.7876   -0.0540   80.88%   96.69%
sem Race                                        16     0.7862   -0.0553   81.83%   97.00%
sem Family History - Psychiatric Diagnosis      16     0.7855   -0.0560   83.00%   95.81%
sem Number of Sensory Sensitivities             16     0.7855   -0.0560   78.60%   97.10%
sem Social Phobia                               16     0.7765   -0.0650   79.38%   96.59%
sem Frequency Temper Tantrums                   16     0.7745   -0.0671   82.38%   95.53%
sem Number of Bio. Parents                      16     0.7660   -0.0755   78.08%   96.07%
sem Number of Sleep Disturbances                16     0.7486   -0.0930   84.75%   96.28%
sem Sex                                         16     0.7376   -0.1040   78.62%   95.06%
sem Age                                         16     0.7208   -0.1207   77.83%   93.18%
sem Number of Impairments                       16     0.7148   -0.1268   74.12%   95.84%
sem Number of Type B Stressors                  16     0.7122   -0.1293   75.67%   95.94%
sem Poverty Status                              16     0.6666   -0.1749   71.25%   96.83%
sem Number of Siblings                          16     0.6366   -0.2049   73.00%   95.58%
```

**Nenhuma das 17 remocoes individuais melhora o Kappa** — a mais barata
(ODD) ainda perde 0.0147. Confirma a conclusao da secao 3.1 de forma exaustiva.

## Etapa 4 — Tuning de hiperparametros (o que realmente ajudou)

Script: `scripts/sad/analysis/mc_v1_tuning.py`. Testado sobre o baseline
completo (17 features):

```
max_depth=4                      Kappa=0.8415 (+0.0628) <-- MELHOR
max_depth=3                      Kappa=0.8224 (+0.0436) <-- MELHOR
n_estimators=50                  Kappa=0.7542 (-0.0246)
max_depth=5                      Kappa=0.7525 (-0.0263)
max_depth=8                      Kappa=0.7440 (-0.0348)
max_depth=10                     Kappa=0.7409 (-0.0379)
n_estimators=200                 Kappa=0.7379 (-0.0409)
SMOTETomek                       Kappa=0.7359 (-0.0428)
SMOTEENN                         Kappa=0.7324 (-0.0463)
N_HARD=25, sorteio=18            Kappa=0.7313 (-0.0475)
max_depth=7                      Kappa=0.7282 (-0.0505)
N_HARD=15, sorteio=10            Kappa=0.6553 (-0.1235)
```

**Achado central: `max_depth=4` (arvores mais RASAS) e o oposto exato do que
ajudou GAD** (`max_depth=8`, arvores mais profundas). Faz sentido em conjunto
com a Etapa 2: a Permutation Importance de SAD tinha 12/17 features negativas
(sinal ruidoso, indicando um modelo mais propenso a se ajustar a ruido
especifico do treino). Arvores mais rasas tem menos capacidade de decorar esse
ruido nos hard samples durante o retreino do Monte Carlo — o oposto do
problema de GAD, que precisava de MAIS capacidade para capturar interacoes
reais entre features.

Testamos tambem Smoothing (eps=0.1) sobre a config final (17f, depth=4):
Kappa cai para 0.8269 (sem smoothing) -> 0.8082 -> piora. Mesma logica de GAD:
smoothing so ajuda quando a arvore e profunda o suficiente para overfitar nos
hard samples; aqui reduzimos a profundidade, entao nao ha overfitting a
corrigir — smoothing so remove sinal util.

## Etapa 5 — Segunda rodada de tuning (regularizacao, balanceamento, threshold)

Script: `scripts/sad/analysis/mc_v1_tuning_round2.py`. Testado sobre o
baseline ja com `max_depth=4` — objetivo era ver se dava pra melhorar ainda
mais o Kappa 0.8415.

```
colsample_bytree=0.8             Kappa=0.8378 (-0.0038)
scale_pos_weight=2               Kappa=0.8168 (-0.0248)
colsample_bytree=0.6             Kappa=0.8150 (-0.0265)
subsample=0.8+colsample=0.8      Kappa=0.8084 (-0.0331)
learning_rate=0.2                Kappa=0.7974 (-0.0441)
reg_lambda=3.0                   Kappa=0.7884 (-0.0531)
scale_pos_weight=3               Kappa=0.7789 (-0.0627)
learning_rate=0.05               Kappa=0.7742 (-0.0674)
subsample=0.8                    Kappa=0.7710 (-0.0705)
n_estimators=50 (com depth=4)    Kappa=0.7618 (-0.0797)
ADASYN                           Kappa=0.7544 (-0.0871)
subsample=0.6                    Kappa=0.7440 (-0.0975)
BorderlineSMOTE                  Kappa=0.7432 (-0.0983)
reg_alpha=1.0                    Kappa=0.7413 (-0.1002)
min_child_weight=3               Kappa=0.7028 (-0.1388)
min_child_weight=5               Kappa=0.5407 (-0.3008)
Youden threshold                 Kappa=0.0638 (-0.7777)
```

**Nenhuma configuracao melhorou.** `colsample_bytree=0.8` chegou perto
(-0.0038, dentro da margem de ruido entre simulacoes) mas nao supera o
baseline. `Youden threshold` colapsou completamente (Kappa 0.0638, Sens
99.62%, Spec 14.62%) — o threshold otimo calculado em cima de so 20 amostras
de teste (os hard samples) e instavel e escolheu um valor perto de zero,
classificando quase tudo como positivo. Mesmo padrao patologico ja visto em
GAD com Youden threshold no Monte Carlo v1 (ver `docs/gad/FEATURE_SELECTION.md`,
secao 5.2).

Neste ponto da investigacao, `max_depth=4` sozinho era o melhor ajuste
encontrado (Kappa 0.8415). **NOTA: isso foi superado na Etapa 8** — ao ajustar
o balanceamento (SMOTE 0.2), a profundidade otima mudou para 3 e o Kappa subiu
para 0.8992. Esta etapa fica como registro da progressao.

## Etapa 6 — Triagem rapida via CV (filtro barato antes do Monte Carlo)

Script: `scripts/sad/analysis/cv_screening.py`. Ideia: CV 10-fold e ~20x mais
barato que o Monte Carlo v1 (10 treinos vs 200 simulacoes x2). Usa CV como
filtro rapido e so promove ao Monte Carlo (caro) o que parecer promissor.
Testou: leave-one-out via CV (sanity check), pares entre as 5 features menos
prejudiciais, subconjuntos guiados pelos agrupamentos de Spearman, e RFECV
(Recursive Feature Elimination, busca sistematica do sklearn).

Melhores sinais encontrados na CV (Kappa CV baseline = 0.3040 — escala bem
mais baixa que o Monte Carlo porque CV nao tem o vazamento de dados do MC v1,
ver `resultados/gad/relatorio_completo_experimentos.md` secao 0):

```
sem Number of Type B Stressors + Poverty Status   Kappa=0.3228 (+0.0188)
sem Number of Type B Stressors                    Kappa=0.3140 (+0.0100)
RFECV (12f, remove Race/BioParents/Siblings/Poverty/ODD)  Kappa=0.2840 (-0.0200 vs CV, mas selecionado por criterio proprio do RFECV)
```

**Validacao no Monte Carlo v1 (as duas mais promissoras):**

```
Config                                                 N    Kappa     Sens     Spec
-------------------------------------------------------------------------------------
baseline 17f                                         287   0.8415   82.38%   98.62%
sem TypeB+Poverty (melhor par na CV)                 287   0.6874   74.75%   96.78%
RFECV 12f (sem Race,BioParents,Siblings,Poverty,ODD)   287   0.7458   82.17%   95.29%
```

**As duas falharam feio no Monte Carlo** (-0.1541 e -0.0957 respectivamente),
apesar de parecerem boas na CV. Essa e a terceira vez que isso acontece para
SAD (a primeira foi Poverty Status sozinha, com o bug de amostra; esta e a
segunda vez com o bug ja corrigido, ou seja, e um efeito real, nao um
artefato). **Conclusao metodologica importante**: para SAD, CV e Monte Carlo
v1 divergem de forma muito mais forte e consistente do que para GAD. Isso
sugere que os hard samples de SAD (os casos mais dificeis de classificar)
dependem estruturalmente de features que nao aparecem como importantes na CV
media — ou seja, a dificuldade de SAD nao esta distribuida uniformemente
pelas features, esta concentrada em poucas features que so importam nos casos
de fronteira. Isso e coerente com o achado da Etapa 2 (Permutation Importance
muito mais ruidosa em SAD que em GAD).

Isso fecha definitivamente a busca por remocao de features: testamos 18
remocoes individuais + 2 combinacoes guiadas por CV/RFECV = **20 tentativas de
remocao, todas rejeitadas pelo Monte Carlo v1**.

## Etapa 7 — Algoritmo alternativo e engenharia de features (busca estrutural)

Ultima rodada, testando alavancas fora de "features do XGBoost": trocar o
classificador inteiro e substituir blocos de features correlacionadas por
composites, em vez de remove-las.

### 7.1 — Engenharia de features (blocos compostos via Spearman)

Script: `scripts/sad/analysis/mc_v1_feature_engineering.py`. Em vez de
remover features de blocos correlacionados, substitui cada bloco por uma
media padronizada dos seus membros.

```
Cenario                                  Feat    Kappa    Delta     Sens     Spec
------------------------------------------------------------------------------
baseline (17 features cruas)               17   0.8415        -   82.38%   98.62%
so bloco_comportamental composto           13   0.7960  -0.0456   85.58%   94.07%
so bloco_social composto                   16   0.7795  -0.0620   79.50%   96.43%
so bloco_familia composto                  16   0.7712  -0.0704   80.62%   95.91%
todos os 3 blocos compostos                11   0.7449  -0.0966   81.17%   92.93%
```

Todas as composicoes pioraram. Confirma o padrao: para SAD, cada feature
individual carrega sinal proprio que se perde ao combinar — nao e so
redundancia que pode ser compactada sem custo (diferente do que o Spearman
por si so sugeria).

### 7.2 — Troca de algoritmo

Script: `scripts/sad/analysis/mc_v1_algoritmos.py`. Mesma config (17
features), trocando XGBoost por LightGBM, CatBoost, RandomForest e
HistGradientBoosting:

```
HistGradientBoosting (default)   Kappa=0.8448 (+0.0032) <-- empate tecnico
LightGBM depth=4                 Kappa=0.8070 (-0.0346)
LightGBM depth=-1 (default)      Kappa=0.7875 (-0.0540)
RandomForest (default)           Kappa=0.7840 (-0.0576)
CatBoost depth=6 (default)       Kappa=0.7435 (-0.0981)
CatBoost depth=4                 Kappa=0.7215 (-0.1200)
HistGradientBoosting max_depth=4 Kappa=0.6454 (-0.1961)
RandomForest max_depth=6         Kappa=0.6189 (-0.2226)
LightGBM depth=3                 Kappa=0.5371 (-0.3045)
```

`HistGradientBoosting` (default do sklearn) empata tecnicamente com XGBoost
(+0.0032, dentro do ruido entre simulacoes — nao e uma melhora real).
Mantido XGBoost por continuidade com o resto do projeto (GAD, scripts
genericos). Todos os outros algoritmos (LightGBM, CatBoost, RandomForest)
foram claramente piores nesta tarefa especifica.

Nenhum algoritmo alternativo superou o XGBoost. **NOTA: o teto de 0.8415 desta
etapa foi quebrado na Etapa 8** ao investigar o balanceamento a fundo (o
`sampling_strategy` do SMOTE ainda nao tinha sido variado ate aqui) — chegou a
0.8992.

## Etapa 8 — Balanceamento, interacoes e a armadilha do "Kappa 0.90"

Investigacao do balanceamento (scripts `mc_v1_balanceamento.py`,
`mc_v1_balanc_x_depth.py`, `mc_v1_interacoes.py`). As 4 features removidas no
pre-processamento (Depression, Type A Stressors, Physical Symptoms, Family
History - Substance Abuse) foram mantidas fora por decisao do projeto, mesmo
Physical Symptoms tendo rho=+0.324 com SAD (3a maior do dataset bruto).

### 8.1 — Interacoes e composites: nao ajudaram

Adicionar features de interacao (produtos dos top preditores) ou substituir
blocos correlacionados por medias: todas pioraram (ver
`output/sad/feature_removal_runs/04_interacoes/`). Para SAD, adicionar
qualquer feature piora — mais colunas = mais espaco pro modelo overfitar nos
hard samples.

### 8.2 — O ajuste decisivo: balanceamento leve (SMOTE 0.2)

SAD e pouco desbalanceado (16% positivos, vs 15% do GAD — na pratica os dois
sao parecidos, mas SAD tem positivos suficientes). O SMOTE completo (1.0, que
balanceia 50/50) estava criando sinteticos demais e distorcendo a
distribuicao. O grid balanceamento x depth (`mc_v1_balanc_x_depth.py`) mostrou
que reduzir o SMOTE melhora, e que a profundidade otima muda junto:

```
config               Kappa_MC    Sens     F1      Spec
smote0.2 depth=3      0.8992     87.25%   90.65%   99.75%   <-- ADOTADO
smote0.2 depth=5      0.8905     86.75%   89.88%   99.61%
none     depth=5      0.8844     87.25%   89.35%   99.39%
smote0.2 depth=4      0.8819     86.25%   89.10%   99.50%
smote1.0 depth=4      0.8415     82.38%   86.84%   98.62%   (config anterior)
```

**`SMOTE 0.2 + max_depth=3` ganha em TODAS as metricas** do Monte Carlo v1
(a mesma regra usada para escolher a config do GAD): Kappa, sensibilidade, F1
e especificidade todas sobem. Com isso SAD chega a Kappa 0.8992, batendo o GAD
(0.8832).

### 8.3 — Ressalva honesta para a discussao da dissertacao

Vale registrar (sem mudar a decisao): no CV 10-fold honesto — protocolo
diferente, sem a selecao de hard samples do Monte Carlo — o SMOTE leve rende
um pouco menos de sensibilidade media que o SMOTE completo. Isso acontece
porque o Monte Carlo v1 avalia a estabilidade do modelo nos casos de fronteira
(hard samples), enquanto o CV mede a deteccao media. Sao duas perguntas
diferentes. Como o projeto inteiro (incluindo GAD) usa o Monte Carlo v1 como
metrica oficial, a config do SAD foi escolhida pela mesma regra. Se em algum
momento a dissertacao priorizar sensibilidade media de triagem em vez de
estabilidade nos hard samples, reavaliar o balanceamento para SAD faz sentido.
Os numeros de CV honesto dos dois alvos ficam registrados em
`resultados/gad/relatorio_completo_experimentos.md` para essa comparacao.

## Conclusao final

**Nenhuma feature foi removida para SAD.** `FEATURE_DROP_COLUMNS_BY_TARGET["SAD"]`
continua vazia — decisao validada por 20 testes de remocao (5 candidatas + 12
leave-one-out + 2 combinacoes de CV/RFECV + composicoes/interacoes), todas
pioraram.

**Configuracao final**: 17 features, `max_depth=3`
(`XGB_MAX_DEPTH_BY_TARGET["SAD"]`) e `SMOTE sampling_strategy=0.2`
(`SMOTE_STRATEGY_BY_TARGET["SAD"]`), ambos em `scripts/config.py`. Sem smoothing.

Evolucao: Kappa 0.7788 (17f, depth=6, SMOTE 1.0) -> 0.8415 (depth=4) ->
**0.8992** (depth=3 + SMOTE 0.2). Melhora total de **15.5%**, vinda de
hiperparametro + balanceamento (nenhuma feature removida — em SAD remover
sempre piora, diferente de GAD). SAD agora empata com o GAD (0.8832) pela
metrica oficial do projeto.

### Nota metodologica: dois numeros ligeiramente diferentes para o mesmo Kappa

O pipeline "oficial" (`scripts/sad/experimento_hard_samples/`, passo1→passo2→passo3)
da Kappa **0.8134** para a mesma configuracao (17f, depth=4), nao 0.8415. A
diferenca vem de uma escolha de design pre-existente (igual em GAD): o passo2
sempre identifica os hard samples usando um modelo de referencia com
`max_depth` default (6), independente do hiperparametro tunado usado depois no
passo3. Os scripts `mc_v1_*.py` (usados nesta analise) usam o mesmo
`max_depth` tunado tanto para identificar hard samples quanto para
retreinar/avaliar — por isso os numeros nao batem exatamente. Ambos sao
validos, mas medem coisas ligeiramente diferentes; os dois estao registrados
em `resultados/sad/melhor_resultado_monte_carlo.txt`. GAD tem essa mesma
caracteristica (nao investigada anteriormente) — vale reconciliar os dois
pipelines em uma proxima rodada de limpeza, para os dois alvos.
