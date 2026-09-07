# Pré-processamento e Resultados — Predição de GAD e SAD em Crianças
**Dissertação de Mestrado | Aprendizado de Máquina**

---

## 1. Contexto do Estudo

Este trabalho tem como objetivo desenvolver modelos de aprendizado de máquina capazes de predizer dois transtornos de ansiedade em crianças: o **GAD** (Transtorno de Ansiedade Generalizada) e o **SAD** (Transtorno de Ansiedade Social). Ambos são condições frequentes na infância e o diagnóstico precoce é fundamental para garantir intervenção e tratamento adequados.

O **GAD** se manifesta como preocupação excessiva e persistente sobre situações do dia a dia — escola, saúde, família, desempenho —, diferindo da ansiedade normal por ser desproporcional, difícil de controlar e por causar prejuízo funcional significativo. O **SAD**, por sua vez, caracteriza-se por medo intenso e persistente de situações sociais ou de desempenho, como falar em público ou interagir com desconhecidos, levando à evitação dessas situações com sofrimento marcante.

---

## 2. Os Dados

Foram utilizados dois datasets distintos, com origens diferentes, o que confere validade ao processo de avaliação dos modelos:

- **Dataset de Treino — PAS**: 917 crianças
- **Dataset de Teste — PTRTS**: 307 crianças (27 variáveis clínicas, demográficas e socioeconômicas)

O dataset de teste continha originalmente **8.289 medições**, com apenas **20 campos sem preenchimento** (0,24% de incompletude), o que representa uma taxa muito baixa de dados faltantes. As duas variáveis-alvo apresentaram a seguinte distribuição no conjunto de teste:

| Transtorno | Positivos (com) | Negativos (sem) | % Positivos |
|---|---|---|---|
| GAD | 44 crianças | 243 crianças | 15,3% |
| SAD | 46 crianças | 241 crianças | 16,0% |

Essa distribuição revela um **desbalanceamento severo de aproximadamente 85/15**, que é um dos principais desafios metodológicos do trabalho e motivou o uso extensivo de técnicas de balanceamento de classes.

---

## 3. Pré-processamento

O pré-processamento foi aplicado integralmente ao dataset de teste (PTRTS), passando pelas etapas descritas a seguir.

### 3.1 Tratamento de Valores Ausentes

Foram identificadas duas colunas com dados faltantes:

| Coluna | Registros ausentes | % do total |
|---|---|---|
| Poverty Status | 19 | 6,19% |
| Social Phobia | 1 | 0,33% |

Os registros com valores ausentes foram **removidos via `dropna()`**, resultando em um dataset final de **287 amostras** para treinamento e avaliação dos modelos.

### 3.2 Eliminação de Variáveis

Das 27 colunas originais, **8 foram removidas**, por dois critérios distintos:

**Por critério clínico ou técnico** — variáveis que não deveriam integrar os modelos preditivos:

| Coluna removida | Motivo |
|---|---|
| Depression | Requer estudo clínico específico; apenas 17 casos positivos (5,5%) — volume insuficiente |
| Number of Type A Stressors | Ausência do valor 0 (distribuição problemática; mín=1, máx=7) |
| Number of Physical Symptoms | Fora do escopo dos modelos definidos |
| Family History - Substance Abuse | Fora do escopo dos modelos definidos |

**Por serem identificadores ou pesos externos** — colunas que não são features preditivas:

| Coluna removida | Motivo |
|---|---|
| Subject | Identificador único do paciente |
| GAD Probability - Gamma | Probabilidade externa pré-calculada; usá-la como feature geraria vazamento de informação |
| SAD Probability - Gamma | Idem |
| Sample Weight | Peso amostral externo; não é uma feature do indivíduo |

Resultado: de 27 colunas originais, **restaram 19 após as remoções**.

### 3.3 Transformações e Padronização

Com as colunas definidas, foram aplicadas as seguintes transformações:

**Codificação de variáveis categóricas:**

| Variável | Transformação |
|---|---|
| Sex | Codificação binária: M → 0, F → 1 |

**Binarização:**

| Variável | Antes | Depois | Critério |
|---|---|---|---|
| Number of Siblings | Contagem ordinal: {0:82, 1:128, 2:60, 3:37} | Binário: {0:82, 1:225} | 0 = sem irmãos; 1 = tem pelo menos 1 irmão |

A binarização de *Number of Siblings* foi adotada por reduzir ruído na variável sem perder a informação clinicamente relevante: a presença ou ausência de irmãos, não o número exato.

**Normalização customizada:**

| Variável | Mapeamento |
|---|---|
| Number of Bio. Parents | 0 pais → 0.0 / 1 pai → 0.5 / 2 pais → 1.0 |

**Escalonamento MinMax [0, 1]** — aplicado em 9 variáveis contínuas:

- Age
- Number of Impairments
- Number of Type B Stressors
- Frequency Temper Tantrums
- Frequency Irritable Mood
- Number of Sleep Disturbances
- Number of Sensory Sensitivities

> **Importante:** nas execuções com validação cruzada, o `MinMaxScaler` foi **fitado exclusivamente nos dados de treino** de cada fold e aplicado ao conjunto de teste sem re-fitting. Isso garante a **prevenção de data leakage** na etapa de normalização.

### 3.4 Definição Final do Dataset

Após todo o pré-processamento:

| Parâmetro | Valor |
|---|---|
| Total de amostras | 287 |
| Features por modelo | 17 |
| Variáveis-alvo | GAD e SAD (binárias, modeladas separadamente) |
| Regra de separação | Quando o alvo é GAD, SAD é removida das features (e vice-versa) |

---

## 4. Análise Exploratória

Antes da modelagem, foi realizada uma análise exploratória com os seguintes componentes:

- **Mapeamento dos tipos de variáveis**: classificação de cada coluna em booleana (0/1), inteira, decimal ou texto.
- **Análise de balanceamento**: confirmação do desbalanceamento ~85/15 para ambas as variáveis-alvo.
- **Matriz de correlação de Spearman**: calculada entre todas as variáveis numéricas. A correlação de Spearman foi escolhida por ser mais adequada a dados categóricos e ordinais, que predominam no dataset.

---

## 5. Configuração dos Experimentos

### 5.1 Validação

- **10-fold Stratified Cross-Validation**: mantém a proporção de classes (~15% positivos) em cada fold.
- **Busca de hiperparâmetros**: Nested CV com `RandomizedSearchCV` (CV externo 10-fold + CV interno 5-fold; n\_iter = 50 rápida / 200 completa).

### 5.2 Algoritmos Avaliados

| Algoritmo | Observação |
|---|---|
| **XGBoost** | Algoritmo principal; melhor desempenho geral |
| **SVM (kernel RBF)** | Segundo algoritmo principal |
| ADTree (via Weka) | Modelo interpretável de referência |
| HistGradientBoosting | Avaliado como alternativa |
| Extra Trees | Avaliado como alternativa |

### 5.3 Técnicas de Balanceamento Testadas (13 no total)

| Categoria | Técnicas |
|---|---|
| Oversampling | SMOTE, ADASYN, BorderlineSMOTE, SMOTEENN, SMOTETomek |
| Undersampling | NearMiss, Random Undersampling (RUS), IHT |
| Ensemble balanceado | EasyEnsemble, Balanced Random Forest, RUSBoost |
| Ponderação de classes | Class Weighting (scale\_pos\_weight natural ~5.5× e extremo ~20×) |
| Baseline | Sem balanceamento |

> **Critério fundamental de metodologia:** SMOTE e Undersampling foram sempre aplicados **dentro de cada fold** de treinamento, nunca antes da divisão dos folds. Isso evita o data leakage que ocorreria se amostras sintéticas geradas a partir do conjunto completo "vazassem" para o conjunto de teste.

### 5.4 Métricas Adotadas

Dado o desbalanceamento severo (~85/15), a acurácia foi descartada como métrica principal. Um modelo que classifique **todas** as amostras como negativo atingiria 84,7% de acurácia com **sensibilidade = 0%** — o que é clinicamente inútil.

| Métrica | Papel |
|---|---|
| **Sensibilidade (Recall)** | Métrica primária: detectar casos reais de GAD/SAD |
| **Kappa de Cohen** | Concordância além do acaso (escala Landis & Koch, 1977) |
| **F1-Score** | Equilíbrio entre precisão e sensibilidade |
| Score Composto | √(Sensibilidade × F1); penaliza queda em qualquer um dos dois |
| Especificidade | Complementar; avalia o trade-off de falsos positivos |

**Escala de interpretação do Kappa (Landis & Koch, 1977):**

| Faixa | Interpretação |
|---|---|
| < 0,20 | Concordância leve |
| 0,21 – 0,40 | Concordância razoável |
| 0,41 – 0,60 | Concordância moderada |
| 0,61 – 0,80 | Concordância substancial |
| > 0,80 | Concordância quase perfeita |

Também foram exploradas estratégias de **otimização de threshold**:
- **Threshold F2-score** (Van Rijsbergen, 1979): falsos negativos custam 4× mais que falsos positivos.
- **Youden's J** (Youden, 1950): maximiza Sensibilidade + Especificidade – 1.
- **Curva Precisão-Recall** (Davis & Goadrich, 2006).

---

## 6. Resultados — GAD (Ansiedade Generalizada)

### 6.1 Resultados Base (XGBoost e SVM com IC 95%)

| Modelo | Técnica | Acurácia | Sensibilidade | Especificidade | F1 | Kappa |
|---|---|---|---|---|---|---|
| XGBoost | Sem balanceamento | 85,7% ± 2,9% | 25,0% ± ? | 96,7% ± 2,3% | 34,9% ± ? | 0,283 |
| XGBoost | Class Weighting | 83,6% ± 3,7% | 38,0% ± 12,4% | 91,8% ± 3,4% | 41,1% ± 12,0% | 0,325 |
| XGBoost | SMOTE | 85,0% ± 2,4% | 34,5% ± 16,5% | 94,3% ± 3,7% | 37,5% ± 15,9% | 0,307 ± 0,151 |
| XGBoost | Undersampling | 64,9% ± 8,5% | 67,0% ± 18,9% | 64,6% ± 8,9% | 37,4% ± 9,8% | 0,195 |
| SVM | Sem balanceamento | 84,7% ± 1,2% | **0,0% ± 0,0%** | 100,0% ± 0,0% | 0,0% ± 0,0% | 0,000 |
| SVM | Class Weighting | 77,0% ± 4,4% | 39,0% ± 11,9% | 84,1% ± 5,9% | 34,0% ± 7,4% | 0,210 |
| SVM | SMOTE | 75,7% ± 4,1% | 22,5% ± 11,0% | 85,3% ± 3,8% | 22,5% ± 10,5% | 0,084 ± 0,127 |
| SVM | Undersampling | 66,6% ± 4,7% | 65,0% ± 18,5% | 67,1% ± 6,1% | 36,3% ± 7,5% | 0,189 |
| ADTree | Sem balanceamento | 82,3% | 11,5% | 95,0% | 13,8% | 0,081 |
| ADTree | Class Weighting | 64,1% | 59,0% | 65,0% | 33,9% | 0,153 |
| ADTree | SMOTE | 32,8% | 84,1% | 23,5% | 27,7% | 0,028 |
| ADTree | Undersampling | 65,9% | 56,8% | 67,5% | 33,8% | 0,156 |

### 6.2 Melhores Modelos GAD — Foco em Equilíbrio (Kappa)

| Rank | Modelo / Técnica | Acurácia | Sensibilidade | F1 | Especificidade | Kappa |
|---|---|---|---|---|---|---|
| 1 | **XGBoost + BorderlineSMOTE** | 85,4% | 41,0% ± 12,8% | 45,2% ± 10,6% | 93,4% | **0,372** ★ melhor kappa |
| 2 | XGBoost + scale\_pos\_weight=20 | 82,3% | 45,5% | 44,4% | 88,9% | 0,340 |
| 3 | XGBoost + SMOTETomek | 85,4% | 38,5% | 43,2% | 93,8% | 0,354 |
| 4 | XGBoost + ADASYN | 85,6% | 38,0% | 43,1% | 94,2% | 0,354 |
| 5 | XGBoost + Threshold F2-score | 81,1% | **47,5%** | 43,7% | 87,2% | 0,326 ★ melhor equilíbrio Sens+F1 |
| 6 | XGBoost + Class Weighting | 83,6% | 38,0% | 41,1% | 91,8% | 0,319 |
| 7 | XGBoost + SMOTE (baseline) | 85,0% | 34,5% | 37,5% | 94,3% | 0,307 |

> Nenhum modelo GAD atingiu concordância moderada (Kappa > 0,40). O melhor Kappa absoluto foi **0,372** (BorderlineSMOTE), no limite superior da faixa razoável.

### 6.3 Melhores Modelos GAD — Foco em Sensibilidade (Triagem)

| Rank | Modelo / Técnica | Acurácia | Sensibilidade | F1 | Especificidade | Kappa |
|---|---|---|---|---|---|---|
| 1 | SMOTEENN + SPW + Threshold F2 | 64,3% | **71,5% ± 19,6%** | 38,7% | 63,0% ± 8,3% | 0,209 ★ máxima sensibilidade |
| 2 | **EasyEnsemble** | 66,3% | 68,5% ± 13,0% | 38,9% | 65,9% ± 7,4% | **0,217** ★ melhor Kappa neste grupo |
| 3 | XGBoost + Undersampling | 64,9% | 67,0% | 37,4% | 64,6% | 0,195 |
| 4 | SVM + Undersampling | 66,6% | 65,0% | 36,3% | 67,1% | 0,189 |
| 5 | SMOTEENN (sem combo) | 73,9% | 52,5% | 39,1% | 77,8% | 0,243 ← bom equilíbrio Sens+Spec |

**Matriz de confusão média (10 folds) — 287 amostras: 44 positivos / 243 negativos:**

| Modelo | VP | FN | FP | VN | Acurácia | Crianças com GAD detectadas |
|---|---|---|---|---|---|---|
| EasyEnsemble | 30 | 14 | 83 | 160 | 66,2% | 30 de 44 |
| SMOTEENN + Combo | 31 | 13 | 90 | 153 | 64,1% | 31 de 44 |
| SMOTE base | 15 | 29 | 14 | 229 | 85,0% | 15 de 44 |

### 6.4 Busca de Hiperparâmetros (200 iterações, XGBoost, GAD)

| Rank | Experimento | Acurácia | Sensibilidade | F1 | Especificidade | Kappa |
|---|---|---|---|---|---|---|
| 1 | BorderlineSMOTE + recall opt. | 15,3% | 100,0% | 26,5% | 0,0% | 0,000 ← inutilizável |
| 2 | SMOTEENN + recall opt. | 20,2% | 100,0% | 27,8% | 5,8% | 0,020 ← inutilizável |
| 3 | SMOTE + recall opt. | 17,3% | 97,5% | 26,4% | 2,4% | -0,001 ← inutilizável |
| — | — | — | — | — | — | — |
| **4** | **BorderlineSMOTE + F2 opt.** | **61,3%** | **83,5% ± 11,8%** | **41,2% ± 5,9%** | **57,3%** | **0,236** ★ melhor com hiperparâmetros |
| 5 | SMOTE + F2 opt. | 62,5% | 73,5% | 37,6% | 60,1% | 0,193 |
| 6 | ADASYN + F2 opt. | 61,8% | 73,5% | 37,4% | 59,2% | 0,191 |

**Melhores hiperparâmetros encontrados (BorderlineSMOTE + F2, n\_iter=200):**

| Parâmetro | Valor |
|---|---|
| learning\_rate | 0,0184 |
| max\_depth | 5–6 |
| n\_estimators | 340 |
| scale\_pos\_weight | 24,8 |
| min\_child\_weight | 8 |
| colsample\_bytree | 0,549 |
| subsample | 0,683 |
| BorderlineSMOTE k\_neighbors | 4 |
| BorderlineSMOTE m\_neighbors | 12 |

> Otimizar por recall puro gerou Sensibilidade=100% e Acurácia≈15%: o modelo passou a classificar **tudo** como positivo (clinicamente inutilizável). Otimizar por F2 gerou o melhor equilíbrio real, com Sensibilidade=83,5% e F1=41,2%.

---

## 7. Resultados — SAD (Ansiedade Social)

### 7.1 Melhores Modelos SAD

| Rank | Modelo / Técnica | Acurácia | Sensibilidade | F1 | Especificidade | Kappa |
|---|---|---|---|---|---|---|
| 1 | **SVM + Class Weighting** | 74,9% ± 6,6% | 52,5% ± 17,1% | 39,7% ± 12,8% | 79,3% ± 7,1% | **0,255** ★ melhor F1+Kappa |
| 2 | XGBoost + SMOTE | 82,6% ± 3,9% | 30,5% ± 14,1% | 34,7% ± 15,5% | 92,6% ± 3,6% | **0,256** ★ melhor Kappa |
| 3 | XGBoost + Class Weighting | 80,1% ± 6,2% | 35,0% ± 16,6% | 36,2% ± 16,9% | 88,8% ± 5,6% | 0,250 |
| 4 | **XGBoost + Undersampling** | 66,9% ± 8,9% | **68,0% ± 14,1%** | **40,9% ± 9,3%** | 66,9% ± 9,7% | 0,239 ★ melhor sensibilidade |
| 5 | SVM + Undersampling | 66,6% ± 7,0% | 59,5% ± 17,8% | 35,8% ± 11,7% | 68,1% ± 7,9% | 0,181 |
| 6 | XGBoost + Sem balanceamento | 82,9% ± 4,3% | 22,0% ± 9,7% | 29,4% ± 13,1% | 94,6% ± 3,7% | 0,213 |
| — | SVM + Sem balanceamento | 83,3% ± 1,5% | **0,0% ± 0,0%** | 0,0% | 99,2% ± 1,3% | -0,012 ← inútil |

> O SAD mostrou ser **mais difícil de predizer que o GAD** (Kappa máximo 0,256 vs. 0,372). Uma diferença relevante: para SAD, o **SVM com Class Weighting se destacou** — ao contrário do GAD, onde o SVM foi consistentemente inferior ao XGBoost.

---

## 8. Comparativo GAD vs SAD

### 8.1 Com SMOTE (IC 95%)

| Algoritmo | Target | Acurácia | Sensibilidade | Especificidade | Kappa |
|---|---|---|---|---|---|
| XGBoost | GAD | 85,0% ± 2,4% | 34,5% ± 16,5% | 94,3% ± 3,7% | 0,307 ± 0,151 |
| XGBoost | SAD | 82,6% ± 3,9% | 30,5% ± 14,1% | 92,6% ± 3,6% | 0,256 ± 0,172 |
| SVM | GAD | 75,7% ± 4,1% | 22,5% ± 11,0% | 85,3% ± 3,8% | 0,084 ± 0,127 |
| SVM | SAD | 74,2% ± 5,9% | 28,5% ± 12,3% | 83,0% ± 7,3% | 0,112 ± 0,128 |

### 8.2 Melhor Kappa por Algoritmo e Target

| | GAD | SAD |
|---|---|---|
| XGBoost | **0,372** (BorderlineSMOTE) | **0,256** (SMOTE) |
| SVM | 0,210 (Class Weighting) | **0,255** (Class Weighting) |
| ADTree | 0,156 (Undersampling) | — |

---

## 9. Feature Importance (XGBoost, média de 10 folds)

Uma das descobertas mais relevantes do trabalho é que os perfis de risco são **completamente distintos** entre GAD e SAD.

### 9.1 Top 5 Features — GAD

| Rank | Feature | Importância | IC 95% |
|---|---|---|---|
| 1 | Number of Impairments | 0,163 | ± 0,046 |
| 2 | Race | 0,112 | ± 0,024 |
| 3 | ODD (Oppositional Defiant Disorder) | 0,083 | ± 0,047 |
| 4 | Social Phobia | 0,080 | ± 0,022 |
| 5 | ADHD | 0,078 | ± 0,035 |

Para o GAD, o modelo é dominado por **fatores de comorbidade e comprometimento funcional**: o número de deficiências é o fator mais relevante, seguido de condições como TOD, Fobia Social e TDAH. Isso faz sentido clínico — crianças com múltiplas comorbidades apresentam maior risco de ansiedade generalizada.

### 9.2 Top 5 Features — SAD

| Rank | Feature | Importância | IC 95% |
|---|---|---|---|
| 1 | Number of Sleep Disturbances | 0,192 | ± 0,041 |
| 2 | Number of Sensory Sensitivities | 0,191 | ± 0,074 |
| 3 | Social Phobia | 0,099 | ± 0,039 |
| 4 | Family History - Psychiatric | 0,063 | ± 0,007 |
| 5 | CD (Conduct Disorder) | 0,049 | ± 0,013 |

Para o SAD, os fatores dominantes são **sensoriais e do sono**: distúrbios do sono e hipersensibilidades sensoriais aparecem com importâncias quase idênticas e muito superiores ao restante. Isso representa um achado clínico interessante e pouco explorado na literatura.

### 9.3 Interpretação do Contraste

| Perfil de risco | GAD | SAD |
|---|---|---|
| Fator principal | Comprometimento funcional múltiplo | Distúrbios do sono e sensoriais |
| Comorbidades relevantes | ODD, ADHD | Social Phobia, CD |
| Componente familiar | Menos relevante | Histórico psiquiátrico familiar presente |

Esse contraste sugere que **GAD e SAD são fenômenos com etiologias distintas** que exigem abordagens clínicas diferentes — e é um dos achados com maior potencial de contribuição científica do trabalho.

---

## 10. Análise Crítica

### 10.1 Por que a acurácia é enganosa aqui

Em datasets com desbalanceamento severo, a acurácia infla artificialmente devido à classe majoritária. O exemplo mais claro no trabalho:

- **SVM sem balanceamento (GAD):** Acurácia = 84,7%, Sensibilidade = **0%**
- O modelo classificou **todas as 287 amostras como "sem GAD"**
- Se qualquer pessoa simplesmente dissesse "nenhuma criança tem GAD", acertaria 84,7% das vezes

Por isso, o **Kappa de Cohen** é a métrica principal: ele mede a concordância *além do acaso*, descontando o que um classificador aleatório já acertaria pela distribuição das classes.

### 10.2 O trade-off Sensibilidade vs. Especificidade

Este é o dilema central do trabalho. Há dois cenários de uso possíveis para os modelos:

**Cenário 1 — Apoio à decisão clínica (equilíbrio):** prioriza Kappa e F1. O modelo serve como ferramenta auxiliar ao diagnóstico, e falsos positivos são custosos (encaminhamentos desnecessários). Melhor opção: **XGBoost + BorderlineSMOTE** para GAD, **SVM + Class Weighting** para SAD.

**Cenário 2 — Triagem em larga escala (máxima sensibilidade):** aceita mais falsos positivos para não perder casos reais. Em contexto de screening populacional, é preferível encaminhar uma criança sadia para avaliação do que deixar uma com transtorno sem identificação. Melhor opção: **EasyEnsemble** para GAD, **XGBoost + Undersampling** para SAD.

### 10.3 Por que os resultados são modestos

| Limitação | Impacto |
|---|---|
| Dataset pequeno (n=287) | ~4–5 amostras positivas por fold de teste — um acerto/erro a mais altera drasticamente a sensibilidade |
| Desbalanceamento 85/15 | Todas as técnicas de balanceamento ajudam, mas nenhuma resolve a escassez de informação da classe positiva |
| Hiperparâmetros não otimizados nos experimentos base | GridSearch/RandomSearch poderia melhorar o Kappa significativamente |
| 17 features | Pode não capturar toda a complexidade dos transtornos; feature selection poderia reduzir ruído |

Os intervalos de confiança amplos confirmam a instabilidade: **Sensibilidade = 34,5% ± 16,5%** (XGBoost + SMOTE, GAD) significa que o IC 95% vai de 18% a 51% — variação que decorre diretamente do pequeno número de casos positivos por fold.

---

## 11. Recomendações Finais por Cenário

### Cenário 1 — Melhor equilíbrio geral (apoio à decisão clínica)

| Target | Modelo | Acurácia | Sensibilidade | F1 | Kappa |
|---|---|---|---|---|---|
| GAD | XGBoost + BorderlineSMOTE | 85,4% | 41,0% ± 12,8% | 45,2% ± 10,6% | **0,372** |
| SAD | SVM + Class Weighting | 74,9% | 52,5% ± 17,1% | 39,7% ± 12,8% | **0,255** |

### Cenário 2 — Máxima sensibilidade aceitável (triagem em larga escala)

| Target | Modelo | Acurácia | Sensibilidade | F1 | Kappa |
|---|---|---|---|---|---|
| GAD | EasyEnsemble | 66,2% | 68,5% ± 13,0% | 38,9% | 0,217 |
| SAD | XGBoost + Undersampling | 66,9% | 68,0% ± 14,1% | 40,9% ± 9,3% | 0,239 |

### Cenário 3 — Máxima sensibilidade absoluta (rastreamento inicial)

| Target | Modelo | Acurácia | Sensibilidade | F1 | Kappa |
|---|---|---|---|---|---|
| GAD | BorderlineSMOTE + F2 opt. (hiperparâmetros) | 61,3% | **83,5% ± 11,8%** | 41,2% ± 5,9% | 0,236 |
| GAD | SMOTEENN + SPW + Threshold F2 | 64,3% | 71,5% ± 19,6% | 38,7% | 0,209 |

### Ranking final dos algoritmos

| Target | 1º | 2º | 3º |
|---|---|---|---|
| GAD | XGBoost (Kappa=0,372) | SVM (0,210) | ADTree (0,156) |
| SAD | XGBoost (Kappa=0,256) ≈ SVM (0,255) | — | — |

---

## 12. Limitações e Trabalhos Futuros

**Limitações reconhecidas:**

- Nenhum modelo atingiu concordância moderada (Kappa > 0,40) para nenhum target.
- Dataset reduzido (n=287; 44 GAD+, 46 SAD+) gera ICs amplos, especialmente em sensibilidade.
- Alta variabilidade entre folds (sensibilidade varia ±10–19 pp dependendo do modelo).
- Acurácia é métrica enganosa aqui: um classificador que chuta tudo negativo tem Acc=84,7%.
- Os modelos não substituem avaliação clínica: devem ser entendidos como **ferramenta auxiliar de triagem**.

**Trabalhos futuros por impacto esperado:**

| Impacto | Ação |
|---|---|
| Alto | Aumentar o dataset (mais amostras, especialmente positivas) |
| Alto | Busca de hiperparâmetros com GridSearch/RandomSearch |
| Alto | Feature selection para reduzir ruído |
| Médio | Testar Random Forest, LightGBM, redes neurais com mais dados |
| Médio | Curvas ROC e AUC para comparação visual |
| Médio | Ensemble/stacking combinando os melhores modelos |
| Baixo | Testes estatísticos entre modelos (ex: teste de McNemar) |
| Baixo | Curvas de aprendizado (learning curves) |
| Baixo | Validação externa com dataset independente |
