# Feature Selection — GAD

## Resumo

O dataset original possui 17 features apos pre-processamento. Duas foram removidas
por redundancia ou ruido, resultando em **15 features** no modelo final.

| Feature removida                       | Motivo                        | Metodo de deteccao         | Impacto Monte Carlo v1       |
| -------------------------------------- | ----------------------------- | -------------------------- | ---------------------------- |
| CD                                     | Redundante com ODD            | Spearman correlation       | Kappa +0.028 (0.813 → 0.841)|
| Family History - Psychiatric Diagnosis | Ruido / importancia negativa  | Permutation Importance     | Kappa +0.085 (0.813 → 0.898)|

Efeito combinado (sem CD + sem Family History): **Kappa 0.898**, Sensibilidade 93.0%, F1 91.9%.


## Estrategia de selecao

A selecao de features seguiu uma abordagem em 3 etapas: filtro, ranking e validacao.

1. **Filtro por correlacao de Spearman**: calcula-se a correlacao monotonica entre
   cada feature e o alvo (GAD), e entre todos os pares de features. Pares com alta
   correlacao entre si (|rho| > 0.40) e onde uma das features tem correlacao mais
   fraca com GAD indicam redundancia — a mais fraca e candidata a remocao.

2. **Ranking por Permutation Importance**: para cada feature, embaralha-se seus
   valores e mede-se a queda no Kappa via CV 10-fold (XGBoost + SMOTE). Features
   com importancia negativa estao atrapalhando o modelo — sao candidatas prioritarias.

3. **Validacao por Monte Carlo v1**: cada candidata identificada nos passos 1 e 2
   e testada individualmente no Monte Carlo v1 (200 simulacoes). Somente features
   cuja remocao melhora o Monte Carlo sao aceitas. Isso evita decisoes baseadas
   apenas em CV, que nem sempre concordam com o Monte Carlo (exemplo: Bio Parents
   melhorava CV mas piorava Monte Carlo).

Essa estrategia combina um metodo estatistico classico (Spearman), um metodo
baseado no modelo (Permutation Importance) e uma validacao experimental robusta
(Monte Carlo), garantindo que cada remocao e defensavel por multiplas evidencias.


## Metodos utilizados para identificar candidatas

### 1. Correlacao de Spearman

Calcula a correlacao monotonica entre cada feature e o alvo (GAD), e entre pares de
features. Features com baixa correlacao com GAD e alta correlacao entre si indicam
redundancia.

Criterios:
- |rho| < 0.15 com GAD → fraca relevancia
- |rho| > 0.40 entre features → alta redundancia

Exemplo: CD (rho=0.24 com GAD) e ODD (rho=0.29 com GAD) tem rho=0.50 entre si.
ODD carrega mais sinal, entao CD foi removida.

### 2. Permutation Importance

Embaralha cada feature individualmente e mede a queda no Kappa (CV 10-fold,
XGBoost + SMOTE, 10 repeticoes por fold). Se embaralhar nao piora o score, a
feature nao contribui. Se o score melhora, a feature esta atrapalhando.

Resultado (ordenado por importancia):
```
  #    Feature                                   Importancia
  1    Frequency Irritable Mood                      -0.0184
  2    Sex                                           -0.0181
  3    Family History - Psychiatric Diagnosis        -0.0125  ← REMOVIDA
  4    Race                                          -0.0100
  5    Number of Siblings                            -0.0054
  6    Number of Type B Stressors                    -0.0037
  7    ODD                                           -0.0009
  8    ADHD                                          -0.0004
  9    Poverty Status                                +0.0019
  10   Number of Bio. Parents                        +0.0110
  11   Number of Sensory Sensitivities               +0.0121
  12   Social Phobia                                 +0.0169
  13   Number of Impairments                         +0.0183
  14   Number of Sleep Disturbances                  +0.0279
  15   Frequency Temper Tantrums                     +0.0765
  16   Age                                           +0.1098
```

### 3. Validacao por Monte Carlo v1

Cada candidata foi testada no Monte Carlo v1 (200 simulacoes, sorteio 15/20 hard
samples) para confirmar o impacto antes de ser removida. Somente features que
melhoraram o Monte Carlo foram aceitas.

Candidatas testadas que NAO foram removidas (pioraram Monte Carlo):
- Number of Bio. Parents: CV melhora, MC piora (Kappa 0.841 → 0.709)
- ADHD: MC piora (Kappa 0.841 → 0.814)
- Sex: MC piora (Kappa 0.813 → 0.692)
- Number of Siblings: MC piora levemente (Kappa 0.813 → 0.803)
- Poverty Status: MC piora (Kappa 0.302 → 0.239 no CV)
- Age: MC piora muito (Kappa -0.166 no CV)
- Frequency Temper Tantrums: MC piora (Kappa -0.050 no CV)


## Por que cada feature foi removida

### CD (Conduct Disorder)

- Correlacao com GAD: rho = 0.241 (moderada, mas mais fraca que ODD)
- Correlacao com ODD: rho = 0.502 (alta redundancia)
- Logica clinica: CD e ODD medem transtornos de comportamento disruptivo.
  Com ambas presentes, o XGBoost divide splits entre elas de forma instavel.
  Removendo CD (a mais fraca), o modelo usa ODD de forma consistente.
- CV 10-fold: Kappa +0.008
- Monte Carlo v1: Kappa +0.028 (0.813 → 0.841)

### Family History - Psychiatric Diagnosis

- Correlacao com GAD: rho = 0.122 (fraca, apenas significativa a p<0.05)
- Permutation Importance: -0.0125 (negativa — embaralhar MELHORA o score)
- Logica clinica: historico familiar e um fator de risco geral para transtornos
  psiquiatricos, mas nao e especifico para GAD. O modelo trata como ruido.
- Monte Carlo v1: Kappa +0.085 (0.813 → 0.898), Sensibilidade 84.3% → 93.0%


## Features mantidas (15)

```
  Sex
  Race
  Age
  Number of Bio. Parents
  Number of Siblings
  Poverty Status
  Social Phobia
  ADHD
  ODD
  Number of Impairments
  Number of Type B Stressors
  Frequency Temper Tantrums
  Frequency Irritable Mood
  Number of Sleep Disturbances
  Number of Sensory Sensitivities
```


## Como reproduzir

```bash
# Permutation Importance
python3 -m scripts.analysis.permutation_importance

# Correlacao de Spearman
python3 -m scripts.analysis.correlation

# Monte Carlo v1 comparativo completo
python3 -m scripts.analysis.mc_v1_comparativo_completo
```

## Configuracao

As features removidas estao definidas em `scripts/config.py` na variavel
`FEATURE_DROP_COLUMNS`. A funcao `preparar_dados()` aplica a remocao
automaticamente.
