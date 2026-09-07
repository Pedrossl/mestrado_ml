# Resultados e plano de limpeza

## Leitura atual dos resultados

### GAD

Nos resultados principais, o XGBoost tem melhor equilibrio geral que o SVM no comparativo entre algoritmos:

- XGBoost: accuracy 85.0%, sensitivity 34.5%, specificity 94.3%, F1 37.5%, Kappa 0.307.
- SVM: accuracy 75.7%, sensitivity 22.5%, specificity 85.3%, F1 22.5%, Kappa 0.084.

O problema central nao e accuracy, e sim baixa sensibilidade. O modelo acerta muito bem a classe negativa, mas perde muitos casos positivos.

Na analise de threshold com XGBoost + SMOTE:

- Threshold default 0.50: sensitivity 34.1%, specificity 94.2%, Kappa 0.329.
- Threshold Youden 0.22: sensitivity 52.3%, specificity 86.8%, Kappa 0.355.
- Impacto: detecta 8 criancas a mais com GAD, ao custo de 18 falsos positivos adicionais.

Na busca focada em sensibilidade para GAD:

- Melhor sensibilidade: `M18_GridSearch_Recall`, sensitivity 73.5%, specificity 68.0%, Kappa 0.265.
- Melhor compromisso clinico/estatistico entre os metodos de alta sensibilidade parece ser `M16_FocalLoss`, com sensitivity 59.5%, specificity 81.5%, F1 45.2% e Kappa 0.326.
- `M20_HistGBM` tambem e forte para compromisso geral: sensitivity 50.5%, specificity 86.4%, F1 45.0%, Kappa 0.337.

### SAD

No XGBoost:

- Default: sensitivity 22.0%, specificity 94.6%, Kappa 0.213.
- Class weighting: sensitivity 35.0%, specificity 88.8%, Kappa 0.250.
- Undersampling: sensitivity 68.0%, specificity 66.9%, Kappa 0.239.
- SMOTE + threshold: sensitivity 68.0%, specificity 71.8%, F1 43.6%, Kappa 0.279.

Na analise de threshold com XGBoost + SMOTE:

- Threshold default 0.50: sensitivity 30.4%, specificity 92.5%, Kappa 0.262.
- Threshold Youden 0.12: sensitivity 60.9%, specificity 77.6%, Kappa 0.292.
- Threshold para sensitivity >= 70%: 0.06, com sensitivity 71.7% e specificity 66.0%.

SAD responde melhor que GAD ao ajuste de threshold, principalmente se o objetivo for triagem.

## Leitura das features

A correlacao de Spearman entre pares de features nao mostrou redundancia extrema. Os pares mais fortes ficam por volta de 0.50:

- ODD x Number of Impairments: 0.528.
- CD x ODD: 0.502.
- ODD x Frequency Irritable Mood: 0.501.
- Number of Impairments x Frequency Irritable Mood: 0.473.
- Frequency Temper Tantrums x Frequency Irritable Mood: 0.468.

Isso sugere blocos clinicos relacionados, nao duplicatas obvias.

Candidatas para teste de remocao por baixo sinal ou redundancia potencial:

- `Poverty Status`
- `Number of Siblings`
- `ADHD`
- `Number of Bio. Parents`
- `Frequency Temper Tantrums`, testando contra `Frequency Irritable Mood`

Features que merecem cuidado antes de remover:

- `Race`, `Sex`, `Poverty Status` e `Number of Bio. Parents`: podem ter implicacoes de vies/fairness.
- `Number of Physical Symptoms`: foi removida no preprocessamento atual, mas no dado bruto tem correlacao razoavel com GAD/SAD. A decisao de remover precisa ser justificada.
- `Age`: tem VIF moderado, mas aparece com importancia alta para GAD.
- `Number of Sleep Disturbances`: VIF moderado, mas e a feature mais forte para SAD.

Conclusao: a limpeza de features deve ser feita por ablation study, nao por corte manual baseado apenas em correlacao.

## Limpeza federal do codigo

### Etapa 1: limpeza segura

- Remover `__pycache__`, `.pyc` e `.DS_Store` do controle de versao.
- Manter outputs historicos por enquanto, mas impedir que novos outputs sejam adicionados sem intencao.
- Documentar dependencias e comandos de execucao.

### Etapa 2: definir scripts canonicos

Separar claramente:

- Scripts finais da dissertacao.
- Scripts exploratorios.
- Scripts historicos/obsoletos.

Sugestao de scripts canonicos:

- `scripts/analysis/eda.py`
- `scripts/analysis/curva_roc.py`
- `scripts/analysis/analise_threshold.py`
- `scripts/analysis/matriz_confusao_norm.py`
- `scripts/analysis/analise_erros.py`
- `scripts/evaluation/comparativo_algoritmos.py`
- `scripts/evaluation/teste_estatistico.py`
- `scripts/evaluation/learning_curves.py`
- `scripts/hyperparameters/gridsearch.py`
- `scripts/maximizar_sensibilidade_gad.py`

Scripts candidatos a arquivo historico ou revisao:

- `scripts/index.py`
- `scripts/Sergio.py`
- `scripts/ml_avaliacao.py`
- `scripts/hyperparameters/busca_hiperparametros.py`
- `scripts/hyperparameters/busca_hiperparametros_v3.py`
- `scripts/hyperparameters/busca_hiperparametros_fold_ajustado.py`
- `scripts/experimento_hard_samples/`
- `scripts/experimento_hard_samples_v2/`

### Etapa 3: padronizar execucao

Cada script final deve ter:

- `main()`.
- `argparse`.
- `--target GAD|SAD`.
- `--seed`.
- `--output-dir`.
- nenhuma execucao pesada ao importar o modulo.

### Etapa 4: centralizar configuracao

Criar uma configuracao compartilhada para:

- `RANDOM_STATE = 42`
- `N_SPLITS = 10`
- paths de `datasets/` e `output/`
- targets validos
- colunas removidas no preprocessamento
- features sensiveis ou de fairness

### Etapa 5: reduzir duplicacao

Extrair funcoes comuns para:

- construcao de folds;
- aplicacao de scaler, SMOTE, undersampling e class weighting;
- calculo de metricas;
- salvamento padronizado de tabelas;
- pipelines XGBoost/SVM.

### Etapa 6: limpeza de features com evidencia

Criar um script de ablation study que rode:

- baseline com todas as features;
- remocao uma-a-uma;
- remocao por blocos correlacionados;
- comparacao por sensitivity, specificity, F1, Kappa e AUC.

So depois disso decidir a lista final de features.
