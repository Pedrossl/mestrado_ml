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

Os outputs de SAD foram removidos desta branch porque a dissertacao atual trata SAD como trabalho futuro. O codigo canonico ainda permite regenerar esses resultados quando eles entrarem no artigo ou em uma nova versao do texto.

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
- `Number of Physical Symptoms`: foi removida no preprocessamento atual, mas no dado bruto tem correlacao razoavel com GAD. A decisao de remover precisa ser justificada.
- `Age`: tem VIF moderado, mas aparece com importancia alta para GAD.
- `Number of Sleep Disturbances`: VIF moderado, mas tem sinal clinico relevante e nao deve ser removida apenas por VIF.

Conclusao: a limpeza de features deve ser feita por ablation study, nao por corte manual baseado apenas em correlacao.

## Ablation inicial de features

A primeira limpeza ativa esta registrada em `scripts/config.py` (`FEATURE_DROP_COLUMNS`) e remove:

- `Poverty Status`
- `Number of Siblings`
- `Family History - Psychiatric Diagnosis`
- `Number of Bio. Parents`

Comparativo gerado em `output/feature_ablation/comparativo_ablation_gad.txt`:

| Cenario | Amostras | Features | Sensibilidade | Especificidade | F1 | Kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 287 | 17 | 36.00 ± 13.99 | 92.18 ± 4.52 | 39.24 ± 12.41 | 0.3042 ± 0.1370 |
| Sem 4 features, mesmas linhas | 287 | 13 | 39.00 ± 11.90 | 93.40 ± 3.73 | 43.67 ± 9.29 | 0.3580 ± 0.1049 |
| Sem 4 features, mais amostras | 306 | 13 | 37.00 ± 17.54 | 92.69 ± 3.03 | 39.09 ± 16.12 | 0.3085 ± 0.1718 |

Leitura: na comparacao mais controlada, mantendo as mesmas 287 linhas do baseline, a limpeza inicial melhora sensibilidade, especificidade, F1 e Kappa. No uso pratico com 306 amostras validas, o resultado fica proximo do baseline; como os intervalos de confianca se sobrepoem, a conclusao conservadora e que a limpeza nao piora o modelo e deixa o conjunto mais simples.

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

Scripts removidos nesta limpeza por serem exploratorios, historicos ou substituidos por fluxos canonicos:

- `scripts/index.py`
- `scripts/Sergio.py`
- `scripts/ml_avaliacao.py`
- `scripts/compare_columns.py`
- `scripts/visualization/`
- `scripts/hyperparameters/busca_hiperparametros.py`
- `scripts/hyperparameters/busca_hiperparametros_v3.py`
- `scripts/hyperparameters/busca_hiperparametros_fold_ajustado.py`
- `scripts/hyperparameters/busca_rf.py`

Scripts de Monte Carlo preservados:

- `scripts/experimento_hard_samples/`: fluxo v1 ainda em uso.
- `scripts/experimento_hard_samples_v2/monte_carlo_corrigido.py`
- `scripts/experimento_hard_samples_v2/convergencia_monte_carlo.py`

O v1 foi restaurado apos a limpeza porque ainda faz parte do fluxo atual de analise. O comparativo antes/depois da limpeza de features esta em `output/experimento_hard_samples/comparativo_pre_pos_feature_cleanup_v1.txt`.

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

O script `scripts/analysis/feature_ablation.py` compara o baseline completo com a limpeza ativa. Proximas extensoes:

- remocao uma-a-uma;
- remocao por blocos correlacionados;
- comparacao por sensitivity, specificity, F1, Kappa e AUC.

So depois disso decidir se a lista final de features fica com 13 variaveis ou se novos blocos entram na limpeza.
