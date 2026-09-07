# Manifesto de resultados oficiais

Este documento registra quais resultados sustentam a versao atual da dissertacao e quais pontos ainda precisam de reconciliacao antes da limpeza pesada do codigo.

PDF analisado: `Dissertação___Nova.pdf`, 49 paginas, gerado em 2026-09-07.

## Escopo atual da dissertacao

A dissertacao esta centrada em GAD. O SAD aparece como transtorno correlato e como trabalho futuro. Os outputs de SAD que existiam no repositorio foram removidos desta branch por nao sustentarem a versao atual da dissertacao.

Decisao recomendada para organizacao:

- `GAD`: pipeline oficial da dissertacao atual.
- `SAD`: regenerar futuramente a partir dos scripts canonicos, se entrar no artigo ou em nova versao da dissertacao.
- resultados exploratorios: remover do caminho atual quando nao sustentarem tabelas, figuras ou auditoria metodologica.

## Base e preprocessamento

Resultado declarado no PDF:

- Base original: 307 registros e 27 variaveis.
- Missing values: 19 em `Poverty Status` e 1 em `Social Phobia`.
- Remocao por listwise deletion: 20 pacientes.
- Base final: 287 observacoes.
- GAD final: 44 positivos e 243 negativos.
- Matriz analitica: 17 preditores + 1 alvo.

Fonte no repositorio:

- `datasets/mestrado-teste.csv`
- `scripts/preprocessing/normalizacao.py`
- `scripts/utils.py`

Ponto a reconciliar:

- O PDF afirma que o MinMax e ajustado apenas nos dados de treino de cada fold.
- O pipeline compartilhado atual normaliza o dataset antes da validacao cruzada via `carregar_teste_normalizado()`.
- Para a versao final, o codigo canonico deve aplicar MinMax dentro de cada fold, preferencialmente via pipeline, ou o texto metodologico deve ser corrigido.

## Resultados GAD usados no texto

### Comparativo principal de modelos

Tabela no PDF: Tabela 4.6.

Resultados:

| Modelo | Tecnica | Accuracy | Sensitivity | F1 | Kappa |
| --- | --- | ---: | ---: | ---: | ---: |
| XGBoost | Sem balanceamento | 85.7% | 24.0% | 33.5% | 0.270 |
| XGBoost | Class Weighting | 83.6% | 38.0% | 41.1% | 0.319 |
| XGBoost | SMOTE | 83.6% | 36.0% | 39.2% | 0.304 |
| XGBoost | Undersampling | 64.9% | 67.0% | 37.4% | 0.195 |
| SVM | Sem balanceamento | 84.7% | 0.0% | 0.0% | 0.000 |
| SVM | Class Weighting | 77.0% | 39.0% | 33.9% | 0.210 |
| SVM | SMOTE | 75.7% | 22.5% | 22.5% | 0.084 |
| SVM | Undersampling | 66.6% | 65.0% | 36.3% | 0.189 |
| ADTree | Sem balanceamento | 82.3% | 11.5% | 13.8% | 0.081 |
| ADTree | Class Weighting | 64.1% | 59.0% | 33.9% | 0.153 |
| ADTree | SMOTE | 58.6% | 73.0% | 35.4% | 0.157 |
| ADTree | Undersampling | 65.9% | 56.0% | 34.4% | 0.166 |

Fontes provaveis:

- `output/plots/XGBoost/GAD/comparativo_gad.txt`
- `output/plots/SVM/GAD/comparativo_gad.txt`
- `output/plots/ADtree/GAD/comparativo_gad.txt`
- `output/plots/Comparativo/GAD/comparativo_algoritmos_gad.txt`

Status: coerente para XGBoost e SVM. ADTree precisa ser mantido como resultado oficial mesmo rodando fora do Python/Weka.

### Threshold XGBoost + SMOTE

Tabela no PDF: Tabela 4.5.

Resultados:

| Criterio | Threshold | Sensitivity | Specificity | F1 | Kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Padrao | 0.50 | 34.1% | 94.2% | 41.1% | 0.329 |
| Youden | 0.22 | 52.3% | 86.8% | 46.5% | 0.355 |
| F1 maximo | 0.41 | 43.2% | 93.0% | 47.5% | 0.391 |

Fonte:

- `output/plots/AnaliseThreshold/GAD/analise_threshold_gad.txt`
- script: `scripts/analysis/analise_threshold.py`

Status: coerente e importante.

### Trade-off de estrategias XGBoost

Tabela no PDF: Tabela 4.8.

Resultados declarados:

| Estrategia | Accuracy | Sensitivity | F1 | Specificity | Kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| BorderlineSMOTE | 84.3% | 34.5% | 39.3% | 93.4% | 0.311 |
| Focal Loss | 78.1% | 59.5% | 45.2% | 81.5% | 0.326 |
| GridSearch Recall | 68.7% | 73.5% | 42.6% | 68.0% | 0.265 |
| EasyEnsemble | 66.2% | 68.5% | 38.9% | 65.9% | 0.217 |

Fontes:

- `output/sensibilidade_gad/comparativo_sensibilidade_gad.txt`
- `output/sensibilidade_gad/resultados_individuais/M03_BorderlineSMOTE.txt`
- `output/sensibilidade_gad/resultados_individuais/M16_FocalLoss.txt`
- `scripts/maximizar_sensibilidade_gad.py`

Status: parcialmente coerente.

Ponto critico:

- O resumo e a conclusao do PDF afirmam `Kappa = 0.372` para BorderlineSMOTE.
- O arquivo individual mais atual de BorderlineSMOTE registra `Kappa = 0.3105`.
- O arquivo `output/sensibilidade_gad/INTERPRETACAO.txt` e `output/historico_evolucao.txt` mencionam uma rodada anterior com `Kappa = 0.372`.
- Antes da versao final, escolher qual rodada e oficial e atualizar PDF/codigo/outputs para concordarem.

Recomendacao:

- Tratar `0.3105` como resultado reprodutivel atual, pois aparece no arquivo individual com seed, commit e timestamp.
- Se o `0.372` for mantido, recuperar exatamente o script/commit que gerou esse resultado.

### ROC/AUC

Figura no PDF: Figura 4.2.

Resultados declarados:

- XGBoost + BorderlineSMOTE: AUC 0.763 ± 0.087.
- XGBoost + SMOTE: AUC 0.751 ± 0.103.
- SVM + SMOTE: AUC 0.678 ± 0.069.

Fontes:

- `scripts/analysis/curva_roc.py`
- `scripts/analysis/curva_roc_gad_ic95.py`
- `output/plots/Comparativo/GAD/roc_comparativo_gad_ic95.png`
- `output/sensibilidade_gad/resultados_individuais/M03_BorderlineSMOTE.txt`

Status: manter como resultado oficial, mas verificar a divergencia entre `curva_roc_gad_ic95.py`, que comenta XGBoost + SMOTE como 0.722, e o PDF, que reporta 0.751.

### Monte Carlo / hard samples

Tabelas no PDF: Tabelas 4.9 e 4.10.

Resultados principais:

- Com vazamento: Kappa 0.81, sensitivity 84.30%.
- Apos correcao: Kappa 0.03, sensitivity 9.13%.
- Sem smoothing: Kappa 0.0264 ± 0.1233.
- Com smoothing: Kappa 0.0478 ± 0.1279.

Fontes:

- `output/experimento_hard_samples/monte_carlo_resultado_pre_feature_cleanup.txt`
- `output/experimento_hard_samples/monte_carlo_resultado.txt`
- `output/experimento_hard_samples/comparativo_pre_pos_feature_cleanup_v1.txt`
- `scripts/experimento_hard_samples/`
- `output/experimento_hard_samples_v2/monte_carlo_corrigido_resultado.txt`
- `output/experimento_hard_samples_v2/convergencia_monte_carlo.txt`
- `scripts/experimento_hard_samples_v2/monte_carlo_corrigido.py`
- `scripts/experimento_hard_samples_v2/convergencia_monte_carlo.py`
- depende dos arrays salvos em `output/experimento_hard_samples/`.

Status: importante e deve ser preservado.

Ponto a reconciliar:

- O comentario inicial de `monte_carlo_corrigido.py` diz que avalia no teste completo sem hard samples, mas o PDF descreve avaliacao sobre 38 amostras honestas + 5 hard samples remanescentes por iteracao.
- O codigo parece seguir o protocolo do PDF dentro de `rodar_monte_carlo()`, mas o comentario de topo deve ser corrigido.

### Feature importance

Tabela no PDF: Tabela 4.11.

Resultados declarados:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | Number of Impairments | 0.163 |
| 2 | Race | 0.112 |
| 3 | ODD | 0.083 |
| 4 | Social Phobia | 0.080 |
| 5 | ADHD | 0.078 |

Fonte atual no repositorio:

- `output/plots/XGBoost/GAD/feature_importance_gad.txt`
- script: `scripts/models/modelo_xgboost.py`

Ponto critico:

- O output atual de `feature_importance_gad.txt` mostra outra ordenacao:
  - Number of Impairments 0.1676
  - Age 0.0831
  - ADHD 0.0750
  - Race 0.0748
  - Frequency Irritable Mood 0.0676
- A tabela do PDF parece vir de outra rodada ou outro metodo de importancia.
- A secao 4.3 do PDF contem `Figura ??`, indicando referencia LaTeX quebrada.

Recomendacao:

- Regerar feature importance oficial ou localizar o output que sustenta a Tabela 4.11.
- Depois atualizar o PDF ou congelar o script que reproduz a tabela atual.

## Resultados SAD

Os outputs de SAD foram removidos desta branch porque o PDF afirma que SAD ainda nao foi tratado e aparece como trabalho futuro.

Status recomendado:

- Regenerar SAD futuramente a partir dos scripts canonicos, quando a dissertacao ou artigo decidir incluir esses resultados.
- Nao misturar SAD com o pipeline oficial de GAD ate existir decisao textual clara.

## Arquivos canonicos a preservar

Codigo:

- `scripts/config.py`
- `scripts/utils.py`
- `scripts/preprocessing/minmax.py`
- `scripts/preprocessing/normalizacao.py`
- `scripts/analysis/eda.py`
- `scripts/analysis/correlation.py`
- `scripts/analysis/curva_roc.py`
- `scripts/analysis/curva_roc_gad_ic95.py`
- `scripts/analysis/analise_threshold.py`
- `scripts/analysis/matriz_confusao_norm.py`
- `scripts/analysis/analise_erros.py`
- `scripts/evaluation/comparativo_algoritmos.py`
- `scripts/evaluation/teste_estatistico.py`
- `scripts/evaluation/learning_curves.py`
- `scripts/models/modelo_xgboost.py`
- `scripts/models/modelo_svm.py`
- `scripts/models/modelo_adtree.py`
- `scripts/maximizar_sensibilidade_gad.py`
- `scripts/experimento_hard_samples_v2/monte_carlo_corrigido.py`
- `scripts/experimento_hard_samples_v2/convergencia_monte_carlo.py`

Resultados:

- `output/plots/XGBoost/GAD/`
- `output/plots/SVM/GAD/`
- `output/plots/ADtree/GAD/`
- `output/plots/Comparativo/GAD/`
- `output/plots/AnaliseThreshold/GAD/`
- `output/plots/AnaliseErros/GAD/`
- `output/plots/EDA/GAD/`
- `output/sensibilidade_gad/`
- `output/experimento_hard_samples/`
- `output/experimento_hard_samples_v2/`

## Arquivos removidos na limpeza

Scripts removidos por nao sustentarem diretamente a versao atual da dissertacao:

- `scripts/index.py`
- `scripts/Sergio.py`
- `scripts/ml_avaliacao.py`
- `scripts/compare_columns.py`
- `scripts/visualization/`
- `scripts/hyperparameters/busca_hiperparametros.py`
- `scripts/hyperparameters/busca_hiperparametros_v3.py`
- `scripts/hyperparameters/busca_hiperparametros_fold_ajustado.py`
- `scripts/hyperparameters/busca_rf.py`

Outputs removidos por serem historicos, exploratorios ou fora do escopo GAD atual:

- `output/busca_hiperparametros/`
- `output/busca_hiperparametros_fold_ajustado/`
- `output/busca_hiperparametros_v3/`
- `output/busca_rf/`
- `output/ml_avaliacao/`
- `output/plots/Scatter/`
- outputs SAD em `output/plots/*/SAD/`

## Antes de fechar a limpeza

Resolver obrigatoriamente:

1. Kappa oficial do BorderlineSMOTE: `0.372` ou `0.3105`.
2. Feature importance oficial da Tabela 4.11.
3. Protocolo real de MinMax dentro ou fora dos folds.
4. Figura quebrada `Figura ??`.
5. Status do SAD: futuro ou resultado complementar.
6. Decidir se a limpeza ativa de features (`Poverty Status`, `Number of Siblings`, `Family History - Psychiatric Diagnosis`, `Number of Bio. Parents`) entra na versao final ou fica apenas como experimento.
