# Mestrado ML

Projeto de pesquisa em aprendizado de maquina para classificacao de GAD e SAD a partir de dados clinicos. O repositorio concentra scripts de preprocessamento, avaliacao de modelos, analises estatisticas e geracao de resultados para a dissertacao e apresentacao.

## Visao geral

O fluxo principal usa os datasets em `datasets/`, prepara os dados com as funcoes de `scripts/preprocessing/` e avalia modelos de classificacao em `scripts/models/`, `scripts/evaluation/`, `scripts/analysis/` e `scripts/hyperparameters/`.

Modelos e tecnicas ja implementados:

- ADTree, XGBoost e SVM.
- Baseline sem balanceamento, class weighting, SMOTE e undersampling.
- Metricas: accuracy, sensitivity, specificity, PPV, NPV, F1-score, Kappa, matriz de confusao, ROC/AUC e intervalos de confianca.
- Analises auxiliares: erros, threshold, learning curves, EDA, grid search e testes estatisticos.

## Estrutura

```text
datasets/
  mestrado-treino.csv
  mestrado-teste.csv
  Planilha_mestrado.xlsx

scripts/
  config.py             # paths, seeds, targets e listas de colunas
  utils.py              # metricas, IC e funcoes compartilhadas
  preprocessing/       # normalizacao e transformacoes dos dados
  models/              # ADTree, XGBoost e SVM
  evaluation/          # comparativos, learning curves e testes estatisticos
  analysis/            # ROC, threshold, EDA, matriz de confusao e erros
  hyperparameters/     # grid search canonico
  experimento_hard_samples_v2/ # Monte Carlo corrigido

docs/                  # manifestos e notas de organizacao
output/                # resultados oficiais preservados
SLIDE/ e slide2/       # materiais para apresentacao
```

## Ambiente

Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Os scripts usam imports do pacote local `scripts`, entao execute os comandos a partir da raiz do projeto com `PYTHONPATH=.`.

## Comandos uteis

Rodar a busca de sensibilidade para GAD:

```bash
PYTHONPATH=. .venv/bin/python scripts/maximizar_sensibilidade_gad.py
```

Gerar comparativo entre algoritmos:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluation/comparativo_algoritmos.py
```

Gerar curvas ROC/AUC:

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/curva_roc.py
```

Rodar testes estatisticos:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluation/teste_estatistico.py
```

Gerar matriz de confusao normalizada:

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/matriz_confusao_norm.py
```

Rodar EDA:

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/eda.py
```

Rodar Monte Carlo corrigido dos hard samples:

```bash
PYTHONPATH=. .venv/bin/python scripts/experimento_hard_samples_v2/monte_carlo_corrigido.py
```

Consultar o manifesto de resultados oficiais:

```bash
less docs/manifesto_resultados_oficiais.md
```

Comparar o baseline completo com a limpeza atual de features:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.analysis.feature_ablation
```

## Observacoes metodologicas

- SMOTE e undersampling devem ser aplicados apenas no conjunto de treino em cada fold, para evitar vazamento de informacao.
- Scalers devem ser ajustados apenas no treino e aplicados no teste/validacao.
- A limpeza ativa de features esta registrada em `scripts/config.py` (`FEATURE_DROP_COLUMNS`).
- Alguns scripts usam `random_state=42`; scripts sem semente fixa podem gerar resultados diferentes a cada execucao.
- A validacao final em holdout ainda aparece como pendente no `todo.md`. Antes de escrever os resultados finais, vale revisar se `mestrado-teste.csv` esta sendo usado como conjunto de experimentacao, holdout ou ambos.
- Experimentos com ADTree dependem de Weka/Java e podem precisar de configuracao adicional fora do `pip install`.

## Proximos passos recomendados

1. Reconciliar os resultados oficiais listados em `docs/manifesto_resultados_oficiais.md`.
2. Corrigir a divergencia entre o Kappa 0.372 e 0.3105 do BorderlineSMOTE.
3. Regerar ou localizar a feature importance usada na Tabela 4.11 da dissertacao.
4. Padronizar a normalizacao MinMax dentro dos folds, se essa for a metodologia final.
5. Repetir a ablation com remocao uma-a-uma e blocos correlacionados adicionais.
