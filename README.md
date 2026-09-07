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
  preprocessing/       # normalizacao e transformacoes dos dados
  models/              # ADTree, XGBoost e SVM
  evaluation/          # comparativos, learning curves e testes estatisticos
  analysis/            # ROC, threshold, EDA, matriz de confusao e erros
  hyperparameters/     # buscas de hiperparametros
  utils.py             # metricas, IC e funcoes compartilhadas

output/                # resultados ja gerados
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

Rodar avaliacao geral de combinacoes:

```bash
PYTHONPATH=. .venv/bin/python scripts/ml_avaliacao.py
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

## Observacoes metodologicas

- SMOTE e undersampling devem ser aplicados apenas no conjunto de treino em cada fold, para evitar vazamento de informacao.
- Scalers devem ser ajustados apenas no treino e aplicados no teste/validacao.
- Alguns scripts usam `random_state=42`; scripts sem semente fixa podem gerar resultados diferentes a cada execucao.
- A validacao final em holdout ainda aparece como pendente no `todo.md`. Antes de escrever os resultados finais, vale revisar se `mestrado-teste.csv` esta sendo usado como conjunto de experimentacao, holdout ou ambos.
- Experimentos com ADTree dependem de Weka/Java e podem precisar de configuracao adicional fora do `pip install`.

## Proximos passos recomendados

1. Consolidar um script de validacao holdout final.
2. Fixar sementes nos scripts que ainda usam splits/modelos sem `random_state`.
3. Documentar claramente o tratamento de missing values.
4. Decidir quais resultados de `output/` devem permanecer versionados como artefatos finais.
5. Remover do git, em um commit separado, caches como `__pycache__`, `.pyc` e `.DS_Store` que ja estao rastreados.
