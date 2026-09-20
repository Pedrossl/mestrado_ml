# Mestrado ML

Projeto de pesquisa em aprendizado de maquina para classificacao de GAD e SAD a partir de dados clinicos. O repositorio concentra scripts de preprocessamento, avaliacao de modelos, analises estatisticas e geracao de resultados para a dissertacao e apresentacao.

## Visao geral

O fluxo principal usa os datasets em `datasets/`, prepara os dados com as funcoes de `scripts/preprocessing/` e avalia modelos de classificacao em `scripts/models/`, `scripts/evaluation/`, `scripts/analysis/` e `scripts/hyperparameters/`. Esses modulos sao **genericos**: aceitam `target='GAD'` ou `target='SAD'` e nao precisam ser duplicados.

Trabalho especifico de um alvo (selecao de features, tuning de hiperparametros, experimentos de Monte Carlo) vive em `scripts/gad/` ou `scripts/sad/` — ver secao "Estrutura GAD vs SAD vs comum" abaixo. Ate agora a etapa de selecao de features e tuning do Monte Carlo v1 foi concluida **apenas para GAD**; `scripts/sad/`, `output/sad/`, `resultados/sad/` e `docs/sad/` sao esqueletos prontos para quando esse trabalho comecar (SAD tende a precisar de normalizacao e lista de features removidas diferentes).

Modelos e tecnicas ja implementados:

- ADTree, XGBoost e SVM.
- Baseline sem balanceamento, class weighting, SMOTE e undersampling.
- Metricas: accuracy, sensitivity, specificity, PPV, NPV, F1-score, Kappa, matriz de confusao, ROC/AUC e intervalos de confianca.
- Analises auxiliares: erros, threshold, learning curves, EDA, grid search e testes estatisticos.
- GAD: selecao de features (Spearman + Permutation Importance + validacao Monte Carlo) e tuning do Monte Carlo v1 — ver `docs/gad/FEATURE_SELECTION.md` e `resultados/gad/`.

## Estrutura GAD vs SAD vs comum

```text
datasets/                 # comum — dataset bruto, usado por GAD e SAD
  mestrado-treino.csv
  mestrado-teste.csv
  Planilha_mestrado.xlsx

scripts/
  config.py               # comum — paths, seeds, targets, colunas gerais
                           #   e FEATURE_DROP_COLUMNS_BY_TARGET (lista de
                           #   features a remover, uma entrada por alvo)
  utils.py                # comum — metricas, IC, preparar_dados()
  preprocessing/           # comum — normalizacao e transformacoes
  models/                  # comum — ADTree, XGBoost, SVM (recebem target)
  evaluation/               # comum — comparativos, learning curves, testes estatisticos
  hyperparameters/          # comum — grid search canonico
  analysis/                 # comum — ROC, threshold, EDA, matriz de confusao,
                             #   erros, correlacao, ablation (todos com target)
  gad/                       # especifico de GAD
    analysis/                #   selecao de features, tuning Monte Carlo v1,
                              #   permutation importance, plots de resultado
    experimento_hard_samples/    # Monte Carlo v1 em uso
    experimento_hard_samples_v2/ # Monte Carlo corrigido
  sad/                        # especifico de SAD (vazio ate comecar) — ver scripts/sad/README.md

output/
  plots/                    # comum — gerado pelos scripts genericos, ja se
                             #   separa por alvo internamente (ex: plots/XGBoost/GAD/)
  colunas_com_missing.csv   # comum — relatorio de missing values do dataset bruto
  gad/                      # especifico de GAD — ablation, feature_removal_runs,
                             #   sensibilidade_gad, experimento_hard_samples, etc.
  sad/                      # especifico de SAD (vazio ate comecar)

resultados/                 # resultados oficiais em destaque
  gad/                       # melhor resultado Monte Carlo v1 e relatorio completo de GAD
  sad/                       # vazio ate comecar

docs/
  dataset/                  # comum — descricao/mapeamento das variaveis do dataset bruto
  melhores_resultados.txt   # comum — cobre GAD e SAD
  relatorio_preprocessamento_resultados.md  # comum — cobre GAD e SAD
  slides/apresentacao_1/    # comum — deck antigo, mistura GAD e SAD
  slides/*.txt              # comum — notas de apresentacao
  gad/                       # especifico de GAD
    FEATURE_SELECTION.md     #   estrategia de selecao de features de GAD
    manifesto_resultados_oficiais.md
    resultados_e_limpeza.md
    comparativo_resultados.txt
    dissertacao/              # texto atual da dissertacao (GAD apenas)
    slides/apresentacao_2/    # deck atual, 100% GAD
  sad/                       # vazio ate comecar — ver docs/sad/README.md
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

### Genericos (funcionam para GAD ou SAD)

Gerar comparativo entre algoritmos:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluation/comparativo_algoritmos.py
```

Gerar curvas ROC/AUC:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.analysis.curva_roc
```

Rodar testes estatisticos:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluation/teste_estatistico.py
```

Gerar matriz de confusao normalizada:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.analysis.matriz_confusao_norm
```

Rodar EDA:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.analysis.eda
```

Comparar o baseline completo com a limpeza atual de features (aceita `target="GAD"` ou `"SAD"`):

```bash
PYTHONPATH=. .venv/bin/python -m scripts.analysis.feature_ablation
```

Os snapshots por rodada ficam em `output/<target>/feature_removal_runs/`.

### Especificos de GAD

Rodar a busca de sensibilidade para GAD:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.gad.analysis.maximizar_sensibilidade_gad
```

Rodar Monte Carlo v1 dos hard samples (GAD):

```bash
PYTHONPATH=. .venv/bin/python -m scripts.gad.experimento_hard_samples.executar_tudo
```

Rodar Monte Carlo v2 dos hard samples (GAD):

```bash
PYTHONPATH=. .venv/bin/python -m scripts.gad.experimento_hard_samples_v2.monte_carlo_corrigido
```

Rodar o comparativo completo de features do Monte Carlo v1 (GAD):

```bash
PYTHONPATH=. .venv/bin/python -m scripts.gad.analysis.mc_v1_comparativo_completo
```

Consultar o manifesto de resultados oficiais de GAD:

```bash
less docs/gad/manifesto_resultados_oficiais.md
```

Consultar o melhor resultado oficial de GAD (Monte Carlo v1):

```bash
less resultados/gad/melhor_resultado_monte_carlo.txt
```

## Observacoes metodologicas

- SMOTE e undersampling devem ser aplicados apenas no conjunto de treino em cada fold, para evitar vazamento de informacao.
- Scalers devem ser ajustados apenas no treino e aplicados no teste/validacao.
- A limpeza ativa de features esta registrada em `scripts/config.py` (`FEATURE_DROP_COLUMNS_BY_TARGET`), com uma lista por alvo — GAD ja tem a lista final (ver `docs/gad/FEATURE_SELECTION.md`), SAD ainda esta vazia.
- Alguns scripts usam `random_state=42`; scripts sem semente fixa podem gerar resultados diferentes a cada execucao.
- A validacao final em holdout ainda aparece como pendente no `todo.md`. Antes de escrever os resultados finais, vale revisar se `mestrado-teste.csv` esta sendo usado como conjunto de experimentacao, holdout ou ambos.
- Experimentos com ADTree dependem de Weka/Java e podem precisar de configuracao adicional fora do `pip install`.
- O Monte Carlo v1 (GAD) tem vazamento de dados conhecido e aceito (avalia parte dos hard samples que tambem entraram no retreino) — ver secao 0 de `resultados/gad/relatorio_completo_experimentos.md` antes de usar esse protocolo para SAD.

## Proximos passos recomendados

1. Reconciliar os resultados oficiais listados em `docs/gad/manifesto_resultados_oficiais.md`.
2. Corrigir a divergencia entre o Kappa 0.372 e 0.3105 do BorderlineSMOTE.
3. Regerar ou localizar a feature importance usada na Tabela 4.11 da dissertacao.
4. Padronizar a normalizacao MinMax dentro dos folds, se essa for a metodologia final.
5. Iniciar a selecao de features para SAD (Spearman + Permutation Importance + validacao Monte Carlo), preenchendo `scripts/sad/` e `FEATURE_DROP_COLUMNS_BY_TARGET["SAD"]` seguindo o padrao documentado em `docs/gad/FEATURE_SELECTION.md`.
