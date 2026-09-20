# Tecnologias do Projeto

## Visão geral
Este projeto é focado em análise de dados tabulares para pesquisa (mestrado), com scripts em Python que fazem leitura de CSV, limpeza de dados, análise de faltantes, correlação estatística e geração de gráficos.

## Linguagem e execução
- Python 3 (scripts em `scripts/*.py`)
- Execução via linha de comando (CLI) com `python` e `argparse`

## Bibliotecas Python
### Manipulação de dados
- `pandas`
- `numpy`

### Estatística
- `scipy` (`scipy.stats`)

### Visualização
- `matplotlib` (incluindo `matplotlib.pyplot`)
- `seaborn`

### Bibliotecas padrão do Python (stdlib)
- `os`
- `sys`
- `argparse`
- `csv`
- `re`
- `difflib`
- `json`
- `collections` (`defaultdict`)

## Dados e formatos
- Entrada principal: arquivos `CSV` em `datasets/`
- Saídas geradas:
  - `CSV` em `output/`
  - `JSON` em `output/`
  - `PNG` (gráficos/heatmaps) em `output/plots/`

## Ferramentas e abordagem analítica
- Cálculo de correlação: Pearson, Spearman e Kendall
- Cálculo de p-values para correlação (`scipy.stats`)
- Heatmaps de correlação para análise exploratória
- Comparação e sugestão de correspondência de colunas entre bases

## Observações de stack
- Dependências externas listadas em `requirements.txt`.
- Não há indícios de framework web (como Flask/Django/FastAPI) neste estágio do projeto.
