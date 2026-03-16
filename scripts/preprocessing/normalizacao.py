import pandas as pd
from scripts.preprocessing.minmax import normalizar_minmax, COLUNAS_NUMERICAS


def normalizar_teste(df, aplicar_minmax=True):
    """
    Normaliza o dataset de teste aplicando transformações e removendo colunas.

    Transformações:
    - MinMax nas colunas numéricas (Age, Number of Impairments, etc.)
    - Number of Siblings: Converte para binário (0 = sem irmãos, 1 = tem irmãos)
    - Number of Bio. Parents: Mapeia para 0, 0.5, 1 (0, 1, 2 pais)

    Remoções:
    - Depression, Number of Type A Stressors, Number of Physical Symptoms,
      Family History - Substance Abuse

    Args:
        df: DataFrame com os dados de teste
        aplicar_minmax: Se True, aplica MinMax nas colunas numéricas

    Returns:
        DataFrame normalizado
    """
    df_norm = df.copy()

    # Aplicar MinMax nas colunas numéricas
    if aplicar_minmax:
        df_norm, _ = normalizar_minmax(df_norm)

    # Transformar Number of Siblings em binário
    if 'Number of Siblings' in df_norm.columns:
        df_norm['Number of Siblings'] = (df_norm['Number of Siblings'] > 0).astype(int)

    # Transformar Number of Bio. Parents: 0→0, 1→0.5, 2→1
    if 'Number of Bio. Parents' in df_norm.columns:
        df_norm['Number of Bio. Parents'] = df_norm['Number of Bio. Parents'] / 2.0

    # Remover colunas não utilizadas
    colunas_remover = [
        'Depression',
        'Number of Type A Stressors',
        'Number of Physical Symptoms',
        'Family History - Substance Abuse'
    ]

    for col in colunas_remover:
        if col in df_norm.columns:
            df_norm = df_norm.drop(col, axis=1)

    return df_norm


def carregar_teste_normalizado(caminho='datasets/mestrado-teste.csv', aplicar_minmax=True):
    """
    Carrega e normaliza o dataset de teste.

    Args:
        caminho: Caminho para o arquivo CSV
        aplicar_minmax: Se True, aplica MinMax nas colunas numéricas

    Returns:
        DataFrame normalizado
    """
    df = pd.read_csv(caminho)
    return normalizar_teste(df, aplicar_minmax=aplicar_minmax)
