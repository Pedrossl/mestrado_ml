import pandas as pd


def normalizar_teste(df):
    """
    Normaliza o dataset de teste aplicando transformações e removendo colunas.

    Transformações:
    - Number of Siblings: Converte para binário (0 = sem irmãos, 1 = tem irmãos)

    Remoções:
    - Depression, Number of Type A Stressors, Number of Physical Symptoms,
      Family History - Substance Abuse

    Args:
        df: DataFrame com os dados de teste

    Returns:
        DataFrame normalizado
    """
    df_norm = df.copy()

    # Transformar Number of Siblings em binário
    if 'Number of Siblings' in df_norm.columns:
        df_norm['Number of Siblings'] = (df_norm['Number of Siblings'] > 0).astype(int)

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


def carregar_teste_normalizado(caminho='datasets/mestrado-teste.csv'):
    """
    Carrega e normaliza o dataset de teste.

    Args:
        caminho: Caminho para o arquivo CSV

    Returns:
        DataFrame normalizado
    """
    df = pd.read_csv(caminho)
    return normalizar_teste(df)
