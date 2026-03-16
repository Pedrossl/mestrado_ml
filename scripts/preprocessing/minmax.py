import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Colunas numéricas do dataset de TESTE que devem ser normalizadas com MinMax
# Nota: Number of Bio. Parents e Number of Siblings têm tratamento customizado em normalizacao.py
COLUNAS_NUMERICAS = [
    'Age',
    'Number of Impairments',
    'Number of Type A Stressors',
    'Number of Type B Stressors',
    'Frequency Temper Tantrums',
    'Frequency Irritable Mood',
    'Number of Sleep Disturbances',
    'Number of Physical Symptoms',
    'Number of Sensory Sensitivities',
]


def normalizar_minmax(df, scaler=None, colunas=None):
    """
    Aplica normalização MinMax nas colunas numéricas.

    Args:
        df: DataFrame com os dados
        scaler: MinMaxScaler já ajustado (se None, cria um novo)
        colunas: Lista de colunas a normalizar (default: COLUNAS_NUMERICAS)

    Returns:
        tuple: (DataFrame normalizado, MinMaxScaler)
    """
    if colunas is None:
        colunas = COLUNAS_NUMERICAS

    df_norm = df.copy()

    if scaler is None:
        scaler = MinMaxScaler()
        fit = True
    else:
        fit = False

    # Filtrar apenas colunas que existem no DataFrame
    colunas_existentes = [col for col in colunas if col in df_norm.columns]

    if colunas_existentes:
        # Converter para numérico
        for col in colunas_existentes:
            df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')

        if fit:
            df_norm[colunas_existentes] = scaler.fit_transform(df_norm[colunas_existentes])
        else:
            df_norm[colunas_existentes] = scaler.transform(df_norm[colunas_existentes])

    return df_norm, scaler


def carregar_teste_com_minmax(caminho='datasets/mestrado-teste.csv'):
    """
    Carrega e normaliza o dataset de teste com MinMax.

    Args:
        caminho: Caminho para o arquivo CSV

    Returns:
        tuple: (DataFrame normalizado, MinMaxScaler ajustado)
    """
    df = pd.read_csv(caminho)
    return normalizar_minmax(df)
