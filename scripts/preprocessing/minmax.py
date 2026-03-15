import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Colunas numéricas que devem ser normalizadas com MinMax
COLUNAS_NUMERICAS = [
    'Onset Fear about possible harm befalling major attachment figures',
    'Onset Fear about calamitous separation',
    'Onset Avoidance of being alone',
    'Onset Anticipatory distress/resistance to separation',
    'Onset Withdrawal when attachement figure absent',
    'Onset Actual distress when attachment figure absent',
    'Onset Complaints of physical symptoms when separation from major attachment figure is anticipated',
    "Onset Parent's plan disrupted due to child's distress at separation",
    'Onset Complaints of physical symptoms when attendance at school/daycare is anticipated or occurs',
    'Onset Fear/Anxiety about leaving home for daycare/school',
    'Onset Anticipatory fear of daycare/school',
    'Onset Daycare/school non-attendance due to anxiety',
    'Onset Picked up early from daycare/school due to anxiety',
    'Frequency of reluctance to go to sleep',
    'Onset Frequency of reluctance to go to sleep',
    'Frequency of sleeping with family member due to a reluctance to sleep alone',
    'Onset Frequency of sleeping with family member due to a reluctance to sleep alone',
    'Sleep resistence',
    'Hours taken to fall asleep',
    'Freqency of nights child wakes up during the night',
    'How long awak per night',
    'Onset Rising at night to check on family members',
    'Onset Separation dreams',
    'Anxious affect that occurs in certain situations/environments',
    'Anxiety not associated with any particular situation',
    'Frequency of worries',
    'Sampling Weight',
]


def normalizar_minmax_treino(df, colunas=None):
    """
    Aplica normalização MinMax nas colunas numéricas do dataset de treino.

    Args:
        df: DataFrame com os dados de treino
        colunas: Lista de colunas a normalizar (default: COLUNAS_NUMERICAS)

    Returns:
        tuple: (DataFrame normalizado, MinMaxScaler ajustado)
    """
    if colunas is None:
        colunas = COLUNAS_NUMERICAS

    df_norm = df.copy()
    scaler = MinMaxScaler()

    # Filtrar apenas colunas que existem no DataFrame
    colunas_existentes = [col for col in colunas if col in df_norm.columns]

    if colunas_existentes:
        # Substituir valores não numéricos por NaN e converter para float
        for col in colunas_existentes:
            df_norm[col] = pd.to_numeric(
                df_norm[col].astype(str).str.replace(',', '.'),
                errors='coerce'
            )

        # Ajustar e transformar
        df_norm[colunas_existentes] = scaler.fit_transform(df_norm[colunas_existentes])

    return df_norm, scaler


def normalizar_minmax_teste(df, scaler, colunas=None):
    """
    Aplica normalização MinMax nas colunas numéricas do dataset de teste
    usando um scaler já ajustado no treino.

    Args:
        df: DataFrame com os dados de teste
        scaler: MinMaxScaler já ajustado no dataset de treino
        colunas: Lista de colunas a normalizar (default: COLUNAS_NUMERICAS)

    Returns:
        DataFrame normalizado
    """
    if colunas is None:
        colunas = COLUNAS_NUMERICAS

    df_norm = df.copy()

    # Filtrar apenas colunas que existem no DataFrame
    colunas_existentes = [col for col in colunas if col in df_norm.columns]

    if colunas_existentes:
        # Substituir valores não numéricos por NaN e converter para float
        for col in colunas_existentes:
            df_norm[col] = pd.to_numeric(
                df_norm[col].astype(str).str.replace(',', '.'),
                errors='coerce'
            )

        # Transformar usando o scaler do treino
        df_norm[colunas_existentes] = scaler.transform(df_norm[colunas_existentes])

    return df_norm


def carregar_treino_normalizado(caminho='datasets/mestrado-treino.csv'):
    """
    Carrega e normaliza o dataset de treino com MinMax.

    Args:
        caminho: Caminho para o arquivo CSV

    Returns:
        tuple: (DataFrame normalizado, MinMaxScaler ajustado)
    """
    df = pd.read_csv(caminho)
    return normalizar_minmax_treino(df)
