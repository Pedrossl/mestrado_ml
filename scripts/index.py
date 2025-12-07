# IMPORTS - Bibliotecas necessárias
import pandas as pd  # Leitura e manipulação de dados em tabelas
import numpy as np   # Operações matemáticas e arrays
import matplotlib.pyplot as plt  # Para criar gráficos
import seaborn as sns  # Para visualizações estatísticas

# ===== CARREGAR OS DADOS =====
dados_treino = pd.read_csv('datasets/mestrado-treino.csv')  # PAS - 917 crianças
dados_teste = pd.read_csv('datasets/mestrado-teste.csv')    # PTRTS - 307 crianças

print("DATASETS CARREGADOS COM SUCESSO!")
print(f"Treino: {dados_treino.shape[0]} linhas x {dados_treino.shape[1]} colunas")
print(f"Teste: {dados_teste.shape[0]} linhas x {dados_teste.shape[1]} colunas\n")

# ===== VERIFICAR DADOS FALTANTES =====
def verificar_faltantes(df, nome):
    """Verifica quantidade de dados faltantes (NaN) em cada coluna"""
    print(f"\n{'=' * 80}\nDADOS FALTANTES: {nome}\n{'=' * 80}\n")

    faltantes = df.isnull().sum()  # Conta NaN por coluna
    total_linhas = len(df)

    # Filtrar apenas colunas com dados faltantes
    colunas_com_faltantes = faltantes[faltantes > 0]

    if len(colunas_com_faltantes) == 0:
        print("✅ Nenhum dado faltante encontrado!\n")
    else:
        print(f"⚠️  Encontrados dados faltantes em {len(colunas_com_faltantes)} colunas:\n")
        for col, qtd in colunas_com_faltantes.items():
            percentual = (qtd / total_linhas) * 100
            print(f"  • {col}: {qtd} faltantes ({percentual:.2f}%)")

        # Resumo
        total_faltantes = colunas_com_faltantes.sum()
        total_celulas = total_linhas * len(df.columns)
        perc_geral = (total_faltantes / total_celulas) * 100
        print(f"\n  📊 Total de valores faltantes: {total_faltantes} ({perc_geral:.2f}% do dataset)")
    print()

# Verificar faltantes em ambos datasets
verificar_faltantes(dados_treino, "TREINO (PAS)")
verificar_faltantes(dados_teste, "TESTE (PTRTS)")

# ===== FUNÇÃO: MAPEAR TIPOS DE VARIÁVEIS =====
def mapear_tipos_variaveis(df, nome):
    """Classifica cada coluna em: int_bool (0/1), int, float ou string"""

    # Arrays para armazenar nomes por tipo
    cols_bool = []    # Apenas 0 e 1
    cols_int = []     # Inteiros
    cols_float = []   # Decimais
    cols_string = []  # Texto

    for col_nome in df.columns:
        col = df[col_nome]
        tipo = col.dtype

        # CASO 1: Object (pode ser string ou número)
        if tipo == 'object':
            try:
                col_num = pd.to_numeric(col, errors='coerce')
                if col_num.notna().any():
                    if (col_num.dropna() % 1 == 0).all():  # São inteiros
                        vals = col_num.dropna().unique()
                        if set(vals).issubset({0, 1}):  # Apenas 0 e 1
                            cols_bool.append(col_nome)
                        else:
                            cols_int.append(col_nome)
                    else:  # Tem decimais
                        cols_float.append(col_nome)
                else:  # Texto puro
                    cols_string.append(col_nome)
            except:
                cols_string.append(col_nome)

        # CASO 2: Int
        elif 'int' in str(tipo):
            vals = col.dropna().unique()
            if set(vals).issubset({0, 1}):  # Apenas 0 e 1
                cols_bool.append(col_nome)
            else:
                cols_int.append(col_nome)

        # CASO 3: Float
        elif 'float' in str(tipo):
            vals = col.dropna().unique()
            if set(vals).issubset({0.0, 1.0}):  # Apenas 0 e 1
                cols_bool.append(col_nome)
            elif (col.dropna() % 1 == 0).all():  # Inteiros disfarçados
                cols_int.append(col_nome)
            else:
                cols_float.append(col_nome)

        # CASO 4: Outros
        else:
            cols_string.append(col_nome)

    return {
        'int_bool': cols_bool,
        'int': cols_int,
        'float': cols_float,
        'string': cols_string
    }

# ===== EXECUTAR MAPEAMENTO =====
map_treino = mapear_tipos_variaveis(dados_treino, "TREINO (PAS)")
map_teste = mapear_tipos_variaveis(dados_teste, "TESTE (PTRTS)")

# ===== RESUMO ANTES DA NORMALIZAÇÃO =====
print("=" * 80)
print("RESUMO GERAL (ANTES DA NORMALIZAÇÃO)")
print("=" * 80)
print(f"\nTREINO:")
print(f"  • Booleanas: {len(map_treino['int_bool'])}")
print(f"  • Inteiras:  {len(map_treino['int'])}")
print(f"  • Decimais:  {len(map_treino['float'])}")
print(f"  • Texto:     {len(map_treino['string'])}")
print(f"  • TOTAL:     {sum(len(v) for v in map_treino.values())}")

print(f"\nTESTE:")
print(f"  • Booleanas: {len(map_teste['int_bool'])}")
print(f"  • Inteiras:  {len(map_teste['int'])}")
print(f"  • Decimais:  {len(map_teste['float'])}")
print(f"  • Texto:     {len(map_teste['string'])}")
print(f"  • TOTAL:     {sum(len(v) for v in map_teste.values())}")

# ==================== NORMALIZAÇÃO ====================

def normalizar_dados(df, nome_dataset):
    """
    Aplica transformações e remove colunas conforme análise preliminar:

    Transformações:
    - Number of Siblings: Converte para binário (0 = sem irmãos, 1 = tem irmãos)

    Remoções (variáveis não relevantes para o modelo):
    - Depression: Depressão em crianças precisa estudo mais aprofundado
    - Number of Type A Stressors: Não contém valor 0 (distribuição problemática)
    - Number of Physical Symptoms: Não será utilizado nesta análise
    - Family History - Substance Abuse: Não será utilizado nesta análise
    """

    print(f"\n{'=' * 80}\nNORMALIZANDO: {nome_dataset}\n{'=' * 80}\n")

    df_norm = df.copy()  # Criar cópia para não modificar original
    colunas_removidas = []
    colunas_transformadas = []

    # 1. TRANSFORMAR: Number of Siblings → Binário
    if 'Number of Siblings' in df_norm.columns:
        print("🔄 Transformando 'Number of Siblings' em binário...")
        valores_antes = df_norm['Number of Siblings'].value_counts().sort_index()
        print(f"   Valores antes: {dict(valores_antes)}")

        # 0 = sem irmãos, 1 = tem pelo menos 1 irmão
        df_norm['Number of Siblings'] = (df_norm['Number of Siblings'] > 0).astype(int)

        valores_depois = df_norm['Number of Siblings'].value_counts().sort_index()
        print(f"   Valores depois: {dict(valores_depois)}")
        print(f"   ✅ 0 = sem irmãos, 1 = tem irmãos\n")
        colunas_transformadas.append('Number of Siblings')

    # 2. REMOVER: Depression
    if 'Depression' in df_norm.columns:
        print("🗑️  Removendo 'Depression' (requer estudo específico)...")
        df_norm = df_norm.drop('Depression', axis=1)
        colunas_removidas.append('Depression')

    # 3. REMOVER: Number of Type A Stressors
    if 'Number of Type A Stressors' in df_norm.columns:
        print("🗑️  Removendo 'Number of Type A Stressors' (sem valor 0)...")
        # Mostrar distribuição antes de remover
        dist = df_norm['Number of Type A Stressors'].value_counts().sort_index()
        print(f"   Distribuição: {dict(dist)}")
        df_norm = df_norm.drop('Number of Type A Stressors', axis=1)
        colunas_removidas.append('Number of Type A Stressors')

    # 4. REMOVER: Number of Physical Symptoms
    if 'Number of Physical Symptoms' in df_norm.columns:
        print("🗑️  Removendo 'Number of Physical Symptoms' (não utilizado)...")
        df_norm = df_norm.drop('Number of Physical Symptoms', axis=1)
        colunas_removidas.append('Number of Physical Symptoms')

    # 5. REMOVER: Family History - Substance Abuse
    if 'Family History - Substance Abuse' in df_norm.columns:
        print("🗑️  Removendo 'Family History - Substance Abuse' (não utilizado)...")
        df_norm = df_norm.drop('Family History - Substance Abuse', axis=1)
        colunas_removidas.append('Family History - Substance Abuse')

    # Resumo das alterações
    print(f"\n{'─' * 80}")
    print(f"📊 RESUMO DA NORMALIZAÇÃO:")
    print(f"   • Colunas transformadas: {len(colunas_transformadas)}")
    for col in colunas_transformadas:
        print(f"     - {col}")
    print(f"   • Colunas removidas: {len(colunas_removidas)}")
    for col in colunas_removidas:
        print(f"     - {col}")
    print(f"   • Shape antes: {df.shape}")
    print(f"   • Shape depois: {df_norm.shape}")
    print(f"{'─' * 80}\n")

    return df_norm

# Aplicar normalização APENAS no dataset de TESTE
dados_teste_norm = normalizar_dados(dados_teste, "TESTE (PTRTS)")

# ===== RESUMO FINAL (APÓS NORMALIZAÇÃO) =====
print("\n" + "=" * 80)
print("RESUMO FINAL")
print("=" * 80)
print(f"\nTREINO (sem normalização):")
print(f"  • Linhas: {dados_treino.shape[0]}")
print(f"  • Colunas: {dados_treino.shape[1]}")

print(f"\nTESTE (normalizado):")
print(f"  • Linhas: {dados_teste_norm.shape[0]}")
print(f"  • Colunas: {dados_teste_norm.shape[1]}")

print("\n✅ Dataset de teste normalizado e pronto para análise!")


def gerar_correlacao_teste(df):
    """
    Gera matriz de correlação do dataset de teste (27 variáveis)
    Usa Spearman (melhor para dados categóricos/ordinais)
    """
    print(f"\n{'=' * 80}\nMATRIZ DE CORRELAÇÃO - TESTE\n{'=' * 80}\n")

    # Remover colunas não numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    df_numerico = df[colunas_numericas]

    print(f"📊 Calculando correlação de Spearman para {len(colunas_numericas)} variáveis...\n")

    # Calcular correlação de Spearman
    corr_matrix = df_numerico.corr(method='spearman')

    # Visualizar matriz completa (legível com 27 variáveis)
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, cbar_kws={'label': 'Correlação de Spearman'},
                square=True)
    plt.title('Matriz de Correlação - Dataset Teste (Spearman)', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('output/plots/correlacao_teste_completa.png', dpi=300, bbox_inches='tight')
    print("✅ Matriz salva: correlacao_teste_completa.png\n")
    plt.show()

    # Mostrar correlações mais fortes com GAD
    if 'GAD' in df_numerico.columns:
        corr_gad = corr_matrix['GAD'].sort_values(ascending=False)
        print("📌 Correlações com GAD:")
        print(corr_gad.head(10))

    # Mostrar correlações mais fortes com SAD
    if 'SAD' in df_numerico.columns:
        corr_sad = corr_matrix['SAD'].sort_values(ascending=False)
        print("\n📌 Correlações com SAD:")
        print(corr_sad.head(10))

    print(f"\n{'─' * 80}\n")

    return corr_matrix

# Executar análise de correlação no TESTE
matriz_corr_teste = gerar_correlacao_teste(dados_teste_norm)
