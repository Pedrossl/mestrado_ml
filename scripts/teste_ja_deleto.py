import pandas as pd
import sys

# script rápido para listar colunas com missing
file_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/mestrado-teste.csv"
file_path_train = sys.argv[1] if len(sys.argv) > 1 else "datasets/mestrado-treino.csv"
na_vals = ['', 'NA', 'N/A', 'n/a', '.', '...', ' .', 'nan', 'NaN']

try:
    df = pd.read_csv(file_path, na_values=na_vals, keep_default_na=True, dtype=str)
except FileNotFoundError:
    sys.stderr.write(f"Arquivo não encontrado: {file_path}\n")
    sys.exit(1)

# limpar nomes e espaços
df.columns = df.columns.str.strip()
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()

# tentar converter object numérico em número real
for col in df.columns:
    sample = df[col].dropna().astype(str).head(100)
    if not sample.empty and sample.str.match(r'^[\d\-\+ ,\.eE]+$').all():
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.').str.replace(' ', ''),
            errors='coerce'
        )

# cálculo de missing
total_rows = len(df)
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\n===== RESUMO DO DATASET =====")
print(f"Total de linhas: {total_rows}")
print(f"Total de colunas: {len(df.columns)}")
print(f"Colunas sem nenhum valor faltante: {(missing == 0).sum()}/{len(df.columns)}")
print("==============================\n")

if missing.empty:
    print("Nenhuma coluna com valores faltantes.")
else:
    print("Colunas com valores faltantes:\n")

    # criar dataframe com porcentagem
    out = missing.reset_index()
    out.columns = ['column', 'missing_count']
    out["missing_pct"] = (out["missing_count"] / total_rows * 100).round(2)

    # exibir formatado
    for _, row in out.iterrows():
        print(f"{row['column']} - {row['missing_count']} ({row['missing_pct']}%)")

    # exportar CSV
    out.to_csv('output/colunas_com_missing.csv', index=False)
    print("\nArquivo salvo em: output/colunas_com_missing.csv")


def analisar_balanceamento(df_to_analyze, vars_to_check, all_rows_count):
    """
    Imprime a contagem e a porcentagem de cada valor nas colunas-alvo especificadas.

    :param df_to_analyze: DataFrame pandas contendo os dados.
    :param vars_to_check: Lista de strings com os nomes das colunas-alvo.
    :param all_rows_count: Número total de linhas no dataframe (para pct).
    """
    print("\n\n===== BALANCEAMENTO DAS VARIÁVEIS ALVO =====")

    for var in vars_to_check:
        if var in df_to_analyze.columns:
            print(f"\n--- Distribuição da variável: {var} ---")

            # Obter contagem de valores (inclui NaNs para ver se há alvos faltando)
            counts = df_to_analyze[var].value_counts(dropna=False)

            # Calcular porcentagens
            percentages = (counts / all_rows_count * 100).round(2)

            # Criar um DataFrame de resumo para impressão limpa
            balance_df = pd.DataFrame({
                'Valor': counts.index.astype(str), # Usar .astype(str) para lidar com NaNs
                'Contagem': counts.values,
                'Porcentagem (%)': percentages.values
            })

            print(balance_df.to_string(index=False))
            print("---------------------------------------")
        else:
            print(f"\nAVISO: A coluna alvo '{var}' não foi encontrada no dataset.")

    print("=============================================")


    # Definir as variáveis-alvo (se ainda não o fez)
target_vars = ['SAD', 'GAD']

# Chamar a função
analisar_balanceamento(df, target_vars, total_rows)
