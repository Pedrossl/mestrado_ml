import pandas as pd
import numpy as np
import sys

# Permite passar o arquivo CSV pela linha de comando:
#   python index.py caminho/para/arquivo.csv
file_path = sys.argv[1] if len(sys.argv) > 1 else "mestrado-teste.csv"
print(f"Usando arquivo: {file_path}")

# tokens comuns que representam NA no CSV
na_vals = ['', 'NA', 'N/A', 'n/a', '.', '...', ' .', 'nan', 'NaN']
try:
	# ler inicialmente como string para podermos normalizar vírgulas e espaços
	df = pd.read_csv(file_path, na_values=na_vals, keep_default_na=True, dtype=str)
except FileNotFoundError:
	sys.stderr.write(f"Arquivo não encontrado: {file_path}\n")
	sys.exit(1)

# limpar nomes de colunas e espaços nas strings
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# tentar converter colunas que parecem numéricas (substituindo vírgula decimal por ponto)
for col in df.columns:
	col_series = df[col]
	sample = col_series.dropna().astype(str).head(100)
	if sample.empty:
		continue
	# se as amostras contêm apenas dígitos, sinais, vírgula/ponto, espaço ou expoente
	if sample.str.match(r'^[\d\-\+ ,\.eE]+$').all():
		converted = pd.to_numeric(sample.str.replace(',', '.').str.replace(' ', ''), errors='coerce')
		# se a conversão produziu números, converter a coluna inteira
		if converted.notna().sum() > 0:
			df[col] = pd.to_numeric(col_series.astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce')

# calcular contagens e percentuais de faltantes
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100

print("=== Valores faltando por coluna (contagem) ===")
print(missing_count)

print("\n=== Percentual faltando por coluna (%) ===")
print(missing_pct.round(2))

total_missing = int(missing_count.sum())
total_cells = df.shape[0] * df.shape[1]
print(f"\nTotal células faltantes: {total_missing} de {total_cells} ({total_missing/total_cells*100:.2f}%)")

# linhas com pelo menos um NA
linhas_faltando = df[df.isnull().any(axis=1)]

print("\n=== Linhas com dados faltando (primeiras 20 linhas) ===")
pd.set_option('display.max_columns', None)
print(linhas_faltando.head(20))
print(f"\nTotal de linhas com pelo menos 1 NA: {len(linhas_faltando)} de {len(df)} ({len(linhas_faltando)/len(df)*100:.2f}%)")

# distribuição do número de NAs por linha
dist = df.isnull().sum(axis=1).value_counts().sort_index()
print("\n=== Distribuição do número de NAs por linha ===")
print(dist)

# colunas com mais de um limite (ex.: 5%) de faltantes
threshold = 5.0
cols_over = missing_pct[missing_pct > threshold].sort_values(ascending=False)
if not cols_over.empty:
	print(f"\nColunas com mais de {threshold}% de faltantes:")
	print(cols_over)
else:
	print(f"\nNenhuma coluna com mais de {threshold}% de faltantes.")

# salvar linhas faltantes
linhas_faltando.to_csv("linhas_faltando.csv", index=False)
print("\nAs linhas com dados faltando foram salvas em 'linhas_faltando.csv'")
