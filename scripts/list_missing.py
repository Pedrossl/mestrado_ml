import pandas as pd
import sys

# script rápido para listar colunas com missing
file_path = sys.argv[1] if len(sys.argv) > 1 else "mestrado-teste.csv"
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

for col in df.columns:
    sample = df[col].dropna().astype(str).head(100)
    if not sample.empty and sample.str.match(r'^[\d\-\+ ,\.eE]+$').all():
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce')

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("Nenhuma coluna com valores faltantes.")
else:
    print("Colunas com valores faltantes e contagens:\n")
    print(missing.to_string())
    out = missing.reset_index()
    out.columns = ['column', 'missing_count']
    out.to_csv('colunas_com_missing.csv', index=False)
    print("\nSalvo em: colunas_com_missing.csv")
