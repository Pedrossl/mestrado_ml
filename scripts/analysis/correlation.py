# ...existing code...
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_NA = ['', 'NA', 'N/A', 'n/a', '.', '...', ' .', 'nan', 'NaN']

def ensure_dirs():
    os.makedirs("output/plots", exist_ok=True)
    os.makedirs("output", exist_ok=True)

def save_fig(fname):
    path = os.path.join("output/plots", fname)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Figura salva: {path}")

def try_parse_numeric(df, sample_size=50):
    """Tenta converter colunas que parecem numéricas (troca vírgula por ponto)."""
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(sample_size)
        if sample.empty:
            continue
        if sample.str.match(r'^[\d\-\+ ,\.eE%]+$').all():
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(' ', '').str.replace('%', '').str.replace(',', '.'),
                errors='coerce'
            )
            if converted.notna().sum() > 0:
                df[col] = converted
    return df

def list_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("Nenhuma coluna com valores faltantes.")
    else:
        print("Colunas com valores faltantes e contagens:\n")
        for col, cnt in missing.items():
            pct = round(cnt / len(df) * 100, 2)
            print(f"{col}: {cnt} ({pct}%)")
        out = missing.reset_index()
        out.columns = ["column", "missing_count"]
        out.to_csv("output/colunas_com_missing.csv", index=False)
        print("\nSalvo: output/colunas_com_missing.csv")
    return missing

def corr_matrix(df, method='pearson'):
    """Retorna a matriz de correlação usando pandas (pairwise, ignora NaNs por par).

    method: 'pearson', 'spearman' ou 'kendall'.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] <= 1:
        return None
    return num_df.corr(method=method)


def corr_with_pvalues(df, method='pearson'):
    """Calcula matriz de correlação e matriz de p-values por par de variáveis.

    Retorna (corr_df, pval_df). Usa scipy.stats para calcular correlações e p-values.
    """
    num_df = df.select_dtypes(include=[np.number])
    cols = num_df.columns
    n = len(cols)
    corr_mat = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    pval_mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    for i in range(n):
        for j in range(i + 1, n):
            x = num_df.iloc[:, i]
            y = num_df.iloc[:, j]
            # dropna pairwise
            valid = x.notna() & y.notna()
            x_valid = x[valid]
            y_valid = y[valid]
            if len(x_valid) < 2:
                r = np.nan
                p = np.nan
            else:
                if method == 'pearson':
                    r, p = stats.pearsonr(x_valid, y_valid)
                elif method == 'spearman':
                    r, p = stats.spearmanr(x_valid, y_valid)
                elif method == 'kendall':
                    r, p = stats.kendalltau(x_valid, y_valid)
                else:
                    raise ValueError(f"Método desconhecido: {method}")
            corr_mat.iat[i, j] = corr_mat.iat[j, i] = r
            pval_mat.iat[i, j] = pval_mat.iat[j, i] = p

    return corr_mat, pval_mat


def plot_correlation_matrix(corr, method='pearson', out_png='output/plots/correlation_heatmap.png', annot_threshold=12):
    """Plota um heatmap da matriz de correlação e salva como PNG (não salva CSV aqui)."""
    if corr is None:
        print('Poucas colunas numéricas para correlação.')
        return

    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(max(8, corr.shape[1] * 0.5), max(6, corr.shape[0] * 0.5)))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    annot = True if corr.shape[0] <= annot_threshold else False
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=annot,
        fmt='.2f' if annot else None,
        linewidths=0.5,
        cbar_kws={'shrink': 0.75}
    )
    plt.title(f'Mapa de correlação ({method})')
    # save_fig espera apenas o nome do arquivo dentro de output/plots
    save_fig(os.path.basename(out_png))


def compute_and_save_all_correlations(df, methods=('pearson', 'spearman', 'kendall'), save_csv=False):
    """Computa matrizes de correlação e p-values para cada método, salva CSVs e heatmaps.

    Retorna dicionário com entradas {method: (corr_df, pval_df)}.
    """
    results = {}
    for m in methods:
        print(f"\nCalculando correlação ({m})...")
        corr = corr_matrix(df, method=m)
        if corr is None:
            print('Pulando — poucas colunas numéricas')
            results[m] = (None, None)
            continue
        # p-values
        corr_p, pvals = corr_with_pvalues(df, method=m)
        # salvar CSVs apenas se save_csv True
        if save_csv:
            corr_path = f'output/correlation_matrix_{m}.csv'
            pval_path = f'output/correlation_pvalues_{m}.csv'
            corr.to_csv(corr_path)
            pvals.to_csv(pval_path)
            print(f'Salvo: {corr_path} e {pval_path}')
        else:
            print('Salvando apenas PNG (CSV desabilitado)')
        # plot (sempre salva PNG)
        png_name = f'correlation_heatmap_{m}.png'
        plot_correlation_matrix(corr, method=m, out_png=f'output/plots/{png_name}')
        results[m] = (corr, pvals)
    return results

def main(file_path, method):
    ensure_dirs()
    na_vals = DEFAULT_NA

    try:
        df = pd.read_csv(file_path, na_values=na_vals, keep_default_na=True, dtype=str)
    except FileNotFoundError:
        sys.stderr.write(f"Arquivo não encontrado: {file_path}\n")
        sys.exit(1)

    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].str.strip()

    df = try_parse_numeric(df)

    print("\n==== Dataset carregado ====")
    print(df.head())

    print("\n==== Valores faltantes ====\n")
    list_missing(df)

    print("\n==== Correlação ====\n")
    # se method == 'all' calcula para pearson, spearman e kendall
    if method == 'all':
        results = compute_and_save_all_correlations(df, methods=('pearson', 'spearman', 'kendall'), save_csv=False)
    else:
        results = compute_and_save_all_correlations(df, methods=(method,), save_csv=False)

    for m, (corr, pvals) in results.items():
        if corr is None:
            continue
        print(f"\nMatriz de correlação ({m}) — primeiras linhas:\n")
        print(corr.round(3).iloc[:8, :8])

    print("\nConcluído. Arquivos em output/ e output/plots/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerar mapa de correlação do dataset")
    parser.add_argument("path", nargs="?", default="datasets/mestrado-teste.csv", help="caminho para o CSV")
    parser.add_argument("--method", choices=['pearson', 'spearman', 'kendall'], default='pearson', help="método de correlação")
    args = parser.parse_args()
    main(args.path, args.method)
