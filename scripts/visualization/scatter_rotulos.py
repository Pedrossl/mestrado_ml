"""
Scatter Plot com Rótulos Verdadeiros
======================================

Visualiza a separabilidade das classes (GAD/SAD) no espaço de features,
usando redução de dimensionalidade:

  1. PCA       — rápido, linear, boa visão geral
  2. t-SNE     — não-linear, preserva estrutura local (grupos)
  3. UMAP      — não-linear, preserva estrutura global e local

COMO RODAR:
  python3 scripts/visualization/scatter_rotulos.py --target GAD
  python3 scripts/visualization/scatter_rotulos.py --target SAD
  python3 scripts/visualization/scatter_rotulos.py --target GAD --metodo pca
  python3 scripts/visualization/scatter_rotulos.py --target GAD --metodo tsne
  python3 scripts/visualization/scatter_rotulos.py --target GAD --metodo umap
  python3 scripts/visualization/scatter_rotulos.py --target GAD --metodo todos

Autor: Dissertação de Mestrado — Março 2026
"""

import numpy as np
import os
import sys
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.utils import preparar_dados

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output/plots/Scatter'


def _reduzir_pca(X):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_std)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    xlabel = f'PC1 ({var1:.1f}% var.)'
    ylabel = f'PC2 ({var2:.1f}% var.)'
    return coords, xlabel, ylabel


def _reduzir_tsne(X):
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    X_std = StandardScaler().fit_transform(X)
    coords = TSNE(n_components=2, random_state=42, perplexity=30,
                  max_iter=1000, learning_rate='auto', init='pca').fit_transform(X_std)
    return coords, 't-SNE Dim 1', 't-SNE Dim 2'


def _reduzir_umap(X):
    try:
        import umap
    except ImportError:
        print("  UMAP não instalado. Instale com: pip install umap-learn")
        return None, None, None
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)
    coords = umap.UMAP(n_components=2, random_state=42,
                       n_neighbors=15, min_dist=0.1).fit_transform(X_std)
    return coords, 'UMAP Dim 1', 'UMAP Dim 2'


def _plotar(coords, y, xlabel, ylabel, titulo, caminho, target):
    """Gera o scatter plot colorido pelos rótulos verdadeiros."""
    pos_label = f'{target} Positivo'
    neg_label = f'{target} Negativo'

    cor_pos = '#e74c3c'   # vermelho — caso positivo
    cor_neg = '#3498db'   # azul    — caso negativo

    mask_pos = y == 1
    mask_neg = y == 0

    fig, ax = plt.subplots(figsize=(9, 7))

    # Negativos primeiro (fundo), positivos na frente
    ax.scatter(coords[mask_neg, 0], coords[mask_neg, 1],
               c=cor_neg, alpha=0.55, s=35, linewidths=0,
               label=f'{neg_label} (n={mask_neg.sum()})')
    ax.scatter(coords[mask_pos, 0], coords[mask_pos, 1],
               c=cor_pos, alpha=0.75, s=50, linewidths=0.4,
               edgecolors='white', label=f'{pos_label} (n={mask_pos.sum()})')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Percentual de positivos
    pct = mask_pos.sum() / len(y) * 100
    ax.text(0.01, 0.01, f'Desbalanceamento: {pct:.1f}% positivos',
            transform=ax.transAxes, fontsize=9, color='gray',
            verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {caminho}")


def gerar_scatter(target='GAD', metodo='todos'):
    out_dir = f'{OUTPUT_DIR}/{target}'
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Scatter Plot — Rótulos Verdadeiros | {target}")
    print(f"{'='*60}")

    df, target_name = preparar_dados(target)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    print(f"  Dataset: {X.shape[0]} amostras | {X.shape[1]} features")
    print(f"  Positivos: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Negativos: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)\n")

    metodos = ['pca', 'tsne', 'umap'] if metodo == 'todos' else [metodo]

    for m in metodos:
        print(f"  Reduzindo com {m.upper()}...")

        if m == 'pca':
            coords, xl, yl = _reduzir_pca(X)
            titulo = f'PCA — Separabilidade das Classes ({target})\nRótulos Verdadeiros'
            fname = f'{out_dir}/scatter_pca_{target.lower()}.png'

        elif m == 'tsne':
            coords, xl, yl = _reduzir_tsne(X)
            titulo = f't-SNE — Separabilidade das Classes ({target})\nRótulos Verdadeiros'
            fname = f'{out_dir}/scatter_tsne_{target.lower()}.png'

        elif m == 'umap':
            coords, xl, yl = _reduzir_umap(X)
            if coords is None:
                continue
            titulo = f'UMAP — Separabilidade das Classes ({target})\nRótulos Verdadeiros'
            fname = f'{out_dir}/scatter_umap_{target.lower()}.png'

        _plotar(coords, y, xl, yl, titulo, fname, target)

    # Painel comparativo (PCA + t-SNE lado a lado)
    if metodo == 'todos':
        print(f"\n  Gerando painel comparativo PCA + t-SNE...")
        _painel_comparativo(X, y, target, out_dir)


def _painel_comparativo(X, y, target, out_dir):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(X_std)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    coords_tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                       max_iter=1000, learning_rate='auto', init='pca').fit_transform(X_std)

    cor_pos = '#e74c3c'
    cor_neg = '#3498db'
    mask_pos = y == 1
    mask_neg = y == 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, coords, xl, yl, subtitulo in [
        (axes[0], coords_pca,  f'PC1 ({var1:.1f}%)', f'PC2 ({var2:.1f}%)', 'PCA'),
        (axes[1], coords_tsne, 't-SNE Dim 1',        't-SNE Dim 2',        't-SNE'),
    ]:
        ax.scatter(coords[mask_neg, 0], coords[mask_neg, 1],
                   c=cor_neg, alpha=0.5, s=30, linewidths=0,
                   label=f'{target} Negativo (n={mask_neg.sum()})')
        ax.scatter(coords[mask_pos, 0], coords[mask_pos, 1],
                   c=cor_pos, alpha=0.75, s=45, linewidths=0.4,
                   edgecolors='white', label=f'{target} Positivo (n={mask_pos.sum()})')
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(subtitulo, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    pct = mask_pos.sum() / len(y) * 100
    fig.suptitle(f'Separabilidade das Classes — {target} | Rótulos Verdadeiros\n'
                 f'Desbalanceamento: {pct:.1f}% positivos',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = f'{out_dir}/scatter_comparativo_{target.lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {fname}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scatter plot com rótulos verdadeiros (PCA, t-SNE, UMAP).'
    )
    parser.add_argument('--target',  choices=['GAD', 'SAD'], default='GAD')
    parser.add_argument('--metodo',  choices=['pca', 'tsne', 'umap', 'todos'], default='todos')
    args = parser.parse_args()

    gerar_scatter(target=args.target, metodo=args.metodo)
