"""
Scatter 3D com Rótulos Verdadeiros
=====================================

Versão 3D do scatter plot usando PCA e t-SNE com 3 componentes.
Útil para apresentações e para ver estrutura que a projeção 2D pode esconder.

COMO RODAR:
  python3 scripts/visualization/scatter_3d.py --target GAD
  python3 scripts/visualization/scatter_3d.py --target SAD

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.utils import preparar_dados

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output/plots/Scatter'


def gerar_scatter_3d(target='GAD'):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    out_dir = f'{OUTPUT_DIR}/{target}'
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Scatter 3D — Rótulos Verdadeiros | {target}")
    print(f"{'='*60}")

    df, target_name = preparar_dados(target)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    print(f"  Dataset: {X.shape[0]} amostras | {X.shape[1]} features")

    X_std = StandardScaler().fit_transform(X)

    # PCA 3D
    print("  Calculando PCA 3D...")
    pca = PCA(n_components=3, random_state=42)
    coords_pca = pca.fit_transform(X_std)
    var = pca.explained_variance_ratio_ * 100

    # t-SNE 3D
    print("  Calculando t-SNE 3D...")
    coords_tsne = TSNE(n_components=3, random_state=42, perplexity=30,
                       max_iter=1000, learning_rate='auto', init='pca').fit_transform(X_std)

    cor_pos = '#e74c3c'
    cor_neg = '#3498db'
    mask_pos = y == 1
    mask_neg = y == 0

    fig = plt.figure(figsize=(18, 7))

    for idx, (coords, labels, titulo) in enumerate([
        (coords_pca,  [f'PC1 ({var[0]:.1f}%)', f'PC2 ({var[1]:.1f}%)', f'PC3 ({var[2]:.1f}%)'], 'PCA 3D'),
        (coords_tsne, ['t-SNE 1', 't-SNE 2', 't-SNE 3'],                                         't-SNE 3D'),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        ax.scatter(coords[mask_neg, 0], coords[mask_neg, 1], coords[mask_neg, 2],
                   c=cor_neg, alpha=0.4, s=25, label=f'{target} Negativo (n={mask_neg.sum()})')
        ax.scatter(coords[mask_pos, 0], coords[mask_pos, 1], coords[mask_pos, 2],
                   c=cor_pos, alpha=0.8, s=40, label=f'{target} Positivo (n={mask_pos.sum()})')

        ax.set_xlabel(labels[0], fontsize=9)
        ax.set_ylabel(labels[1], fontsize=9)
        ax.set_zlabel(labels[2], fontsize=9)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.9)

    pct = mask_pos.sum() / len(y) * 100
    fig.suptitle(f'Scatter 3D — {target} | Rótulos Verdadeiros | {pct:.1f}% positivos',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = f'{out_dir}/scatter_3d_{target.lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {fname}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scatter 3D com rótulos verdadeiros (PCA e t-SNE).'
    )
    parser.add_argument('--target', choices=['GAD', 'SAD'], default='GAD')
    args = parser.parse_args()

    gerar_scatter_3d(target=args.target)
