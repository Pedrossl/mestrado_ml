"""
Scatter por Features Individuais
===================================

Plota pares das features mais importantes contra os rótulos verdadeiros.
Útil para identificar quais variáveis clínicas melhor separam os casos.

COMO RODAR:
  python3 scripts/visualization/scatter_features.py --target GAD
  python3 scripts/visualization/scatter_features.py --target GAD --top 6

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.utils import preparar_dados

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output/plots/Scatter'


def _top_features(X, y, feature_names, top_n):
    """Seleciona as top N features por correlação ponto-bisserial com o target."""
    from scipy.stats import pointbiserialr
    corrs = []
    for i, nome in enumerate(feature_names):
        r, _ = pointbiserialr(y, X[:, i])
        corrs.append((nome, abs(r), i))
    corrs.sort(key=lambda x: x[1], reverse=True)
    return corrs[:top_n]


def gerar_scatter_features(target='GAD', top_n=6):
    out_dir = f'{OUTPUT_DIR}/{target}'
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Scatter Features — Rótulos Verdadeiros | {target}")
    print(f"{'='*60}")

    df, target_name = preparar_dados(target)
    feature_names = [c for c in df.columns if c != target_name]
    X = df[feature_names].values
    y = df[target_name].values

    top = _top_features(X, y, feature_names, top_n)
    print(f"\n  Top {top_n} features por correlação com {target}:")
    for nome, r, _ in top:
        print(f"    {nome:<35} |r| = {r:.3f}")

    cor_pos = '#e74c3c'
    cor_neg = '#3498db'
    mask_pos = y == 1
    mask_neg = y == 0

    # ── Pairplot das top features ──────────────────────────────────────────────
    n = len(top)
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))

    for i, (nome_i, _, idx_i) in enumerate(top):
        for j, (nome_j, _, idx_j) in enumerate(top):
            ax = axes[i][j]

            if i == j:
                # Diagonal: histograma por classe
                ax.hist(X[mask_neg, idx_i], bins=20, alpha=0.5, color=cor_neg,
                        density=True, label=f'{target}-')
                ax.hist(X[mask_pos, idx_i], bins=20, alpha=0.6, color=cor_pos,
                        density=True, label=f'{target}+')
                ax.set_xlabel(nome_i, fontsize=8)
            else:
                # Off-diagonal: scatter
                ax.scatter(X[mask_neg, idx_j], X[mask_neg, idx_i],
                           c=cor_neg, alpha=0.35, s=15, linewidths=0)
                ax.scatter(X[mask_pos, idx_j], X[mask_pos, idx_i],
                           c=cor_pos, alpha=0.65, s=20, linewidths=0)
                if i == n - 1:
                    ax.set_xlabel(nome_j, fontsize=8)
                if j == 0:
                    ax.set_ylabel(nome_i, fontsize=8)

            ax.tick_params(labelsize=7)

    # Legenda global
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cor_neg, alpha=0.6, label=f'{target} Negativo (n={mask_neg.sum()})'),
        Patch(facecolor=cor_pos, alpha=0.8, label=f'{target} Positivo (n={mask_pos.sum()})'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               bbox_to_anchor=(1.0, 1.0))

    pct = mask_pos.sum() / len(y) * 100
    fig.suptitle(f'Pairplot Top {top_n} Features — {target} | Rótulos Verdadeiros\n'
                 f'{pct:.1f}% positivos',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    fname = f'{out_dir}/scatter_features_{target.lower()}.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Salvo: {fname}")

    # ── Scatter simples feature mais importante vs segunda ─────────────────────
    nome1, r1, idx1 = top[0]
    nome2, r2, idx2 = top[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[mask_neg, idx1], X[mask_neg, idx2],
               c=cor_neg, alpha=0.5, s=35, linewidths=0,
               label=f'{target} Negativo (n={mask_neg.sum()})')
    ax.scatter(X[mask_pos, idx1], X[mask_pos, idx2],
               c=cor_pos, alpha=0.75, s=50, linewidths=0.4,
               edgecolors='white', label=f'{target} Positivo (n={mask_pos.sum()})')

    ax.set_xlabel(f'{nome1}  (|r|={r1:.3f})', fontsize=12)
    ax.set_ylabel(f'{nome2}  (|r|={r2:.3f})', fontsize=12)
    ax.set_title(f'Top 2 Features vs {target} — Rótulos Verdadeiros', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname2 = f'{out_dir}/scatter_top2_features_{target.lower()}.png'
    plt.savefig(fname2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {fname2}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scatter das top features por correlação com o target.'
    )
    parser.add_argument('--target', choices=['GAD', 'SAD'], default='GAD')
    parser.add_argument('--top', type=int, default=6,
                        help='Número de features a plotar (padrão: 6)')
    args = parser.parse_args()

    gerar_scatter_features(target=args.target, top_n=args.top)
