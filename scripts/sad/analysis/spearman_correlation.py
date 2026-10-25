"""
Correlacao de Spearman — SAD.

Espelha a analise feita para GAD (ver docs/gad/FEATURE_SELECTION.md, etapa 1):
- Correlacao de cada feature com o alvo (SAD).
- Pares de features com alta correlacao entre si (candidatos a redundancia).

Nao remove nada sozinho: so aponta candidatas. A decisao final precisa passar
por Permutation Importance e validacao Monte Carlo, igual foi feito para GAD.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.utils import preparar_dados

ALVO = "SAD"
REDUNDANCIA_MIN = 0.40
RELEVANCIA_MAX = 0.15

OUTPUT_DIR = "output/sad/correlacao"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, target = preparar_dados(ALVO, aplicar_limpeza_features=False)
    features = [c for c in df.columns if c != target]

    corr = df.corr(method="spearman")

    corr_target = corr[target].drop(target).sort_values(key=abs, ascending=False)

    print(f"\nCorrelacao de Spearman com {ALVO} | {len(features)} features")
    print("=" * 70)
    for feat, rho in corr_target.items():
        flag = " <- fraca (candidata)" if abs(rho) < RELEVANCIA_MAX else ""
        print(f"  {feat:<45} rho={rho:+.3f}{flag}")

    txt_path = f"{OUTPUT_DIR}/correlacao_com_{ALVO.lower()}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Correlacao de Spearman com {ALVO}\n")
        f.write("=" * 70 + "\n\n")
        for feat, rho in corr_target.items():
            flag = " <- fraca (candidata a remocao por baixo sinal)" if abs(rho) < RELEVANCIA_MAX else ""
            f.write(f"{feat:<45} rho={rho:+.3f}{flag}\n")
    print(f"\nSalvo: {txt_path}")

    pares_redundantes = []
    corr_features = corr.loc[features, features]
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            rho = corr_features.loc[f1, f2]
            if abs(rho) >= REDUNDANCIA_MIN:
                pares_redundantes.append((f1, f2, rho, corr_target[f1], corr_target[f2]))

    pares_redundantes.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n\nPares com |rho| >= {REDUNDANCIA_MIN} entre si (candidatos a redundancia)")
    print("=" * 70)
    if not pares_redundantes:
        print("  Nenhum par encontrado.")
    for f1, f2, rho, r1, r2 in pares_redundantes:
        mais_fraca = f1 if abs(r1) < abs(r2) else f2
        print(f"  {f1} <-> {f2}: rho={rho:+.3f}  "
              f"({f1} vs {ALVO}={r1:+.3f}, {f2} vs {ALVO}={r2:+.3f})  "
              f"candidata a remover: {mais_fraca}")

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n\nPares com |rho| >= {REDUNDANCIA_MIN} entre si (candidatos a redundancia)\n")
        f.write("=" * 70 + "\n")
        if not pares_redundantes:
            f.write("Nenhum par encontrado.\n")
        for f1, f2, rho, r1, r2 in pares_redundantes:
            mais_fraca = f1 if abs(r1) < abs(r2) else f2
            f.write(f"{f1} <-> {f2}: rho={rho:+.3f}  "
                    f"({f1} vs {ALVO}={r1:+.3f}, {f2} vs {ALVO}={r2:+.3f})  "
                    f"candidata a remover: {mais_fraca}\n")

    corr_plot = corr.mask(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_plot.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title(f"Matriz de Correlacao de Spearman ({ALVO})\nTriangulo inferior",
                 fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    png_path = f"{OUTPUT_DIR}/matriz_spearman_{ALVO.lower()}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nMatriz salva: {png_path}")


if __name__ == "__main__":
    main()
