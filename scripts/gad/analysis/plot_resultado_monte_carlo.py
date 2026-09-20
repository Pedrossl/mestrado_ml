import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TXT_PATH = ROOT / "resultados" / "melhor_resultado_monte_carlo.txt"
OUT_PATH = ROOT / "output" / "plots" / "resultado_monte_carlo_bonito.png"


def extrair_metricas(caminho: Path):
    texto = caminho.read_text(encoding="utf-8")

    def get_block(start_marker: str, end_marker: str):
        i = texto.index(start_marker)
        j = texto.index(end_marker, i)
        return texto[i:j]

    atual_block = get_block(
        "RESULTADO ATUAL (15 features, max_depth=8) — Sem Smoothing",
        "RESULTADO ATUAL (15 features, max_depth=8) — Com Smoothing",
    )
    original_block = get_block(
        "RESULTADO ANTERIOR (17 features) — Baseline da dissertacao",
        "Observacao: os valores +/- sao IC 95% (intervalo de confianca da media).",
    )

    def metricas(block: str):
        valores = {
            "accuracy": float(re.search(r"accuracy\s+([0-9.]+)%", block).group(1)),
            "sensitivity": float(re.search(r"sensitivity\s+([0-9.]+)%", block).group(1)),
            "specificity": float(re.search(r"specificity\s+([0-9.]+)%", block).group(1)),
            "f1": float(re.search(r"f1\s+([0-9.]+)%", block).group(1)),
            "kappa": float(re.search(r"kappa\s+([0-9.]+)\s", block).group(1)),
        }
        return valores

    return metricas(atual_block), metricas(original_block)


atual, original = extrair_metricas(TXT_PATH)

labels = ["Accuracy", "Sensitivity", "Specificity", "F1", "Kappa"]
orig_vals = [original["accuracy"], original["sensitivity"], original["specificity"], original["f1"], original["kappa"]]
new_vals = [atual["accuracy"], atual["sensitivity"], atual["specificity"], atual["f1"], atual["kappa"]]

# Ajuste de estilo para visual mais elegante
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

x = range(len(labels))
bar_width = 0.35
bars1 = ax.bar([i - bar_width/2 for i in x], orig_vals, width=bar_width, color="#6c757d", edgecolor="black", linewidth=0.7, alpha=0.8, label="Baseline (17 features)")
bars2 = ax.bar([i + bar_width/2 for i in x], new_vals, width=bar_width, color="#0d6efd", edgecolor="black", linewidth=0.7, label="Melhor Monte Carlo (15 features)")

ax.set_title("Comparativo do melhor resultado do Monte Carlo\nGAD — 200 simulações", fontsize=16, weight="bold", pad=18)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Valor (%) / Kappa", fontsize=12)
ax.set_ylim(0, max(max(orig_vals), max(new_vals)) * 1.18)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars1 + bars2:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + max(0.5, max(orig_vals + new_vals) * 0.015),
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        weight="bold",
    )

ax.legend(frameon=True, loc="upper right", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Destaque do melhor ganho
kappa_delta = new_vals[-1] - orig_vals[-1]
ax.text(
    0.02,
    0.96,
    f"Melhor Kappa: {new_vals[-1]:.3f} (+{kappa_delta:.3f} vs baseline)",
    transform=ax.transAxes,
    fontsize=10,
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "#e8f1ff", "edgecolor": "#0d6efd"},
    color="#0b2545",
)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"PNG gerado em: {OUT_PATH}")
print(f"Kappa baseline: {orig_vals[-1]:.4f}")
print(f"Kappa atual: {new_vals[-1]:.4f}")
