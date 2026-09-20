from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "output" / "plots" / "resultado_monte_carlo_tabela.png"

rows = [
    ["Métrica", "Base", "Sem Smoothing", "Com Smoothing"],
    ["Acurácia", "84,21%", "82,88% ± 2,09%", "82,93% ± 2,23%"],
    ["Sensibilidade", "0,00%", "9,13% ± 9,96%", "11,01% ± 10,16%"],
    ["Especificidade", "94,12%", "93,18% ± 1,58%", "92,96% ± 1,89%"],
    ["F1-Score", "0,00%", "11,23% ± 12,15%", "13,52% ± 12,35%"],
    ["Kappa", "-0,08", "0,0264 ± 0,1233", "0,0478 ± 0,1279"],
]

fig = plt.figure(figsize=(9.5, 4.8), dpi=220)
ax = fig.add_axes([0.05, 0.12, 0.9, 0.76])
ax.axis("off")

# Tabela com estética de artigo
col_widths = [0.28, 0.22, 0.25, 0.25]
table = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    loc="center",
    cellLoc="center",
    colWidths=col_widths,
)

# Ajuste do estilo da tabela
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.8)
    cell.set_fontsize(10)
    if row == 0:
        cell.set_facecolor("#f2f2f2")
        cell.set_text_props(weight="bold")
    else:
        cell.set_facecolor("white" if row % 2 == 1 else "#fafafa")

# Estilo do título e da legenda
fig.text(
    0.5,
    0.92,
    "Tabela 4.10 – Resultados da simulação de Monte Carlo",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.5,
    0.83,
    "200 simulações com sorteio de 15 dos 20 casos difíceis e avaliação sobre 43 amostras",
    ha="center",
    va="center",
    fontsize=10,
    style="italic",
)

fig.text(
    0.5,
    0.06,
    "Fonte: resultados oficiais do experimento de Monte Carlo (GAD).",
    ha="center",
    va="center",
    fontsize=8,
    color="dimgray",
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Tabela PNG salva em: {OUT_PATH}")
