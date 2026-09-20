from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "output" / "plots" / "comparativo_monte_carlo_final.png"

rows = [
    ["Métrica", "Original (17f)", "Atual (15f, d8)", "Delta"],
    ["accuracy", "93.20%", "96.12%", "+2.92"],
    ["sensitivity", "84.30%", "93.25%", "+8.95"],
    ["specificity", "96.17%", "96.84%", "+0.67"],
    ["f1", "85.67%", "90.73%", "+5.06"],
    ["kappa", "0.8130", "0.8832", "+0.0702"],
    ["sigma_kappa", "0.1277", "0.1144", "-0.0133"],
    ["sigma_sensitivity", "15.25", "11.41", "-3.84"],
]

fig = plt.figure(figsize=(11, 5.8), dpi=220)
ax = fig.add_axes([0.04, 0.10, 0.92, 0.76])
ax.axis('off')

table = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    loc='center',
    cellLoc='center',
    colWidths=[0.25, 0.25, 0.25, 0.20],
)

for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(0.8)
    cell.set_edgecolor('#1f1f1f')
    if row == 0:
        cell.set_facecolor('#f2f2f2')
        cell.set_text_props(weight='bold', fontsize=11)
    else:
        cell.set_facecolor('#ffffff' if row % 2 == 1 else '#f9f9f9')
        cell.set_text_props(fontsize=10)

# destaque positivo vs negativo no Delta
for r in range(1, len(rows)):
    val = rows[r][3]
    if val.startswith('+'):
        table[(r, 3)].set_facecolor('#eaf7ee')
    elif val.startswith('-'):
        table[(r, 3)].set_facecolor('#fdecec')
    table[(r, 3)].set_text_props(weight='bold')

# título e subtítulo bem parecidos com tabela de artigo
fig.text(
    0.5,
    0.92,
    'Comparativo com resultado original (17 features)',
    ha='center',
    va='center',
    fontsize=15,
    weight='bold',
)
fig.text(
    0.5,
    0.82,
    'Monte Carlo — GAD | 200 simulações | sorteio 15/20 hard samples',
    ha='center',
    va='center',
    fontsize=10,
    color='#444444',
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"PNG gerado em: {OUT_PATH}")
