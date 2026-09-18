from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "output" / "plots" / "resultado_monte_carlo_padrao_dissertacao.png"

# Dados do bloco "COMPARATIVO COM RESULTADO ORIGINAL"
labels = ["accuracy", "sensitivity", "specificity", "f1", "kappa", "sigma_kappa", "sigma_sensitivity"]
orig = [93.20, 84.30, 96.17, 85.67, 0.8130, 0.1277, 15.25]
novo = [96.12, 93.25, 96.84, 90.73, 0.8832, 0.1144, 11.41]

def fmt(v):
    return f"{v:.2f}"

fig = plt.figure(figsize=(10.5, 6), dpi=220)
ax = fig.add_axes([0.04, 0.12, 0.92, 0.74])
ax.axis("off")

# Tabela principal
rows = [
    ["Métrica", "Original (17f)", "Atual (15f, d8)", "Delta"],
    ["accuracy", fmt(orig[0]) + "%", fmt(novo[0]) + "%", "+2.92"],
    ["sensitivity", fmt(orig[1]) + "%", fmt(novo[1]) + "%", "+8.95"],
    ["specificity", fmt(orig[2]) + "%", fmt(novo[2]) + "%", "+0.67"],
    ["f1", fmt(orig[3]) + "%", fmt(novo[3]) + "%", "+5.06"],
    ["kappa", fmt(orig[4]), fmt(novo[4]), "+0.0702"],
    ["sigma_kappa", fmt(orig[5]), fmt(novo[5]), "-0.0133"],
    ["sigma_sensitivity", fmt(orig[6]), fmt(novo[6]), "-3.84"],
]

table = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    loc='center',
    cellLoc='center',
    colWidths=[0.26, 0.24, 0.24, 0.18],
)

# Diagrama sem excessos visualmente
for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(0.8)
    cell.set_edgecolor('#2f2f2f')
    if row == 0:
        cell.set_facecolor('#f2f2f2')
        cell.set_text_props(weight='bold', fontsize=11)
    else:
        cell.set_facecolor('white' if row % 2 == 1 else '#fafafa')
        cell.set_text_props(fontsize=10)

# destaque no Delta
for r in range(1, len(rows)):
    val = rows[r][3]
    if val.startswith('+'):
        table[(r, 3)].set_facecolor('#e8f5e9')
    elif val.startswith('-'):
        table[(r, 3)].set_facecolor('#fbeaea')
    table[(r, 3)].set_text_props(weight='bold')

fig.text(
    0.5,
    0.92,
    'Comparativo com resultado original (17 features)',
    ha='center',
    va='center',
    fontsize=16,
    fontweight='bold',
)
fig.text(
    0.5,
    0.82,
    'Monte Carlo — GAD | 200 simulações | sorteio 15/20 hard samples',
    ha='center',
    va='center',
    fontsize=10,
    color='#3a3a3a',
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f'PNG gerado em: {OUT_PATH}')
