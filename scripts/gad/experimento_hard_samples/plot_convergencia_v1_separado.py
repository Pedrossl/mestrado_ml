"""
Gera gráficos de convergência da v1 em arquivos separados (um por métrica).
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = 'output/experimento_hard_samples'
PLOTS  = f'{OUTPUT}/plots'
os.makedirs(PLOTS, exist_ok=True)

METRICAS = ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']
CONDICOES = ['sem_smoothing', 'com_smoothing']
CORES = {'sem_smoothing': '#1f77b4', 'com_smoothing': '#d62728'}
LABELS = {'sem_smoothing': 'Sem smoothing', 'com_smoothing': 'Com smoothing'}

TITULOS = {
    'accuracy': 'Accuracy',
    'sensitivity': 'Sensibilidade',
    'specificity': 'Especificidade',
    'f1': 'F1-Score',
    'kappa': 'Kappa',
}

data = {c: {m: {'N': [], 'media': [], 'lo': [], 'hi': []} for m in METRICAS}
        for c in CONDICOES}

with open(f'{OUTPUT}/convergencia_curva_v1.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        c = row['condicao']
        m = row['metrica']
        data[c][m]['N'].append(int(row['N']))
        data[c][m]['media'].append(float(row['sigma_media']))
        data[c][m]['lo'].append(float(row['sigma_min']))
        data[c][m]['hi'].append(float(row['sigma_max']))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

for m in METRICAS:
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for c in CONDICOES:
        d = data[c][m]
        N = np.array(d['N'])
        media = np.array(d['media'])
        lo = np.array(d['lo'])
        hi = np.array(d['hi'])
        ax.plot(N, media, color=CORES[c], lw=1.8, label=LABELS[c])
        ax.fill_between(N, lo, hi, color=CORES[c], alpha=0.15)

    ax.axvline(200, color='gray', ls='--', lw=1, label='N = 200')
    ax.set_xlabel('N simulações')
    ax.set_ylabel('σ da métrica')
    ax.set_title(f'Convergência — {TITULOS[m]} (V1)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = f'{PLOTS}/convergencia_v1_{m}.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  {path}')

print('Concluído.')
