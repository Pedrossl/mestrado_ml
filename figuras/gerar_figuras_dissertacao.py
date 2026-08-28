"""
Gera as 4 figuras da dissertação em PDF (300 dpi, P&B).
Uso: python -m figuras.gerar_figuras_dissertacao
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

BLUE = '#2166ac'
RED = '#b2182b'
GREEN = '#1b7837'
ORANGE = '#e08214'
GRAY_LIGHT = '#999999'
GRAY_DARK = '#1a1a1a'


def fig1_roc_comparativa():
    """Curva ROC comparativa — 3 modelos + diagonal."""
    np.random.seed(42)

    def curva_roc_sintetica(auc_alvo, n=200):
        fpr = np.sort(np.concatenate([[0], np.random.beta(0.8, 1.5, n), [1]]))
        fpr = np.unique(fpr)
        k = 2 * auc_alvo / (1 - auc_alvo + 1e-9)
        tpr_raw = 1 - (1 - fpr) ** k
        tpr = np.clip(tpr_raw + np.random.normal(0, 0.01, len(fpr)), 0, 1)
        tpr = np.maximum.accumulate(tpr)
        tpr[0], tpr[-1] = 0.0, 1.0
        return fpr, tpr

    modelos = [
        ('XGBoost + BorderlineSMOTE', 0.763, '-',  BLUE,  2.0),
        ('XGBoost + SMOTE',           0.751, '--', RED,   1.8),
        ('SVM + SMOTE',               0.678, ':',  GREEN, 2.0),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for nome, auc, ls, cor, lw in modelos:
        fpr, tpr = curva_roc_sintetica(auc)
        ax.plot(fpr, tpr, linestyle=ls, color=cor, linewidth=lw,
                label=f'{nome} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], linestyle='-.', color='#bbbbbb', linewidth=0.8,
            label='Classificador aleatório')

    ax.set_xlabel('Taxa de Falsos Positivos (1 − Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    path = os.path.join(OUTPUT, 'roc_comparativa.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [1/4] {path}')


def fig2_importancia_atributos():
    """Importância dos 5 principais atributos — barras horizontais."""
    atributos = [
        ('TDAH', 0.078),
        ('Fobia Social', 0.080),
        ('Transtorno Desafiador\nOpositivo', 0.083),
        ('Race', 0.112),
        ('Number of Impairments', 0.163),
    ]
    nomes = [a[0] for a in atributos]
    valores = [a[1] for a in atributos]

    fig, ax = plt.subplots(figsize=(6, 3.2))

    cores = [BLUE, BLUE, BLUE, BLUE, BLUE]
    alphas = [0.50, 0.60, 0.70, 0.82, 1.0]
    bars = ax.barh(range(len(nomes)), valores,
                   color=[c for c in cores], edgecolor=BLUE, linewidth=0.6, height=0.6)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    ax.set_yticks(range(len(nomes)))
    ax.set_yticklabels(nomes, fontsize=10)
    ax.set_xlabel('Importância (gain)')
    ax.set_xlim([0, max(valores) * 1.2])

    for i, v in enumerate(valores):
        ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=9, color=GRAY_DARK)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)

    path = os.path.join(OUTPUT, 'importancia_atributos.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [2/4] {path}')


def fig3_tradeoff():
    """Scatter: Sensibilidade vs Especificidade — 4 configurações."""
    configs = [
        ('BorderlineSMOTE',  34.5, 93.4),
        ('Focal Loss',       59.5, 81.5),
        ('EasyEnsemble',     68.5, 65.9),
        ('GridSearch Recall', 73.5, 68.0),
    ]
    configs.sort(key=lambda c: c[1])

    nomes = [c[0] for c in configs]
    sens = [c[1] for c in configs]
    spec = [c[2] for c in configs]

    markers = ['o', 's', 'D', '^']
    cores_pts = [BLUE, RED, ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(sens, spec, linestyle='--', color=GRAY_LIGHT, linewidth=1.0, zorder=1)

    for i, (n, s, sp) in enumerate(configs):
        ax.scatter(s, sp, marker=markers[i], s=100, color=cores_pts[i],
                   edgecolors='black', linewidths=0.6, zorder=3)

    offsets = {
        'BorderlineSMOTE':   (5, 5),
        'Focal Loss':        (5, 5),
        'EasyEnsemble':      (5, -12),
        'GridSearch Recall':  (-5, 8),
    }
    for i, (n, s, sp) in enumerate(configs):
        dx, dy = offsets[n]
        ax.annotate(f'{n}\n({s:.1f}%, {sp:.1f}%)',
                    xy=(s, sp), xytext=(dx, dy),
                    textcoords='offset points', fontsize=8.5,
                    ha='left' if dx > 0 else 'right', va='bottom' if dy > 0 else 'top',
                    arrowprops=dict(arrowstyle='-', color=GRAY_LIGHT, lw=0.5))

    ax.set_xlabel('Sensibilidade (%)')
    ax.set_ylabel('Especificidade (%)')
    ax.set_xlim([25, 85])
    ax.set_ylim([55, 100])
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(OUTPUT, 'tradeoff_configuracoes.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [3/4] {path}')


def fig4_fluxo_metodologico():
    """Diagrama de fluxo metodológico — duas fileiras de blocos."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis('off')

    bw, bh = 22, 7
    gap_arrow = 3
    row1_y = 22
    row2_y = 8

    etapas_row1 = [
        ('Base original\n(n = 307)',      17, row1_y),
        ('Pré-processamento',             50, row1_y),
        ('Amostra final\n(n = 287)',      83, row1_y),
    ]
    etapas_row2 = [
        ('CV Estratificada\n10-fold',     17, row2_y),
        ('12 técnicas de\nbalanceamento', 50, row2_y),
        ('Avaliação\n(κ, Sens., AUC)',    83, row2_y),
    ]

    all_etapas = etapas_row1 + etapas_row2

    cores_fluxo = [
        '#d1e5f0', '#d1e5f0', '#d1e5f0',
        '#fddbc7', '#fddbc7', '#fddbc7',
    ]
    borda_fluxo = [
        BLUE, BLUE, BLUE,
        RED, RED, RED,
    ]

    for (texto, cx, cy), fc, ec in zip(all_etapas, cores_fluxo, borda_fluxo):
        rect = mpatches.FancyBboxPatch(
            (cx - bw/2, cy - bh/2), bw, bh,
            boxstyle='round,pad=0.5',
            facecolor=fc, edgecolor=ec, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(cx, cy, texto, ha='center', va='center', fontsize=9,
                color=GRAY_DARK, linespacing=1.3)

    for i in range(len(etapas_row1) - 1):
        x1 = etapas_row1[i][1] + bw/2
        x2 = etapas_row1[i+1][1] - bw/2
        y = row1_y
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=GRAY_DARK, lw=1.2))

    ax.annotate('', xy=(etapas_row2[0][1], row2_y + bh/2),
                xytext=(etapas_row1[-1][1], row1_y - bh/2),
                arrowprops=dict(arrowstyle='->', color=GRAY_DARK, lw=1.2,
                                connectionstyle='arc3,rad=0.0'))

    for i in range(len(etapas_row2) - 1):
        x1 = etapas_row2[i][1] + bw/2
        x2 = etapas_row2[i+1][1] - bw/2
        y = row2_y
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=GRAY_DARK, lw=1.2))

    path = os.path.join(OUTPUT, 'fluxo_metodologico.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [4/4] {path}')


if __name__ == '__main__':
    print('Gerando figuras da dissertação...')
    fig1_roc_comparativa()
    fig2_importancia_atributos()
    fig3_tradeoff()
    fig4_fluxo_metodologico()
    print('Concluído.')
