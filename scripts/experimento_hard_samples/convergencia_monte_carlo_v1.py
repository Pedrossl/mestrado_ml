"""
Análise de convergência do Monte Carlo — V1 (avalia nos 20 hard samples).

Mesmo protocolo da v2 (convergencia_monte_carlo.py), mas usando a
metodologia original da v1: avalia nos 20 hard samples, não no teste honesto.
"""

import os
import csv
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import calcular_metricas_fold

N_MAX           = 500
GRID_N          = [50, 100, 200, 500]
R_SEMENTES      = 5
TAMANHO_SORTEIO = 15
SMOOTH_EPSILON  = 0.1
ALVO            = 'GAD'

METRICAS = ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']

OUTPUT = 'output/experimento_hard_samples'
PLOTS  = f'{OUTPUT}/plots'
os.makedirs(PLOTS, exist_ok=True)

X_treino = np.load(f'{OUTPUT}/X_treino.npy')
y_treino = np.load(f'{OUTPUT}/y_treino.npy')
X_hard   = np.load(f'{OUTPUT}/X_hard.npy')
y_hard   = np.load(f'{OUTPUT}/y_hard.npy')

smote = SMOTE(random_state=42)
X_tr_res, y_tr_res = smote.fit_resample(X_treino, y_treino)

modelo_base = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
modelo_base.fit(X_tr_res, y_tr_res)

print(f"\n{'=' * 70}")
print(f"  CONVERGÊNCIA DO MONTE CARLO V1 — {ALVO}")
print(f"  N até {N_MAX} | {R_SEMENTES} sementes | avaliação nos 20 hard samples")
print(f"{'=' * 70}\n")


def rodar_monte_carlo(usar_smoothing, semente, n_simulacoes):
    resultados = []
    rng = np.random.default_rng(seed=semente)

    for _ in range(n_simulacoes):
        idx = rng.choice(len(X_hard), size=TAMANHO_SORTEIO, replace=False)
        X_sim = X_hard[idx]
        y_sim = y_hard[idx]

        X_combinado = np.vstack([X_tr_res, X_sim])
        y_combinado = np.concatenate([y_tr_res, y_sim])

        pesos_treino = np.ones(len(y_tr_res))
        if usar_smoothing:
            probas_hard = modelo_base.predict_proba(X_sim)[:, 1]
            margem_hard = np.abs(probas_hard - 0.5)
            pesos_hard = SMOOTH_EPSILON + (1 - SMOOTH_EPSILON) * (margem_hard / 0.5)
        else:
            pesos_hard = np.ones(len(y_sim))

        pesos = np.concatenate([pesos_treino, pesos_hard])

        modelo = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
        modelo.fit(X_combinado, y_combinado, sample_weight=pesos)

        y_pred = modelo.predict(X_hard)
        resultados.append(calcular_metricas_fold(y_hard.astype(int), y_pred.astype(int)))

    return resultados


CONDICOES = [('sem_smoothing', False), ('com_smoothing', True)]

series = {nome: {m: np.zeros((R_SEMENTES, N_MAX)) for m in METRICAS}
          for nome, _ in CONDICOES}

for nome, usar_smoothing in CONDICOES:
    for r in range(R_SEMENTES):
        print(f"  {nome:<16} semente {r + 1}/{R_SEMENTES}...", end=" ", flush=True)
        res = rodar_monte_carlo(usar_smoothing, semente=r, n_simulacoes=N_MAX)
        for m in METRICAS:
            series[nome][m][r] = [x[m] for x in res]
        print("OK")


def sigma_ate(valores, n):
    return float(np.std(valores[:n], ddof=1))


GRID_CURVA = list(range(10, N_MAX + 1, 10))
curvas = {nome: {m: np.array([[sigma_ate(series[nome][m][r], n) for n in GRID_CURVA]
                              for r in range(R_SEMENTES)])
                 for m in METRICAS}
          for nome, _ in CONDICOES}

with open(f'{OUTPUT}/convergencia_curva_v1.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['condicao', 'metrica', 'N', 'sigma_media', 'sigma_min', 'sigma_max'])
    for nome, _ in CONDICOES:
        for m in METRICAS:
            c = curvas[nome][m]
            media, lo, hi = c.mean(axis=0), c.min(axis=0), c.max(axis=0)
            for i, n in enumerate(GRID_CURVA):
                w.writerow([nome, m, n, f"{media[i]:.4f}", f"{lo[i]:.4f}", f"{hi[i]:.4f}"])

print(f"\n  CSV salvo em: {OUTPUT}/convergencia_curva_v1.csv")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("  [aviso] matplotlib indisponível — gráfico não gerado.\n")
else:
    fig, axes = plt.subplots(1, len(METRICAS), figsize=(4 * len(METRICAS), 3.6),
                             sharex=True)
    cores = {'sem_smoothing': '#1f77b4', 'com_smoothing': '#d62728'}

    for eixo, m in zip(axes, METRICAS):
        for nome, _ in CONDICOES:
            c = curvas[nome][m]
            media, lo, hi = c.mean(axis=0), c.min(axis=0), c.max(axis=0)
            eixo.plot(GRID_CURVA, media, color=cores[nome], lw=1.6,
                      label=nome.replace('_', ' '))
            eixo.fill_between(GRID_CURVA, lo, hi, color=cores[nome], alpha=0.15)
        eixo.axvline(200, color='gray', ls='--', lw=1)
        eixo.set_title(m)
        eixo.set_xlabel('N simulações')
        eixo.grid(alpha=0.3)

    axes[0].set_ylabel('σ da métrica')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Convergência do desvio padrão do Monte Carlo V1 — {ALVO} '
                 f'(banda = {R_SEMENTES} sementes independentes)', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/convergencia_sigma_v1.png', dpi=160)
    print(f"  Gráfico salvo em: {PLOTS}/convergencia_sigma_v1.png\n")
