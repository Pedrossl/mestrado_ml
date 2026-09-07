# =============================================================================
# PASSO 3 — Monte Carlo nos Hard Samples: com e sem Smoothing
#
# Smoothing aqui = Label Smoothing nas probabilidades de treino
# Suaviza as predições extremas (0 ou 1) para evitar overfitting nos hard cases
#
# Metodologia:
#   - N simulações sorteando subconjuntos aleatórios dos hard samples
#   - Compara estabilidade das métricas com e sem smoothing
# =============================================================================

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from scripts.utils import calcular_metricas_fold, agregar_metricas_com_ic

N_SIMULACOES  = 200    # Número de sorteios Monte Carlo
TAMANHO_SORTEIO = 15   # Quantos dos 20 hard samples usar por simulação
SMOOTH_EPSILON  = 0.1  # Grau de suavização (0 = sem, 0.1 = leve, 0.3 = forte)
ALVO = 'GAD'

OUTPUT = 'output/experimento_hard_samples'
os.makedirs(OUTPUT, exist_ok=True)

# Carrega dados
X_treino = np.load(f'{OUTPUT}/X_treino.npy')
y_treino = np.load(f'{OUTPUT}/y_treino.npy')
X_hard   = np.load(f'{OUTPUT}/X_hard.npy')
y_hard   = np.load(f'{OUTPUT}/y_hard.npy')

print(f"\n{'=' * 60}")
print(f"  MONTE CARLO — Hard Samples ({ALVO})")
print(f"  {N_SIMULACOES} simulações | Sorteio de {TAMANHO_SORTEIO}/{len(X_hard)} hard samples")
print(f"{'=' * 60}\n")

# Treina modelo base nos 80% de treino com SMOTE
smote = SMOTE(random_state=42)
X_tr_res, y_tr_res = smote.fit_resample(X_treino, y_treino)

modelo_base = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
modelo_base.fit(X_tr_res, y_tr_res)

# =============================================================================
# Função de smoothing: suaviza labels do treino
# Ex: com epsilon=0.1, labels ficam entre 0.1 e 0.9 ao invés de 0 e 1
# Isso evita que o modelo fique muito confiante nos casos difíceis
# =============================================================================
def aplicar_smoothing(y, epsilon):
    return y * (1 - epsilon) + (1 - y) * epsilon

# =============================================================================
# Monte Carlo: sorteia subconjuntos dos hard samples e avalia
# =============================================================================

def rodar_monte_carlo(usar_smoothing):
    resultados = []
    rng = np.random.default_rng(seed=0)

    for _ in range(N_SIMULACOES):
        # Sorteia TAMANHO_SORTEIO índices aleatórios dos hard samples
        idx = rng.choice(len(X_hard), size=TAMANHO_SORTEIO, replace=False)
        X_sim = X_hard[idx]
        y_sim = y_hard[idx]

        # Concatena hard samples com os dados de treino (expandindo o treino)
        X_combinado = np.vstack([X_tr_res, X_sim])
        y_combinado = np.concatenate([y_tr_res, y_sim])

        # Pesos de amostra: hard samples recebem peso reduzido com smoothing,
        # refletindo incerteza — treino normal recebe peso 1.0
        pesos_treino = np.ones(len(y_tr_res))
        if usar_smoothing:
            # Hard samples "suavizados": peso proporcional à margem de incerteza
            # Quanto mais incerto (prob próxima de 0.5), menor o peso
            probas_hard = modelo_base.predict_proba(X_sim)[:, 1]
            margem_hard = np.abs(probas_hard - 0.5)               # 0=incerto, 0.5=certo
            pesos_hard = SMOOTH_EPSILON + (1 - SMOOTH_EPSILON) * (margem_hard / 0.5)
        else:
            pesos_hard = np.ones(len(y_sim))

        pesos = np.concatenate([pesos_treino, pesos_hard])

        # Treina novo modelo com hard samples incluídos
        modelo = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
        modelo.fit(X_combinado, y_combinado, sample_weight=pesos)

        # Avalia nos próprios hard samples (o que nos interessa: estabilidade)
        y_pred = modelo.predict(X_hard)
        metricas = calcular_metricas_fold(y_hard.astype(int), y_pred.astype(int))
        resultados.append(metricas)

    return resultados

print("Rodando sem smoothing...", end=" ", flush=True)
res_sem = rodar_monte_carlo(usar_smoothing=False)
print("OK")

print("Rodando com smoothing...", end=" ", flush=True)
res_com = rodar_monte_carlo(usar_smoothing=True)
print("OK")

# =============================================================================
# COMPARAR RESULTADOS
# =============================================================================

agg_sem = agregar_metricas_com_ic(res_sem)
agg_com = agregar_metricas_com_ic(res_com)

print(f"\n{'=' * 70}")
print(f"  RESULTADOS MONTE CARLO — {ALVO} | {N_SIMULACOES} simulações")
print(f"{'=' * 70}")
print(f"\n  {'Métrica':<16} {'Sem Smoothing':>20} {'Com Smoothing (ε={:.1f})'.format(SMOOTH_EPSILON):>22}")
print(f"  {'-' * 60}")

for m in ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']:
    v_sem = f"{agg_sem[m]:.1f}% ± {agg_sem[m+'_ic']:.1f}%" if m != 'kappa' else f"{agg_sem[m]:.3f} ± {agg_sem[m+'_ic']:.3f}"
    v_com = f"{agg_com[m]:.1f}% ± {agg_com[m+'_ic']:.1f}%" if m != 'kappa' else f"{agg_com[m]:.3f} ± {agg_com[m+'_ic']:.3f}"
    print(f"  {m:<16} {v_sem:>20} {v_com:>22}")

# Impacto do smoothing na variabilidade (desvio padrão)
print(f"\n  Impacto do Smoothing na variabilidade (desvio padrão):")
print(f"  {'Métrica':<16} {'σ Sem':>10} {'σ Com':>10} {'Redução':>10}")
print(f"  {'-' * 50}")
for m in ['sensitivity', 'specificity', 'kappa']:
    s_sem = agg_sem[m + '_std']
    s_com = agg_com[m + '_std']
    reducao = (s_sem - s_com) / s_sem * 100 if s_sem > 0 else 0
    sinal = "↓" if reducao > 0 else "↑"
    print(f"  {m:<16} {s_sem:>10.3f} {s_com:>10.3f} {sinal}{abs(reducao):>8.1f}%")

print(f"\n{'=' * 70}")

# Salva resultado
with open(f'{OUTPUT}/monte_carlo_resultado.txt', 'w') as f:
    f.write(f"MONTE CARLO — {ALVO}\n")
    f.write(f"{N_SIMULACOES} simulações | Sorteio de {TAMANHO_SORTEIO}/{len(X_hard)} hard samples | ε={SMOOTH_EPSILON}\n\n")
    f.write(f"{'Métrica':<16} {'Sem Smoothing':>20} {'Com Smoothing':>22}\n")
    f.write("-" * 60 + "\n")
    for m in ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']:
        v_sem = f"{agg_sem[m]:.2f} ± {agg_sem[m+'_ic']:.2f}"
        v_com = f"{agg_com[m]:.2f} ± {agg_com[m+'_ic']:.2f}"
        f.write(f"{m:<16} {v_sem:>20} {v_com:>22}\n")

print(f"\n  Resultado salvo em: {OUTPUT}/monte_carlo_resultado.txt\n")
