# =============================================================================
# MONTE CARLO CORRIGIDO — Hard Samples + Avaliação Honesta
#
# DIFERENÇA da v1:
#   v1 (errado): avaliava nos próprios hard samples usados no treino → data leakage
#   v2 (correto): avalia no conjunto de teste COMPLETO (58 amostras)
#                 os hard samples só entram no treino, nunca na avaliação
#
# Metodologia:
#   - Modelo base: XGBoost + SMOTE nos 80% de treino
#   - 200 simulações: sorteia subconjunto dos hard samples, inclui no treino,
#     retreina e avalia no teste COMPLETO (sem os hard samples)
#   - Compara com e sem Label Smoothing (via sample_weight)
# =============================================================================

import os
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

N_SIMULACOES    = 200
TAMANHO_SORTEIO = 15    # Quantos dos 20 hard samples incluir por simulação
SMOOTH_EPSILON  = 0.1   # Peso mínimo dos hard samples com smoothing
ALVO            = 'GAD'

OUTPUT_V1 = 'output/experimento_hard_samples'
OUTPUT    = 'output/experimento_hard_samples_v2'
os.makedirs(OUTPUT, exist_ok=True)

# Carrega splits do passo 1 (já salvos pela v1)
X_treino = np.load(f'{OUTPUT_V1}/X_treino.npy')
X_teste  = np.load(f'{OUTPUT_V1}/X_teste.npy')
y_treino = np.load(f'{OUTPUT_V1}/y_treino.npy')
y_teste  = np.load(f'{OUTPUT_V1}/y_teste.npy')
X_hard   = np.load(f'{OUTPUT_V1}/X_hard.npy')
y_hard   = np.load(f'{OUTPUT_V1}/y_hard.npy')

# Identifica quais índices do teste são hard samples (para removê-los da avaliação)
# Compara linha a linha para encontrar os índices correspondentes
idx_hard_no_teste = []
for i, x_h in enumerate(X_hard):
    for j, x_t in enumerate(X_teste):
        if np.allclose(x_h, x_t):
            idx_hard_no_teste.append(j)
            break

# Conjunto de teste SEM os hard samples — avaliação honesta
mask_nao_hard = np.ones(len(X_teste), dtype=bool)
mask_nao_hard[idx_hard_no_teste] = False
X_teste_honesto = X_teste[mask_nao_hard]
y_teste_honesto = y_teste[mask_nao_hard]

print(f"\n{'=' * 65}")
print(f"  MONTE CARLO CORRIGIDO — {ALVO}")
print(f"  {N_SIMULACOES} simulações | Sorteio de {TAMANHO_SORTEIO}/{len(X_hard)} hard samples")
print(f"{'=' * 65}")
print(f"\n  Teste completo:        {len(X_teste)} amostras")
print(f"  Hard samples:          {len(X_hard)} (usados só no treino)")
print(f"  Teste para avaliação:  {len(X_teste_honesto)} amostras (sem hard samples)")

# Treina modelo base nos 80%
smote = SMOTE(random_state=42)
X_tr_res, y_tr_res = smote.fit_resample(X_treino, y_treino)

modelo_base = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
modelo_base.fit(X_tr_res, y_tr_res)

# Resultado do modelo BASE (sem hard samples) — linha de referência
y_pred_base = modelo_base.predict(X_teste_honesto)
metricas_base = calcular_metricas_fold(y_teste_honesto.astype(int), y_pred_base.astype(int))
print(f"\n  Modelo base (sem hard samples):")
print(f"    Sensitivity={metricas_base['sensitivity']:.1f}%  Specificity={metricas_base['specificity']:.1f}%  Kappa={metricas_base['kappa']:.3f}")

# =============================================================================
# Monte Carlo
# =============================================================================

def rodar_monte_carlo(usar_smoothing):
    resultados = []
    rng = np.random.default_rng(seed=0)

    for _ in range(N_SIMULACOES):
        idx = rng.choice(len(X_hard), size=TAMANHO_SORTEIO, replace=False)
        X_sim = X_hard[idx]
        y_sim = y_hard[idx]

        X_combinado = np.vstack([X_tr_res, X_sim])
        y_combinado = np.concatenate([y_tr_res, y_sim])

        pesos_treino = np.ones(len(y_tr_res))
        if usar_smoothing:
            probas_hard = modelo_base.predict_proba(X_sim)[:, 1]
            margem_hard = np.abs(probas_hard - 0.5)
            pesos_hard  = SMOOTH_EPSILON + (1 - SMOOTH_EPSILON) * (margem_hard / 0.5)
        else:
            pesos_hard = np.ones(len(y_sim))

        pesos = np.concatenate([pesos_treino, pesos_hard])

        modelo = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
        modelo.fit(X_combinado, y_combinado, sample_weight=pesos)

        # Avalia no teste HONESTO (sem os hard samples)
        y_pred = modelo.predict(X_teste_honesto)
        resultados.append(calcular_metricas_fold(y_teste_honesto.astype(int), y_pred.astype(int)))

    return resultados

print(f"\nRodando sem smoothing...", end=" ", flush=True)
res_sem = rodar_monte_carlo(usar_smoothing=False)
print("OK")

print(f"Rodando com smoothing...", end=" ", flush=True)
res_com = rodar_monte_carlo(usar_smoothing=True)
print("OK")

agg_sem = agregar_metricas_com_ic(res_sem)
agg_com = agregar_metricas_com_ic(res_com)

# =============================================================================
# EXIBIR E SALVAR
# =============================================================================

print(f"\n{'=' * 65}")
print(f"  RESULTADOS — {N_SIMULACOES} simulações | teste honesto ({len(X_teste_honesto)} amostras)")
print(f"{'=' * 65}")
print(f"\n  {'Métrica':<16} {'Base':>12} {'Sem Smooth':>14} {'Com Smooth':>14}")
print(f"  {'-' * 58}")

for m in ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']:
    base = f"{metricas_base[m]:.1f}%" if m != 'kappa' else f"{metricas_base[m]:.3f}"
    v_sem = f"{agg_sem[m]:.1f}%±{agg_sem[m+'_ic']:.1f}" if m != 'kappa' else f"{agg_sem[m]:.3f}±{agg_sem[m+'_ic']:.3f}"
    v_com = f"{agg_com[m]:.1f}%±{agg_com[m+'_ic']:.1f}" if m != 'kappa' else f"{agg_com[m]:.3f}±{agg_com[m+'_ic']:.3f}"
    print(f"  {m:<16} {base:>12} {v_sem:>14} {v_com:>14}")

print(f"\n  Impacto na variabilidade (desvio padrão):")
print(f"  {'Métrica':<16} {'σ Base':>10} {'σ Sem':>10} {'σ Com':>10}")
print(f"  {'-' * 48}")
for m in ['sensitivity', 'specificity', 'kappa']:
    s_sem = agg_sem[m + '_std']
    s_com = agg_com[m + '_std']
    print(f"  {m:<16} {'—':>10} {s_sem:>10.3f} {s_com:>10.3f}")

with open(f'{OUTPUT}/monte_carlo_corrigido_resultado.txt', 'w') as f:
    f.write(f"MONTE CARLO CORRIGIDO — {ALVO}\n")
    f.write(f"Avaliação honesta: teste sem hard samples ({len(X_teste_honesto)} amostras)\n")
    f.write(f"{N_SIMULACOES} simulações | Sorteio {TAMANHO_SORTEIO}/{len(X_hard)} | ε={SMOOTH_EPSILON}\n\n")
    f.write(f"PROBLEMA DA V1: avaliava nos próprios hard samples usados no treino (data leakage)\n")
    f.write(f"CORREÇÃO V2:    avalia no restante do teste (excluindo os hard samples)\n\n")
    f.write(f"{'Métrica':<16} {'Base':>12} {'Sem Smooth':>16} {'Com Smooth':>16}\n")
    f.write("-" * 62 + "\n")
    for m in ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']:
        base = f"{metricas_base[m]:.2f}"
        v_sem = f"{agg_sem[m]:.2f} ± {agg_sem[m+'_ic']:.2f}"
        v_com = f"{agg_com[m]:.2f} ± {agg_com[m+'_ic']:.2f}"
        f.write(f"{m:<16} {base:>12} {v_sem:>16} {v_com:>16}\n")

print(f"\n  Salvo em: {OUTPUT}/monte_carlo_corrigido_resultado.txt\n")
