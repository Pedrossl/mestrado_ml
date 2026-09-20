# =============================================================================
# ANÁLISE DE CONVERGÊNCIA DO MONTE CARLO
#
# Objetivo: justificar empiricamente a escolha de N = 200 simulações, em vez
# de apelar para "é o que se costuma usar".
#
# Método:
#   - Roda N_MAX = 500 simulações por condição (sem/com Label Smoothing),
#     repetindo tudo com R = 5 sementes independentes.
#   - Como as simulações são i.i.d., o desvio padrão estimado com as N
#     primeiras é exatamente o que se obteria rodando o experimento com N.
#     Isso permite ler a curva σ(N) de um único run por semente.
#   - As R sementes dão a incerteza da própria estimativa de σ: se σ(200)
#     varia pouco entre sementes e já não muda em relação a σ(500), N = 200
#     está no platô.
#
# Saídas:
#   output/experimento_hard_samples_v2/convergencia_monte_carlo.txt
#   output/experimento_hard_samples_v2/convergencia_monte_carlo.csv
#   output/experimento_hard_samples_v2/plots/convergencia_sigma.png
# =============================================================================

import os
import csv
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import calcular_metricas_fold

N_MAX           = 500          # maior N avaliado
GRID_N          = [50, 100, 200, 500]
R_SEMENTES      = 5            # réplicas independentes para medir a incerteza de σ
TAMANHO_SORTEIO = 15
SMOOTH_EPSILON  = 0.1
ALVO            = 'GAD'

METRICAS = ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']

OUTPUT_V1 = 'output/experimento_hard_samples'
OUTPUT    = 'output/experimento_hard_samples_v2'
PLOTS     = f'{OUTPUT}/plots'
os.makedirs(PLOTS, exist_ok=True)

# =============================================================================
# Dados — mesmos splits usados no monte_carlo_corrigido.py
# =============================================================================

X_treino = np.load(f'{OUTPUT_V1}/X_treino.npy')
X_teste  = np.load(f'{OUTPUT_V1}/X_teste.npy')
y_treino = np.load(f'{OUTPUT_V1}/y_treino.npy')
y_teste  = np.load(f'{OUTPUT_V1}/y_teste.npy')
X_hard   = np.load(f'{OUTPUT_V1}/X_hard.npy')
y_hard   = np.load(f'{OUTPUT_V1}/y_hard.npy')

idx_hard_no_teste = []
for x_h in X_hard:
    for j, x_t in enumerate(X_teste):
        if np.allclose(x_h, x_t):
            idx_hard_no_teste.append(j)
            break

mask_nao_hard = np.ones(len(X_teste), dtype=bool)
mask_nao_hard[idx_hard_no_teste] = False
X_teste_honesto = X_teste[mask_nao_hard]
y_teste_honesto = y_teste[mask_nao_hard]

smote = SMOTE(random_state=42)
X_tr_res, y_tr_res = smote.fit_resample(X_treino, y_treino)

modelo_base = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
modelo_base.fit(X_tr_res, y_tr_res)

print(f"\n{'=' * 70}")
print(f"  CONVERGÊNCIA DO MONTE CARLO — {ALVO}")
print(f"  N até {N_MAX} | {R_SEMENTES} sementes independentes | grade {GRID_N}")
print(f"{'=' * 70}")
print(f"  Teste para avaliação: {len(X_teste_honesto)} amostras (sem hard samples)")
print(f"  Total de treinos:     {2 * R_SEMENTES * N_MAX} modelos\n")

# =============================================================================
# Monte Carlo — devolve as métricas simulação a simulação
# =============================================================================

def rodar_monte_carlo(usar_smoothing, semente, n_simulacoes):
    """Roda n_simulacoes e devolve a lista de métricas, uma por simulação."""
    resultados = []
    rng = np.random.default_rng(seed=semente)

    for _ in range(n_simulacoes):
        idx_treino = rng.choice(len(X_hard), size=TAMANHO_SORTEIO, replace=False)
        X_sim_treino, y_sim_treino = X_hard[idx_treino], y_hard[idx_treino]

        idx_sobra = np.setdiff1d(np.arange(len(X_hard)), idx_treino)
        X_sim_teste, y_sim_teste = X_hard[idx_sobra], y_hard[idx_sobra]

        X_combinado = np.vstack([X_tr_res, X_sim_treino])
        y_combinado = np.concatenate([y_tr_res, y_sim_treino])

        pesos_treino = np.ones(len(y_tr_res))
        if usar_smoothing:
            probas_hard = modelo_base.predict_proba(X_sim_treino)[:, 1]
            margem_hard = np.abs(probas_hard - 0.5)
            pesos_hard  = SMOOTH_EPSILON + (1 - SMOOTH_EPSILON) * (margem_hard / 0.5)
        else:
            pesos_hard = np.ones(len(y_sim_treino))

        pesos = np.concatenate([pesos_treino, pesos_hard])

        modelo = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
        modelo.fit(X_combinado, y_combinado, sample_weight=pesos)

        X_aval = np.vstack([X_teste_honesto, X_sim_teste])
        y_aval = np.concatenate([y_teste_honesto, y_sim_teste])

        y_pred = modelo.predict(X_aval)
        resultados.append(calcular_metricas_fold(y_aval.astype(int), y_pred.astype(int)))

    return resultados


# =============================================================================
# Coleta: para cada condição e semente, guarda a série de N_MAX simulações
# =============================================================================

CONDICOES = [('sem_smoothing', False), ('com_smoothing', True)]

# series[condicao][metrica] -> array (R_SEMENTES, N_MAX)
series = {nome: {m: np.zeros((R_SEMENTES, N_MAX)) for m in METRICAS}
          for nome, _ in CONDICOES}

for nome, usar_smoothing in CONDICOES:
    for r in range(R_SEMENTES):
        print(f"  {nome:<16} semente {r + 1}/{R_SEMENTES}...", end=" ", flush=True)
        res = rodar_monte_carlo(usar_smoothing, semente=r, n_simulacoes=N_MAX)
        for m in METRICAS:
            series[nome][m][r] = [x[m] for x in res]
        print("OK")

# =============================================================================
# σ(N): desvio padrão amostral usando as N primeiras simulações
# =============================================================================

def sigma_ate(valores, n):
    """Desvio padrão amostral (ddof=1) das n primeiras simulações."""
    return float(np.std(valores[:n], ddof=1))

# sigmas[condicao][metrica] -> array (R_SEMENTES, len(GRID_N))
sigmas = {nome: {m: np.array([[sigma_ate(series[nome][m][r], n) for n in GRID_N]
                              for r in range(R_SEMENTES)])
                 for m in METRICAS}
          for nome, _ in CONDICOES}

# Curva contínua de σ(N) para o gráfico (passo de 10)
GRID_CURVA = list(range(10, N_MAX + 1, 10))
curvas = {nome: {m: np.array([[sigma_ate(series[nome][m][r], n) for n in GRID_CURVA]
                              for r in range(R_SEMENTES)])
                 for m in METRICAS}
          for nome, _ in CONDICOES}

# =============================================================================
# Relatório
# =============================================================================

linhas_csv = [['condicao', 'metrica', 'N', 'sigma_media', 'sigma_min', 'sigma_max',
               'amplitude_entre_sementes_pct', 'delta_vs_N_anterior_pct',
               'delta_vs_N500_pct', 'erro_padrao_media']]

texto = []
texto.append(f"ANÁLISE DE CONVERGÊNCIA DO MONTE CARLO — {ALVO}")
texto.append(f"N_MAX={N_MAX} | sementes independentes={R_SEMENTES} | "
             f"sorteio {TAMANHO_SORTEIO}/{len(X_hard)} | eps={SMOOTH_EPSILON}")
texto.append(f"Avaliacao: teste sem hard samples ({len(X_teste_honesto)} amostras) "
             f"+ {len(X_hard) - TAMANHO_SORTEIO} hard samples sobressalentes")
texto.append("")
texto.append("sigma(N) = desvio padrao da metrica entre as N simulacoes.")
texto.append("Media e faixa [min, max] calculadas sobre as sementes independentes.")
texto.append("amp = amplitude de sigma entre as sementes, (max-min)/media. Mede quao")
texto.append("      REPRODUTIVEL e a estimativa de sigma para aquele N — e o criterio que")
texto.append("      de fato discrimina os Ns, ja que a media de sigma estabiliza cedo.")
texto.append("delta = variacao relativa de sigma em relacao ao N anterior da grade.")
texto.append("EP = erro padrao da media = sigma / sqrt(N) (precisao do ponto estimado).")
texto.append("")

for nome, _ in CONDICOES:
    texto.append("=" * 78)
    texto.append(f"CONDICAO: {nome}")
    texto.append("=" * 78)
    for m in METRICAS:
        s = sigmas[nome][m]                 # (R, len(GRID_N))
        media = s.mean(axis=0)
        smin, smax = s.min(axis=0), s.max(axis=0)
        texto.append("")
        texto.append(f"  {m}")
        texto.append(f"  {'N':>5} {'sigma':>9} {'[min, max]':>20} {'amp':>8} "
                     f"{'d vs N-1':>10} {'d vs 500':>10} {'EP media':>10}")
        texto.append("  " + "-" * 76)
        for i, n in enumerate(GRID_N):
            d_ant = (media[i] - media[i - 1]) / media[i - 1] * 100 if i > 0 and media[i - 1] > 0 else float('nan')
            d_500 = (media[i] - media[-1]) / media[-1] * 100 if media[-1] > 0 else float('nan')
            amp = (smax[i] - smin[i]) / media[i] * 100 if media[i] > 0 else float('nan')
            ep = media[i] / np.sqrt(n)
            faixa = f"[{smin[i]:.3f}, {smax[i]:.3f}]"
            d_ant_s = "     —" if i == 0 else f"{d_ant:+9.1f}%"
            texto.append(f"  {n:>5} {media[i]:>9.3f} {faixa:>20} {amp:>7.1f}% "
                         f"{d_ant_s:>10} {d_500:>+9.1f}% {ep:>10.3f}")
            linhas_csv.append([nome, m, n, f"{media[i]:.4f}", f"{smin[i]:.4f}",
                               f"{smax[i]:.4f}", f"{amp:.2f}",
                               "" if i == 0 else f"{d_ant:.2f}",
                               f"{d_500:.2f}", f"{ep:.4f}"])
    texto.append("")

# Leitura automática: N mínimo da grade em que σ já está a menos de 5% de σ(500)
texto.append("=" * 78)
texto.append("LEITURA — menor N da grade que satisfaz cada criterio")
texto.append("=" * 78)
texto.append("  (A) sigma medio dentro de 5% de sigma(N=500)")
texto.append("  (B) amplitude de sigma entre sementes <= 10%  [criterio exigente]")
texto.append("")
texto.append(f"  {'condicao':<16} {'metrica':<14} {'(A)':>8} {'(B)':>8}")
texto.append("  " + "-" * 48)
for nome, _ in CONDICOES:
    for m in METRICAS:
        s = sigmas[nome][m]
        media, smin, smax = s.mean(axis=0), s.min(axis=0), s.max(axis=0)
        alvo = media[-1]
        n_a = next((n for i, n in enumerate(GRID_N)
                    if alvo > 0 and abs(media[i] - alvo) / alvo <= 0.05), None)
        n_b = next((n for i, n in enumerate(GRID_N)
                    if media[i] > 0 and (smax[i] - smin[i]) / media[i] <= 0.10), None)
        texto.append(f"  {nome:<16} {m:<14} {str(n_a or '>500'):>8} {str(n_b or '>500'):>8}")
texto.append("")
texto.append("  Nota: a amplitude min-max sobre apenas 5 sementes e ela propria uma")
texto.append("  estatistica ruidosa; o que sustenta a leitura e a consistencia do padrao")
texto.append("  nas 10 combinacoes condicao x metrica, nao cada valor isolado.")
texto.append("")

relatorio = "\n".join(texto)
print("\n" + relatorio)

with open(f'{OUTPUT}/convergencia_monte_carlo.txt', 'w') as f:
    f.write(relatorio)

with open(f'{OUTPUT}/convergencia_monte_carlo.csv', 'w', newline='') as f:
    csv.writer(f).writerows(linhas_csv)

# Curva completa de sigma(N) — permite refazer o gráfico sem rodar tudo de novo
with open(f'{OUTPUT}/convergencia_curva.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['condicao', 'metrica', 'N', 'sigma_media', 'sigma_min', 'sigma_max'])
    for nome, _ in CONDICOES:
        for m in METRICAS:
            c = curvas[nome][m]
            media, lo, hi = c.mean(axis=0), c.min(axis=0), c.max(axis=0)
            for i, n in enumerate(GRID_CURVA):
                w.writerow([nome, m, n, f"{media[i]:.4f}", f"{lo[i]:.4f}", f"{hi[i]:.4f}"])

# =============================================================================
# Gráfico — σ(N) por métrica, com a banda entre sementes
# =============================================================================

def gerar_grafico():
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
    fig.suptitle(f'Convergência do desvio padrão do Monte Carlo — {ALVO} '
                 f'(banda = {R_SEMENTES} sementes independentes)', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/convergencia_sigma.png', dpi=160)


print(f"  Salvo em: {OUTPUT}/convergencia_monte_carlo.txt")
print(f"            {OUTPUT}/convergencia_monte_carlo.csv")
print(f"            {OUTPUT}/convergencia_curva.csv")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("  [aviso] matplotlib indisponível — gráfico não gerado "
          "(a curva está em convergencia_curva.csv).\n")
else:
    gerar_grafico()
    print(f"            {PLOTS}/convergencia_sigma.png\n")
