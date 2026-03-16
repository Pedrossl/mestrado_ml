"""
Busca de Hiperparâmetros — Random Forest
=========================================

Baseado na estrutura do busca_hiperparametros_v3.py, adaptado para Random Forest.

Classificador:
  RandomForestClassifier — ensemble de árvores com bagging e feature subsampling

Samplers:
  SMOTEENN, BorderlineSMOTE, SMOTE, RandomUnderSampler, NearMiss(v1), IHT

Scorers:
  gmean   = √(Sens × Spec)              — elimina colapso recall
  clinic  = Sens × F1 × min(Spec/0.5,1) — racional clínico
  f2_pen  = F2 × (Spec/0.3)²            — F2 penalizado

COMO RODAR:
  Rápido (~20-40 min): python3 scripts/hyperparameters/busca_rf.py --modo rapido
  Médio  (~2-4 horas): python3 scripts/hyperparameters/busca_rf.py --modo medio

Referências:
  - Breiman (2001). Random Forests. Machine Learning 45(1), 5-32.
  - NearMiss: Zhang & Mani (2003). KDD Workshop on Learning from Imbalanced Data.

Autor: Dissertação de Mestrado — Março 2026
"""

import numpy as np
import os
import sys
import time
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import fbeta_score, confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, InstanceHardnessThreshold
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.utils import (
    preparar_dados, calcular_metricas_fold,
    agregar_metricas_com_ic,
)

warnings.filterwarnings('ignore')

# ─── Configuração ──────────────────────────────────────────────────────────────

MODOS = {
    'rapido':   {'n_iter': 50,  'desc': '~50 combinações | ~20-40 min'},
    'medio':    {'n_iter': 200, 'desc': '~200 combinações | ~2-4 horas'},
    'completo': {'n_iter': 500, 'desc': '~500 combinações | ~8-14 horas'},
}

TARGET     = 'GAD'
OUTPUT_DIR = 'output/busca_rf'
PLOTS_DIR  = f'{OUTPUT_DIR}/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Scorers inteligentes ──────────────────────────────────────────────────────

def _scorer_gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    vn, fp, fn, vp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    sens = vp/(vp+fn) if (vp+fn)>0 else 0.0
    spec = vn/(vn+fp) if (vn+fp)>0 else 0.0
    return np.sqrt(sens * spec)

def _scorer_clinic(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    vn, fp, fn, vp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    sens = vp/(vp+fn) if (vp+fn)>0 else 0.0
    spec = vn/(vn+fp) if (vn+fp)>0 else 0.0
    prec = vp/(vp+fp) if (vp+fp)>0 else 0.0
    f1   = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0.0
    return sens * f1 * min(1.0, spec/0.50)

def _scorer_f2_pen(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    vn, fp = cm[0,0], cm[0,1]
    spec = vn/(vn+fp) if (vn+fp)>0 else 0.0
    f2   = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    return f2 * min(1.0, (spec/0.30)**2)

scorer_gmean  = make_scorer(_scorer_gmean)
scorer_clinic = make_scorer(_scorer_clinic)
scorer_f2_pen = make_scorer(_scorer_f2_pen)

# ─── Espaços de busca ─────────────────────────────────────────────────────────

SPACE_RF = {
    'clf__n_estimators':      randint(100, 800),
    'clf__max_depth':         [None, 5, 10, 15, 20, 30],
    'clf__min_samples_split': randint(2, 20),
    'clf__min_samples_leaf':  randint(1, 15),
    'clf__max_features':      ['sqrt', 'log2', 0.3, 0.5, 0.7],
    'clf__class_weight':      ['balanced', 'balanced_subsample', None],
    'clf__bootstrap':         [True, False],
    'clf__max_samples':       uniform(0.5, 0.5),   # só se bootstrap=True
    'clf__criterion':         ['gini', 'entropy'],
}

SPACE_SAMPLER_SMOTE      = {'sampler__k_neighbors': randint(2, 10)}
SPACE_SAMPLER_BORDERLINE = {
    'sampler__k_neighbors': randint(2, 10),
    'sampler__m_neighbors': randint(5, 20),
}
SPACE_SAMPLER_RUS        = {'sampler__sampling_strategy': uniform(0.3, 0.7)}
SPACE_SAMPLER_NEARMISS   = {'sampler__n_neighbors': randint(2, 7)}

# ─── Funções auxiliares ────────────────────────────────────────────────────────

def _combinar_param_spaces(*spaces):
    result = {}
    for s in spaces:
        result.update(s)
    return result

def _extrair_consenso(lista_params):
    from collections import Counter
    consenso = {}
    for key in lista_params[0].keys():
        vals = [p[key] for p in lista_params]
        all_numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals)
        if all_numeric:
            consenso[key] = float(np.median(vals))
        else:
            consenso[key] = Counter(vals).most_common(1)[0][0]
    return consenso

def _construir_pipeline(clf, sampler):
    return ImbPipeline([('sampler', sampler), ('clf', clf)])

def _buscar(clf_factory, sampler_factory, param_space, n_iter, scoring, nome_scoring):
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)

    pipeline = _construir_pipeline(clf=clf_factory(), sampler=sampler_factory())

    search = RandomizedSearchCV(
        pipeline, param_distributions=param_space,
        n_iter=n_iter, scoring=scoring, cv=inner_cv,
        refit=True, n_jobs=-1, random_state=42, verbose=0,
    )

    metricas_folds = []
    params_folds   = []
    thresholds     = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        search.fit(X_tr, y_tr)
        params_folds.append(search.best_params_)

        # Threshold tuning via F2 no treino (sem leakage)
        y_prob_tr = search.best_estimator_.predict_proba(X_tr)[:, 1]
        melhor_t, melhor_f2 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            score = fbeta_score(y_tr, (y_prob_tr >= t).astype(int), beta=2, zero_division=0)
            if score > melhor_f2:
                melhor_f2, melhor_t = score, t
        thresholds.append(melhor_t)

        y_prob_te = search.best_estimator_.predict_proba(X_te)[:, 1]
        y_pred    = (y_prob_te >= melhor_t).astype(int)
        metricas_folds.append(calcular_metricas_fold(y_te, y_pred))

        m = metricas_folds[-1]
        print(f"    Fold {fold_idx+1:2d}/10 → "
              f"Sens={m['sensitivity']:.1f}%  "
              f"Spec={m['specificity']:.1f}%  "
              f"F1={m['f1']:.1f}%  "
              f"t≈{melhor_t:.2f}")

    metricas = agregar_metricas_com_ic(metricas_folds)
    sc = float(np.sqrt((metricas['sensitivity']/100) * (metricas['f1']/100)))
    return metricas, _extrair_consenso(params_folds), thresholds, sc

# ─── Definição dos experimentos ────────────────────────────────────────────────

def _definir_experimentos(n_iter, scorer, nome_scorer):
    def rf():       return RandomForestClassifier(random_state=42, n_jobs=1)
    def smoteenn(): return SMOTEENN(random_state=42)
    def blsmote():  return BorderlineSMOTE(random_state=42)
    def smote():    return SMOTE(random_state=42)
    def rus():      return RandomUnderSampler(random_state=42)
    def nearmiss(): return NearMiss(version=1)
    def iht():      return InstanceHardnessThreshold(random_state=42, cv=3)

    # (nome, clf_fn, sampler_fn, param_space_dict)
    exps = [
        ('RF_SMOTEENN',
            rf, smoteenn,
            SPACE_RF),
        ('RF_BorderlineSMOTE',
            rf, blsmote,
            _combinar_param_spaces(SPACE_RF, SPACE_SAMPLER_BORDERLINE)),
        ('RF_SMOTE',
            rf, smote,
            _combinar_param_spaces(SPACE_RF, SPACE_SAMPLER_SMOTE)),
        ('RF_NearMiss',
            rf, nearmiss,
            _combinar_param_spaces(SPACE_RF, SPACE_SAMPLER_NEARMISS)),
        ('RF_RUS',
            rf, rus,
            _combinar_param_spaces(SPACE_RF, SPACE_SAMPLER_RUS)),
        ('RF_IHT',
            rf, iht,
            SPACE_RF),
    ]

    return exps

# ─── Script principal ─────────────────────────────────────────────────────────

def main(modo='medio'):
    n_iter = MODOS[modo]['n_iter']
    desc   = MODOS[modo]['desc']

    print("\n" + "=" * 80)
    print("   BUSCA DE HIPERPARÂMETROS — Random Forest")
    print(f"   Modo: {modo.upper()} | {desc}")
    print(f"   Scorers: GMean | Clínico | F2-Penalizado")
    print("=" * 80)

    df, target_name = preparar_dados(TARGET)
    dist = df[target_name].value_counts()
    print(f"\n  Dataset: {df.shape[0]} amostras | {df.shape[1]-1} features")
    print(f"  GAD positivos: {dist[1]} ({dist[1]/len(df)*100:.1f}%)\n")

    scorers = [
        (scorer_gmean,  'gmean'),
        (scorer_clinic, 'clinic'),
        (scorer_f2_pen, 'f2_pen'),
    ]

    todos_resultados = {}
    log_path = f'{OUTPUT_DIR}/busca_rf_{modo}.txt'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Progresso global
    _exps_amostra = _definir_experimentos(n_iter, scorer_gmean, 'gmean')
    total_exps = len(scorers) * len(_exps_amostra)
    exp_atual  = 0
    t_inicio   = time.time()

    with open(log_path, 'w') as log:
        log.write("BUSCA DE HIPERPARÂMETROS — Random Forest | GAD\n")
        log.write(f"Modo: {modo} | n_iter={n_iter}\n")
        log.write("=" * 80 + "\n\n")

        for scorer, nome_scorer in scorers:
            exps = _definir_experimentos(n_iter, scorer, nome_scorer)

            print(f"\n{'═'*75}")
            print(f"  SCORER: {nome_scorer.upper()}")
            print(f"{'═'*75}")

            for nome_exp, clf_fn, sampler_fn, param_space in exps:
                chave = f"{nome_exp}_{nome_scorer}"
                exp_atual += 1
                pct       = exp_atual / total_exps * 100
                elapsed   = time.time() - t_inicio
                eta_str   = ""
                if exp_atual > 1:
                    eta_s   = elapsed / (exp_atual - 1) * (total_exps - exp_atual + 1)
                    eta_str = f" | ETA ≈ {eta_s/60:.0f} min"

                print(f"\n  {'─'*65}")
                print(f"  [{chave}]  —  Progresso: {exp_atual}/{total_exps} ({pct:.1f}%){eta_str}")
                print(f"  {'─'*65}")

                t0 = time.time()
                try:
                    metricas, consenso, thresholds, sc = _buscar(
                        clf_factory=clf_fn,
                        sampler_factory=sampler_fn,
                        param_space=param_space,
                        n_iter=n_iter,
                        scoring=scorer,
                        nome_scoring=nome_scorer,
                    )
                    duracao = time.time() - t0
                    t_med   = float(np.mean(thresholds))

                    todos_resultados[chave] = {
                        'metricas': metricas,
                        'consenso': consenso,
                        'thresholds': thresholds,
                        'threshold_medio': t_med,
                        'duracao': duracao,
                        'score_composto': sc,
                    }

                    print(f"\n  Resultado [{chave}] ({duracao/60:.1f} min):")
                    print(f"    Sensitivity: {metricas['sensitivity']:.1f}% ± {metricas['sensitivity_ic']:.1f}%")
                    print(f"    Specificity: {metricas['specificity']:.1f}%")
                    print(f"    F1-Score:    {metricas['f1']:.1f}% ± {metricas['f1_ic']:.1f}%")
                    print(f"    Kappa:       {metricas['kappa']:.3f}")
                    print(f"    Threshold:   {t_med:.3f}")
                    print(f"    Composto:    {sc:.4f}")

                    log.write(f"\n[{chave}] ({duracao/60:.1f} min)\n")
                    log.write(f"  Scorer:       {nome_scorer}\n")
                    log.write(f"  Sensitivity:  {metricas['sensitivity']:.2f}% ± {metricas['sensitivity_ic']:.2f}%\n")
                    log.write(f"  Specificity:  {metricas['specificity']:.2f}%\n")
                    log.write(f"  F1-Score:     {metricas['f1']:.2f}% ± {metricas['f1_ic']:.2f}%\n")
                    log.write(f"  PPV:          {metricas['ppv']:.2f}%\n")
                    log.write(f"  Kappa:        {metricas['kappa']:.4f}\n")
                    log.write(f"  Threshold:    {t_med:.3f}\n")
                    log.write(f"  Composto:     {sc:.4f}\n")
                    log.write(f"  Matriz: VN={metricas['vn']} FP={metricas['fp']} FN={metricas['fn']} VP={metricas['vp']}\n")
                    log.write(f"  Parâmetros (consenso dos 10 folds):\n")
                    for k, v in sorted(consenso.items()):
                        nc = k.replace('clf__','').replace('sampler__','sampler.')
                        log.write(f"    {nc:<28}: {v:.4f}\n" if isinstance(v, float) else f"    {nc:<28}: {v}\n")

                except Exception as e:
                    print(f"  ERRO em {chave}: {e}")
                    import traceback; traceback.print_exc()
                    log.write(f"\n[{chave}] ERRO: {e}\n")

    # ── Ranking final ──────────────────────────────────────────────────────────
    if not todos_resultados:
        print("\nNenhum resultado gerado."); return

    ord_comp = sorted(todos_resultados.items(), key=lambda x: x[1]['score_composto'], reverse=True)
    ord_sens = sorted(todos_resultados.items(), key=lambda x: x[1]['metricas']['sensitivity'], reverse=True)

    print("\n" + "=" * 90)
    print("  RANKING FINAL — Score Composto √(Sens × F1)")
    print("=" * 90)
    print(f"\n  {'Rank':<4} {'Experimento':<38} {'Comp':>7} {'Sens':>12} {'Spec':>8} {'F1':>12} {'Kappa':>7} {'t':>5}")
    print("  " + "─" * 90)
    for rank, (chave, dados) in enumerate(ord_comp, 1):
        m  = dados['metricas']
        t  = dados['threshold_medio']
        sc = dados['score_composto']
        print(f"  {rank:<4} {chave:<38} {sc:>7.4f} "
              f"{m['sensitivity']:.1f}%±{m['sensitivity_ic']:.1f}% "
              f"{m['specificity']:>7.1f}% "
              f"{m['f1']:.1f}%±{m['f1_ic']:.1f}% "
              f"{m['kappa']:>7.3f} "
              f"t≈{t:.2f}")

    # Salvar ranking
    relatorio = f'{OUTPUT_DIR}/ranking_rf_{modo}.txt'
    with open(relatorio, 'w') as f:
        f.write("=" * 95 + "\n")
        f.write(f"RANKING FINAL — Random Forest | GAD | Modo: {modo} | n_iter={n_iter}\n")
        f.write("Score Composto = √(Sens × F1)\n")
        f.write("=" * 95 + "\n\n")

        f.write("─── Ranking por Score Composto ───\n\n")
        hdr = f"{'Rank':<5} {'Experimento':<40} {'Composto':>9} {'Sensitivity':>18} {'F1':>16} {'Spec':>10} {'Kappa':>8} {'T':>6}\n"
        f.write(hdr)
        f.write("─" * 115 + "\n")
        for rank, (chave, dados) in enumerate(ord_comp, 1):
            m  = dados['metricas']
            t  = dados['threshold_medio']
            sc = dados['score_composto']
            sens = f"{m['sensitivity']:.1f}±{m['sensitivity_ic']:.1f}%"
            f1   = f"{m['f1']:.1f}±{m['f1_ic']:.1f}%"
            f.write(f"{rank:<5} {chave:<40} {sc:>9.4f} {sens:>18} {f1:>16} {m['specificity']:>9.1f}% {m['kappa']:>8.3f} {t:>5.2f}\n")

        f.write("\n\n─── Ranking por Sensibilidade (referência) ───\n\n")
        f.write(hdr)
        f.write("─" * 115 + "\n")
        for rank, (chave, dados) in enumerate(ord_sens, 1):
            m  = dados['metricas']
            t  = dados['threshold_medio']
            sc = dados['score_composto']
            sens = f"{m['sensitivity']:.1f}±{m['sensitivity_ic']:.1f}%"
            f1   = f"{m['f1']:.1f}±{m['f1_ic']:.1f}%"
            f.write(f"{rank:<5} {chave:<40} {sc:>9.4f} {sens:>18} {f1:>16} {m['specificity']:>9.1f}% {m['kappa']:>8.3f} {t:>5.2f}\n")

        # Detalhamento
        f.write("\n\n" + "=" * 95 + "\n")
        f.write("DETALHAMENTO — PARÂMETROS ÓTIMOS POR EXPERIMENTO\n")
        f.write("=" * 95 + "\n\n")
        for rank, (chave, dados) in enumerate(ord_comp, 1):
            m  = dados['metricas']
            c  = dados['consenso']
            ts = dados['thresholds']
            f.write(f"[{rank}] {chave}\n")
            f.write(f"  Score Composto: {dados['score_composto']:.4f}\n")
            f.write(f"  Sensitivity:   {m['sensitivity']:.2f}% ± {m['sensitivity_ic']:.2f}%\n")
            f.write(f"  Specificity:   {m['specificity']:.2f}%\n")
            f.write(f"  F1-Score:      {m['f1']:.2f}% ± {m['f1_ic']:.2f}%\n")
            f.write(f"  PPV:           {m['ppv']:.2f}%\n")
            f.write(f"  Kappa:         {m['kappa']:.4f}\n")
            f.write(f"  Matriz: VN={m['vn']} FP={m['fp']} FN={m['fn']} VP={m['vp']}\n")
            f.write(f"  Threshold: média={dados['threshold_medio']:.3f} | min={min(ts):.3f} | max={max(ts):.3f}\n")
            f.write(f"  Parâmetros ótimos (mediana dos 10 folds):\n")
            for k, v in sorted(c.items()):
                nc = k.replace('clf__','').replace('sampler__','sampler.')
                f.write(f"    {nc:<28}: {v:.4f}\n" if isinstance(v, float) else f"    {nc:<28}: {v}\n")
            f.write("\n")

        melhor_chave, melhor_dados = ord_comp[0]
        m  = melhor_dados['metricas']
        t  = melhor_dados['threshold_medio']
        sc = melhor_dados['score_composto']
        f.write("=" * 95 + "\n")
        f.write(f"MELHOR MODELO (Score Composto): {melhor_chave}\n")
        f.write("=" * 95 + "\n\n")
        f.write(f"Score Composto  √(Sens×F1): {sc:.4f}\n")
        f.write(f"Sensibilidade:  {m['sensitivity']:.1f}% ± {m['sensitivity_ic']:.1f}%\n")
        f.write(f"F1-Score:       {m['f1']:.1f}% ± {m['f1_ic']:.1f}%\n")
        f.write(f"Especificidade: {m['specificity']:.1f}%\n")
        f.write(f"Threshold (F2): {t:.3f}\n\n")

    print(f"\n  Log:     {log_path}")
    print(f"  Ranking: {relatorio}")

    # ── Gráficos ───────────────────────────────────────────────────────────────
    top_n = min(18, len(ord_comp))
    top   = ord_comp[:top_n]
    nomes = [c for c, _ in top]
    scs   = [d['score_composto'] for _, d in top]
    senss = [d['metricas']['sensitivity'] for _, d in top]
    f1s   = [d['metricas']['f1'] for _, d in top]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, max(6, top_n * 0.55)))
    y_pos = np.arange(top_n)

    c1 = plt.cm.BuGn(np.linspace(0.2, 0.9, top_n))
    c2 = plt.cm.RdYlGn(np.linspace(0.15, 0.85, top_n))
    c3 = plt.cm.RdYlBu(np.linspace(0.15, 0.85, top_n))

    axes[0].barh(y_pos, [scs[i] for i in range(top_n-1,-1,-1)], color=c1, edgecolor='white')
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels([nomes[i] for i in range(top_n-1,-1,-1)], fontsize=7)
    axes[0].set_xlabel('Score Composto √(Sens×F1)', fontweight='bold')
    axes[0].set_title('Score Composto', fontweight='bold')

    axes[1].barh(y_pos, [senss[i] for i in range(top_n-1,-1,-1)], color=c2, edgecolor='white')
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels([nomes[i] for i in range(top_n-1,-1,-1)], fontsize=7)
    axes[1].axvline(x=50, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Sensibilidade (%)', fontweight='bold')
    axes[1].set_title('Sensibilidade', fontweight='bold')

    axes[2].barh(y_pos, [f1s[i] for i in range(top_n-1,-1,-1)], color=c3, edgecolor='white')
    axes[2].set_yticks(y_pos); axes[2].set_yticklabels([nomes[i] for i in range(top_n-1,-1,-1)], fontsize=7)
    axes[2].set_xlabel('F1-Score (%)', fontweight='bold')
    axes[2].set_title('F1-Score', fontweight='bold')

    plt.suptitle(f'Busca Hiperparâmetros — Random Forest | GAD | {modo} ({n_iter} iter)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    graf_path = f'{PLOTS_DIR}/ranking_rf_{modo}.png'
    plt.savefig(graf_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico: {graf_path}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Busca de hiperparâmetros — Random Forest (GAD).'
    )
    parser.add_argument(
        '--modo', choices=['rapido', 'medio', 'completo'], default='medio',
        help='rapido (~20-40 min) | medio (~2-4h, padrão) | completo (~8-14h)',
    )
    args = parser.parse_args()
    print(f"\n  Modo: {args.modo.upper()} — {MODOS[args.modo]['desc']}")
    print(f"  Classificador: RandomForestClassifier")
    print(f"  Samplers: SMOTEENN | BorderlineSMOTE | SMOTE | NearMiss | RUS | IHT")
    print(f"  Scorers: gmean | clinic | f2_pen")
    print(f"  Experimentos: 6 configs × 3 scorers = 18 buscas")
    print(f"  Resultados: output/busca_rf/\n")

    t_total = time.time()
    main(modo=args.modo)
    print(f"  Tempo total: {(time.time()-t_total)/60:.1f} minutos")
