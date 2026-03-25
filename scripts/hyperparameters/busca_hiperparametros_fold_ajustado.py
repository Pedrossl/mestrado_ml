"""
Busca de Hiperparâmetros — Fold Ajustado (sem data leakage na normalização)
=============================================================================

DIFERENÇA em relação ao busca_hiperparametros_v3.py
----------------------------------------------------
No v3, o MinMaxScaler era aplicado no dataset INTEIRO antes da CV começar,
via carregar_teste_normalizado() → normalizar_minmax(df). Isso é data leakage:
os dados de teste de cada fold "contaminavam" o scaler com seus min/max.

Aqui o MinMaxScaler entra como primeiro passo do ImbPipeline, sendo:
  - fit_transform aplicado APENAS nos dados de treino de cada fold
  - transform aplicado nos dados de teste de cada fold

Os dados são carregados com aplicar_minmax=False, e todas as outras
transformações fixas (binarização de siblings, mapeamento de bio parents,
codificação de sex, remoção de colunas) continuam sendo aplicadas antes da CV
— o que é correto, pois são regras determinísticas sem parâmetros aprendidos.

Referências:
  - Kuhn & Johnson (2013). Applied Predictive Modeling. Springer.
    [Cap. 3 — pré-processamento deve ser estimado apenas no treino]
  - Kaufman et al. (2012). Leakage in Data Mining. TKDD 6(4).
    [definição formal de data leakage em CV]
  - Ke et al. (2017). LightGBM. NIPS 3149-3157.
  - Geurts et al. (2006). ExtraTrees. Machine Learning 63(1), 3-42.
  - Zhang & Mani (2003). NearMiss. KDD Workshop on Imbalanced Data.

COMO RODAR:
  Rápido (~30-60 min): python scripts/hyperparameters/busca_hiperparametros_fold_ajustado.py --modo rapido
  Médio  (~3-6 horas): python scripts/hyperparameters/busca_hiperparametros_fold_ajustado.py --modo medio

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

from scipy.stats import randint, uniform, loguniform
from collections import Counter

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import fbeta_score, confusion_matrix, make_scorer
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, InstanceHardnessThreshold
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.preprocessing.normalizacao import carregar_teste_normalizado
from scripts.utils import calcular_metricas_fold, agregar_metricas_com_ic

warnings.filterwarnings('ignore')

# ─── Configuração ──────────────────────────────────────────────────────────────

MODOS = {
    'rapido':   {'n_iter': 50,  'desc': '~50 combinações | ~30-60 min'},
    'medio':    {'n_iter': 200, 'desc': '~200 combinações | ~3-6 horas'},
    'completo': {'n_iter': 500, 'desc': '~500 combinações | ~12-20 horas'},
}

TARGET     = 'GAD'
OUTPUT_DIR = 'output/busca_hiperparametros_fold_ajustado'
PLOTS_DIR  = f'{OUTPUT_DIR}/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Carregamento sem MinMax ───────────────────────────────────────────────────

def _carregar_dados(target=TARGET):
    """
    Carrega o dataset aplicando todas as transformações fixas (sem MinMax).
    O MinMax será aplicado dentro de cada fold para evitar data leakage.

    Transformações fixas aplicadas aqui (sem parâmetros aprendidos):
      - Number of Siblings → binário (0/1)
      - Number of Bio. Parents → mapeado para 0 / 0.5 / 1
      - Sex → codificado (M=0, F=1)
      - Remoção de colunas não preditivas
      - dropna()
    """
    df = carregar_teste_normalizado(aplicar_minmax=False)

    colunas_remover = [
        'Subject',
        'GAD Probabiliy - Gamma',
        'SAD Probability - Gamma',
        'Sample Weight',
    ]
    if target == 'GAD':
        colunas_remover.append('SAD')
    else:
        colunas_remover.append('GAD')

    df = df.drop(columns=[c for c in colunas_remover if c in df.columns])

    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})

    cols = [c for c in df.columns if c != target] + [target]
    df = df[cols].dropna()

    return df, target

# ─── Scorers inteligentes ──────────────────────────────────────────────────────

def _scorer_gmean(y_true, y_pred):
    """G-Mean = sqrt(Sens x Spec). Impede modelos degenerados (Spec=0%)."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    vn, fp, fn, vp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    sens = vp/(vp+fn) if (vp+fn)>0 else 0.0
    spec = vn/(vn+fp) if (vn+fp)>0 else 0.0
    return np.sqrt(sens * spec)

def _scorer_clinic(y_true, y_pred):
    """Clinico = Sens x F1 x min(Spec/0.5, 1). Penaliza Spec < 50%."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    vn, fp, fn, vp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    sens = vp/(vp+fn) if (vp+fn)>0 else 0.0
    spec = vn/(vn+fp) if (vn+fp)>0 else 0.0
    prec = vp/(vp+fp) if (vp+fp)>0 else 0.0
    f1   = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0.0
    return sens * f1 * min(1.0, spec/0.50)

def _scorer_f2_pen(y_true, y_pred):
    """F2 x penalidade quadratica quando Spec < 30%."""
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

SPACE_XGB = {
    'clf__n_estimators':      randint(50, 500),
    'clf__max_depth':         randint(2, 10),
    'clf__learning_rate':     loguniform(0.001, 0.3),
    'clf__subsample':         uniform(0.5, 0.5),
    'clf__colsample_bytree':  uniform(0.4, 0.6),
    'clf__colsample_bylevel': uniform(0.4, 0.6),
    'clf__min_child_weight':  randint(1, 15),
    'clf__gamma':             loguniform(1e-4, 2.0),
    'clf__reg_lambda':        loguniform(0.1, 10.0),
    'clf__reg_alpha':         loguniform(1e-4, 2.0),
    'clf__scale_pos_weight':  loguniform(1.0, 20.0),
}

SPACE_HISTGBM = {
    'clf__max_iter':          randint(50, 400),
    'clf__max_depth':         randint(2, 10),
    'clf__learning_rate':     loguniform(0.001, 0.3),
    'clf__min_samples_leaf':  randint(5, 50),
    'clf__max_leaf_nodes':    randint(10, 63),
    'clf__l2_regularization': loguniform(1e-4, 1.0),
    'clf__max_bins':          randint(64, 255),
    'clf__class_weight':      [None, 'balanced'],
}

SPACE_ET = {
    'clf__n_estimators':      randint(50, 400),
    'clf__max_depth':         randint(3, 20),
    'clf__min_samples_split': randint(2, 20),
    'clf__min_samples_leaf':  randint(1, 15),
    'clf__max_features':      ['sqrt', 'log2', 0.3, 0.5, 0.7],
    'clf__class_weight':      ['balanced', 'balanced_subsample', None],
}

SPACE_SAMPLER_SMOTE      = {'sampler__k_neighbors': randint(2, 10)}
SPACE_SAMPLER_BORDERLINE = {'sampler__k_neighbors': randint(2, 10),
                             'sampler__m_neighbors': randint(5, 20)}
SPACE_SAMPLER_RUS        = {'sampler__sampling_strategy': uniform(0.3, 0.7)}
SPACE_SAMPLER_NEARMISS   = {'sampler__n_neighbors': randint(2, 7)}
SPACE_SAMPLER_IHT        = {}

# ─── Funções auxiliares ────────────────────────────────────────────────────────

def _combinar(*spaces):
    result = {}
    for s in spaces:
        result.update(s)
    return result

def _extrair_consenso(lista_params):
    consenso = {}
    for key in lista_params[0].keys():
        vals = [p[key] for p in lista_params]
        all_numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals)
        consenso[key] = float(np.median(vals)) if all_numeric else Counter(vals).most_common(1)[0][0]
    return consenso

def _construir_pipeline(clf, sampler):
    """
    Pipeline: MinMaxScaler (fitado só no treino) → sampler → clf.
    O scaler entra sempre aqui, ao contrário do v3 onde era pré-aplicado.
    """
    return ImbPipeline([
        ('scaler',  MinMaxScaler()),
        ('sampler', sampler),
        ('clf',     clf),
    ])

# ─── Busca principal ──────────────────────────────────────────────────────────

# Espaço do scaler (sem hiperparâmetros — MinMax não tem parâmetros tunaveis)
SPACE_SCALER = {}

def _buscar(clf_factory, sampler_factory, param_space, n_iter, scoring):
    df, target_name = _carregar_dados(TARGET)
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

        # MinMaxScaler é fitado apenas em X_tr dentro do pipeline
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
        print(f"    Fold {fold_idx+1:2d}/10 -> "
              f"Sens={m['sensitivity']:.1f}%  "
              f"Spec={m['specificity']:.1f}%  "
              f"F1={m['f1']:.1f}%  "
              f"t~{melhor_t:.2f}")

    metricas = agregar_metricas_com_ic(metricas_folds)
    sc = float(np.sqrt((metricas['sensitivity']/100) * (metricas['f1']/100)))
    return metricas, _extrair_consenso(params_folds), thresholds, sc

# ─── Definição dos experimentos ───────────────────────────────────────────────

def _definir_experimentos(n_iter, scorer):
    def xgb():      return XGBClassifier(random_state=42, use_label_encoder=False,
                                         eval_metric='logloss', verbosity=0, n_jobs=1)
    def hist():     return HistGradientBoostingClassifier(random_state=42)
    def et():       return ExtraTreesClassifier(random_state=42, n_jobs=1)

    def smoteenn(): return SMOTEENN(random_state=42)
    def blsmote():  return BorderlineSMOTE(random_state=42)
    def smote():    return SMOTE(random_state=42)
    def rus():      return RandomUnderSampler(random_state=42)
    def nearmiss(): return NearMiss(version=1)
    def iht():      return InstanceHardnessThreshold(random_state=42, cv=3)

    return [
        # ── XGBoost ───────────────────────────────────────────────────────
        ('XGB_SMOTEENN',        xgb, smoteenn, SPACE_XGB),
        ('XGB_BorderlineSMOTE', xgb, blsmote,  _combinar(SPACE_XGB, SPACE_SAMPLER_BORDERLINE)),
        ('XGB_SMOTE',           xgb, smote,    _combinar(SPACE_XGB, SPACE_SAMPLER_SMOTE)),
        ('XGB_NearMiss',        xgb, nearmiss, _combinar(SPACE_XGB, SPACE_SAMPLER_NEARMISS)),
        ('XGB_RUS',             xgb, rus,      _combinar(SPACE_XGB, SPACE_SAMPLER_RUS)),
        ('XGB_IHT',             xgb, iht,      SPACE_XGB),
        # ── HistGradientBoosting ──────────────────────────────────────────
        ('HistGBM_SMOTEENN',        hist, smoteenn, SPACE_HISTGBM),
        ('HistGBM_BorderlineSMOTE', hist, blsmote,  _combinar(SPACE_HISTGBM, SPACE_SAMPLER_BORDERLINE)),
        ('HistGBM_SMOTE',           hist, smote,    _combinar(SPACE_HISTGBM, SPACE_SAMPLER_SMOTE)),
        ('HistGBM_NearMiss',        hist, nearmiss, _combinar(SPACE_HISTGBM, SPACE_SAMPLER_NEARMISS)),
        ('HistGBM_RUS',             hist, rus,      _combinar(SPACE_HISTGBM, SPACE_SAMPLER_RUS)),
        # ── ExtraTrees ────────────────────────────────────────────────────
        ('ET_SMOTEENN',        et, smoteenn, SPACE_ET),
        ('ET_BorderlineSMOTE', et, blsmote,  _combinar(SPACE_ET, SPACE_SAMPLER_BORDERLINE)),
        ('ET_SMOTE',           et, smote,    _combinar(SPACE_ET, SPACE_SAMPLER_SMOTE)),
        ('ET_NearMiss',        et, nearmiss, _combinar(SPACE_ET, SPACE_SAMPLER_NEARMISS)),
        ('ET_RUS',             et, rus,      _combinar(SPACE_ET, SPACE_SAMPLER_RUS)),
    ]

# ─── Script principal ─────────────────────────────────────────────────────────

def main(modo='medio'):
    n_iter = MODOS[modo]['n_iter']
    desc   = MODOS[modo]['desc']

    print("\n" + "=" * 80)
    print("   BUSCA DE HIPERPARAMETROS — Fold Ajustado (MinMax dentro do fold)")
    print(f"   Modo: {modo.upper()} | {desc}")
    print(f"   Correcao: MinMaxScaler fitado apenas no treino de cada fold")
    print(f"   Scorers: GMean | Clinico | F2-Penalizado")
    print("=" * 80)

    df, target_name = _carregar_dados(TARGET)
    dist = df[target_name].value_counts()
    print(f"\n  Dataset: {df.shape[0]} amostras | {df.shape[1]-1} features")
    print(f"  GAD positivos: {dist[1]} ({dist[1]/len(df)*100:.1f}%)\n")

    scorers = [
        (scorer_gmean,  'gmean'),
        (scorer_clinic, 'clinic'),
        (scorer_f2_pen, 'f2_pen'),
    ]

    todos_resultados = {}
    log_path = f'{OUTPUT_DIR}/busca_{modo}_fold_ajustado.txt'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exps_amostra = _definir_experimentos(n_iter, scorer_gmean)
    total_exps   = len(scorers) * len(exps_amostra)
    exp_atual    = 0
    t_inicio     = time.time()

    with open(log_path, 'w') as log:
        log.write("BUSCA DE HIPERPARAMETROS — Fold Ajustado (sem leakage na normalizacao)\n")
        log.write(f"Modo: {modo} | n_iter={n_iter}\n")
        log.write("Correcao: MinMaxScaler aplicado dentro de cada fold (fit apenas no treino)\n")
        log.write("=" * 80 + "\n\n")

        for scorer, nome_scorer in scorers:
            exps = _definir_experimentos(n_iter, scorer)

            print(f"\n{'='*75}")
            print(f"  SCORER: {nome_scorer.upper()}")
            print(f"{'='*75}")

            for nome_exp, clf_fn, sampler_fn, param_space in exps:
                chave     = f"{nome_exp}_{nome_scorer}"
                exp_atual += 1
                pct        = exp_atual / total_exps * 100
                elapsed    = time.time() - t_inicio
                eta_str    = ""
                if exp_atual > 1:
                    eta_s   = elapsed / (exp_atual - 1) * (total_exps - exp_atual + 1)
                    eta_str = f" | ETA ~{eta_s/60:.0f} min"

                print(f"\n  [{chave}]  —  {exp_atual}/{total_exps} ({pct:.1f}%){eta_str}")
                print(f"  {'─'*65}")

                t0 = time.time()
                try:
                    metricas, consenso, thresholds, sc = _buscar(
                        clf_factory=clf_fn,
                        sampler_factory=sampler_fn,
                        param_space=param_space,
                        n_iter=n_iter,
                        scoring=scorer,
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
                    print(f"    Sensitivity: {metricas['sensitivity']:.1f}% +- {metricas['sensitivity_ic']:.1f}%")
                    print(f"    Specificity: {metricas['specificity']:.1f}%")
                    print(f"    F1-Score:    {metricas['f1']:.1f}% +- {metricas['f1_ic']:.1f}%")
                    print(f"    Kappa:       {metricas['kappa']:.3f}")
                    print(f"    Threshold:   {t_med:.3f}")
                    print(f"    Composto:    {sc:.4f}")

                    log.write(f"\n[{chave}] ({duracao/60:.1f} min)\n")
                    log.write(f"  Scorer:      {nome_scorer}\n")
                    log.write(f"  Sensitivity: {metricas['sensitivity']:.2f}% +- {metricas['sensitivity_ic']:.2f}%\n")
                    log.write(f"  Specificity: {metricas['specificity']:.2f}%\n")
                    log.write(f"  F1-Score:    {metricas['f1']:.2f}% +- {metricas['f1_ic']:.2f}%\n")
                    log.write(f"  PPV:         {metricas['ppv']:.2f}%\n")
                    log.write(f"  Kappa:       {metricas['kappa']:.4f}\n")
                    log.write(f"  Threshold:   {t_med:.3f}\n")
                    log.write(f"  Composto:    {sc:.4f}\n")
                    log.write(f"  Matriz: VN={metricas['vn']} FP={metricas['fp']} FN={metricas['fn']} VP={metricas['vp']}\n")
                    log.write(f"  Parametros (consenso dos 10 folds):\n")
                    for k, v in sorted(consenso.items()):
                        nc = k.replace('clf__','').replace('sampler__','sampler.').replace('scaler__','scaler.')
                        log.write(f"    {nc:<28}: {v:.4f}\n" if isinstance(v, float) else f"    {nc:<28}: {v}\n")

                except Exception as e:
                    print(f"  ERRO em {chave}: {e}")
                    import traceback; traceback.print_exc()
                    log.write(f"\n[{chave}] ERRO: {e}\n")

    if not todos_resultados:
        print("\nNenhum resultado gerado."); return

    ord_comp = sorted(todos_resultados.items(), key=lambda x: x[1]['score_composto'], reverse=True)
    ord_sens = sorted(todos_resultados.items(), key=lambda x: x[1]['metricas']['sensitivity'], reverse=True)

    print("\n" + "=" * 90)
    print("  RANKING FINAL — Score Composto sqrt(Sens x F1)")
    print("=" * 90)
    print(f"\n  {'Rank':<4} {'Experimento':<42} {'Comp':>7} {'Sens':>12} {'Spec':>8} {'F1':>12} {'Kappa':>7} {'t':>5}")
    print("  " + "─" * 95)
    for rank, (chave, dados) in enumerate(ord_comp[:20], 1):
        m  = dados['metricas']
        t  = dados['threshold_medio']
        sc = dados['score_composto']
        print(f"  {rank:<4} {chave:<42} {sc:>7.4f} "
              f"{m['sensitivity']:.1f}%+-{m['sensitivity_ic']:.1f}% "
              f"{m['specificity']:>7.1f}% "
              f"{m['f1']:.1f}%+-{m['f1_ic']:.1f}% "
              f"{m['kappa']:>7.3f} "
              f"t~{t:.2f}")

    relatorio = f'{OUTPUT_DIR}/ranking_final_{modo}_fold_ajustado.txt'
    with open(relatorio, 'w') as f:
        f.write("=" * 95 + "\n")
        f.write(f"RANKING FINAL — Fold Ajustado | GAD | Modo: {modo} | n_iter={n_iter}\n")
        f.write("MinMaxScaler fitado apenas no treino de cada fold (sem data leakage)\n")
        f.write("Score Composto = sqrt(Sens x F1)\n")
        f.write("=" * 95 + "\n\n")

        f.write("─── Ranking por Score Composto ───\n\n")
        hdr = f"{'Rank':<5} {'Experimento':<45} {'Composto':>9} {'Sensitivity':>18} {'F1':>16} {'Spec':>10} {'Kappa':>8} {'T':>6}\n"
        f.write(hdr)
        f.write("─" * 120 + "\n")
        for rank, (chave, dados) in enumerate(ord_comp, 1):
            m  = dados['metricas']
            t  = dados['threshold_medio']
            sc = dados['score_composto']
            f.write(f"{rank:<5} {chave:<45} {sc:>9.4f} "
                    f"{m['sensitivity']:.1f}+-{m['sensitivity_ic']:.1f}%  "
                    f"{m['f1']:.1f}+-{m['f1_ic']:.1f}%  "
                    f"{m['specificity']:>8.1f}%  "
                    f"{m['kappa']:>8.3f}  {t:>5.2f}\n")

        f.write("\n\n─── Ranking por Sensibilidade (referência) ───\n\n")
        f.write(hdr)
        f.write("─" * 120 + "\n")
        for rank, (chave, dados) in enumerate(ord_sens, 1):
            m  = dados['metricas']
            t  = dados['threshold_medio']
            sc = dados['score_composto']
            f.write(f"{rank:<5} {chave:<45} {sc:>9.4f} "
                    f"{m['sensitivity']:.1f}+-{m['sensitivity_ic']:.1f}%  "
                    f"{m['f1']:.1f}+-{m['f1_ic']:.1f}%  "
                    f"{m['specificity']:>8.1f}%  "
                    f"{m['kappa']:>8.3f}  {t:>5.2f}\n")

        f.write("\n\n" + "=" * 95 + "\n")
        f.write("DETALHAMENTO — PARAMETROS OTIMOS POR EXPERIMENTO\n")
        f.write("=" * 95 + "\n\n")
        for rank, (chave, dados) in enumerate(ord_comp, 1):
            m  = dados['metricas']
            c  = dados['consenso']
            ts = dados['thresholds']
            f.write(f"[{rank}] {chave}\n")
            f.write(f"  Score Composto: {dados['score_composto']:.4f}\n")
            f.write(f"  Sensitivity:   {m['sensitivity']:.2f}% +- {m['sensitivity_ic']:.2f}%\n")
            f.write(f"  Specificity:   {m['specificity']:.2f}%\n")
            f.write(f"  F1-Score:      {m['f1']:.2f}% +- {m['f1_ic']:.2f}%\n")
            f.write(f"  PPV:           {m['ppv']:.2f}%\n")
            f.write(f"  Kappa:         {m['kappa']:.4f}\n")
            f.write(f"  Matriz: VN={m['vn']} FP={m['fp']} FN={m['fn']} VP={m['vp']}\n")
            f.write(f"  Threshold: media={dados['threshold_medio']:.3f} | min={min(ts):.3f} | max={max(ts):.3f}\n")
            f.write(f"  Parametros otimos (mediana dos 10 folds):\n")
            for k, v in sorted(c.items()):
                nc = k.replace('clf__','').replace('sampler__','sampler.').replace('scaler__','scaler.')
                f.write(f"    {nc:<28}: {v:.4f}\n" if isinstance(v, float) else f"    {nc:<28}: {v}\n")
            f.write("\n")

    print(f"\n  Log:     {log_path}")
    print(f"  Ranking: {relatorio}")

    # ── Gráfico ────────────────────────────────────────────────────────────────
    top_n = min(20, len(ord_comp))
    top   = ord_comp[:top_n]
    nomes = [c for c, _ in top]
    scs   = [d['score_composto']              for _, d in top]
    senss = [d['metricas']['sensitivity']     for _, d in top]
    f1s   = [d['metricas']['f1']              for _, d in top]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, max(6, top_n * 0.55)))
    y_pos = np.arange(top_n)
    rev   = list(range(top_n - 1, -1, -1))

    c1 = plt.cm.BuGn(np.linspace(0.2, 0.9, top_n))
    c2 = plt.cm.RdYlGn(np.linspace(0.15, 0.85, top_n))
    c3 = plt.cm.RdYlBu(np.linspace(0.15, 0.85, top_n))

    axes[0].barh(y_pos, [scs[i]   for i in rev], color=c1, edgecolor='white')
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels([nomes[i] for i in rev], fontsize=6)
    axes[0].set_xlabel('Score Composto sqrt(Sens x F1)', fontweight='bold')
    axes[0].set_title('Score Composto', fontweight='bold')

    axes[1].barh(y_pos, [senss[i] for i in rev], color=c2, edgecolor='white')
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels([nomes[i] for i in rev], fontsize=6)
    axes[1].axvline(x=50, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Sensibilidade (%)', fontweight='bold')
    axes[1].set_title('Sensibilidade', fontweight='bold')

    axes[2].barh(y_pos, [f1s[i]   for i in rev], color=c3, edgecolor='white')
    axes[2].set_yticks(y_pos); axes[2].set_yticklabels([nomes[i] for i in rev], fontsize=6)
    axes[2].set_xlabel('F1-Score (%)', fontweight='bold')
    axes[2].set_title('F1-Score', fontweight='bold')

    plt.suptitle(f'Busca Hiperparametros — Fold Ajustado | GAD | {modo} ({n_iter} iter)\n'
                 f'MinMaxScaler fitado apenas no treino de cada fold',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    graf_path = f'{PLOTS_DIR}/ranking_{modo}_fold_ajustado.png'
    plt.savefig(graf_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Grafico: {graf_path}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Busca de hiperparametros com MinMax corrigido para dentro do fold.'
    )
    parser.add_argument(
        '--modo', choices=['rapido', 'medio', 'completo'], default='medio',
        help='rapido (~30-60 min) | medio (~3-6h, padrao) | completo (~12-20h)',
    )
    args = parser.parse_args()
    print(f"\n  Modo: {args.modo.upper()} — {MODOS[args.modo]['desc']}")
    print(f"  MinMaxScaler: fitado apenas no treino de cada fold (sem data leakage)")
    print(f"  Scorers: gmean | clinic | f2_pen")
    print(f"  Experimentos: 16 configs x 3 scorers = 48 buscas")
    print(f"  Resultados: {OUTPUT_DIR}/\n")

    t_total = time.time()
    main(modo=args.modo)
    print(f"  Tempo total: {(time.time()-t_total)/60:.1f} minutos")
