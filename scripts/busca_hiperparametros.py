"""
Busca Exaustiva de Hiperparâmetros — Maximizar Sensibilidade + F1 para GAD
===========================================================================

Usa RandomizedSearchCV com nested cross-validation para explorar um
espaço enorme de combinações de hiperparâmetros do XGBoost + técnicas de
balanceamento.

PROBLEMA CORRIGIDO na v2:
  Na v1, otimizar puramente por 'recall' produzia modelos degenerados:
  o algoritmo aprendia que prever TUDO como positivo maximizava recall
  (100% Sensibilidade, 0% Especificidade, F1 ~26%). Clinicamente inútil.

SOLUÇÃO v2 — Três novos scorers inteligentes:
  1. G-Mean  (√(Sens × Spec)): força equilíbrio matemático entre as duas classes
  2. Clínico (Sens × F1, penaliza Spec < 50%): captura o dilema clínico
  3. F2 Penalizado (F2 × penalidade se Spec < 30%): F2 mais robusto ao colapso

Adicionalmente, o threshold de decisão é otimizado (via F2) APÓS encontrar
os melhores hiperparâmetros — separando as duas etapas de tuning.

Referências:
  - Kubat & Matwin (1997). Addressing the Curse of Imbalancedness.
    ECML. [G-Mean como métrica para dados desbalanceados]
  - Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization.
    JMLR 13, 281-305.
  - Varma & Simon (2006). Bias in error estimation when using cross-validation.
    BMC Bioinformatics 7, 91.

COMO RODAR:
  Rápido  (~10-30 min): python scripts/busca_hiperparametros.py --modo rapido
  Médio   (~2-4 horas): python scripts/busca_hiperparametros.py --modo medio
  Completo (~8-12h):    python scripts/busca_hiperparametros.py --modo completo

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

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    fbeta_score, f1_score, make_scorer,
    confusion_matrix, recall_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    preparar_dados, calcular_metricas_fold,
    agregar_metricas_com_ic, calcular_ic,
)

warnings.filterwarnings('ignore')

# ─── Configuração de modos ──────────────────────────────────────────────────────

MODOS = {
    'rapido':   {'n_iter': 50,   'desc': '~50 combinações   | ~10-30 min'},
    'medio':    {'n_iter': 200,  'desc': '~200 combinações  | ~2-4 horas'},
    'completo': {'n_iter': 500,  'desc': '~500 combinações  | ~8-12 horas'},
}

TARGET = 'GAD'
OUTPUT_DIR = 'output/busca_hiperparametros'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Scorers customizados ──────────────────────────────────────────────────────

def _scorer_gmean(y_true, y_pred):
    """
    G-Mean = √(Sensibilidade × Especificidade).

    Força equilíbrio entre as duas classes. Valor máximo = 1.0 apenas quando
    ambas as taxas são altas. Um modelo que prevê tudo como positivo terá
    G-Mean = √(1.0 × 0.0) = 0.0 — eliminando modelos degenerados.

    Ref: Kubat & Matwin (1997). Necessitating classifiers for skewed classes.
    ECML, 179-182.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    vn, fp, fn, vp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    spec = vn / (vn + fp) if (vn + fp) > 0 else 0.0
    return np.sqrt(sens * spec)


def _scorer_clinic(y_true, y_pred):
    """
    Scorer Clínico = Sens × F1, com penalidade multiplicativa quando Spec < 50%.

    Racional clínico: em saúde mental, queremos alta sensibilidade (não perder
    casos) e boa F1 (poucos falsos alarmes). A penalidade garante que o modelo
    não degere para "classificar tudo como positivo".

    Penalidade: min(1.0, Spec / 0.5) — cresce linearmente de 0 a 1 enquanto
    a especificidade vai de 0% a 50%.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    vn, fp, fn, vp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    spec = vn / (vn + fp) if (vn + fp) > 0 else 0.0

    prec = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    # Penalidade proporcional à especificidade (zera se Spec=0%)
    penalidade = min(1.0, spec / 0.50)
    return sens * f1 * penalidade


def _scorer_f2_penalizado(y_true, y_pred):
    """
    F2-Score com penalidade quando Especificidade < 30%.

    O F2 normal (beta=2) já prioriza recall 4× sobre precisão. Adicionamos
    uma penalidade quadrática quando a especificidade é muito baixa, evitando
    modelos que "tudo predizem como positivo".

    Penalidade: min(1.0, (Spec/0.30)^2).
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    vn, fp, fn, vp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    spec = vn / (vn + fp) if (vn + fp) > 0 else 0.0
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    penalidade = min(1.0, (spec / 0.30) ** 2)
    return f2 * penalidade


# ─── Espaço de busca ──────────────────────────────────────────────────────────

PARAM_SPACE_XGBOOST = {
    'clf__n_estimators':      randint(50, 600),
    'clf__max_depth':         randint(2, 10),
    'clf__learning_rate':     loguniform(0.001, 0.4),
    'clf__subsample':         uniform(0.5, 0.5),
    'clf__colsample_bytree':  uniform(0.4, 0.6),
    'clf__colsample_bylevel': uniform(0.4, 0.6),
    'clf__min_child_weight':  randint(1, 15),
    'clf__gamma':             loguniform(1e-4, 2.0),
    'clf__reg_lambda':        loguniform(0.1, 10.0),
    'clf__reg_alpha':         loguniform(1e-4, 2.0),
    # scale_pos_weight mais controlado: 1x a 20x.
    # Valores muito altos (>20) tendem a colapsar para "prever tudo positivo"
    'clf__scale_pos_weight':  loguniform(1.0, 20.0),
}

PARAM_SPACE_SMOTE = {
    **PARAM_SPACE_XGBOOST,
    'sampler__k_neighbors': randint(2, 10),
}

PARAM_SPACE_ADASYN = {
    **PARAM_SPACE_XGBOOST,
    'sampler__n_neighbors': randint(2, 8),
}

PARAM_SPACE_BORDERLINE = {
    **PARAM_SPACE_XGBOOST,
    'sampler__k_neighbors': randint(2, 10),
    'sampler__m_neighbors': randint(5, 20),
}

PARAM_SPACE_SMOTEENN = {**PARAM_SPACE_XGBOOST}
PARAM_SPACE_SMOTETOMEK = {**PARAM_SPACE_XGBOOST}


# ─── Função principal de busca ─────────────────────────────────────────────────

def buscar_hiperparametros(sampler, param_space, nome_sampler, n_iter,
                           scoring, nome_scoring, otimizar_threshold=True):
    """
    Executa RandomizedSearchCV com nested CV.

    Outer CV (10-fold): estimativa não-viesada do desempenho real
    Inner CV (5-fold):  seleciona hiperparâmetros (dentro de cada fold de treino)

    Opcional: após encontrar os melhores hiperparâmetros, otimiza o threshold
    de decisão via F2 nos dados de treino (sem data leakage).
    """
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)

    pipeline = ImbPipeline([
        ('sampler', sampler),
        ('clf', XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=1,
        )),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    metricas_folds = []
    melhores_params_folds = []
    thresholds_usados = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        search.fit(X_tr, y_tr)
        melhores_params_folds.append(search.best_params_)

        # Threshold tuning pós-busca: otimizar F2 nos dados de TREINO (sem leakage)
        if otimizar_threshold and hasattr(search.best_estimator_, 'predict_proba'):
            y_prob_tr = search.best_estimator_.predict_proba(X_tr)[:, 1]
            melhor_t, melhor_f2 = 0.5, 0.0
            for t in np.arange(0.05, 0.95, 0.01):
                y_pred_t = (y_prob_tr >= t).astype(int)
                score = fbeta_score(y_tr, y_pred_t, beta=2, zero_division=0)
                if score > melhor_f2:
                    melhor_f2, melhor_t = score, t
            thresholds_usados.append(melhor_t)

            y_prob_te = search.best_estimator_.predict_proba(X_te)[:, 1]
            y_pred = (y_prob_te >= melhor_t).astype(int)
        else:
            thresholds_usados.append(0.5)
            y_pred = search.predict(X_te)

        metricas_folds.append(calcular_metricas_fold(y_te, y_pred))

        m = metricas_folds[-1]
        print(f"    Fold {fold_idx+1:2d}/10 → "
              f"Sens={m['sensitivity']:.1f}%  "
              f"Spec={m['specificity']:.1f}%  "
              f"F1={m['f1']:.1f}%  "
              f"t≈{thresholds_usados[-1]:.2f}  "
              f"depth={search.best_params_.get('clf__max_depth','?')} "
              f"spw={search.best_params_.get('clf__scale_pos_weight', 0.0):.1f}")

    return agregar_metricas_com_ic(metricas_folds), melhores_params_folds, thresholds_usados


def _extrair_consenso_params(lista_params):
    """Retorna a mediana/moda dos parâmetros entre os folds."""
    consenso = {}
    for key in lista_params[0].keys():
        valores = [p[key] for p in lista_params]
        if isinstance(valores[0], (int, float, np.integer, np.floating)):
            consenso[key] = float(np.median(valores))
        else:
            from collections import Counter
            consenso[key] = Counter(valores).most_common(1)[0][0]
    return consenso


# ─── Script principal ─────────────────────────────────────────────────────────

def main(modo='medio'):
    n_iter = MODOS[modo]['n_iter']
    desc   = MODOS[modo]['desc']

    print("\n" + "=" * 75)
    print("   BUSCA DE HIPERPARÂMETROS v2 — Maximizar Sensibilidade + F1 (GAD)")
    print(f"   Modo: {modo.upper()} | {desc}")
    print(f"   Algoritmo: XGBoost | RandomizedSearchCV + Nested 10-fold CV")
    print(f"   Scorers: GMean, Clínico, F2-Penalizado (sem colapso recall)")
    print("=" * 75)

    df, target_name = preparar_dados(TARGET)
    dist = df[target_name].value_counts()
    print(f"\n  Dataset: {df.shape[0]} amostras | {df.shape[1]-1} features")
    print(f"  GAD positivos: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"\n  Cada configuração: {n_iter} combinações × 5-fold inner × 10-fold outer")
    print()

    # ── Scorers ──────────────────────────────────────────────────────────────
    scorer_gmean    = make_scorer(_scorer_gmean)
    scorer_clinic   = make_scorer(_scorer_clinic)
    scorer_f2_pen   = make_scorer(_scorer_f2_penalizado)
    scorer_f2_orig  = make_scorer(fbeta_score, beta=2, zero_division=0)

    # ── Experimentos ─────────────────────────────────────────────────────────
    # Tupla: (sampler_class, param_space, nome, scorer, nome_scoring)
    experimentos = [
        # --- Scorer GMean (força equilíbrio Sens × Spec) ---
        (SMOTE,            PARAM_SPACE_SMOTE,       'SMOTE',           scorer_gmean,   'gmean'),
        (BorderlineSMOTE,  PARAM_SPACE_BORDERLINE,  'BorderlineSMOTE', scorer_gmean,   'gmean'),
        (SMOTEENN,         PARAM_SPACE_SMOTEENN,    'SMOTEENN',        scorer_gmean,   'gmean'),
        (ADASYN,           PARAM_SPACE_ADASYN,      'ADASYN',          scorer_gmean,   'gmean'),
        (SMOTETomek,       PARAM_SPACE_SMOTETOMEK,  'SMOTETomek',      scorer_gmean,   'gmean'),

        # --- Scorer Clínico (Sens × F1, penaliza Spec < 50%) ---
        (SMOTE,            PARAM_SPACE_SMOTE,       'SMOTE',           scorer_clinic,  'clinic'),
        (BorderlineSMOTE,  PARAM_SPACE_BORDERLINE,  'BorderlineSMOTE', scorer_clinic,  'clinic'),
        (SMOTEENN,         PARAM_SPACE_SMOTEENN,    'SMOTEENN',        scorer_clinic,  'clinic'),

        # --- Scorer F2 Penalizado (melhora o F2 original) ---
        (SMOTE,            PARAM_SPACE_SMOTE,       'SMOTE',           scorer_f2_pen,  'f2_pen'),
        (BorderlineSMOTE,  PARAM_SPACE_BORDERLINE,  'BorderlineSMOTE', scorer_f2_pen,  'f2_pen'),
        (SMOTEENN,         PARAM_SPACE_SMOTEENN,    'SMOTEENN',        scorer_f2_pen,  'f2_pen'),
        (SMOTETomek,       PARAM_SPACE_SMOTETOMEK,  'SMOTETomek',      scorer_f2_pen,  'f2_pen'),

        # --- F2 original mantido para comparação ---
        (SMOTE,            PARAM_SPACE_SMOTE,       'SMOTE',           scorer_f2_orig,  'f2'),
        (BorderlineSMOTE,  PARAM_SPACE_BORDERLINE,  'BorderlineSMOTE', scorer_f2_orig,  'f2'),
    ]

    todos_resultados = {}
    log_path = f'{OUTPUT_DIR}/busca_{modo}_v2.txt'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(log_path, 'w') as log:
        log.write(f"BUSCA DE HIPERPARÂMETROS v2 — GAD | Modo: {modo} | n_iter={n_iter}\n")
        log.write("Scorers: gmean=G-Mean, clinic=Clínico, f2_pen=F2-Penalizado, f2=F2-Original\n")
        log.write("Threshold: otimizado via F2 nos dados de treino de cada fold\n")
        log.write("=" * 75 + "\n\n")

        for sampler_cls, param_space, nome_sampler, scoring, nome_scoring in experimentos:
            chave = f"{nome_sampler}_{nome_scoring}"

            print(f"\n  {'─' * 65}")
            print(f"  [{chave}] Buscando {n_iter} combinações | scorer={nome_scoring}")
            print(f"  {'─' * 65}")

            t0 = time.time()
            try:
                metricas, params_folds, thresholds = buscar_hiperparametros(
                    sampler=sampler_cls(random_state=42),
                    param_space=param_space,
                    nome_sampler=nome_sampler,
                    n_iter=n_iter,
                    scoring=scoring,
                    nome_scoring=nome_scoring,
                    otimizar_threshold=True,
                )
                duracao = time.time() - t0
                consenso = _extrair_consenso_params(params_folds)
                t_medio = float(np.mean(thresholds))

                # Score composto = √(Sens × F1) — para ranking equilibrado
                score_composto = np.sqrt(
                    (metricas['sensitivity'] / 100) * (metricas['f1'] / 100)
                )

                todos_resultados[chave] = {
                    'metricas': metricas,
                    'consenso': consenso,
                    'params_folds': params_folds,
                    'thresholds': thresholds,
                    'threshold_medio': t_medio,
                    'duracao': duracao,
                    'score_composto': score_composto,
                }

                print(f"\n  Resultado [{chave}] ({duracao/60:.1f} min):")
                print(f"    Sensitivity:    {metricas['sensitivity']:.1f}% ± {metricas['sensitivity_ic']:.1f}%")
                print(f"    Specificity:    {metricas['specificity']:.1f}%")
                print(f"    F1-Score:       {metricas['f1']:.1f}% ± {metricas['f1_ic']:.1f}%")
                print(f"    Kappa:          {metricas['kappa']:.3f}")
                print(f"    Threshold(med): {t_medio:.3f}")
                print(f"    Score Composto: {score_composto:.4f} (√(Sens×F1))")

                log.write(f"\n[{chave}] ({duracao/60:.1f} min)\n")
                log.write(f"  Scorer usado:   {nome_scoring}\n")
                log.write(f"  Sensitivity:    {metricas['sensitivity']:.2f}% ± {metricas['sensitivity_ic']:.2f}%\n")
                log.write(f"  Specificity:    {metricas['specificity']:.2f}%\n")
                log.write(f"  F1-Score:       {metricas['f1']:.2f}% ± {metricas['f1_ic']:.2f}%\n")
                log.write(f"  PPV:            {metricas['ppv']:.2f}%\n")
                log.write(f"  Kappa:          {metricas['kappa']:.4f}\n")
                log.write(f"  Threshold(med): {t_medio:.3f}\n")
                log.write(f"  Score Composto: {score_composto:.4f}\n")
                log.write(f"  Matriz: VN={metricas['vn']} FP={metricas['fp']} FN={metricas['fn']} VP={metricas['vp']}\n")
                log.write(f"  Parâmetros (consenso dos 10 folds):\n")
                for k, v in sorted(consenso.items()):
                    nome_curto = k.replace('clf__', '').replace('sampler__', 'sampler.')
                    log.write(f"    {nome_curto:<25}: {v:.4f}\n" if isinstance(v, float) else f"    {nome_curto:<25}: {v}\n")

            except Exception as e:
                print(f"  ERRO em {chave}: {e}")
                import traceback; traceback.print_exc()
                log.write(f"\n[{chave}] ERRO: {e}\n")

    # ── Relatório final ────────────────────────────────────────────────────────

    if not todos_resultados:
        print("\nNenhum resultado gerado.")
        return

    # Ordenar por Score Composto (√(Sens × F1)) — equilíbrio clínico ideal
    ordenados_composto = sorted(
        todos_resultados.items(),
        key=lambda x: x[1]['score_composto'],
        reverse=True,
    )

    # Ordenar por Sensibilidade (para referência)
    ordenados_sens = sorted(
        todos_resultados.items(),
        key=lambda x: x[1]['metricas']['sensitivity'],
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("  RANKING FINAL — ORDENADO POR SCORE COMPOSTO √(Sens × F1)")
    print("  (equilíbrio entre sensibilidade e F1 — rejeita modelos degenerados)")
    print("=" * 80)
    print(f"\n  {'Rank':<4} {'Experimento':<30} {'Composto':>10} {'Sensitivity':>14} {'F1':>12} {'Spec':>8} {'Kappa':>7} {'Threshold':>10}")
    print("  " + "─" * 95)
    for rank, (chave, dados) in enumerate(ordenados_composto, 1):
        m = dados['metricas']
        t = dados['threshold_medio']
        sc = dados['score_composto']
        print(f"  {rank:<4} {chave:<30} {sc:>10.4f} "
              f"{m['sensitivity']:.1f}%±{m['sensitivity_ic']:.1f}%{'':<2} "
              f"{m['f1']:.1f}%±{m['f1_ic']:.1f}%{'':<1} "
              f"{m['specificity']:>7.1f}% "
              f"{m['kappa']:>7.3f} "
              f"     t≈{t:.2f}")

    relatorio_final = f'{OUTPUT_DIR}/ranking_final_{modo}_v2.txt'
    with open(relatorio_final, 'w') as f:
        f.write("=" * 95 + "\n")
        f.write(f"RANKING FINAL v2 — Busca de Hiperparâmetros para GAD\n")
        f.write(f"Modo: {modo} | n_iter={n_iter} | Score Composto = √(Sens × F1)\n")
        f.write("Scorers: gmean=G-Mean | clinic=Clínico | f2_pen=F2-Penalizado | f2=F2-Original\n")
        f.write("Threshold otimizado via F2 no treino de cada fold (sem data leakage)\n")
        f.write("=" * 95 + "\n\n")

        # Tabela por Score Composto
        f.write("─── Ranking por Score Composto √(Sens × F1) ───\n\n")
        f.write(f"{'Rank':<5} {'Experimento':<32} {'Composto':>10} {'Sensitivity':>18} {'F1':>16} {'Spec':>10} {'Kappa':>8} {'Threshold':>11}\n")
        f.write("─" * 115 + "\n")

        for rank, (chave, dados) in enumerate(ordenados_composto, 1):
            m = dados['metricas']
            t = dados['threshold_medio']
            sc = dados['score_composto']
            sens = f"{m['sensitivity']:.1f}±{m['sensitivity_ic']:.1f}%"
            f1   = f"{m['f1']:.1f}±{m['f1_ic']:.1f}%"
            f.write(f"{rank:<5} {chave:<32} {sc:>10.4f} {sens:>18} {f1:>16} {m['specificity']:>9.1f}% {m['kappa']:>8.3f}     t≈{t:.2f}\n")

        # Tabela por Sensibilidade (referência)
        f.write("\n\n─── Ranking por Sensibilidade (referência) ───\n\n")
        f.write(f"{'Rank':<5} {'Experimento':<32} {'Sensitivity':>18} {'F1':>16} {'Spec':>10} {'Kappa':>8} {'Threshold':>11}\n")
        f.write("─" * 100 + "\n")
        for rank, (chave, dados) in enumerate(ordenados_sens, 1):
            m = dados['metricas']
            t = dados['threshold_medio']
            sens = f"{m['sensitivity']:.1f}±{m['sensitivity_ic']:.1f}%"
            f1   = f"{m['f1']:.1f}±{m['f1_ic']:.1f}%"
            f.write(f"{rank:<5} {chave:<32} {sens:>18} {f1:>16} {m['specificity']:>9.1f}% {m['kappa']:>8.3f}     t≈{t:.2f}\n")

        # Detalhamento por método
        f.write("\n\n" + "=" * 95 + "\n")
        f.write("DETALHAMENTO — MELHORES PARÂMETROS POR EXPERIMENTO\n")
        f.write("=" * 95 + "\n\n")

        for rank, (chave, dados) in enumerate(ordenados_composto, 1):
            m = dados['metricas']
            c = dados['consenso']
            t = dados['threshold_medio']
            ts = dados['thresholds']
            sc = dados['score_composto']
            f.write(f"[{rank}] {chave}\n")
            f.write(f"  Score Composto: {sc:.4f}\n")
            f.write(f"  Sensitivity:    {m['sensitivity']:.2f}% ± {m['sensitivity_ic']:.2f}%\n")
            f.write(f"  F1-Score:       {m['f1']:.2f}% ± {m['f1_ic']:.2f}%\n")
            f.write(f"  Specificity:    {m['specificity']:.2f}%\n")
            f.write(f"  PPV:            {m['ppv']:.2f}%\n")
            f.write(f"  Kappa:          {m['kappa']:.4f}\n")
            f.write(f"  Matriz:  VN={m['vn']}  FP={m['fp']}  FN={m['fn']}  VP={m['vp']}\n")
            f.write(f"  Duração:        {dados['duracao']/60:.1f} min\n")
            f.write(f"  Threshold (F2 no treino):\n")
            f.write(f"    Média={t:.3f}  Min={min(ts):.3f}  Max={max(ts):.3f}\n")
            f.write(f"    Por fold: {[f'{x:.2f}' for x in ts]}\n")
            f.write(f"  Parâmetros ótimos (mediana dos 10 folds):\n")
            for k, v in sorted(c.items()):
                nome_curto = k.replace('clf__', '').replace('sampler__', 'sampler.')
                f.write(f"    {nome_curto:<25}: {v:.4f}\n" if isinstance(v, float) else f"    {nome_curto:<25}: {v}\n")
            f.write("\n")

        # Como usar o melhor modelo
        melhor_chave, melhor_dados = ordenados_composto[0]
        c = melhor_dados['consenso']
        m = melhor_dados['metricas']
        t = melhor_dados['threshold_medio']
        f.write("=" * 95 + "\n")
        f.write(f"COMO USAR O MELHOR MODELO (Score Composto): {melhor_chave}\n")
        f.write("=" * 95 + "\n\n")
        f.write(f"Score Composto √(Sens×F1): {melhor_dados['score_composto']:.4f}\n")
        f.write(f"Sensibilidade:  {m['sensitivity']:.1f}% ± {m['sensitivity_ic']:.1f}%\n")
        f.write(f"F1-Score:       {m['f1']:.1f}% ± {m['f1_ic']:.1f}%\n")
        f.write(f"Especificidade: {m['specificity']:.1f}%\n")
        f.write(f"Threshold (F2 otimizado): {t:.3f}\n\n")
        f.write("Código Python para replicar:\n\n")
        sampler_nome = melhor_chave.split('_')[0]
        sampler_imports = {
            'SMOTE': 'from imblearn.over_sampling import SMOTE',
            'ADASYN': 'from imblearn.over_sampling import ADASYN',
            'BorderlineSMOTE': 'from imblearn.over_sampling import BorderlineSMOTE',
            'SMOTEENN': 'from imblearn.combine import SMOTEENN',
            'SMOTETomek': 'from imblearn.combine import SMOTETomek',
        }
        f.write(sampler_imports.get(sampler_nome, f'from imblearn.over_sampling import {sampler_nome}') + "\n")
        f.write("from xgboost import XGBClassifier\n\n")
        xgb_params = {k.replace('clf__', ''): v for k, v in c.items() if k.startswith('clf__')}
        f.write("model = XGBClassifier(\n")
        for k, v in sorted(xgb_params.items()):
            f.write(f"    {k}={v:.4f},\n" if isinstance(v, float) else f"    {k}={v},\n")
        f.write(f"    random_state=42, use_label_encoder=False,\n")
        f.write(f"    eval_metric='logloss', verbosity=0,\n")
        f.write(f")\n\n")
        f.write(f"# Após treinar com o sampler, usar threshold otimizado:\n")
        f.write(f"# y_pred = (model.predict_proba(X_test)[:, 1] >= {t:.3f}).astype(int)\n")

    print(f"\n  Log detalhado:  {log_path}")
    print(f"  Ranking final:  {relatorio_final}")

    # ── Gráfico de ranking ─────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(22, max(6, len(todos_resultados) * 0.55)))
    plt.style.use('seaborn-v0_8-whitegrid')
    cores_r  = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(ordenados_composto)))
    cores_f1 = plt.cm.RdYlBu(np.linspace(0.15, 0.85, len(ordenados_composto)))
    cores_sc = plt.cm.BuGn(np.linspace(0.15, 0.85, len(ordenados_composto)))
    y_pos = np.arange(len(ordenados_composto))

    # Gráfico 1: Sensibilidade
    senss    = [d['metricas']['sensitivity'] for _, d in ordenados_sens]
    senss_ic = [d['metricas']['sensitivity_ic'] for _, d in ordenados_sens]
    nomes_s  = [c for c, _ in ordenados_sens]
    axes[0].barh(y_pos, senss[::-1], xerr=senss_ic[::-1], color=cores_r, edgecolor='white', capsize=3, error_kw={'elinewidth': 1.2})
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(nomes_s[::-1], fontsize=7)
    axes[0].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[0].set_xlabel('Sensibilidade (%)', fontweight='bold')
    axes[0].set_title('Sensibilidade', fontweight='bold')
    axes[0].legend(fontsize=7)

    # Gráfico 2: F1-Score
    ord_f1   = sorted(todos_resultados.items(), key=lambda x: x[1]['metricas']['f1'], reverse=True)
    f1s      = [d['metricas']['f1'] for _, d in ord_f1]
    f1s_ic   = [d['metricas']['f1_ic'] for _, d in ord_f1]
    nomes_f1 = [c for c, _ in ord_f1]
    axes[1].barh(y_pos, f1s[::-1], xerr=f1s_ic[::-1], color=cores_f1, edgecolor='white', capsize=3, error_kw={'elinewidth': 1.2})
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels(nomes_f1[::-1], fontsize=7)
    axes[1].set_xlabel('F1-Score (%)', fontweight='bold')
    axes[1].set_title('F1-Score', fontweight='bold')

    # Gráfico 3: Score Composto
    scs      = [d['score_composto'] for _, d in ordenados_composto]
    nomes_sc = [c for c, _ in ordenados_composto]
    axes[2].barh(y_pos, scs[::-1], color=cores_sc, edgecolor='white')
    axes[2].set_yticks(y_pos); axes[2].set_yticklabels(nomes_sc[::-1], fontsize=7)
    axes[2].set_xlabel('Score Composto √(Sens × F1)', fontweight='bold')
    axes[2].set_title('Ranking Equilibrado', fontweight='bold')

    plt.suptitle(f'Busca de Hiperparâmetros v2 — GAD | {modo} ({n_iter} iter)\n'
                 f'Scorers: GMean | Clínico | F2-Penalizado | Threshold via F2',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    graf_path = f'{PLOTS_DIR}/ranking_{modo}_v2.png'
    plt.savefig(graf_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico:        {graf_path}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Busca de hiperparâmetros v2 — Maximizar Sensibilidade + F1 (GAD).'
    )
    parser.add_argument(
        '--modo',
        choices=['rapido', 'medio', 'completo'],
        default='medio',
        help='rapido (~30 min) | medio (~2-4h, padrão) | completo (~8-12h)',
    )
    args = parser.parse_args()

    print(f"\n  Modo selecionado: {args.modo.upper()}")
    print(f"  {MODOS[args.modo]['desc']}")
    print(f"  Scorers: GMean, Clínico, F2-Penalizado, F2-Original")
    print(f"  Experimentos: 14 (5×gmean + 3×clinic + 4×f2_pen + 2×f2)")
    print(f"  Threshold: otimizado via F2 no treino de cada fold")
    print(f"  Resultados salvos em: output/busca_hiperparametros/\n")

    t_total = time.time()
    main(modo=args.modo)
    print(f"  Tempo total: {(time.time() - t_total)/60:.1f} minutos")
