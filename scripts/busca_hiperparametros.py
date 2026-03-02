"""
Busca Exaustiva de Hiperparâmetros — Maximizar Sensibilidade para GAD
======================================================================

Usa RandomizedSearchCV com nested cross-validation para explorar um
espaço enorme de combinações de hiperparâmetros do XGBoost + técnicas de
balanceamento, otimizando para recall (sensibilidade) e F2-score.

Por que RandomizedSearch em vez de GridSearch?
  GridSearch = testa TODAS as combinações (ex: 3×4×5×6×4×4 = 5760 combinações)
  RandomSearch = sorteia N combinações aleatórias e acha ótimos ~similares
  em ~10% do tempo. (Bergstra & Bengio 2012 provaram isso empiricamente.)

Referências:
  - Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization.
    JMLR 13, 281-305.
  - Varma & Simon (2006). Bias in error estimation when using cross-validation
    for model selection. BMC Bioinformatics 7, 91.

COMO RODAR (pode demorar horas):
  Rápido  (~10-20 min):   python scripts/busca_hiperparametros.py --modo rapido
  Médio   (~1-2 horas):   python scripts/busca_hiperparametros.py --modo medio
  Completo (~4-8 horas):  python scripts/busca_hiperparametros.py --modo completo
  Padrão  (médio):        python scripts/busca_hiperparametros.py

Autor: Dissertação de Mestrado — Fevereiro 2026
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
    cross_val_predict,
)
from sklearn.metrics import (
    recall_score, fbeta_score, f1_score, make_scorer,
    confusion_matrix, classification_report,
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
    'rapido':   {'n_iter': 50,   'desc': '~50 combinações   | ~10-20 min'},
    'medio':    {'n_iter': 200,  'desc': '~200 combinações  | ~1-2 horas'},
    'completo': {'n_iter': 1000, 'desc': '~1000 combinações | ~4-8 horas'},
}

TARGET = 'GAD'
OUTPUT_DIR = 'output/busca_hiperparametros'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Espaço de busca — TODOS os hiperparâmetros relevantes do XGBoost ─────────

PARAM_SPACE_XGBOOST = {
    # Número de árvores
    'clf__n_estimators': randint(50, 600),

    # Profundidade máxima de cada árvore
    # Menor = mais simples/menos overfitting; Maior = mais capacidade
    'clf__max_depth': randint(2, 10),

    # Taxa de aprendizado (shrinkage)
    # Menor + mais árvores = geralmente melhor, mas mais lento
    'clf__learning_rate': loguniform(0.001, 0.4),

    # Fração de amostras usada para treinar cada árvore (subsampling)
    # < 1.0 previne overfitting e adiciona aleatoriedade
    'clf__subsample': uniform(0.5, 0.5),   # 0.5 a 1.0

    # Fração de features usada por árvore (como Random Forest)
    'clf__colsample_bytree': uniform(0.4, 0.6),   # 0.4 a 1.0

    # Fração de features usada por nível da árvore
    'clf__colsample_bylevel': uniform(0.4, 0.6),  # 0.4 a 1.0

    # Peso mínimo de instâncias em um nó folha
    # Maior = mais conservador, evita dividir nós com poucos positivos
    # MUITO IMPORTANTE para dados desbalanceados
    'clf__min_child_weight': randint(1, 15),

    # Redução mínima de loss para fazer split (regularização)
    'clf__gamma': loguniform(1e-4, 2.0),

    # Regularização L2 (Ridge)
    'clf__reg_lambda': loguniform(0.1, 10.0),

    # Regularização L1 (Lasso) — força esparsidade
    'clf__reg_alpha': loguniform(1e-4, 2.0),

    # CHAVE para desbalanceamento: peso da classe positiva
    # Valores altos = modelo foca muito mais em não errar os positivos
    # Testamos de 1x (neutro) a 50x (extremamente agressivo)
    'clf__scale_pos_weight': loguniform(1.0, 50.0),
}

# Espaços menores para cada sampler específico
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

PARAM_SPACE_SMOTEENN = {
    **PARAM_SPACE_XGBOOST,
    # SMOTEENN tem poucos hiperparâmetros acessíveis
}

PARAM_SPACE_SMOTETOMEK = {
    **PARAM_SPACE_XGBOOST,
}

# ─── Função principal de busca ─────────────────────────────────────────────────

def buscar_hiperparametros(sampler, param_space, nome_sampler, n_iter, scoring, nome_scoring):
    """
    Executa RandomizedSearchCV com nested CV.

    Outer CV (10-fold): avalia o modelo no conjunto de teste (estimativa não-viesada)
    Inner CV (5-fold): seleciona os melhores hiperparâmetros (no fold de treino)
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
            n_jobs=1,  # nested paralelismo gerenciado pelo GridSearch
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

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        search.fit(X_tr, y_tr)
        y_pred = search.predict(X_te)
        metricas_folds.append(calcular_metricas_fold(y_te, y_pred))
        melhores_params_folds.append(search.best_params_)

        sens = metricas_folds[-1]['sensitivity']
        f1   = metricas_folds[-1]['f1']
        print(f"    Fold {fold_idx+1:2d}/10 → Sens={sens:.1f}%  F1={f1:.1f}%  "
              f"Melhores: depth={search.best_params_.get('clf__max_depth','?')} "
              f"spw={search.best_params_.get('clf__scale_pos_weight', '?'):.1f}")

    return agregar_metricas_com_ic(metricas_folds), melhores_params_folds


def _extrair_consenso_params(lista_params):
    """Retorna a mediana/moda dos parâmetros mais frequentes entre os folds."""
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
    print("   BUSCA DE HIPERPARÂMETROS — Maximizar Sensibilidade para GAD")
    print(f"   Modo: {modo.upper()} | {desc}")
    print(f"   Algoritmo: XGBoost | RandomizedSearchCV + Nested 10-fold CV")
    print("=" * 75)

    df, target_name = preparar_dados(TARGET)
    dist = df[target_name].value_counts()
    print(f"\n  Dataset: {df.shape[0]} amostras | {df.shape[1]-1} features")
    print(f"  GAD positivos: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"\n  Cada configuração: {n_iter} combinações × 5-fold inner CV = {n_iter*5} fits")
    print(f"  Total estimado por sampler: {n_iter*5*10} fits (10-fold outer)")
    print()

    # Definir experimentos: (sampler, param_space, nome)
    experimentos = [
        (SMOTE(random_state=42),            PARAM_SPACE_SMOTE,      'SMOTE'),
        (ADASYN(random_state=42),           PARAM_SPACE_ADASYN,     'ADASYN'),
        (BorderlineSMOTE(random_state=42),  PARAM_SPACE_BORDERLINE, 'BorderlineSMOTE'),
        (SMOTEENN(random_state=42),         PARAM_SPACE_SMOTEENN,   'SMOTEENN'),
        (SMOTETomek(random_state=42),       PARAM_SPACE_SMOTETOMEK, 'SMOTETomek'),
    ]

    # Métricas alvo
    scorer_recall = make_scorer(recall_score, zero_division=0)
    scorer_f2     = make_scorer(fbeta_score, beta=2, zero_division=0)

    todos_resultados = {}
    log_path = f'{OUTPUT_DIR}/busca_{modo}.txt'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(log_path, 'w') as log:
        log.write(f"BUSCA DE HIPERPARÂMETROS — GAD | Modo: {modo} | n_iter={n_iter}\n")
        log.write("=" * 75 + "\n\n")

        for sampler, param_space, nome_sampler in experimentos:
            for scoring, nome_scoring in [(scorer_recall, 'recall'), (scorer_f2, 'f2')]:
                chave = f"{nome_sampler}_{nome_scoring}"

                print(f"\n  {'─' * 65}")
                print(f"  [{chave}] Buscando {n_iter} combinações com scoring={nome_scoring}...")
                print(f"  {'─' * 65}")

                t0 = time.time()
                try:
                    metricas, params_folds = buscar_hiperparametros(
                        sampler.__class__(random_state=42),
                        param_space, nome_sampler, n_iter,
                        scoring, nome_scoring,
                    )
                    duracao = time.time() - t0
                    consenso = _extrair_consenso_params(params_folds)

                    todos_resultados[chave] = {
                        'metricas': metricas,
                        'consenso': consenso,
                        'params_folds': params_folds,
                        'duracao': duracao,
                    }

                    print(f"\n  Resultado [{chave}] ({duracao/60:.1f} min):")
                    print(f"    Sensitivity: {metricas['sensitivity']:.1f}% ± {metricas['sensitivity_ic']:.1f}%")
                    print(f"    F1-Score:    {metricas['f1']:.1f}% ± {metricas['f1_ic']:.1f}%")
                    print(f"    Specificity: {metricas['specificity']:.1f}%")
                    print(f"    Kappa:       {metricas['kappa']:.3f}")
                    print(f"  Melhores parâmetros (mediana/moda dos 10 folds):")
                    for k, v in sorted(consenso.items()):
                        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

                    # Salvar no log
                    log.write(f"\n[{chave}] ({duracao/60:.1f} min)\n")
                    log.write(f"  Sensitivity: {metricas['sensitivity']:.2f}% ± {metricas['sensitivity_ic']:.2f}%\n")
                    log.write(f"  F1-Score:    {metricas['f1']:.2f}% ± {metricas['f1_ic']:.2f}%\n")
                    log.write(f"  Specificity: {metricas['specificity']:.2f}%\n")
                    log.write(f"  Kappa:       {metricas['kappa']:.4f}\n")
                    log.write(f"  Matriz: VN={metricas['vn']} FP={metricas['fp']} FN={metricas['fn']} VP={metricas['vp']}\n")
                    log.write(f"  Parâmetros (consenso dos 10 folds):\n")
                    for k, v in sorted(consenso.items()):
                        log.write(f"    {k}: {v:.4f}\n" if isinstance(v, float) else f"    {k}: {v}\n")

                except Exception as e:
                    print(f"  ERRO em {chave}: {e}")
                    import traceback; traceback.print_exc()
                    log.write(f"\n[{chave}] ERRO: {e}\n")

    # ── Relatório final ────────────────────────────────────────────────────────

    if not todos_resultados:
        print("\nNenhum resultado gerado.")
        return

    print("\n" + "=" * 75)
    print("  RANKING FINAL — TODOS OS EXPERIMENTOS (por Sensibilidade)")
    print("=" * 75)

    ordenados = sorted(
        todos_resultados.items(),
        key=lambda x: x[1]['metricas']['sensitivity'],
        reverse=True,
    )

    print(f"\n  {'Rank':<4} {'Experimento':<30} {'Sensitivity':>16} {'F1-Score':>14} {'Kappa':>8}")
    print("  " + "─" * 75)
    for rank, (chave, dados) in enumerate(ordenados, 1):
        m = dados['metricas']
        print(f"  {rank:<4} {chave:<30} "
              f"{m['sensitivity']:.1f}%±{m['sensitivity_ic']:.1f}%{'':<5} "
              f"{m['f1']:.1f}%±{m['f1_ic']:.1f}%{'':<3} "
              f"{m['kappa']:.3f}")

    # Salvar ranking no log
    relatorio_final = f'{OUTPUT_DIR}/ranking_final_{modo}.txt'
    with open(relatorio_final, 'w') as f:
        f.write("=" * 75 + "\n")
        f.write(f"RANKING FINAL — Busca de Hiperparâmetros para GAD\n")
        f.write(f"Modo: {modo} | n_iter={n_iter} por experimento\n")
        f.write("=" * 75 + "\n\n")

        f.write(f"{'Rank':<4} {'Experimento':<32} {'Sensitivity':>18} {'F1':>14} {'Spec':>10} {'Kappa':>8}\n")
        f.write("─" * 90 + "\n")
        for rank, (chave, dados) in enumerate(ordenados, 1):
            m = dados['metricas']
            sens = f"{m['sensitivity']:.1f}±{m['sensitivity_ic']:.1f}%"
            f1   = f"{m['f1']:.1f}±{m['f1_ic']:.1f}%"
            f.write(f"{rank:<4} {chave:<32} {sens:>18} {f1:>14} {m['specificity']:>9.1f}% {m['kappa']:>8.3f}\n")

        f.write("\n\n" + "=" * 75 + "\n")
        f.write("DETALHAMENTO — MELHORES PARÂMETROS POR EXPERIMENTO\n")
        f.write("=" * 75 + "\n\n")

        for rank, (chave, dados) in enumerate(ordenados, 1):
            m = dados['metricas']
            c = dados['consenso']
            f.write(f"[{rank}] {chave}\n")
            f.write(f"  Sensitivity:  {m['sensitivity']:.2f}% ± {m['sensitivity_ic']:.2f}%\n")
            f.write(f"  F1-Score:     {m['f1']:.2f}% ± {m['f1_ic']:.2f}%\n")
            f.write(f"  Specificity:  {m['specificity']:.2f}%\n")
            f.write(f"  Kappa:        {m['kappa']:.4f}\n")
            f.write(f"  Duração:      {dados['duracao']/60:.1f} min\n")
            f.write(f"  Parâmetros ótimos (mediana dos 10 folds):\n")
            for k, v in sorted(c.items()):
                nome_curto = k.replace('clf__', '').replace('sampler__', 'sampler.')
                f.write(f"    {nome_curto:<25}: {v:.4f}\n" if isinstance(v, float) else f"    {nome_curto:<25}: {v}\n")
            f.write("\n")

        # Seção de "como usar os melhores parâmetros"
        if ordenados:
            melhor_chave, melhor_dados = ordenados[0]
            c = melhor_dados['consenso']
            m = melhor_dados['metricas']
            f.write("=" * 75 + "\n")
            f.write(f"COMO USAR O MELHOR MODELO ENCONTRADO: {melhor_chave}\n")
            f.write("=" * 75 + "\n\n")
            f.write(f"Sensibilidade atingida: {m['sensitivity']:.1f}% ± {m['sensitivity_ic']:.1f}%\n")
            f.write(f"F1-Score:               {m['f1']:.1f}% ± {m['f1_ic']:.1f}%\n\n")
            f.write("Código Python para replicar:\n\n")
            f.write("from xgboost import XGBClassifier\n")
            sampler_nome = melhor_chave.split('_')[0]
            f.write(f"from imblearn.over_sampling import {sampler_nome}  # ou combine/\n\n")
            f.write(f"model = XGBClassifier(\n")
            xgb_params = {k.replace('clf__', ''): v for k, v in c.items() if k.startswith('clf__')}
            for k, v in sorted(xgb_params.items()):
                f.write(f"    {k}={v:.4f},\n" if isinstance(v, float) else f"    {k}={v},\n")
            f.write(f"    random_state=42, use_label_encoder=False,\n")
            f.write(f"    eval_metric='logloss', verbosity=0,\n")
            f.write(f")\n")

    print(f"\n  Log detalhado:  {log_path}")
    print(f"  Ranking final:  {relatorio_final}")

    # ── Gráfico de ranking ─────────────────────────────────────────────────────

    nomes = [c for c, _ in ordenados]
    senss = [d['metricas']['sensitivity'] for _, d in ordenados]
    senss_ic = [d['metricas']['sensitivity_ic'] for _, d in ordenados]
    f1s   = [d['metricas']['f1'] for _, d in ordenados]

    plt.style.use('seaborn-v0_8-whitegrid')
    cores = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(ordenados)))

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(ordenados) * 0.5)))

    # Sensibilidade
    y_pos = np.arange(len(ordenados))
    axes[0].barh(y_pos, senss[::-1], xerr=senss_ic[::-1],
                 color=cores, edgecolor='white', capsize=4,
                 error_kw={'elinewidth': 1.5})
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(nomes[::-1], fontsize=8)
    axes[0].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% (mínimo clínico)')
    axes[0].set_xlabel('Sensibilidade (%)', fontweight='bold')
    axes[0].set_title('Sensibilidade por Experimento\n(RandomizedSearchCV)',
                      fontweight='bold')
    axes[0].legend(fontsize=8)

    # F1-score
    f1s_ordenados_f1 = [d['metricas']['f1'] for _, d in sorted(
        todos_resultados.items(), key=lambda x: x[1]['metricas']['f1'], reverse=True)]
    nomes_f1 = [c for c, _ in sorted(
        todos_resultados.items(), key=lambda x: x[1]['metricas']['f1'], reverse=True)]
    f1_ics = [d['metricas']['f1_ic'] for _, d in sorted(
        todos_resultados.items(), key=lambda x: x[1]['metricas']['f1'], reverse=True)]

    axes[1].barh(y_pos, f1s_ordenados_f1[::-1], xerr=f1_ics[::-1],
                 color=plt.cm.RdYlBu(np.linspace(0.15, 0.85, len(ordenados))),
                 edgecolor='white', capsize=4, error_kw={'elinewidth': 1.5})
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(nomes_f1[::-1], fontsize=8)
    axes[1].set_xlabel('F1-Score (%)', fontweight='bold')
    axes[1].set_title('F1-Score por Experimento\n(RandomizedSearchCV)',
                      fontweight='bold')

    plt.suptitle(f'Busca de Hiperparâmetros — GAD | Modo: {modo} ({n_iter} iter/experimento)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    graf_path = f'{PLOTS_DIR}/ranking_{modo}.png'
    plt.savefig(graf_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico:        {graf_path}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Busca de hiperparâmetros para maximizar sensibilidade no GAD.'
    )
    parser.add_argument(
        '--modo',
        choices=['rapido', 'medio', 'completo'],
        default='medio',
        help=(
            'rapido (~10-20 min) | '
            'medio (~1-2 horas, padrão) | '
            'completo (~4-8 horas)'
        ),
    )
    args = parser.parse_args()

    print(f"\n  Modo selecionado: {args.modo.upper()}")
    print(f"  {MODOS[args.modo]['desc']}")
    print(f"  Samplers testados: SMOTE, ADASYN, BorderlineSMOTE, SMOTEENN, SMOTETomek")
    print(f"  Scorings: recall (sensibilidade) e F2-score")
    print(f"  Total de experimentos: 10 (5 samplers × 2 scorings)")
    print(f"  Resultados salvos em: output/busca_hiperparametros/\n")

    t_total = time.time()
    main(modo=args.modo)
    print(f"  Tempo total: {(time.time() - t_total)/60:.1f} minutos")
