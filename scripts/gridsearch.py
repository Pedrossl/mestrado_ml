"""
GridSearch de hiperparametros com Nested Cross-Validation.

Otimiza XGBoost e SVM com 4 tecnicas de balanceamento cada.
- Outer CV: 10-fold stratified (avaliacao final)
- Inner CV: 5-fold stratified (selecao de hiperparametros)
- SMOTE/Undersampling aplicado dentro de cada fold via imblearn Pipeline
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
from collections import Counter

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from utils import (
    preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic,
    exibir_resultados, comparativo_modelos,
    plotar_comparativo_grafico, plotar_comparativo_tabela,
    plotar_gridsearch_heatmap
)

# --- Grids de hiperparametros ---

XGBOOST_GRID = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__subsample': [0.8, 1.0],
}

SVM_GRID = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 0.001, 0.01, 0.1],
}


def _criar_pipeline_xgboost(balanceamento, peso_classe=None):
    """Cria pipeline XGBoost com a tecnica de balanceamento especificada."""
    clf_params = dict(
        random_state=42, use_label_encoder=False,
        eval_metric='logloss', verbosity=0
    )
    if peso_classe is not None:
        clf_params['scale_pos_weight'] = peso_classe

    steps = []
    if balanceamento == 'smote':
        steps.append(('sampler', SMOTE(random_state=42)))
    elif balanceamento == 'undersampling':
        steps.append(('sampler', RandomUnderSampler(random_state=42)))
    steps.append(('clf', XGBClassifier(**clf_params)))

    return ImbPipeline(steps)


def _criar_pipeline_svm(balanceamento):
    """Cria pipeline SVM com scaler e tecnica de balanceamento."""
    clf_params = dict(kernel='rbf', random_state=42)
    if balanceamento == 'weighted':
        clf_params['class_weight'] = 'balanced'

    steps = [('scaler', StandardScaler())]
    if balanceamento == 'smote':
        steps.append(('sampler', SMOTE(random_state=42)))
    elif balanceamento == 'undersampling':
        steps.append(('sampler', RandomUnderSampler(random_state=42)))
    steps.append(('clf', SVC(**clf_params)))

    return ImbPipeline(steps)


def _moda_params(lista_params):
    """Retorna os hiperparametros mais frequentes entre os folds."""
    consenso = {}
    for key in lista_params[0].keys():
        valores = [p[key] for p in lista_params]
        consenso[key] = Counter(valores).most_common(1)[0][0]
    return consenso


def gridsearch_nested_cv(algoritmo, target='GAD'):
    """Executa GridSearch com nested CV para um algoritmo.

    Args:
        algoritmo: 'xgboost' ou 'svm'
        target: 'GAD' ou 'SAD'

    Returns:
        dict com resultados por tecnica de balanceamento
    """
    df, target_name = preparar_dados(target)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    dist = df[target_name].value_counts()
    peso_classe = dist[0] / dist[1]

    nome_algo = 'XGBoost' if algoritmo == 'xgboost' else 'SVM'
    output_path = f'output/plots/GridSearch/{nome_algo}/{target.upper()}'
    plots_path = f'{output_path}/plots'
    os.makedirs(plots_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  GRIDSEARCH NESTED CV - {nome_algo} ({target})")
    print("=" * 70)
    print(f"\n  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1}")
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    param_grid = XGBOOST_GRID if algoritmo == 'xgboost' else SVM_GRID
    n_combinacoes = 1
    for v in param_grid.values():
        n_combinacoes *= len(v)
    print(f"  Grid: {n_combinacoes} combinacoes | Inner CV: 5-fold | Outer CV: 10-fold")

    tecnicas = {
        'Sem Balanceamento': 'none',
        'Class Weighting': 'weighted',
        'SMOTE': 'smote',
        'Undersampling': 'undersampling'
    }

    resultados_dict = {}
    ultimo_cv_results = None

    for nome_tecnica, bal in tecnicas.items():
        print(f"\n  --- {nome_tecnica} ---")

        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        metricas_folds = []
        best_params_folds = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if algoritmo == 'xgboost':
                peso = peso_classe if bal == 'weighted' else None
                pipeline = _criar_pipeline_xgboost(bal, peso)
            else:
                pipeline = _criar_pipeline_svm(bal)

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=inner_cv,
                scoring='f1', n_jobs=-1, refit=True
            )
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            metricas_folds.append(calcular_metricas_fold(y_test, y_pred))
            best_params_folds.append(grid_search.best_params_)

            if fold_idx == 0:
                ultimo_cv_results = grid_search.cv_results_

        consenso = _moda_params(best_params_folds)
        print(f"    Melhores params (consenso): {consenso}")

        metricas = agregar_metricas_com_ic(metricas_folds)

        sufixo = bal if bal != 'none' else 'baseline'
        output_file = f'{output_path}/{algoritmo}_{target.lower()}_gridsearch_{sufixo}.txt'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(f"GridSearch Nested CV - {nome_algo} {nome_tecnica} - {target}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Melhores hiperparametros (consenso dos 10 folds):\n")
            for k, v in consenso.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nParametros por fold:\n")
            for i, p in enumerate(best_params_folds):
                f.write(f"  Fold {i+1}: {p}\n")
            f.write(f"\nMETRICAS (media +/- IC 95%)\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy:    {metricas['accuracy']:.2f}% +/- {metricas['accuracy_ic']:.2f}%\n")
            f.write(f"Sensitivity: {metricas['sensitivity']:.2f}% +/- {metricas['sensitivity_ic']:.2f}%\n")
            f.write(f"Specificity: {metricas['specificity']:.2f}% +/- {metricas['specificity_ic']:.2f}%\n")
            f.write(f"PPV:         {metricas['ppv']:.2f}% +/- {metricas['ppv_ic']:.2f}%\n")
            f.write(f"NPV:         {metricas['npv']:.2f}% +/- {metricas['npv_ic']:.2f}%\n")
            f.write(f"F1-Score:    {metricas['f1']:.2f}% +/- {metricas['f1_ic']:.2f}%\n")
            f.write(f"Kappa:       {metricas['kappa']:.4f} +/- {metricas['kappa_ic']:.4f}\n\n")
            f.write(f"Matriz de Confusao Agregada:\n")
            f.write(f"VN={metricas['vn']} | FP={metricas['fp']}\n")
            f.write(f"FN={metricas['fn']} | VP={metricas['vp']}\n")

        print(f"    Acc={metricas['accuracy']:.1f}% Sens={metricas['sensitivity']:.1f}% "
              f"Spec={metricas['specificity']:.1f}% F1={metricas['f1']:.1f}% "
              f"Kappa={metricas['kappa']:.3f}")
        print(f"    Salvo em: {output_file}")

        resultados_dict[nome_tecnica] = metricas

    # Heatmap do GridSearch (ultimo fold, primeira tecnica com resultado)
    if ultimo_cv_results is not None:
        if algoritmo == 'svm':
            plotar_gridsearch_heatmap(
                ultimo_cv_results, 'clf__C', 'clf__gamma',
                f'GridSearch {nome_algo} - F1 Score ({target})',
                f'{plots_path}/gridsearch_heatmap_{target.lower()}.png'
            )
        else:
            plotar_gridsearch_heatmap(
                ultimo_cv_results, 'clf__max_depth', 'clf__learning_rate',
                f'GridSearch {nome_algo} - F1 Score ({target})',
                f'{plots_path}/gridsearch_heatmap_{target.lower()}.png'
            )

    # Tabela comparativa
    resultados = comparativo_modelos(
        resultados_dict, target, f"{nome_algo} (GridSearch)", output_path
    )
    plotar_comparativo_grafico(resultados, target, f"{nome_algo} (GridSearch)", plots_path)
    plotar_comparativo_tabela(resultados, target, f"{nome_algo} (GridSearch)", plots_path)

    return resultados_dict


if __name__ == "__main__":
    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 70)
        print(f"{'':>20}GRIDSEARCH - {target}")
        print("#" * 70)

        gridsearch_nested_cv('xgboost', target)
        gridsearch_nested_cv('svm', target)
