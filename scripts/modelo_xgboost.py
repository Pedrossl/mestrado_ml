import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from utils import (
    preparar_dados, calcular_ic, calcular_metricas_fold, agregar_metricas_com_ic,
    exibir_resultados, comparativo_modelos,
    plotar_comparativo_grafico, plotar_comparativo_tabela
)

OUTPUT_PATH = 'output/plots/XGBoost'
PLOTS_PATH = 'output/plots/XGBoost/plots'


def treinar_xgboost(target='GAD'):
    """Treina XGBoost sem balanceamento."""
    output_path = f'output/plots/XGBoost/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 70)
    print("           MODELO XGBoost - SEM BALANCEAMENTO")
    print("=" * 70)

    print("\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_por_fold = []

    print(f"\n[TREINAMENTO]")
    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metricas_por_fold.append(calcular_metricas_fold(y_test, y_pred))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_por_fold)
    output_file = f'{output_path}/xgboost_{target.lower()}_metricas.txt'
    exibir_resultados(metricas, target_name, "XGBoost (sem balanceamento)", output_file)

    return metricas


def treinar_xgboost_weighted(target='GAD'):
    """Treina XGBoost com scale_pos_weight."""
    output_path = f'output/plots/XGBoost/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 70)
    print("          MODELO XGBoost - COM CLASS WEIGHTING")
    print("=" * 70)

    print("\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    peso = dist[0] / dist[1]
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"  scale_pos_weight: {peso:.2f}")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_por_fold = []

    print(f"\n[TREINAMENTO]")
    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=peso, random_state=42,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metricas_por_fold.append(calcular_metricas_fold(y_test, y_pred))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_por_fold)
    output_file = f'{output_path}/xgboost_{target.lower()}_weighted_metricas.txt'
    exibir_resultados(metricas, target_name, f"XGBoost (scale_pos_weight={peso:.1f})", output_file)

    return metricas


def treinar_xgboost_smote(target='GAD'):
    """Treina XGBoost com SMOTE dentro de cada fold."""
    output_path = f'output/plots/XGBoost/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 70)
    print("         MODELO XGBoost - COM SMOTE (CORRIGIDO)")
    print("=" * 70)

    print("\n[DATASET ORIGINAL]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    print("\n[CROSS-VALIDATION COM SMOTE POR FOLD]")
    print("  SMOTE aplicado apenas no treino (sem data leakage)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_por_fold = []

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        metricas_por_fold.append(calcular_metricas_fold(y_test, y_pred))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_por_fold)
    output_file = f'{output_path}/xgboost_{target.lower()}_smote_metricas.txt'
    exibir_resultados(metricas, target_name, "XGBoost (com SMOTE)", output_file)

    return metricas


def treinar_xgboost_undersampling(target='GAD'):
    """Treina XGBoost com Undersampling dentro de cada fold."""
    output_path = f'output/plots/XGBoost/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 70)
    print("      MODELO XGBoost - COM UNDERSAMPLING (CORRIGIDO)")
    print("=" * 70)

    print("\n[DATASET ORIGINAL]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    print("\n[CROSS-VALIDATION COM UNDERSAMPLING POR FOLD]")
    print("  Undersampling aplicado apenas no treino (sem data leakage)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_por_fold = []

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        undersampler = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        metricas_por_fold.append(calcular_metricas_fold(y_test, y_pred))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_por_fold)
    output_file = f'{output_path}/xgboost_{target.lower()}_undersampling_metricas.txt'
    exibir_resultados(metricas, target_name, "XGBoost (com Undersampling)", output_file)

    return metricas


def analisar_importancia_features(target='GAD'):
    """Analisa importância das features usando XGBoost com SMOTE (média dos 10 folds)."""
    output_path = f'output/plots/XGBoost/{target.upper()}'
    plots_path = f'output/plots/XGBoost/{target.upper()}/plots'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print(f"     FEATURE IMPORTANCE - XGBoost + SMOTE ({target})")
    print("=" * 60)

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    feature_names = df.drop(columns=[target_name]).columns.tolist()

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    importancias_folds = []

    print(f"\n  Coletando importâncias dos {n_folds} folds...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train_res, y_train_res)
        importancias_folds.append(model.feature_importances_)

    print("OK")

    importancias_array = np.array(importancias_folds)
    resultados_features = []

    for i, nome in enumerate(feature_names):
        valores = importancias_array[:, i]
        media, desvio, ic = calcular_ic(valores)
        resultados_features.append({
            'Feature': nome,
            'Importancia': media,
            'Desvio': desvio,
            'IC': ic
        })

    resultados_features.sort(key=lambda x: x['Importancia'], reverse=True)

    print(f"\n  {'Rank':<5} {'Feature':<30} {'Importância':>12} {'± IC 95%':>10}")
    print("  " + "-" * 60)
    for i, r in enumerate(resultados_features):
        print(f"  {i+1:<5} {r['Feature']:<30} {r['Importancia']:>12.4f} ± {r['IC']:.4f}")

    output_file = f'{output_path}/feature_importance_{target.lower()}.txt'
    with open(output_file, 'w') as f:
        f.write(f"FEATURE IMPORTANCE - XGBoost + SMOTE - {target_name}\n")
        f.write("=" * 60 + "\n")
        f.write("Método: média das importâncias dos 10 folds (IC 95%)\n\n")
        f.write(f"{'Rank':<5} {'Feature':<30} {'Importância':>12} {'± IC 95%':>10}\n")
        f.write("-" * 60 + "\n")
        for i, r in enumerate(resultados_features):
            f.write(f"{i+1:<5} {r['Feature']:<30} {r['Importancia']:>12.4f} ± {r['IC']:.4f}\n")

    print(f"\n  Resultados salvos em: {output_file}")

    # Gráfico

    top_n = min(15, len(resultados_features))
    top_features = resultados_features[:top_n]
    top_features.reverse()

    nomes = [r['Feature'] for r in top_features]
    valores = [r['Importancia'] for r in top_features]
    erros = [r['IC'] for r in top_features]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.5)))

    cores = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))
    barras = ax.barh(range(top_n), valores, xerr=erros, color=cores,
                     edgecolor='white', linewidth=0.5,
                     capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(nomes, fontsize=10)
    ax.set_xlabel('Importância (média dos 10 folds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance - XGBoost + SMOTE ({target_name})\nTop {top_n} variáveis mais importantes (com IC 95%)',
                 fontsize=14, fontweight='bold', pad=20)

    for barra, valor in zip(barras, valores):
        ax.text(barra.get_width() + max(erros) * 0.3, barra.get_y() + barra.get_height() / 2,
                f'{valor:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    grafico_file = f'{plots_path}/feature_importance_{target.lower()}.png'
    plt.savefig(grafico_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico salvo em: {grafico_file}")

    print("\n" + "=" * 60 + "\n")

    return resultados_features


if __name__ == "__main__":
    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 60)
        print(f"{'':>15}ANÁLISE XGBoost - {target}")
        print("#" * 60)

        output_path = f'output/plots/XGBoost/{target.upper()}'
        plots_path = f'output/plots/XGBoost/{target.upper()}/plots'
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)

        resultados_dict = {}

        m1 = treinar_xgboost(target)
        resultados_dict['Sem Balanceamento'] = m1

        print("\n" + "@" * 70 + "\n")

        m2 = treinar_xgboost_weighted(target)
        resultados_dict['Class Weighting'] = m2

        print("\n" + "@" * 70 + "\n")

        m3 = treinar_xgboost_smote(target)
        resultados_dict['SMOTE'] = m3

        print("\n" + "@" * 70 + "\n")

        m4 = treinar_xgboost_undersampling(target)
        resultados_dict['Undersampling'] = m4

        print("\n" + "@" * 70 + "\n")

        resultados = comparativo_modelos(resultados_dict, target, "XGBoost", output_path)
        plotar_comparativo_grafico(resultados, target, "XGBoost", plots_path)
        plotar_comparativo_tabela(resultados, target, "XGBoost", plots_path)

        analisar_importancia_features(target)
