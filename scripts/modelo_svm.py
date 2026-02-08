import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from utils import (
    preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic,
    exibir_resultados, comparativo_modelos,
    plotar_comparativo_grafico, plotar_comparativo_tabela
)

OUTPUT_PATH = 'output/plots/SVM'
PLOTS_PATH = 'output/plots/SVM/plots'


def treinar_svm(target='GAD'):
    """Treina SVM sem balanceamento."""
    output_path = f'output/plots/SVM/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print("          MODELO SVM - SEM BALANCEAMENTO")
    print("=" * 60)

    print("\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

    print(f"\n[TREINAMENTO]")
    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metricas_folds.append(calcular_metricas_fold(y_test, y_pred))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_folds)
    output_file = f'{output_path}/svm_{target.lower()}_metricas.txt'
    exibir_resultados(metricas, target_name, "SVM (sem balanceamento)", output_file)

    return metricas


def treinar_svm_weighted(target='GAD'):
    """Treina SVM com class_weight='balanced'."""
    output_path = f'output/plots/SVM/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print("        MODELO SVM - COM CLASS WEIGHTING")
    print("=" * 60)

    print("\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"  Usando class_weight='balanced'")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

    print(f"\n[TREINAMENTO]")
    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metricas_folds.append(calcular_metricas_fold(y_test, y_pred))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_folds)
    output_file = f'{output_path}/svm_{target.lower()}_weighted_metricas.txt'
    exibir_resultados(metricas, target_name, "SVM (class_weight='balanced')", output_file)

    return metricas


def treinar_svm_smote(target='GAD'):
    """Treina SVM com SMOTE aplicado dentro de cada fold."""
    output_path = f'output/plots/SVM/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print("       MODELO SVM - COM SMOTE (CORRIGIDO)")
    print("=" * 60)

    print("\n[DATASET ORIGINAL]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    print("\n[CROSS-VALIDATION COM SMOTE POR FOLD]")
    print("  SMOTE aplicado apenas no treino (sem data leakage)")

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_scaled)

        metricas_folds.append(calcular_metricas_fold(y_test, y_pred))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_folds)
    output_file = f'{output_path}/svm_{target.lower()}_smote_metricas.txt'
    exibir_resultados(metricas, target_name, "SVM (com SMOTE)", output_file)

    return metricas


def treinar_svm_undersampling(target='GAD'):
    """Treina SVM com Undersampling aplicado dentro de cada fold."""
    output_path = f'output/plots/SVM/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print("    MODELO SVM - COM UNDERSAMPLING (CORRIGIDO)")
    print("=" * 60)

    print("\n[DATASET ORIGINAL]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    print("\n[CROSS-VALIDATION COM UNDERSAMPLING POR FOLD]")
    print("  Undersampling aplicado apenas no treino (sem data leakage)")

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        undersampler = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_scaled, y_train_res)
        y_pred = model.predict(X_test_scaled)

        metricas_folds.append(calcular_metricas_fold(y_test, y_pred))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_folds)
    output_file = f'{output_path}/svm_{target.lower()}_undersampling_metricas.txt'
    exibir_resultados(metricas, target_name, "SVM (com Undersampling)", output_file)

    return metricas


if __name__ == "__main__":
    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 60)
        print(f"{'':>15}ANÁLISE SVM - {target}")
        print("#" * 60)

        output_path = f'output/plots/SVM/{target.upper()}'
        plots_path = f'output/plots/SVM/{target.upper()}/plots'
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)

        resultados_dict = {}

        m1 = treinar_svm(target)
        resultados_dict['Sem Balanceamento'] = m1

        print("\n" + "@" * 60 + "\n")

        m2 = treinar_svm_weighted(target)
        resultados_dict['Class Weighting'] = m2

        print("\n" + "@" * 60 + "\n")

        m3 = treinar_svm_smote(target)
        resultados_dict['SMOTE'] = m3

        print("\n" + "@" * 60 + "\n")

        m4 = treinar_svm_undersampling(target)
        resultados_dict['Undersampling'] = m4

        print("\n" + "@" * 60 + "\n")

        resultados = comparativo_modelos(resultados_dict, target, "SVM", output_path)
        plotar_comparativo_grafico(resultados, target, "SVM", plots_path)
        plotar_comparativo_tabela(resultados, target, "SVM", plots_path)
