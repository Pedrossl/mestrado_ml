

import pandas as pd
import logging
import os

# Desabilitar logs do Weka
logging.getLogger("weka").setLevel(logging.ERROR)
os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.util.logging.config.file=/dev/null"

import weka.core.jvm as jvm
from weka.core.dataset import Instances, Attribute, Instance
from weka.classifiers import Classifier, Evaluation

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from scripts.utils import (
    preparar_dados, calcular_metricas_fold_cm, agregar_metricas_com_ic,
    exibir_resultados, comparativo_modelos,
    plotar_comparativo_grafico, plotar_comparativo_tabela
)

OUTPUT_PATH = 'output/plots/ADtree'
PLOTS_PATH = 'output/plots/ADtree/plots'


def criar_dataset_weka(df):
    """Cria dataset Weka a partir de DataFrame com atributos nominais."""
    atts = []
    for col in df.columns:
        valores_unicos = sorted(df[col].dropna().unique())
        valores_str = [str(int(v)) if v == int(v) else str(v) for v in valores_unicos]
        atts.append(Attribute.create_nominal(col, valores_str))

    dataset = Instances.create_instances("dados_teste", atts, len(df))

    for _, row in df.iterrows():
        values = []
        for i, col in enumerate(df.columns):
            val = row[col]
            valores_unicos = sorted(df[col].dropna().unique())
            valores_str = [str(int(v)) if v == int(v) else str(v) for v in valores_unicos]
            val_str = str(int(val)) if val == int(val) else str(val)
            values.append(valores_str.index(val_str))
        inst = Instance.create_instance(values)
        dataset.add_instance(inst)

    dataset.class_is_last()
    return dataset


def treinar_adtree(target='GAD'):
    """Treina modelo ADTree sem class weighting com CV manual para IC."""
    output_path = f'output/plots/ADtree/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print("         MODELO ADTree - SEM BALANCEAMENTO")
    print("=" * 60)

    print("\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    feature_names = df.drop(columns=[target_name]).columns.tolist()

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

    print(f"\n[TREINAMENTO]")
    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_train[target_name] = y_train
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_name] = y_test

        dataset_train = criar_dataset_weka(df_train)
        dataset_test = criar_dataset_weka(df_test)

        adtree = Classifier(classname="weka.classifiers.trees.ADTree")
        adtree.options = ["-B", "10", "-E", "-3"]
        adtree.build_classifier(dataset_train)

        evaluation = Evaluation(dataset_train)
        evaluation.test_model(adtree, dataset_test)

        metricas_folds.append(calcular_metricas_fold_cm(evaluation.confusion_matrix))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_folds)
    output_file = f'{output_path}/adtree_{target.lower()}_metricas.txt'
    exibir_resultados(metricas, target_name, "ADTree (sem balanceamento)", output_file)

    return metricas


def treinar_adtree_weighted(target='GAD'):
    """Treina modelo ADTree com Class Weighting usando CV manual para IC."""
    output_path = f'output/plots/ADtree/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print("        MODELO ADTree - COM CLASS WEIGHTING")
    print("=" * 60)

    print("\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    peso = dist[0] / dist[1]
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"  Peso aplicado à classe 1: {peso:.2f}x")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    feature_names = df.drop(columns=[target_name]).columns.tolist()

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

    print(f"\n[TREINAMENTO]")
    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_train[target_name] = y_train
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_name] = y_test

        dataset_train = criar_dataset_weka(df_train)
        dataset_test = criar_dataset_weka(df_test)

        cost_matrix = f"[0 1; {peso:.1f} 0]"
        cost_sensitive = Classifier(classname="weka.classifiers.meta.CostSensitiveClassifier")
        cost_sensitive.options = [
            "-cost-matrix", cost_matrix,
            "-S", "1",
            "-W", "weka.classifiers.trees.ADTree",
            "--", "-B", "10", "-E", "-3"
        ]
        cost_sensitive.build_classifier(dataset_train)

        evaluation = Evaluation(dataset_train)
        evaluation.test_model(cost_sensitive, dataset_test)

        metricas_folds.append(calcular_metricas_fold_cm(evaluation.confusion_matrix))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_folds)
    output_file = f'{output_path}/adtree_{target.lower()}_weighted_metricas.txt'
    exibir_resultados(metricas, target_name, f"ADTree (class weighting {peso:.1f}x)", output_file)

    return metricas


def treinar_adtree_smote(target='GAD'):
    """Treina modelo ADTree com SMOTE aplicado dentro de cada fold."""
    output_path = f'output/plots/ADtree/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print("       MODELO ADTree - COM SMOTE (CORRIGIDO)")
    print("=" * 60)

    print("\n[DATASET ORIGINAL]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    feature_names = df.drop(columns=[target_name]).columns.tolist()

    print("\n[CROSS-VALIDATION COM SMOTE POR FOLD]")
    print("  SMOTE aplicado apenas no treino (sem data leakage)")

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        df_train = pd.DataFrame(X_train_res, columns=feature_names)
        df_train[target_name] = y_train_res
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_name] = y_test

        dataset_train = criar_dataset_weka(df_train)
        dataset_test = criar_dataset_weka(df_test)

        adtree = Classifier(classname="weka.classifiers.trees.ADTree")
        adtree.options = ["-B", "10", "-E", "-3"]
        adtree.build_classifier(dataset_train)

        evaluation = Evaluation(dataset_train)
        evaluation.test_model(adtree, dataset_test)

        metricas_folds.append(calcular_metricas_fold_cm(evaluation.confusion_matrix))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_folds)
    output_file = f'{output_path}/adtree_{target.lower()}_smote_metricas.txt'
    exibir_resultados(metricas, target_name, "ADTree (com SMOTE)", output_file)

    return metricas


def treinar_adtree_undersampling(target='GAD'):
    """Treina modelo ADTree com Undersampling aplicado dentro de cada fold."""
    output_path = f'output/plots/ADtree/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    df, target_name = preparar_dados(target)

    print("\n" + "=" * 60)
    print("    MODELO ADTree - COM UNDERSAMPLING (CORRIGIDO)")
    print("=" * 60)

    print("\n[DATASET ORIGINAL]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    feature_names = df.drop(columns=[target_name]).columns.tolist()

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

        df_train = pd.DataFrame(X_train_res, columns=feature_names)
        df_train[target_name] = y_train_res
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_name] = y_test

        dataset_train = criar_dataset_weka(df_train)
        dataset_test = criar_dataset_weka(df_test)

        adtree = Classifier(classname="weka.classifiers.trees.ADTree")
        adtree.options = ["-B", "10", "-E", "-3"]
        adtree.build_classifier(dataset_train)

        evaluation = Evaluation(dataset_train)
        evaluation.test_model(adtree, dataset_test)

        metricas_folds.append(calcular_metricas_fold_cm(evaluation.confusion_matrix))

    print("OK")

    metricas = agregar_metricas_com_ic(metricas_folds)
    output_file = f'{output_path}/adtree_{target.lower()}_undersampling_metricas.txt'
    exibir_resultados(metricas, target_name, "ADTree (com Undersampling)", output_file)

    return metricas


def instalar_pacote_adtree():
    """Instala o pacote ADTree do Weka se não estiver instalado."""
    import weka.core.packages as packages

    pkg_name = "alternatingDecisionTrees"
    installed = [p.name for p in packages.installed_packages()]

    if pkg_name not in installed:
        print(f"Instalando pacote {pkg_name}...")
        packages.install_package(pkg_name)
        print("Pacote instalado! Reinicie o script.")
        return False
    return True


if __name__ == "__main__":
    import sys
    from io import StringIO

    old_stderr = sys.stderr
    sys.stderr = StringIO()
    jvm.start(packages=True, logging_level=logging.ERROR)
    sys.stderr = old_stderr

    try:
        if instalar_pacote_adtree():
            for target in ['GAD', 'SAD']:
                print("\n" + "#" * 60)
                print(f"{'':>15}ANÁLISE ADTree - {target}")
                print("#" * 60)

                output_path = f'output/plots/ADtree/{target.upper()}'
                plots_path = f'output/plots/ADtree/{target.upper()}/plots'
                os.makedirs(output_path, exist_ok=True)
                os.makedirs(plots_path, exist_ok=True)

                resultados_dict = {}

                m1 = treinar_adtree(target)
                resultados_dict['Sem Balanceamento'] = m1

                print("\n" + "@" * 60 + "\n")

                m2 = treinar_adtree_weighted(target)
                resultados_dict['Class Weighting'] = m2

                print("\n" + "@" * 60 + "\n")

                m3 = treinar_adtree_smote(target)
                resultados_dict['SMOTE'] = m3

                print("\n" + "@" * 60 + "\n")

                m4 = treinar_adtree_undersampling(target)
                resultados_dict['Undersampling'] = m4

                print("\n" + "@" * 60 + "\n")

                resultados = comparativo_modelos(resultados_dict, target, "ADTree", output_path)
                plotar_comparativo_grafico(resultados, target, "ADTree", plots_path)
                plotar_comparativo_tabela(resultados, target, "ADTree", plots_path)

    finally:
        jvm.stop()
