"""
Comparativo de Algoritmos: ADTree vs XGBoost vs SVM
Compara o melhor modelo de cada algoritmo (usando SMOTE corrigido)
Inclui Intervalos de Confiança (IC 95%)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from scripts.utils import (
    preparar_dados, calcular_metricas_fold, calcular_metricas_fold_cm,
    agregar_metricas_com_ic, plotar_comparativo_grafico, plotar_comparativo_tabela
)

OUTPUT_PATH = 'output/plots/Comparativo'
PLOTS_PATH = 'output/plots/Comparativo/plots'


def treinar_xgboost_smote(X, y):
    """Treina XGBoost com SMOTE dentro de cada fold."""
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

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

        metricas_folds.append(calcular_metricas_fold(y_test, y_pred))

    return agregar_metricas_com_ic(metricas_folds)


def treinar_svm_smote(X, y):
    """Treina SVM com SMOTE dentro de cada fold."""
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metricas_folds = []

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

    return agregar_metricas_com_ic(metricas_folds)


def treinar_adtree_smote(X, y, feature_names, target_name):
    """Treina ADTree com SMOTE dentro de cada fold (requer Weka)."""
    try:
        import logging
        logging.getLogger("weka").setLevel(logging.ERROR)
        os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.util.logging.config.file=/dev/null"

        import weka.core.jvm as jvm
        from weka.core.dataset import Instances, Attribute, Instance
        from weka.classifiers import Classifier, Evaluation

        import sys
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        jvm.start(packages=True, logging_level=logging.ERROR)
        sys.stderr = old_stderr

        import weka.core.packages as packages
        pkg_name = "alternatingDecisionTrees"
        installed = [p.name for p in packages.installed_packages()]
        if pkg_name not in installed:
            print("  [AVISO] Pacote ADTree não instalado. Instalando...")
            packages.install_package(pkg_name)
            print("  [AVISO] Reinicie o script após instalação.")
            jvm.stop()
            return None

        def criar_dataset_weka(df):
            atts = []
            for col in df.columns:
                valores_unicos = sorted(df[col].dropna().unique())
                valores_str = [str(int(v)) if v == int(v) else str(v) for v in valores_unicos]
                atts.append(Attribute.create_nominal(col, valores_str))

            dataset = Instances.create_instances("dados", atts, len(df))

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

        n_folds = 10
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        metricas_folds = []

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

        jvm.stop()
        return agregar_metricas_com_ic(metricas_folds)

    except Exception as e:
        print(f"  [ERRO] ADTree: {e}")
        return None


def comparativo_algoritmos(resultados_dict, target_name):
    """Exibe tabela comparativa dos algoritmos com IC."""
    print("\n" + "=" * 110)
    print("              COMPARATIVO DE ALGORITMOS - ADTree vs XGBoost vs SVM (com IC 95%)")
    print("=" * 110)

    print(f"\n{'Algoritmo':<15} {'Accuracy':>18} {'Sensitivity':>18} {'Specificity':>18} {'F1-Score':>18} {'Kappa':>16}")
    print("-" * 110)

    resultados = []
    for nome, m in resultados_dict.items():
        if m is not None:
            acc_str = f"{m['accuracy']:.1f}±{m['accuracy_ic']:.1f}"
            sens_str = f"{m['sensitivity']:.1f}±{m['sensitivity_ic']:.1f}"
            spec_str = f"{m['specificity']:.1f}±{m['specificity_ic']:.1f}"
            f1_str = f"{m['f1']:.1f}±{m['f1_ic']:.1f}"
            kappa_str = f"{m['kappa']:.3f}±{m['kappa_ic']:.3f}"

            print(f"{nome:<15} {acc_str:>18} {sens_str:>18} {spec_str:>18} {f1_str:>18} {kappa_str:>16}")

            resultados.append({
                'Modelo': nome,
                'Accuracy': m['accuracy'],
                'Accuracy_IC': m['accuracy_ic'],
                'Sensitivity': m['sensitivity'],
                'Sensitivity_IC': m['sensitivity_ic'],
                'Specificity': m['specificity'],
                'Specificity_IC': m['specificity_ic'],
                'PPV': m['ppv'],
                'PPV_IC': m['ppv_ic'],
                'F1-Score': m['f1'],
                'F1-Score_IC': m['f1_ic'],
                'Kappa': m['kappa'],
                'Kappa_IC': m['kappa_ic']
            })

    print("-" * 110)

    output_path_target = f'{OUTPUT_PATH}/{target_name.upper()}'
    os.makedirs(output_path_target, exist_ok=True)
    output_file = f'{output_path_target}/comparativo_algoritmos_{target_name.lower()}.txt'
    with open(output_file, 'w') as f:
        f.write(f"COMPARATIVO DE ALGORITMOS - {target_name} (com IC 95%)\n")
        f.write("ADTree vs XGBoost vs SVM (todos com SMOTE corrigido)\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"{'Algoritmo':<15} {'Accuracy':>18} {'Sensitivity':>18} {'Specificity':>18} {'F1-Score':>18} {'Kappa':>16}\n")
        f.write("-" * 110 + "\n")
        for r in resultados:
            acc_str = f"{r['Accuracy']:.1f}±{r['Accuracy_IC']:.1f}"
            sens_str = f"{r['Sensitivity']:.1f}±{r['Sensitivity_IC']:.1f}"
            spec_str = f"{r['Specificity']:.1f}±{r['Specificity_IC']:.1f}"
            f1_str = f"{r['F1-Score']:.1f}±{r['F1-Score_IC']:.1f}"
            kappa_str = f"{r['Kappa']:.3f}±{r['Kappa_IC']:.3f}"
            f.write(f"{r['Modelo']:<15} {acc_str:>18} {sens_str:>18} {spec_str:>18} {f1_str:>18} {kappa_str:>16}\n")

    print(f"\n  Comparativo salvo em: {output_file}")
    print("\n" + "=" * 110 + "\n")

    return resultados


if __name__ == "__main__":
    cores_algoritmos = ['#e67e22', '#27ae60', '#8e44ad']  # Laranja, Verde, Roxo

    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 90)
        print(f"         COMPARATIVO DE ALGORITMOS: ADTree vs XGBoost vs SVM - {target}")
        print("         Todos usando SMOTE com correção de data leakage")
        print("         Inclui Intervalos de Confiança (IC 95%)")
        print("#" * 90)

        df, target_name = preparar_dados(target)
        X = df.drop(columns=[target_name]).values
        y = df[target_name].values
        feature_names = df.drop(columns=[target_name]).columns.tolist()

        print(f"\n[DATASET]")
        print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
        dist = df[target_name].value_counts()
        print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

        plots_path_target = f'{PLOTS_PATH}/{target_name.upper()}'
        os.makedirs(plots_path_target, exist_ok=True)

        resultados_dict = {}

        print("\n" + "-" * 60)
        print("  Treinando XGBoost + SMOTE...", end=" ")
        m_xgb = treinar_xgboost_smote(X, y)
        print("OK")
        resultados_dict['XGBoost'] = m_xgb

        print("  Treinando SVM + SMOTE...", end=" ")
        m_svm = treinar_svm_smote(X, y)
        print("OK")
        resultados_dict['SVM'] = m_svm

        print("  Treinando ADTree + SMOTE...", end=" ")
        m_adtree = treinar_adtree_smote(X, y, feature_names, target_name)
        if m_adtree:
            print("OK")
            resultados_dict['ADTree'] = m_adtree
        else:
            print("FALHOU (Weka não disponível)")

        print("-" * 60)

        resultados = comparativo_algoritmos(resultados_dict, target_name)
        plotar_comparativo_grafico(resultados, target_name, "Algoritmos", plots_path_target, cores=cores_algoritmos)
        plotar_comparativo_tabela(resultados, target_name, "Algoritmos", plots_path_target, cores=cores_algoritmos)
