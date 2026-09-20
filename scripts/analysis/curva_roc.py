"""
Curva ROC + AUC para todos os algoritmos e técnicas de balanceamento.
Gera curvas ROC médias com banda de ± 1σ usando 10-fold stratified CV.

Saídas por algoritmo (XGBoost, SVM):
  - Gráfico ROC com 4 técnicas de balanceamento
  - Arquivo TXT com métricas AUC por fold e IC 95%

Saída comparativa:
  - Gráfico ROC com o melhor modelo de cada algoritmo (SMOTE)
  - Arquivo TXT com AUC comparativo
"""

import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from scripts.utils import (
    preparar_dados, coletar_roc_folds,
    plotar_curvas_roc, salvar_auc_metricas
)


# ============================================================
#  XGBoost - Coleta de probabilidades por técnica
# ============================================================

def roc_xgboost_sem_balanceamento(X, y):
    """XGBoost sem balanceamento - coleta predict_proba por fold."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_scores = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train, y_train)
        y_scores.append(model.predict_proba(X_test)[:, 1])
        y_trues.append(y_test)

    return coletar_roc_folds(y_trues, y_scores)


def roc_xgboost_weighted(X, y):
    """XGBoost com scale_pos_weight - coleta predict_proba por fold."""
    peso = np.sum(y == 0) / np.sum(y == 1)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_scores = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=peso, random_state=42,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        model.fit(X_train, y_train)
        y_scores.append(model.predict_proba(X_test)[:, 1])
        y_trues.append(y_test)

    return coletar_roc_folds(y_trues, y_scores)


def roc_xgboost_smote(X, y):
    """XGBoost com SMOTE (por fold) - coleta predict_proba por fold."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_scores = [], []

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
        y_scores.append(model.predict_proba(X_test)[:, 1])
        y_trues.append(y_test)

    return coletar_roc_folds(y_trues, y_scores)


def roc_xgboost_undersampling(X, y):
    """XGBoost com Undersampling (por fold) - coleta predict_proba por fold."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_scores = [], []

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
        y_scores.append(model.predict_proba(X_test)[:, 1])
        y_trues.append(y_test)

    return coletar_roc_folds(y_trues, y_scores)


# ============================================================
#  SVM - Coleta de decision_function por técnica
# ============================================================

def roc_svm_sem_balanceamento(X, y):
    """SVM sem balanceamento - coleta decision_function por fold."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_scores = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_scaled, y_train)
        y_scores.append(model.decision_function(X_test_scaled))
        y_trues.append(y_test)

    return coletar_roc_folds(y_trues, y_scores)


def roc_svm_weighted(X, y):
    """SVM com class_weight='balanced' - coleta decision_function por fold."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_scores = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=1.0, gamma='scale',
                     class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)
        y_scores.append(model.decision_function(X_test_scaled))
        y_trues.append(y_test)

    return coletar_roc_folds(y_trues, y_scores)


def roc_svm_smote(X, y):
    """SVM com SMOTE (por fold) - coleta decision_function por fold."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_scores = [], []

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
        y_scores.append(model.decision_function(X_test_scaled))
        y_trues.append(y_test)

    return coletar_roc_folds(y_trues, y_scores)


def roc_svm_undersampling(X, y):
    """SVM com Undersampling (por fold) - coleta decision_function por fold."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_scores = [], []

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
        y_scores.append(model.decision_function(X_test_scaled))
        y_trues.append(y_test)

    return coletar_roc_folds(y_trues, y_scores)


# ============================================================
#  ADTree (opcional - requer Weka/JVM)
# ============================================================

def roc_adtree_tecnica(X, y, feature_names, target_name, tecnica='sem_balanceamento'):
    """ADTree - coleta probabilidades via Weka distribution_for_instance.

    Args:
        tecnica: 'sem_balanceamento', 'weighted', 'smote', 'undersampling'
    Returns:
        roc_dados dict ou None se Weka não disponível
    """
    try:
        import logging
        logging.getLogger("weka").setLevel(logging.ERROR)
        os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.util.logging.config.file=/dev/null"

        import weka.core.jvm as jvm
        from weka.core.dataset import Instances, Attribute, Instance
        from weka.classifiers import Classifier, Evaluation

        import sys
        from io import StringIO
        import pandas as pd

        if not jvm.started:
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            jvm.start(packages=True, logging_level=logging.ERROR)
            sys.stderr = old_stderr

        import weka.core.packages as packages
        pkg_name = "alternatingDecisionTrees"
        installed = [p.name for p in packages.installed_packages()]
        if pkg_name not in installed:
            print("  [AVISO] Pacote ADTree não instalado.")
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

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_trues, y_scores = [], []
        peso = np.sum(y == 0) / np.sum(y == 1)

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if tecnica == 'smote':
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            elif tecnica == 'undersampling':
                undersampler = RandomUnderSampler(random_state=42)
                X_train, y_train = undersampler.fit_resample(X_train, y_train)

            df_train = pd.DataFrame(X_train, columns=feature_names)
            df_train[target_name] = y_train
            df_test = pd.DataFrame(X_test, columns=feature_names)
            df_test[target_name] = y_test

            dataset_train = criar_dataset_weka(df_train)
            dataset_test = criar_dataset_weka(df_test)

            if tecnica == 'weighted':
                cost_matrix = f"[0 1; {peso:.1f} 0]"
                classifier = Classifier(classname="weka.classifiers.meta.CostSensitiveClassifier")
                classifier.options = [
                    "-cost-matrix", cost_matrix, "-S", "1",
                    "-W", "weka.classifiers.trees.ADTree",
                    "--", "-B", "10", "-E", "-3"
                ]
            else:
                classifier = Classifier(classname="weka.classifiers.trees.ADTree")
                classifier.options = ["-B", "10", "-E", "-3"]

            classifier.build_classifier(dataset_train)

            scores = []
            for i in range(dataset_test.num_instances):
                inst = dataset_test.get_instance(i)
                dist = classifier.distribution_for_instance(inst)
                scores.append(dist[1])

            y_scores.append(np.array(scores))
            y_trues.append(y_test)

        return coletar_roc_folds(y_trues, y_scores)

    except Exception as e:
        print(f"  [AVISO] ADTree não disponível: {e}")
        return None


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    cores_tecnicas = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    cores_algoritmos = ['#e67e22', '#27ae60', '#8e44ad']

    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 70)
        print(f"         CURVA ROC + AUC - {target}")
        print(f"         10-Fold Stratified CV (random_state=42)")
        print("#" * 70)

        df, target_name = preparar_dados(target)
        X = df.drop(columns=[target_name]).values
        y = df[target_name].values
        feature_names = df.drop(columns=[target_name]).columns.tolist()

        print(f"\n[DATASET]")
        print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
        dist = df[target_name].value_counts()
        print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

        # ==================== XGBoost ====================
        print("\n" + "=" * 60)
        print("  XGBoost - Coletando probabilidades...")
        print("=" * 60)

        roc_xgb = {}
        print("    Sem Balanceamento...", end=" ")
        roc_xgb['Sem Balanceamento'] = roc_xgboost_sem_balanceamento(X, y)
        print(f"AUC = {roc_xgb['Sem Balanceamento']['mean_auc']:.3f}")

        print("    Class Weighting...", end=" ")
        roc_xgb['Class Weighting'] = roc_xgboost_weighted(X, y)
        print(f"AUC = {roc_xgb['Class Weighting']['mean_auc']:.3f}")

        print("    SMOTE...", end=" ")
        roc_xgb['SMOTE'] = roc_xgboost_smote(X, y)
        print(f"AUC = {roc_xgb['SMOTE']['mean_auc']:.3f}")

        print("    Undersampling...", end=" ")
        roc_xgb['Undersampling'] = roc_xgboost_undersampling(X, y)
        print(f"AUC = {roc_xgb['Undersampling']['mean_auc']:.3f}")

        xgb_plots = f'output/plots/XGBoost/{target}/plots'
        xgb_output = f'output/plots/XGBoost/{target}'
        plotar_curvas_roc(roc_xgb, target, "XGBoost", f'{xgb_plots}/roc_xgboost_{target.lower()}.png', cores_tecnicas)
        salvar_auc_metricas(roc_xgb, target, "XGBoost", f'{xgb_output}/xgboost_{target.lower()}_auc.txt')

        # ==================== SVM ====================
        print("\n" + "=" * 60)
        print("  SVM - Coletando decision_function scores...")
        print("=" * 60)

        roc_svm = {}
        print("    Sem Balanceamento...", end=" ")
        roc_svm['Sem Balanceamento'] = roc_svm_sem_balanceamento(X, y)
        print(f"AUC = {roc_svm['Sem Balanceamento']['mean_auc']:.3f}")

        print("    Class Weighting...", end=" ")
        roc_svm['Class Weighting'] = roc_svm_weighted(X, y)
        print(f"AUC = {roc_svm['Class Weighting']['mean_auc']:.3f}")

        print("    SMOTE...", end=" ")
        roc_svm['SMOTE'] = roc_svm_smote(X, y)
        print(f"AUC = {roc_svm['SMOTE']['mean_auc']:.3f}")

        print("    Undersampling...", end=" ")
        roc_svm['Undersampling'] = roc_svm_undersampling(X, y)
        print(f"AUC = {roc_svm['Undersampling']['mean_auc']:.3f}")

        svm_plots = f'output/plots/SVM/{target}/plots'
        svm_output = f'output/plots/SVM/{target}'
        plotar_curvas_roc(roc_svm, target, "SVM", f'{svm_plots}/roc_svm_{target.lower()}.png', cores_tecnicas)
        salvar_auc_metricas(roc_svm, target, "SVM", f'{svm_output}/svm_{target.lower()}_auc.txt')

        # ==================== ADTree (opcional) ====================
        print("\n" + "=" * 60)
        print("  ADTree - Tentando coletar probabilidades via Weka...")
        print("=" * 60)

        roc_adtree = {}
        adtree_disponivel = False

        for tecnica_nome, tecnica_id in [('Sem Balanceamento', 'sem_balanceamento'),
                                          ('Class Weighting', 'weighted'),
                                          ('SMOTE', 'smote'),
                                          ('Undersampling', 'undersampling')]:
            print(f"    {tecnica_nome}...", end=" ")
            resultado = roc_adtree_tecnica(X, y, feature_names, target_name, tecnica_id)
            if resultado is not None:
                roc_adtree[tecnica_nome] = resultado
                adtree_disponivel = True
                print(f"AUC = {resultado['mean_auc']:.3f}")
            else:
                print("IGNORADO")
                break

        if adtree_disponivel and roc_adtree:
            adt_plots = f'output/plots/ADtree/{target}/plots'
            adt_output = f'output/plots/ADtree/{target}'
            plotar_curvas_roc(roc_adtree, target, "ADTree", f'{adt_plots}/roc_adtree_{target.lower()}.png', cores_tecnicas)
            salvar_auc_metricas(roc_adtree, target, "ADTree", f'{adt_output}/adtree_{target.lower()}_auc.txt')
        else:
            print("  ADTree ignorado (Weka não disponível)")

        # ==================== Comparativo ====================
        print("\n" + "=" * 60)
        print("  COMPARATIVO - Melhor técnica de cada algoritmo (SMOTE)")
        print("=" * 60)

        roc_comparativo = {
            'XGBoost + SMOTE': roc_xgb['SMOTE'],
            'SVM + SMOTE': roc_svm['SMOTE'],
        }

        if adtree_disponivel and 'SMOTE' in roc_adtree:
            roc_comparativo['ADTree + SMOTE'] = roc_adtree['SMOTE']

        comp_plots = f'output/plots/Comparativo/{target}'
        os.makedirs(comp_plots, exist_ok=True)
        plotar_curvas_roc(roc_comparativo, target, "Comparativo de Algoritmos",
                          f'{comp_plots}/roc_comparativo_{target.lower()}.png', cores_algoritmos)
        salvar_auc_metricas(roc_comparativo, target, "Comparativo",
                            f'{comp_plots}/comparativo_{target.lower()}_auc.txt')

        # ==================== Resumo ====================
        print("\n" + "=" * 60)
        print(f"  RESUMO AUC - {target}")
        print("=" * 60)

        print(f"\n  {'Modelo':<30} {'AUC':>8}")
        print("  " + "-" * 40)

        todos_modelos = {}
        for nome, dados in roc_xgb.items():
            todos_modelos[f'XGBoost {nome}'] = dados['mean_auc']
        for nome, dados in roc_svm.items():
            todos_modelos[f'SVM {nome}'] = dados['mean_auc']
        if adtree_disponivel:
            for nome, dados in roc_adtree.items():
                todos_modelos[f'ADTree {nome}'] = dados['mean_auc']

        for nome, auc_val in sorted(todos_modelos.items(), key=lambda x: x[1], reverse=True):
            print(f"  {nome:<30} {auc_val:>8.4f}")

        print("\n" + "#" * 70 + "\n")

    # Fechar JVM do Weka se foi iniciada
    try:
        import weka.core.jvm as jvm
        if jvm.started:
            jvm.stop()
    except ImportError:
        pass
