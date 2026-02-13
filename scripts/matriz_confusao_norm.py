"""
Matrizes de Confusao Normalizadas para todos os algoritmos e tecnicas.

Gera matrizes de confusao normalizadas por classe real (cada linha soma 100%),
permitindo visualizar proporcoes de acertos/erros independente do desbalanceamento.

Saidas por algoritmo (XGBoost, SVM):
  - Grid 2x2 com as 4 tecnicas de balanceamento (PNG)

Saida comparativa:
  - Grid 1x2 ou 1x3 com SMOTE de cada algoritmo (PNG)
"""

import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from utils import (
    preparar_dados, plotar_grid_matrizes_confusao, plotar_matriz_confusao_normalizada
)

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
#  Coleta de matrizes de confusao agregadas
# ============================================================

def coletar_cm_xgboost(X, y, tecnica='smote'):
    """Treina XGBoost e retorna CM agregada [[VN,FP],[FN,VP]]."""
    peso = np.sum(y == 0) / np.sum(y == 1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cm_total = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        kwargs = {}
        if tecnica == 'weighted':
            kwargs['scale_pos_weight'] = peso
        elif tecnica == 'smote':
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        elif tecnica == 'undersampling':
            undersampler = RandomUnderSampler(random_state=42)
            X_train, y_train = undersampler.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0, **kwargs
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm_total += confusion_matrix(y_test, y_pred)

    return cm_total


def coletar_cm_svm(X, y, tecnica='smote'):
    """Treina SVM e retorna CM agregada [[VN,FP],[FN,VP]]."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cm_total = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if tecnica == 'smote':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        elif tecnica == 'undersampling':
            undersampler = RandomUnderSampler(random_state=42)
            X_train, y_train = undersampler.fit_resample(X_train, y_train)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        kwargs = {}
        if tecnica == 'weighted':
            kwargs['class_weight'] = 'balanced'

        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, **kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm_total += confusion_matrix(y_test, y_pred)

    return cm_total


def coletar_cm_adtree(X, y, feature_names, target_name, tecnica='smote'):
    """Treina ADTree via Weka e retorna CM agregada [[VN,FP],[FN,VP]]."""
    try:
        import logging
        logging.getLogger("weka").setLevel(logging.ERROR)
        os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.util.logging.config.file=/dev/null"

        import weka.core.jvm as jvm
        from weka.core.dataset import Instances, Attribute, Instance
        from weka.classifiers import Classifier, Evaluation

        import sys
        from io import StringIO

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

        peso = np.sum(y == 0) / np.sum(y == 1)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cm_total = np.zeros((2, 2), dtype=int)

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

            evaluation = Evaluation(dataset_train)
            evaluation.test_model(classifier, dataset_test)

            cm_fold = np.array(evaluation.confusion_matrix, dtype=int)
            cm_total += cm_fold

        return cm_total

    except Exception as e:
        print(f"  [AVISO] ADTree não disponível: {e}")
        return None


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    tecnicas = [
        ('sem_balanceamento', 'Sem Balanceamento'),
        ('weighted', 'Class Weighting'),
        ('smote', 'SMOTE'),
        ('undersampling', 'Undersampling'),
    ]

    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 70)
        print(f"    MATRIZES DE CONFUSAO NORMALIZADAS - {target}")
        print("#" * 70)

        df, target_name = preparar_dados(target)
        X = df.drop(columns=[target_name]).values
        y = df[target_name].values

        print(f"\n[DATASET]")
        print(f"  Amostras: {len(y)} | Features: {X.shape[1]} | Target: {target_name}")
        dist_vals = np.bincount(y.astype(int))
        print(f"  Classe 0: {dist_vals[0]} ({dist_vals[0]/len(y)*100:.1f}%) | "
              f"Classe 1: {dist_vals[1]} ({dist_vals[1]/len(y)*100:.1f}%)")

        # ==================== XGBoost ====================
        print("\n  XGBoost - Coletando matrizes de confusao...")
        cms_xgb = {}
        for tecnica_id, tecnica_nome in tecnicas:
            print(f"    {tecnica_nome}...", end=" ")
            cm = coletar_cm_xgboost(X, y, tecnica_id)
            cms_xgb[tecnica_nome] = cm
            vn, fp, fn, vp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            print(f"VN={vn} FP={fp} FN={fn} VP={vp}")

        xgb_plots = f'output/plots/XGBoost/{target}/plots'
        plotar_grid_matrizes_confusao(
            cms_xgb, target_name, "XGBoost",
            f'{xgb_plots}/matrizes_confusao_norm_xgboost_{target.lower()}.png')

        # ==================== SVM ====================
        print("\n  SVM - Coletando matrizes de confusao...")
        cms_svm = {}
        for tecnica_id, tecnica_nome in tecnicas:
            print(f"    {tecnica_nome}...", end=" ")
            cm = coletar_cm_svm(X, y, tecnica_id)
            cms_svm[tecnica_nome] = cm
            vn, fp, fn, vp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            print(f"VN={vn} FP={fp} FN={fn} VP={vp}")

        svm_plots = f'output/plots/SVM/{target}/plots'
        plotar_grid_matrizes_confusao(
            cms_svm, target_name, "SVM",
            f'{svm_plots}/matrizes_confusao_norm_svm_{target.lower()}.png')

        # ==================== ADTree (opcional) ====================
        print("\n  ADTree - Coletando matrizes de confusao...")
        feature_names = df.drop(columns=[target_name]).columns.tolist()
        cms_adtree = {}
        adtree_disponivel = False

        for tecnica_id, tecnica_nome in tecnicas:
            print(f"    {tecnica_nome}...", end=" ")
            cm = coletar_cm_adtree(X, y, feature_names, target_name, tecnica_id)
            if cm is not None:
                cms_adtree[tecnica_nome] = cm
                adtree_disponivel = True
                vn, fp, fn, vp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                print(f"VN={vn} FP={fp} FN={fn} VP={vp}")
            else:
                print("IGNORADO")
                break

        if adtree_disponivel and cms_adtree:
            adt_plots = f'output/plots/ADtree/{target}/plots'
            plotar_grid_matrizes_confusao(
                cms_adtree, target_name, "ADTree",
                f'{adt_plots}/matrizes_confusao_norm_adtree_{target.lower()}.png')

        # ==================== Comparativo (SMOTE) ====================
        print("\n  Comparativo - SMOTE de cada algoritmo...")
        cms_comp = {
            'XGBoost + SMOTE': cms_xgb['SMOTE'],
            'SVM + SMOTE': cms_svm['SMOTE'],
        }

        if adtree_disponivel and 'SMOTE' in cms_adtree:
            cms_comp['ADTree + SMOTE'] = cms_adtree['SMOTE']

        comp_plots = f'output/plots/Comparativo/{target}'
        os.makedirs(comp_plots, exist_ok=True)

        # Plot comparativo em linha
        n_modelos = len(cms_comp)
        fig, axes = plt.subplots(1, n_modelos, figsize=(6 * n_modelos, 5))
        if n_modelos == 1:
            axes = [axes]

        for i, (nome, cm) in enumerate(cms_comp.items()):
            plotar_matriz_confusao_normalizada(cm, target_name, nome, ax=axes[i])

        fig.suptitle(f'Matrizes de Confus\u00e3o Normalizadas - Comparativo ({target_name})\n'
                     f'Normalizado por classe real (linhas somam 100%)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        comp_file = f'{comp_plots}/matrizes_confusao_norm_comparativo_{target.lower()}.png'
        plt.savefig(comp_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Comparativo salvo em: {comp_file}")

        # ==================== Resumo textual ====================
        output_txt = f'{comp_plots}/matrizes_confusao_norm_{target.lower()}.txt'
        with open(output_txt, 'w') as f:
            f.write(f"MATRIZES DE CONFUSAO NORMALIZADAS - {target_name}\n")
            f.write("=" * 70 + "\n")
            f.write("Normalizacao: por classe real (cada linha soma 100%)\n")
            f.write("Agregadas dos 10 folds de cross-validation\n\n")

            todos_algos = [('XGBoost', cms_xgb), ('SVM', cms_svm)]
            if adtree_disponivel and cms_adtree:
                todos_algos.append(('ADTree', cms_adtree))

            for algo_nome, cms_dict in todos_algos:
                f.write(f"\n{'='*70}\n")
                f.write(f"{algo_nome}\n")
                f.write(f"{'='*70}\n\n")

                for tecnica_nome, cm in cms_dict.items():
                    cm_float = cm.astype(float)
                    total_row0 = cm_float[0].sum()
                    total_row1 = cm_float[1].sum()

                    spec = cm_float[0, 0] / total_row0 * 100 if total_row0 > 0 else 0
                    fpr = cm_float[0, 1] / total_row0 * 100 if total_row0 > 0 else 0
                    fnr = cm_float[1, 0] / total_row1 * 100 if total_row1 > 0 else 0
                    sens = cm_float[1, 1] / total_row1 * 100 if total_row1 > 0 else 0

                    f.write(f"  {tecnica_nome}:\n")
                    f.write(f"    Sem {target_name} (n={int(total_row0)}): "
                            f"Correto={spec:.1f}% ({int(cm[0,0])})  "
                            f"Errado={fpr:.1f}% ({int(cm[0,1])})\n")
                    f.write(f"    Com {target_name} (n={int(total_row1)}): "
                            f"Errado={fnr:.1f}% ({int(cm[1,0])})  "
                            f"Correto={sens:.1f}% ({int(cm[1,1])})\n")
                    f.write(f"    -> Especificidade={spec:.1f}%  Sensibilidade={sens:.1f}%\n\n")

        print(f"  Resumo salvo em: {output_txt}")
        print("\n" + "#" * 70 + "\n")
