"""
Comparativo de Algoritmos: ADTree vs XGBoost vs SVM
Compara o melhor modelo de cada algoritmo (usando SMOTE corrigido)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from normalizacao import carregar_teste_normalizado

OUTPUT_PATH = 'output/plots/Comparativo'
PLOTS_PATH = 'output/plots/Comparativo/plots'


def preparar_dados(target='GAD'):
    """Prepara os dados para os modelos."""
    df = carregar_teste_normalizado()

    colunas_remover = [
        'Subject',
        'GAD Probabiliy - Gamma',
        'SAD Probability - Gamma',
        'Sample Weight'
    ]

    if target == 'GAD':
        colunas_remover.append('SAD')
    else:
        colunas_remover.append('GAD')

    df_modelo = df.drop(columns=[c for c in colunas_remover if c in df.columns])

    if 'Sex' in df_modelo.columns:
        df_modelo['Sex'] = df_modelo['Sex'].map({'M': 0, 'F': 1})

    cols = [c for c in df_modelo.columns if c != target] + [target]
    df_modelo = df_modelo[cols]
    df_modelo = df_modelo.dropna()

    return df_modelo, target


def calcular_metricas(y_true, y_pred):
    """Calcula todas as métricas."""
    cm = confusion_matrix(y_true, y_pred)
    vn, fp, fn, vp = cm.ravel()

    total = vn + fp + fn + vp
    accuracy = (vn + vp) / total * 100
    sensitivity = vp / (vp + fn) * 100 if (vp + fn) > 0 else 0
    specificity = vn / (vn + fp) * 100 if (vn + fp) > 0 else 0
    ppv = vp / (vp + fp) * 100 if (vp + fp) > 0 else 0
    npv = vn / (vn + fn) * 100 if (vn + fn) > 0 else 0

    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0

    po = (vn + vp) / total
    pe = ((vn + fp) * (vn + fn) + (fn + vp) * (fp + vp)) / (total * total)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    return {
        'vn': vn, 'fp': fp, 'fn': fn, 'vp': vp,
        'accuracy': accuracy, 'sensitivity': sensitivity,
        'specificity': specificity, 'ppv': ppv, 'npv': npv,
        'f1': f1, 'kappa': kappa
    }


def treinar_xgboost_smote(X, y):
    """Treina XGBoost com SMOTE dentro de cada fold."""
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_y_true, all_y_pred = [], []

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

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return calcular_metricas(all_y_true, all_y_pred)


def treinar_svm_smote(X, y):
    """Treina SVM com SMOTE dentro de cada fold."""
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_y_true, all_y_pred = [], []

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

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return calcular_metricas(all_y_true, all_y_pred)


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

        # Verificar pacote ADTree
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
        total_vp, total_vn, total_fp, total_fn = 0, 0, 0, 0

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

            cm = evaluation.confusion_matrix
            total_vn += int(cm[0][0])
            total_fp += int(cm[0][1])
            total_fn += int(cm[1][0])
            total_vp += int(cm[1][1])

        jvm.stop()

        total = total_vn + total_fp + total_fn + total_vp
        accuracy = (total_vn + total_vp) / total * 100
        sensitivity = total_vp / (total_vp + total_fn) * 100 if (total_vp + total_fn) > 0 else 0
        specificity = total_vn / (total_vn + total_fp) * 100 if (total_vn + total_fp) > 0 else 0
        ppv = total_vp / (total_vp + total_fp) * 100 if (total_vp + total_fp) > 0 else 0
        npv = total_vn / (total_vn + total_fn) * 100 if (total_vn + total_fn) > 0 else 0

        precision = total_vp / (total_vp + total_fp) if (total_vp + total_fp) > 0 else 0
        recall = total_vp / (total_vp + total_fn) if (total_vp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0

        po = (total_vn + total_vp) / total
        pe = ((total_vn + total_fp) * (total_vn + total_fn) + (total_fn + total_vp) * (total_fp + total_vp)) / (total * total)
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

        return {
            'vn': total_vn, 'fp': total_fp, 'fn': total_fn, 'vp': total_vp,
            'accuracy': accuracy, 'sensitivity': sensitivity,
            'specificity': specificity, 'ppv': ppv, 'npv': npv,
            'f1': f1, 'kappa': kappa
        }

    except Exception as e:
        print(f"  [ERRO] ADTree: {e}")
        return None


def comparativo_algoritmos(resultados_dict, target_name):
    """Exibe tabela comparativa dos algoritmos."""
    print("\n" + "=" * 90)
    print("              COMPARATIVO DE ALGORITMOS - ADTree vs XGBoost vs SVM")
    print("=" * 90)

    print(f"\n{'Algoritmo':<20} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'F1':>8} {'Kappa':>8}")
    print("-" * 90)

    resultados = []
    for nome, m in resultados_dict.items():
        if m is not None:
            print(f"{nome:<20} {m['accuracy']:>7.2f}% {m['sensitivity']:>7.2f}% {m['specificity']:>7.2f}% {m['ppv']:>7.2f}% {m['f1']:>7.2f}% {m['kappa']:>8.4f}")
            resultados.append({
                'Algoritmo': nome,
                'Accuracy': m['accuracy'],
                'Sensitivity': m['sensitivity'],
                'Specificity': m['specificity'],
                'PPV': m['ppv'],
                'F1-Score': m['f1'],
                'Kappa': m['kappa']
            })

    print("-" * 90)

    print("\n[MELHORES POR MÉTRICA]")
    metricas = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score', 'Kappa']
    for metrica in metricas:
        melhor = max(resultados, key=lambda x: x[metrica])
        print(f"  {metrica:<12}: {melhor['Algoritmo']} ({melhor[metrica]:.2f}{'%' if metrica != 'Kappa' else ''})")

    # Salvar
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = f'{OUTPUT_PATH}/comparativo_algoritmos_{target_name.lower()}.txt'
    with open(output_file, 'w') as f:
        f.write(f"COMPARATIVO DE ALGORITMOS - {target_name}\n")
        f.write("ADTree vs XGBoost vs SVM (todos com SMOTE corrigido)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Algoritmo':<20} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'F1':>8} {'Kappa':>8}\n")
        f.write("-" * 90 + "\n")
        for r in resultados:
            f.write(f"{r['Algoritmo']:<20} {r['Accuracy']:>7.2f}% {r['Sensitivity']:>7.2f}% {r['Specificity']:>7.2f}% {r['PPV']:>7.2f}% {r['F1-Score']:>7.2f}% {r['Kappa']:>8.4f}\n")
        f.write("-" * 90 + "\n\n")
        f.write("MELHORES POR MÉTRICA\n")
        f.write("-" * 40 + "\n")
        for metrica in metricas:
            melhor = max(resultados, key=lambda x: x[metrica])
            f.write(f"{metrica:<12}: {melhor['Algoritmo']} ({melhor[metrica]:.2f}{'%' if metrica != 'Kappa' else ''})\n")

    print(f"\n  Comparativo salvo em: {output_file}")
    print("\n" + "=" * 90 + "\n")

    return resultados


def plotar_comparativo(resultados, target_name):
    """Gera gráfico e tabela comparativa dos algoritmos."""
    algoritmos = [r['Algoritmo'] for r in resultados]
    metricas = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score', 'Kappa']
    metricas_grafico = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score']

    cores = ['#e67e22', '#27ae60', '#8e44ad']  # Laranja, Verde, Roxo

    # Gráfico de barras
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(metricas_grafico))
    largura = 0.25
    offsets = [-1, 0, 1]

    for i, (algo, cor) in enumerate(zip(algoritmos, cores)):
        valores = [resultados[i][m] for m in metricas_grafico]
        barras = ax.bar(x + offsets[i] * largura, valores, largura,
                        label=algo, color=cor, edgecolor='white', linewidth=0.5)
        for barra, valor in zip(barras, valores):
            ax.annotate(f'{valor:.1f}',
                        xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Porcentagem (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparativo de Algoritmos - {target_name}\nADTree vs XGBoost vs SVM (com SMOTE)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_grafico, fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    grafico_file = f'{PLOTS_PATH}/comparativo_algoritmos_{target_name.lower()}_grafico.png'
    plt.savefig(grafico_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico salvo em: {grafico_file}")

    # Tabela
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    colunas = ['Algoritmo'] + metricas
    dados_tabela = []
    for r in resultados:
        linha = [r['Algoritmo']]
        for m in metricas:
            if m == 'Kappa':
                linha.append(f"{r[m]:.4f}")
            else:
                linha.append(f"{r[m]:.2f}%")
        dados_tabela.append(linha)

    tabela = ax.table(cellText=dados_tabela, colLabels=colunas, cellLoc='center', loc='center',
                      colColours=['#2c3e50'] + ['#34495e'] * len(metricas))
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1.2, 2.2)

    for j in range(len(colunas)):
        tabela[(0, j)].set_text_props(color='white', fontweight='bold')

    for i, r in enumerate(resultados):
        cor_fundo = cores[i] + '30'
        for j in range(len(colunas)):
            tabela[(i + 1, j)].set_facecolor(cor_fundo)

    for j, metrica in enumerate(metricas):
        valores = [r[metrica] for r in resultados]
        melhor_idx = valores.index(max(valores))
        tabela[(melhor_idx + 1, j + 1)].set_text_props(fontweight='bold', color='#27ae60')

    ax.set_title(f'Comparativo de Algoritmos - {target_name}',
                 fontsize=14, fontweight='bold', pad=20, y=0.98)
    plt.tight_layout()

    tabela_file = f'{PLOTS_PATH}/comparativo_algoritmos_{target_name.lower()}_tabela.png'
    plt.savefig(tabela_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Tabela salva em: {tabela_file}")

    return grafico_file, tabela_file


if __name__ == "__main__":
    target = 'GAD'

    print("\n" + "=" * 90)
    print("         COMPARATIVO DE ALGORITMOS: ADTree vs XGBoost vs SVM")
    print("         Todos usando SMOTE com correção de data leakage")
    print("=" * 90)

    # Preparar dados
    df, target_name = preparar_dados(target)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    feature_names = df.drop(columns=[target_name]).columns.tolist()

    print(f"\n[DATASET]")
    print(f"  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1} | Target: {target_name}")
    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    resultados_dict = {}

    # XGBoost
    print("\n" + "-" * 60)
    print("  Treinando XGBoost + SMOTE...", end=" ")
    m_xgb = treinar_xgboost_smote(X, y)
    print("OK")
    resultados_dict['XGBoost'] = m_xgb

    # SVM
    print("  Treinando SVM + SMOTE...", end=" ")
    m_svm = treinar_svm_smote(X, y)
    print("OK")
    resultados_dict['SVM'] = m_svm

    # ADTree
    print("  Treinando ADTree + SMOTE...", end=" ")
    m_adtree = treinar_adtree_smote(X, y, feature_names, target_name)
    if m_adtree:
        print("OK")
        resultados_dict['ADTree'] = m_adtree
    else:
        print("FALHOU (Weka não disponível)")

    print("-" * 60)

    # Comparativo
    resultados = comparativo_algoritmos(resultados_dict, target_name)

    # Gráficos
    plotar_comparativo(resultados, target_name)
