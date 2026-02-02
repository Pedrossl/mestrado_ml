import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from normalizacao import carregar_teste_normalizado

OUTPUT_PATH = 'output/plots/SVM'
PLOTS_PATH = 'output/plots/SVM/plots'


def preparar_dados(target='GAD'):
    """Prepara os dados para o modelo."""
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
    """Calcula todas as métricas a partir de y_true e y_pred."""
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


def exibir_resultados(metricas, target_name, modelo_nome, output_file):
    """Exibe e salva os resultados."""
    vn, fp, fn, vp = metricas['vn'], metricas['fp'], metricas['fn'], metricas['vp']
    total = vn + fp + fn + vp

    print("\n" + "=" * 60)
    print("                      RESULTADOS")
    print("=" * 60)

    print("\n[MÉTRICAS DE DESEMPENHO]")
    print(f"  Accuracy (Acurácia):             {metricas['accuracy']:>6.2f}%")
    print(f"  Sensitivity (Sensibilidade):    {metricas['sensitivity']:>6.2f}%")
    print(f"  Specificity (Especificidade):   {metricas['specificity']:>6.2f}%")
    print(f"  PPV - Precision (Precisão):     {metricas['ppv']:>6.2f}%")
    print(f"  NPV (Valor Pred. Negativo):     {metricas['npv']:>6.2f}%")
    print(f"  F1-Score:                       {metricas['f1']:>6.2f}%")
    print(f"  Kappa:                          {metricas['kappa']:>6.4f}")

    print("\n[MATRIZ DE CONFUSÃO]")
    print(f"                    SEM {target_name}     COM {target_name}")
    print("                  +-----------+-----------+")
    print(f"  Paciente   SEM  |    {vn:^5}  |    {fp:^5}  |")
    print("                  +-----------+-----------+")
    print(f"             COM  |    {fn:^5}  |    {vp:^5}  |")
    print("                  +-----------+-----------+")

    print(f"\n  Total: {total} | Acertos: {vn+vp} | Erros: {fp+fn}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"{modelo_nome} - {target_name}\n")
        f.write("=" * 40 + "\n\n")
        f.write("MÉTRICAS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:    {metricas['accuracy']:.2f}%\n")
        f.write(f"Sensitivity: {metricas['sensitivity']:.2f}%\n")
        f.write(f"Specificity: {metricas['specificity']:.2f}%\n")
        f.write(f"PPV:         {metricas['ppv']:.2f}%\n")
        f.write(f"NPV:         {metricas['npv']:.2f}%\n")
        f.write(f"F1-Score:    {metricas['f1']:.2f}%\n")
        f.write(f"Kappa:       {metricas['kappa']:.4f}\n\n")
        f.write("MATRIZ DE CONFUSÃO\n")
        f.write("-" * 40 + "\n")
        f.write(f"VN={vn} | FP={fp}\n")
        f.write(f"FN={fn} | VP={vp}\n")

    print(f"\n  Métricas salvas em: {output_file}")
    print("\n" + "=" * 60 + "\n")


def treinar_svm(target='GAD'):
    """Treina SVM sem balanceamento."""
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

    all_y_true, all_y_pred = [], []

    print(f"\n[TREINAMENTO]")
    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalizar features (importante para SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print("OK")

    metricas = calcular_metricas(all_y_true, all_y_pred)
    output_file = f'{OUTPUT_PATH}/svm_{target.lower()}_metricas.txt'
    exibir_resultados(metricas, target_name, "SVM (sem balanceamento)", output_file)

    return metricas


def treinar_svm_weighted(target='GAD'):
    """Treina SVM com class_weight='balanced'."""
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

    all_y_true, all_y_pred = [], []

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

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print("OK")

    metricas = calcular_metricas(all_y_true, all_y_pred)
    output_file = f'{OUTPUT_PATH}/svm_{target.lower()}_weighted_metricas.txt'
    exibir_resultados(metricas, target_name, "SVM (class_weight='balanced')", output_file)

    return metricas


def treinar_svm_smote(target='GAD'):
    """Treina SVM com SMOTE aplicado dentro de cada fold."""
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

    all_y_true, all_y_pred = [], []

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalizar antes do SMOTE
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SMOTE apenas no treino
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_scaled)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print("OK")

    metricas = calcular_metricas(all_y_true, all_y_pred)
    output_file = f'{OUTPUT_PATH}/svm_{target.lower()}_smote_metricas.txt'
    exibir_resultados(metricas, target_name, "SVM (com SMOTE)", output_file)

    return metricas


def treinar_svm_undersampling(target='GAD'):
    """Treina SVM com Undersampling aplicado dentro de cada fold."""
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

    all_y_true, all_y_pred = [], []

    print(f"  Executando {n_folds}-fold CV...", end=" ")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Undersampling antes de normalizar
        undersampler = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)

        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_scaled, y_train_res)
        y_pred = model.predict(X_test_scaled)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print("OK")

    metricas = calcular_metricas(all_y_true, all_y_pred)
    output_file = f'{OUTPUT_PATH}/svm_{target.lower()}_undersampling_metricas.txt'
    exibir_resultados(metricas, target_name, "SVM (com Undersampling)", output_file)

    return metricas


def comparativo_modelos(resultados_dict, target_name):
    """Exibe tabela comparativa."""
    print("\n" + "=" * 80)
    print("                      COMPARATIVO DE MODELOS - SVM")
    print("=" * 80)

    print(f"\n{'Modelo':<25} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'F1':>8} {'Kappa':>8}")
    print("-" * 80)

    resultados = []
    for nome, m in resultados_dict.items():
        print(f"{nome:<25} {m['accuracy']:>7.2f}% {m['sensitivity']:>7.2f}% {m['specificity']:>7.2f}% {m['ppv']:>7.2f}% {m['f1']:>7.2f}% {m['kappa']:>8.4f}")
        resultados.append({
            'Modelo': nome,
            'Accuracy': m['accuracy'],
            'Sensitivity': m['sensitivity'],
            'Specificity': m['specificity'],
            'PPV': m['ppv'],
            'F1-Score': m['f1'],
            'Kappa': m['kappa']
        })

    print("-" * 80)

    print("\n[MELHORES POR MÉTRICA]")
    metricas = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score', 'Kappa']
    for metrica in metricas:
        melhor = max(resultados, key=lambda x: x[metrica])
        print(f"  {metrica:<12}: {melhor['Modelo']} ({melhor[metrica]:.2f}{'%' if metrica != 'Kappa' else ''})")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = f'{OUTPUT_PATH}/comparativo_{target_name.lower()}.txt'
    with open(output_file, 'w') as f:
        f.write(f"COMPARATIVO SVM - {target_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Modelo':<25} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'F1':>8} {'Kappa':>8}\n")
        f.write("-" * 80 + "\n")
        for r in resultados:
            f.write(f"{r['Modelo']:<25} {r['Accuracy']:>7.2f}% {r['Sensitivity']:>7.2f}% {r['Specificity']:>7.2f}% {r['PPV']:>7.2f}% {r['F1-Score']:>7.2f}% {r['Kappa']:>8.4f}\n")

    print(f"\n  Comparativo salvo em: {output_file}")
    print("\n" + "=" * 80 + "\n")

    return resultados


def plotar_comparativo(resultados, target_name):
    """Gera gráfico e tabela comparativa."""
    modelos = [r['Modelo'] for r in resultados]
    metricas = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score', 'Kappa']
    metricas_grafico = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score']

    cores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(metricas_grafico))
    largura = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, (modelo, cor) in enumerate(zip(modelos, cores)):
        valores = [resultados[i][m] for m in metricas_grafico]
        barras = ax.bar(x + offsets[i] * largura, valores, largura,
                        label=modelo, color=cor, edgecolor='white', linewidth=0.5)
        for barra, valor in zip(barras, valores):
            ax.annotate(f'{valor:.1f}',
                        xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Porcentagem (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparativo de Modelos SVM - {target_name}\nTécnicas de Balanceamento de Classes',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_grafico, fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    grafico_file = f'{PLOTS_PATH}/comparativo_{target_name.lower()}_grafico.png'
    plt.savefig(grafico_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico salvo em: {grafico_file}")

    # Tabela
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    colunas = ['Modelo'] + metricas
    dados_tabela = []
    for r in resultados:
        linha = [r['Modelo']]
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
    tabela.scale(1.2, 2)

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

    ax.set_title(f'Comparativo de Métricas - SVM ({target_name})',
                 fontsize=14, fontweight='bold', pad=20, y=0.95)
    plt.tight_layout()

    tabela_file = f'{PLOTS_PATH}/comparativo_{target_name.lower()}_tabela.png'
    plt.savefig(tabela_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Tabela salva em: {tabela_file}")

    return grafico_file, tabela_file


if __name__ == "__main__":
    target = 'GAD'
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

    resultados = comparativo_modelos(resultados_dict, target)
    plotar_comparativo(resultados, target)
