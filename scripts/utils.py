import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc

from normalizacao import carregar_teste_normalizado


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


def calcular_ic(valores, confianca=0.95):
    """Calcula média, desvio padrão e intervalo de confiança."""
    n = len(valores)
    media = np.mean(valores)
    desvio = np.std(valores, ddof=1)
    erro_padrao = desvio / np.sqrt(n)
    t_valor = stats.t.ppf((1 + confianca) / 2, n - 1)
    ic = t_valor * erro_padrao
    return media, desvio, ic


def calcular_metricas_fold(y_true, y_pred):
    """Calcula métricas para um único fold a partir de y_true e y_pred."""
    cm = confusion_matrix(y_true, y_pred)
    return calcular_metricas_fold_cm(cm)


def calcular_metricas_fold_cm(cm):
    """Calcula métricas a partir de uma matriz de confusão (para Weka/ADTree)."""
    vn, fp = int(cm[0][0]), int(cm[0][1])
    fn, vp = int(cm[1][0]), int(cm[1][1])

    total = vn + fp + fn + vp
    accuracy = (vn + vp) / total * 100 if total > 0 else 0
    sensitivity = vp / (vp + fn) * 100 if (vp + fn) > 0 else 0
    specificity = vn / (vn + fp) * 100 if (vn + fp) > 0 else 0
    ppv = vp / (vp + fp) * 100 if (vp + fp) > 0 else 0
    npv = vn / (vn + fn) * 100 if (vn + fn) > 0 else 0

    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0

    po = (vn + vp) / total if total > 0 else 0
    pe = ((vn + fp) * (vn + fn) + (fn + vp) * (fp + vp)) / (total * total) if total > 0 else 0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    return {
        'vn': vn, 'fp': fp, 'fn': fn, 'vp': vp,
        'accuracy': accuracy, 'sensitivity': sensitivity,
        'specificity': specificity, 'ppv': ppv, 'npv': npv,
        'f1': f1, 'kappa': kappa
    }


def agregar_metricas_com_ic(metricas_folds):
    """Agrega métricas de todos os folds com IC 95%. Retorna dict flat."""
    metricas_nomes = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'kappa']
    resultado = {}

    resultado['vn'] = sum(m['vn'] for m in metricas_folds)
    resultado['fp'] = sum(m['fp'] for m in metricas_folds)
    resultado['fn'] = sum(m['fn'] for m in metricas_folds)
    resultado['vp'] = sum(m['vp'] for m in metricas_folds)

    for metrica in metricas_nomes:
        valores = [m[metrica] for m in metricas_folds]
        media, desvio, ic = calcular_ic(valores)
        resultado[metrica] = media
        resultado[f'{metrica}_std'] = desvio
        resultado[f'{metrica}_ic'] = ic

    return resultado


def exibir_resultados(metricas, target_name, modelo_nome, output_file):
    """Exibe e salva os resultados com IC."""
    vn, fp, fn, vp = metricas['vn'], metricas['fp'], metricas['fn'], metricas['vp']
    total = vn + fp + fn + vp

    print("\n" + "=" * 60)
    print("                      RESULTADOS")
    print("=" * 60)

    print("\n[MÉTRICAS DE DESEMPENHO] (média ± IC 95%)")
    print(f"  Accuracy:     {metricas['accuracy']:>6.2f}% ± {metricas['accuracy_ic']:.2f}%")
    print(f"  Sensitivity:  {metricas['sensitivity']:>6.2f}% ± {metricas['sensitivity_ic']:.2f}%")
    print(f"  Specificity:  {metricas['specificity']:>6.2f}% ± {metricas['specificity_ic']:.2f}%")
    print(f"  PPV:          {metricas['ppv']:>6.2f}% ± {metricas['ppv_ic']:.2f}%")
    print(f"  NPV:          {metricas['npv']:>6.2f}% ± {metricas['npv_ic']:.2f}%")
    print(f"  F1-Score:     {metricas['f1']:>6.2f}% ± {metricas['f1_ic']:.2f}%")
    print(f"  Kappa:        {metricas['kappa']:>6.4f} ± {metricas['kappa_ic']:.4f}")

    print("\n[MATRIZ DE CONFUSÃO AGREGADA]")
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
        f.write("=" * 50 + "\n\n")
        f.write("MÉTRICAS (média ± IC 95%)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy:    {metricas['accuracy']:.2f}% ± {metricas['accuracy_ic']:.2f}%\n")
        f.write(f"Sensitivity: {metricas['sensitivity']:.2f}% ± {metricas['sensitivity_ic']:.2f}%\n")
        f.write(f"Specificity: {metricas['specificity']:.2f}% ± {metricas['specificity_ic']:.2f}%\n")
        f.write(f"PPV:         {metricas['ppv']:.2f}% ± {metricas['ppv_ic']:.2f}%\n")
        f.write(f"NPV:         {metricas['npv']:.2f}% ± {metricas['npv_ic']:.2f}%\n")
        f.write(f"F1-Score:    {metricas['f1']:.2f}% ± {metricas['f1_ic']:.2f}%\n")
        f.write(f"Kappa:       {metricas['kappa']:.4f} ± {metricas['kappa_ic']:.4f}\n\n")
        f.write("DESVIO PADRÃO\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy:    σ = {metricas['accuracy_std']:.2f}%\n")
        f.write(f"Sensitivity: σ = {metricas['sensitivity_std']:.2f}%\n")
        f.write(f"Specificity: σ = {metricas['specificity_std']:.2f}%\n")
        f.write(f"F1-Score:    σ = {metricas['f1_std']:.2f}%\n")
        f.write(f"Kappa:       σ = {metricas['kappa_std']:.4f}\n\n")
        f.write("MATRIZ DE CONFUSÃO AGREGADA\n")
        f.write("-" * 50 + "\n")
        f.write(f"VN={vn} | FP={fp}\n")
        f.write(f"FN={fn} | VP={vp}\n")

    print(f"\n  Métricas salvas em: {output_file}")
    print("\n" + "=" * 60 + "\n")


def comparativo_modelos(resultados_dict, target_name, algoritmo_nome, output_path):
    """Exibe tabela comparativa com IC."""
    print("\n" + "=" * 100)
    print(f"                      COMPARATIVO DE MODELOS - {algoritmo_nome} (com IC 95%)")
    print("=" * 100)

    print(f"\n{'Modelo':<20} {'Accuracy':>16} {'Sensitivity':>16} {'Specificity':>16} {'F1-Score':>16} {'Kappa':>14}")
    print("-" * 100)

    resultados = []
    for nome, m in resultados_dict.items():
        acc_str = f"{m['accuracy']:.1f}±{m['accuracy_ic']:.1f}"
        sens_str = f"{m['sensitivity']:.1f}±{m['sensitivity_ic']:.1f}"
        spec_str = f"{m['specificity']:.1f}±{m['specificity_ic']:.1f}"
        f1_str = f"{m['f1']:.1f}±{m['f1_ic']:.1f}"
        kappa_str = f"{m['kappa']:.3f}±{m['kappa_ic']:.3f}"

        print(f"{nome:<20} {acc_str:>16} {sens_str:>16} {spec_str:>16} {f1_str:>16} {kappa_str:>14}")

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

    print("-" * 100)

    os.makedirs(output_path, exist_ok=True)
    output_file = f'{output_path}/comparativo_{target_name.lower()}.txt'
    with open(output_file, 'w') as f:
        f.write(f"COMPARATIVO {algoritmo_nome} - {target_name} (com IC 95%)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Modelo':<20} {'Accuracy':>16} {'Sensitivity':>16} {'Specificity':>16} {'F1-Score':>16} {'Kappa':>14}\n")
        f.write("-" * 100 + "\n")
        for r in resultados:
            acc_str = f"{r['Accuracy']:.1f}±{r['Accuracy_IC']:.1f}"
            sens_str = f"{r['Sensitivity']:.1f}±{r['Sensitivity_IC']:.1f}"
            spec_str = f"{r['Specificity']:.1f}±{r['Specificity_IC']:.1f}"
            f1_str = f"{r['F1-Score']:.1f}±{r['F1-Score_IC']:.1f}"
            kappa_str = f"{r['Kappa']:.3f}±{r['Kappa_IC']:.3f}"
            f.write(f"{r['Modelo']:<20} {acc_str:>16} {sens_str:>16} {spec_str:>16} {f1_str:>16} {kappa_str:>14}\n")

    print(f"\n  Comparativo salvo em: {output_file}")
    print("\n" + "=" * 100 + "\n")

    return resultados


def plotar_comparativo_grafico(resultados, target_name, algoritmo_nome, plots_path, cores=None):
    """Gera gráfico comparativo com barras de erro."""
    modelos = [r['Modelo'] for r in resultados]
    metricas_grafico = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score']

    if cores is None:
        cores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(metricas_grafico))
    n_modelos = len(modelos)
    largura = 0.8 / n_modelos
    offsets = [i - (n_modelos - 1) / 2 for i in range(n_modelos)]

    for i, (modelo, cor) in enumerate(zip(modelos, cores)):
        valores = [resultados[i][m] for m in metricas_grafico]
        erros = [resultados[i][f'{m}_IC'] for m in metricas_grafico]

        barras = ax.bar(x + offsets[i] * largura, valores, largura,
                        label=modelo, color=cor, edgecolor='white', linewidth=0.5,
                        yerr=erros, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})

        for barra, valor in zip(barras, valores):
            ax.annotate(f'{valor:.1f}',
                        xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        xytext=(0, 8), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Porcentagem (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparativo de Modelos {algoritmo_nome} - {target_name}\nTécnicas de Balanceamento (com IC 95%)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_grafico, fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(plots_path, exist_ok=True)
    grafico_file = f'{plots_path}/comparativo_{target_name.lower()}_grafico.png'
    plt.savefig(grafico_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico salvo em: {grafico_file}")

    return grafico_file


def plotar_comparativo_tabela(resultados, target_name, algoritmo_nome, plots_path, cores=None):
    """Gera tabela comparativa como imagem."""
    if cores is None:
        cores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    metricas_tabela = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-Score', 'Kappa']
    colunas = ['Modelo'] + metricas_tabela
    dados_tabela = []

    for r in resultados:
        linha = [r['Modelo']]
        for m in metricas_tabela:
            if m == 'Kappa':
                linha.append(f"{r[m]:.3f}±{r[f'{m}_IC']:.3f}")
            else:
                linha.append(f"{r[m]:.1f}±{r[f'{m}_IC']:.1f}%")
        dados_tabela.append(linha)

    tabela = ax.table(cellText=dados_tabela, colLabels=colunas, cellLoc='center', loc='center',
                      colColours=['#2c3e50'] + ['#34495e'] * len(metricas_tabela))
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 2)

    for j in range(len(colunas)):
        tabela[(0, j)].set_text_props(color='white', fontweight='bold')

    for i, r in enumerate(resultados):
        cor_fundo = cores[i] + '30'
        for j in range(len(colunas)):
            tabela[(i + 1, j)].set_facecolor(cor_fundo)

    for j, metrica in enumerate(metricas_tabela):
        valores = [r[metrica] for r in resultados]
        melhor_idx = valores.index(max(valores))
        tabela[(melhor_idx + 1, j + 1)].set_text_props(fontweight='bold', color='#27ae60')

    ax.set_title(f'Comparativo de Métricas - {algoritmo_nome} ({target_name}) com IC 95%',
                 fontsize=14, fontweight='bold', pad=20, y=0.95)
    plt.tight_layout()

    os.makedirs(plots_path, exist_ok=True)
    tabela_file = f'{plots_path}/comparativo_{target_name.lower()}_tabela.png'
    plt.savefig(tabela_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Tabela salva em: {tabela_file}")

    return tabela_file


def coletar_roc_folds(y_trues_folds, y_scores_folds):
    """Calcula curva ROC média e desvio padrão a partir dos folds.

    Para cada fold, interpola a curva ROC em 100 pontos de FPR comuns,
    permitindo calcular a média e desvio padrão da TPR.

    Args:
        y_trues_folds: lista de arrays com y_true de cada fold
        y_scores_folds: lista de arrays com scores/probabilidades de cada fold

    Returns:
        dict com mean_fpr, mean_tpr, std_tpr, tprs_upper, tprs_lower,
        mean_auc, std_auc, aucs
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs_lista = []

    for y_true, y_score in zip(y_trues_folds, y_scores_folds):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs_lista.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr,
        'tprs_upper': tprs_upper,
        'tprs_lower': tprs_lower,
        'mean_auc': np.mean(aucs_lista),
        'std_auc': np.std(aucs_lista),
        'aucs': aucs_lista
    }


def plotar_curvas_roc(roc_dados_dict, target_name, titulo, output_file, cores=None):
    """Plota curvas ROC para múltiplos modelos no mesmo gráfico.

    Inclui banda de ± 1 desvio padrão e linha diagonal de referência.

    Args:
        roc_dados_dict: dict de {nome_modelo: roc_dados} (saída de coletar_roc_folds)
        target_name: nome do alvo (GAD/SAD)
        titulo: título do gráfico (ex: "XGBoost")
        output_file: caminho completo para salvar o PNG
        cores: lista de cores hexadecimais (opcional)
    """
    if cores is None:
        cores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (nome, dados) in enumerate(roc_dados_dict.items()):
        cor = cores[i % len(cores)]
        ax.plot(dados['mean_fpr'], dados['mean_tpr'], color=cor, lw=2,
                label=f"{nome} (AUC = {dados['mean_auc']:.3f} \u00b1 {dados['std_auc']:.3f})")
        ax.fill_between(dados['mean_fpr'], dados['tprs_lower'], dados['tprs_upper'],
                        color=cor, alpha=0.1)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Aleat\u00f3rio (AUC = 0.500)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12, fontweight='bold')
    ax.set_title(f'Curva ROC - {titulo} ({target_name})\n10-Fold Stratified CV com \u00b1 1\u03c3',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Curva ROC salva em: {output_file}")


def salvar_auc_metricas(roc_dados_dict, target_name, titulo, output_file):
    """Salva métricas AUC em arquivo texto com IC 95%.

    Args:
        roc_dados_dict: dict de {nome_modelo: roc_dados}
        target_name: nome do alvo (GAD/SAD)
        titulo: título (ex: "XGBoost")
        output_file: caminho completo para salvar o TXT
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"AUC - {titulo} - {target_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"{'Modelo':<25} {'AUC M\u00e9dia':>12} {'Desvio':>10} {'IC 95%':>12}\n")
        f.write("-" * 60 + "\n")

        for nome, dados in roc_dados_dict.items():
            media, desvio, ic = calcular_ic(dados['aucs'])
            f.write(f"{nome:<25} {media:>12.4f} {desvio:>10.4f} {ic:>12.4f}\n")

        f.write(f"\n\nAUC por fold:\n")
        f.write("-" * 60 + "\n")
        for nome, dados in roc_dados_dict.items():
            f.write(f"\n{nome}:\n")
            for i, auc_val in enumerate(dados['aucs']):
                f.write(f"  Fold {i+1:2d}: {auc_val:.4f}\n")

    print(f"  M\u00e9tricas AUC salvas em: {output_file}")


def plotar_matriz_confusao_normalizada(cm_raw, target_name, titulo, ax=None, cmap='Blues'):
    """Plota matriz de confusao normalizada por linha (true class).

    Cada celula mostra a porcentagem e o valor absoluto entre parenteses.
    Normalizado por linha: cada linha soma 100%.

    Args:
        cm_raw: array 2x2 [[VN, FP], [FN, VP]]
        target_name: nome do alvo (GAD/SAD)
        titulo: titulo do subplot
        ax: eixo matplotlib (se None, cria figura nova)
        cmap: colormap a usar
    """
    cm = np.array(cm_raw, dtype=float)
    cm_norm = np.zeros_like(cm)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i] = cm[i] / row_sum * 100

    criar_fig = ax is None
    if criar_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)

    for i in range(2):
        for j in range(2):
            cor_texto = 'white' if cm_norm[i, j] > 60 else 'black'
            ax.text(j, i, f'{cm_norm[i, j]:.1f}%\n({int(cm[i, j])})',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color=cor_texto)

    labels_true = [f'Sem {target_name}', f'Com {target_name}']
    labels_pred = [f'Predito\nSem', f'Predito\nCom']

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_pred, fontsize=10)
    ax.set_yticklabels(labels_true, fontsize=10)
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_title(titulo, fontsize=12, fontweight='bold', pad=10)

    if criar_fig:
        plt.colorbar(im, ax=ax, label='%', shrink=0.8)
        plt.tight_layout()

    return ax


def plotar_grid_matrizes_confusao(cms_dict, target_name, titulo_principal, output_file,
                                   cores_cmap='Blues'):
    """Plota grid 2x2 de matrizes de confusao normalizadas.

    Args:
        cms_dict: dict de {nome_tecnica: [[VN,FP],[FN,VP]]}
        target_name: nome do alvo (GAD/SAD)
        titulo_principal: titulo da figura (ex: "XGBoost")
        output_file: caminho para salvar PNG
        cores_cmap: colormap a usar
    """
    nomes = list(cms_dict.keys())
    n = len(nomes)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (nome, cm) in enumerate(cms_dict.items()):
        plotar_matriz_confusao_normalizada(cm, target_name, nome, ax=axes[i], cmap=cores_cmap)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Matrizes de Confus\u00e3o Normalizadas - {titulo_principal} ({target_name})\n'
                 f'Normalizado por classe real (linhas somam 100%)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Matrizes salvas em: {output_file}")
