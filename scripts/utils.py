import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

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
