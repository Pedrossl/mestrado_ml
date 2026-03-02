"""
Analise Exploratoria de Dados (EDA).

Gera visualizacoes completas do dataset:
1. Distribuicoes de todas as features (histogramas e barplots)
2. Boxplots comparando classes (GAD/nao-GAD, SAD/nao-SAD)
3. Matriz de correlacao
4. Analise de multicolinearidade (VIF)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import preparar_dados


# Classificacao das features por tipo
FEATURES_BINARIAS = [
    'Sex', 'Race', 'Number of Siblings', 'Poverty Status',
    'Social Phobia', 'ADHD', 'CD', 'ODD',
    'Family History - Psychiatric Diagnosis'
]

FEATURES_ORDINAIS = [
    'Age', 'Number of Bio. Parents',
    'Number of Impairments', 'Number of Type B Stressors',
    'Number of Sleep Disturbances', 'Number of Sensory Sensitivities'
]

FEATURES_CONTINUAS = [
    'Frequency Temper Tantrums', 'Frequency Irritable Mood'
]

# Labels legíveis para features binárias
LABELS_BINARIAS = {
    'Sex': {0: 'Masculino', 1: 'Feminino'},
    'Race': {0: 'Nao-branco', 1: 'Branco'},
    'Number of Siblings': {0: 'Sem irmaos', 1: 'Com irmaos'},
    'Poverty Status': {0: 'Nao', 1: 'Sim'},
    'Social Phobia': {0: 'Nao', 1: 'Sim'},
    'ADHD': {0: 'Nao', 1: 'Sim'},
    'CD': {0: 'Nao', 1: 'Sim'},
    'ODD': {0: 'Nao', 1: 'Sim'},
    'Family History - Psychiatric Diagnosis': {0: 'Nao', 1: 'Sim'},
}


def plotar_distribuicoes(df, target, output_path):
    """Plota distribuicoes de todas as features.

    - Binarias: barplots com contagem
    - Ordinais: barplots com contagem por valor
    - Continuas: histogramas com KDE
    """
    plots_path = f'{output_path}/distribuicoes'
    os.makedirs(plots_path, exist_ok=True)

    features = [c for c in df.columns if c != target]

    # --- Features binarias: grid unico ---
    binarias = [f for f in FEATURES_BINARIAS if f in features]
    n_bin = len(binarias)
    ncols = 3
    nrows = (n_bin + ncols - 1) // ncols

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(binarias):
        ax = axes[i]
        contagem = df[feat].value_counts().sort_index()
        labels = LABELS_BINARIAS.get(feat, {0: '0', 1: '1'})
        bars = ax.bar(
            [labels.get(v, str(v)) for v in contagem.index],
            contagem.values,
            color=['#3498db', '#e74c3c'], edgecolor='white', linewidth=0.8
        )
        for bar, val in zip(bars, contagem.values):
            pct = val / len(df) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_ylabel('Contagem')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Distribuicao - Features Binarias ({target})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/distribuicao_binarias.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Distribuicoes binarias salvas")

    # --- Features ordinais: grid unico ---
    ordinais = [f for f in FEATURES_ORDINAIS if f in features]
    n_ord = len(ordinais)
    ncols = 3
    nrows = (n_ord + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(ordinais):
        ax = axes[i]
        contagem = df[feat].value_counts().sort_index()
        ax.bar(contagem.index.astype(str), contagem.values,
               color='#3498db', edgecolor='white', linewidth=0.8)
        for idx, (val_x, val_y) in enumerate(zip(contagem.index, contagem.values)):
            pct = val_y / len(df) * 100
            ax.text(idx, val_y + 1, f'{val_y}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=8)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_ylabel('Contagem')
        ax.set_xlabel('Valor')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Distribuicao - Features Ordinais ({target})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/distribuicao_ordinais.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Distribuicoes ordinais salvas")

    # --- Features continuas: histogramas com KDE ---
    continuas = [f for f in FEATURES_CONTINUAS if f in features]
    n_cont = len(continuas)

    fig, axes = plt.subplots(1, n_cont, figsize=(6 * n_cont, 5))
    if n_cont == 1:
        axes = [axes]

    for i, feat in enumerate(continuas):
        ax = axes[i]
        ax.hist(df[feat], bins=30, color='#3498db', edgecolor='white',
                linewidth=0.8, alpha=0.7, density=True)
        df[feat].plot.kde(ax=ax, color='#e74c3c', lw=2)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Densidade')

        # Estatísticas descritivas no gráfico
        stats_text = (f'Media: {df[feat].mean():.1f}\n'
                      f'Mediana: {df[feat].median():.1f}\n'
                      f'DP: {df[feat].std():.1f}\n'
                      f'Assimetria: {df[feat].skew():.2f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    fig.suptitle(f'Distribuicao - Features Continuas ({target})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/distribuicao_continuas.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Distribuicoes continuas salvas")


def plotar_boxplots_por_classe(df, target, output_path):
    """Plota boxplots comparando distribuicao de cada feature entre classes.

    Para features binarias: barplots empilhados (proporcao por classe).
    Para ordinais/continuas: boxplots lado a lado.
    """
    plots_path = f'{output_path}/boxplots'
    os.makedirs(plots_path, exist_ok=True)

    features = [c for c in df.columns if c != target]
    label_0 = f'Sem {target}'
    label_1 = f'Com {target}'

    # --- Features binarias: proporcao por classe ---
    binarias = [f for f in FEATURES_BINARIAS if f in features]
    n_bin = len(binarias)
    ncols = 3
    nrows = (n_bin + ncols - 1) // ncols

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(binarias):
        ax = axes[i]
        ct = pd.crosstab(df[feat], df[target], normalize='columns') * 100
        labels = LABELS_BINARIAS.get(feat, {0: '0', 1: '1'})
        ct.index = [labels.get(v, str(v)) for v in ct.index]
        ct.columns = [label_0, label_1]
        ct.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'],
                edgecolor='white', linewidth=0.8)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_ylabel('Proporcao (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Proporcao por Classe - Features Binarias ({target})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/binarias_por_classe.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Barplots binarias por classe salvos")

    # --- Features ordinais: boxplots ---
    ordinais = [f for f in FEATURES_ORDINAIS if f in features]
    n_ord = len(ordinais)
    ncols = 3
    nrows = (n_ord + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(ordinais):
        ax = axes[i]
        df_plot = df[[feat, target]].copy()
        df_plot[target] = df_plot[target].map({0: label_0, 1: label_1})
        sns.boxplot(data=df_plot, x=target, y=feat, hue=target, ax=ax,
                    palette=['#3498db', '#e74c3c'], width=0.5, legend=False)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Boxplots por Classe - Features Ordinais ({target})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/ordinais_por_classe.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Boxplots ordinais por classe salvos")

    # --- Features continuas: boxplots ---
    continuas = [f for f in FEATURES_CONTINUAS if f in features]
    n_cont = len(continuas)

    fig, axes = plt.subplots(1, n_cont, figsize=(6 * n_cont, 5))
    if n_cont == 1:
        axes = [axes]

    for i, feat in enumerate(continuas):
        ax = axes[i]
        df_plot = df[[feat, target]].copy()
        df_plot[target] = df_plot[target].map({0: label_0, 1: label_1})
        sns.boxplot(data=df_plot, x=target, y=feat, hue=target, ax=ax,
                    palette=['#3498db', '#e74c3c'], width=0.5, legend=False)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('')

        # Adicionar medias por classe
        for j_cls, cls_label in enumerate([label_0, label_1]):
            media = df_plot[df_plot[target] == cls_label][feat].mean()
            ax.text(j_cls, media, f'  {media:.1f}', va='center',
                    fontsize=9, fontweight='bold', color='black')

    fig.suptitle(f'Boxplots por Classe - Features Continuas ({target})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/continuas_por_classe.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Boxplots continuas por classe salvos")


def plotar_correlacao(df, target, output_path):
    """Plota matriz de correlacao com anotacoes."""
    plots_path = f'{output_path}/correlacao'
    os.makedirs(plots_path, exist_ok=True)

    corr = df.corr()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Correlacao'},
                annot_kws={'size': 8}, ax=ax)

    ax.set_title(f'Matriz de Correlacao ({target})\nPearson - Triangulo Inferior',
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/matriz_correlacao.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Matriz de correlacao salva")

    # Correlacoes com o target
    corr_target = corr[target].drop(target).sort_values(key=abs, ascending=False)
    txt_file = f'{plots_path}/correlacao_com_{target.lower()}.txt'
    with open(txt_file, 'w') as f:
        f.write(f"Correlacao de Pearson com {target}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Feature':<45} {'r':>8}\n")
        f.write("-" * 55 + "\n")
        for feat, r in corr_target.items():
            significancia = '***' if abs(r) > 0.3 else '**' if abs(r) > 0.2 else '*' if abs(r) > 0.1 else ''
            f.write(f"{feat:<45} {r:>8.4f} {significancia}\n")
        f.write("\n* |r| > 0.1  ** |r| > 0.2  *** |r| > 0.3\n")

    print(f"    Correlacoes com {target} salvas em: {txt_file}")

    # Barplot correlacao com target
    fig, ax = plt.subplots(figsize=(10, 8))
    cores = ['#e74c3c' if r > 0 else '#3498db' for r in corr_target.values]
    bars = ax.barh(range(len(corr_target)), corr_target.values, color=cores,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(corr_target)))
    ax.set_yticklabels(corr_target.index, fontsize=10)
    ax.set_xlabel('Correlacao de Pearson (r)', fontsize=12, fontweight='bold')
    ax.set_title(f'Correlacao com {target}\nOrdenado por valor absoluto',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=-0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.invert_yaxis()

    for bar, val in zip(bars, corr_target.values):
        offset = 0.005 if val >= 0 else -0.005
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha=ha, fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{plots_path}/correlacao_barplot_{target.lower()}.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Barplot correlacao com {target} salvo")


def calcular_vif(df, target, output_path):
    """Calcula Variance Inflation Factor para detectar multicolinearidade.

    VIF > 5: multicolinearidade moderada
    VIF > 10: multicolinearidade alta (considerar remover)
    """
    plots_path = f'{output_path}/vif'
    os.makedirs(plots_path, exist_ok=True)

    X = df.drop(columns=[target])

    # VIF requer ao menos 2 features e valores numericos
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    print(f"\n    VIF - Variance Inflation Factor:")
    print(f"    {'Feature':<45} {'VIF':>8}")
    print(f"    " + "-" * 55)
    for _, row in vif_data.iterrows():
        flag = ' <<<' if row['VIF'] > 5 else ''
        print(f"    {row['Feature']:<45} {row['VIF']:>8.2f}{flag}")

    # Salvar txt
    txt_file = f'{plots_path}/vif_{target.lower()}.txt'
    with open(txt_file, 'w') as f:
        f.write(f"Variance Inflation Factor (VIF) - {target}\n")
        f.write("=" * 60 + "\n\n")
        f.write("VIF mede o quanto a variancia de um coeficiente de regressao\n")
        f.write("e inflada pela correlacao com outras features.\n\n")
        f.write("  VIF = 1: sem multicolinearidade\n")
        f.write("  VIF > 5: multicolinearidade moderada\n")
        f.write("  VIF > 10: multicolinearidade alta\n\n")
        f.write(f"{'Feature':<45} {'VIF':>8} {'Status':>15}\n")
        f.write("-" * 70 + "\n")
        for _, row in vif_data.iterrows():
            if row['VIF'] > 10:
                status = 'ALTO'
            elif row['VIF'] > 5:
                status = 'MODERADO'
            else:
                status = 'OK'
            f.write(f"{row['Feature']:<45} {row['VIF']:>8.2f} {status:>15}\n")

        altos = vif_data[vif_data['VIF'] > 5]
        if len(altos) > 0:
            f.write(f"\nATENCAO: {len(altos)} feature(s) com VIF > 5:\n")
            for _, row in altos.iterrows():
                f.write(f"  - {row['Feature']} (VIF={row['VIF']:.2f})\n")
            f.write("\nConsiderar remover ou combinar features com VIF alto.\n")
        else:
            f.write("\nNenhuma feature com multicolinearidade preocupante (VIF > 5).\n")

    print(f"    VIF salvo em: {txt_file}")

    # Grafico VIF
    fig, ax = plt.subplots(figsize=(10, 8))
    cores = ['#e74c3c' if v > 10 else '#f39c12' if v > 5 else '#2ecc71'
             for v in vif_data['VIF']]
    bars = ax.barh(range(len(vif_data)), vif_data['VIF'], color=cores,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(vif_data)))
    ax.set_yticklabels(vif_data['Feature'], fontsize=10)
    ax.set_xlabel('VIF', fontsize=12, fontweight='bold')
    ax.set_title(f'Variance Inflation Factor - {target}\nVIF > 5: moderado | VIF > 10: alto',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=5, color='#f39c12', linewidth=1.5, linestyle='--', label='VIF = 5')
    ax.axvline(x=10, color='#e74c3c', linewidth=1.5, linestyle='--', label='VIF = 10')
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    for bar, val in zip(bars, vif_data['VIF']):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{plots_path}/vif_{target.lower()}.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Grafico VIF salvo")

    return vif_data


def gerar_sumario(df, target, output_path):
    """Gera sumario estatistico descritivo do dataset."""
    txt_file = f'{output_path}/sumario_descritivo.txt'

    features = [c for c in df.columns if c != target]
    dist = df[target].value_counts()

    with open(txt_file, 'w') as f:
        f.write(f"Sumario Descritivo do Dataset - {target}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Amostras: {len(df)}\n")
        f.write(f"Features: {len(features)}\n")
        f.write(f"Target: {target}\n")
        f.write(f"  Classe 0 (sem {target}): {dist[0]} ({dist[0]/len(df)*100:.1f}%)\n")
        f.write(f"  Classe 1 (com {target}): {dist[1]} ({dist[1]/len(df)*100:.1f}%)\n")
        f.write(f"  Razao desbalanceamento: {dist[0]/dist[1]:.1f}:1\n\n")

        f.write("FEATURES BINARIAS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Feature':<45} {'N(1)':>6} {'%(1)':>8}\n")
        f.write("-" * 70 + "\n")
        for feat in FEATURES_BINARIAS:
            if feat in features:
                n1 = df[feat].sum()
                pct = n1 / len(df) * 100
                f.write(f"{feat:<45} {int(n1):>6} {pct:>7.1f}%\n")

        f.write(f"\nFEATURES ORDINAIS/COUNT\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Feature':<35} {'Min':>5} {'Max':>5} {'Media':>8} {'Mediana':>8} {'DP':>8}\n")
        f.write("-" * 70 + "\n")
        for feat in FEATURES_ORDINAIS + FEATURES_CONTINUAS:
            if feat in features:
                f.write(f"{feat:<35} {df[feat].min():>5.0f} {df[feat].max():>5.0f} "
                        f"{df[feat].mean():>8.2f} {df[feat].median():>8.1f} {df[feat].std():>8.2f}\n")

        f.write(f"\nDADOS FALTANTES\n")
        f.write("-" * 70 + "\n")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            f.write("Nenhum dado faltante (apos dropna na preparacao).\n")
        else:
            for feat, n_miss in missing[missing > 0].items():
                f.write(f"  {feat}: {n_miss} ({n_miss/len(df)*100:.1f}%)\n")

    print(f"    Sumario descritivo salvo em: {txt_file}")


def executar_eda(target='GAD'):
    """Executa EDA completa para um target."""
    df, target_name = preparar_dados(target)

    output_path = f'output/plots/EDA/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  EDA - ANALISE EXPLORATORIA DE DADOS ({target})")
    print("=" * 70)
    print(f"\n  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1}")

    dist = df[target_name].value_counts()
    print(f"  Classe 0: {dist[0]} ({dist[0]/len(df)*100:.1f}%) | "
          f"Classe 1: {dist[1]} ({dist[1]/len(df)*100:.1f}%)")

    print(f"\n  --- Sumario Descritivo ---")
    gerar_sumario(df, target_name, output_path)

    print(f"\n  --- Distribuicoes ---")
    plotar_distribuicoes(df, target_name, output_path)

    print(f"\n  --- Boxplots por Classe ---")
    plotar_boxplots_por_classe(df, target_name, output_path)

    print(f"\n  --- Correlacao ---")
    plotar_correlacao(df, target_name, output_path)

    print(f"\n  --- VIF (Multicolinearidade) ---")
    vif = calcular_vif(df, target_name, output_path)

    print(f"\n  EDA completa! Outputs em: {output_path}/")

    return vif


if __name__ == "__main__":
    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 70)
        print(f"{'':>25}EDA - {target}")
        print("#" * 70)

        executar_eda(target)
