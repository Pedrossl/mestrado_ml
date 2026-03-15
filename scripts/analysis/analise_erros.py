"""
Análise Detalhada de Erros - Identificação de perfis de FP/FN
=============================================================
Identifica características das crianças que o modelo classifica
incorretamente (falsos positivos e falsos negativos), comparando
com os acertos (verdadeiros positivos e verdadeiros negativos).

Usa XGBoost + SMOTE (melhor modelo) com 10-fold Stratified CV.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats

from scripts.utils import preparar_dados
from scripts.preprocessing.normalizacao import carregar_teste_normalizado


OUTPUT_PATH = 'output/plots/AnaliseErros'

# Colunas demográficas e clínicas disponíveis no dataset
COLUNAS_DEMOGRAFICAS = ['Sex', 'Race', 'Age', 'Number of Bio. Parents',
                        'Number of Siblings', 'Poverty Status']
COLUNAS_COMORBIDADES = ['Social Phobia', 'ADHD', 'CD', 'ODD']
COLUNAS_CLINICAS = ['Number of Impairments', 'Number of Type A Stressors',
                    'Number of Type B Stressors', 'Frequency Temper Tantrums',
                    'Frequency Irritable Mood', 'Number of Sleep Disturbances',
                    'Number of Physical Symptoms', 'Number of Sensory Sensitivities',
                    'Family History - Psychiatric Diagnosis']


def coletar_predicoes_por_amostra(target='GAD'):
    """Executa 10-fold CV e coleta predições individuais por amostra.

    Retorna o DataFrame original (com todas as colunas) mais colunas de
    predição, probabilidade e classificação de erro (VP/VN/FP/FN).
    """
    df_completo = carregar_teste_normalizado()

    # Remover colunas não usadas no modelo (mesmo processo do preparar_dados)
    colunas_remover_modelo = ['Subject', 'GAD Probabiliy - Gamma',
                              'SAD Probability - Gamma', 'Sample Weight']
    outro_target = 'SAD' if target == 'GAD' else 'GAD'
    colunas_remover_modelo.append(outro_target)

    # Remover colunas da normalização
    colunas_remover_norm = ['Depression', 'Number of Type A Stressors',
                            'Number of Physical Symptoms',
                            'Family History - Substance Abuse']

    colunas_remover_todas = colunas_remover_modelo + colunas_remover_norm

    df_modelo = df_completo.drop(
        columns=[c for c in colunas_remover_todas if c in df_completo.columns]
    )

    if 'Sex' in df_modelo.columns:
        df_modelo['Sex'] = df_modelo['Sex'].map({'M': 0, 'F': 1})

    cols = [c for c in df_modelo.columns if c != target] + [target]
    df_modelo = df_modelo[cols]

    # Guardar índices antes do dropna para mapear de volta
    mask_validos = df_modelo.dropna().index
    df_modelo = df_modelo.dropna()

    X = df_modelo.drop(columns=[target]).values
    y = df_modelo[target].values
    feature_names = df_modelo.drop(columns=[target]).columns.tolist()

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Arrays para armazenar predições de cada amostra
    y_pred_all = np.full(len(y), -1, dtype=int)
    y_proba_all = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
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

        y_pred_all[test_idx] = model.predict(X_test)
        y_proba_all[test_idx] = model.predict_proba(X_test)[:, 1]
        fold_ids[test_idx] = fold_i

    # Montar DataFrame de resultados com dados originais
    df_resultado = df_completo.loc[mask_validos].copy().reset_index(drop=True)
    df_resultado['y_real'] = y
    df_resultado['y_pred'] = y_pred_all
    df_resultado['y_proba'] = y_proba_all
    df_resultado['fold'] = fold_ids

    # Classificar tipo de erro
    def classificar_erro(row):
        real, pred = row['y_real'], row['y_pred']
        if real == 1 and pred == 1:
            return 'VP'
        elif real == 0 and pred == 0:
            return 'VN'
        elif real == 0 and pred == 1:
            return 'FP'
        elif real == 1 and pred == 0:
            return 'FN'
    df_resultado['tipo'] = df_resultado.apply(classificar_erro, axis=1)

    return df_resultado, feature_names


def analisar_perfil_grupo(df, grupo_nome, colunas):
    """Calcula estatísticas descritivas de um grupo para colunas numéricas."""
    resultados = {}
    for col in colunas:
        if col not in df.columns:
            continue
        valores = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(valores) == 0:
            continue
        resultados[col] = {
            'media': valores.mean(),
            'mediana': valores.median(),
            'std': valores.std(),
            'n': len(valores)
        }
    return resultados


def comparar_grupos(df, grupo_a_nome, grupo_b_nome, colunas):
    """Compara dois grupos usando teste Mann-Whitney U (não paramétrico)."""
    df_a = df[df['tipo'] == grupo_a_nome]
    df_b = df[df['tipo'] == grupo_b_nome]

    comparacoes = []
    for col in colunas:
        if col not in df.columns:
            continue
        vals_a = pd.to_numeric(df_a[col], errors='coerce').dropna()
        vals_b = pd.to_numeric(df_b[col], errors='coerce').dropna()
        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        stat, p_value = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        comparacoes.append({
            'variavel': col,
            f'media_{grupo_a_nome}': vals_a.mean(),
            f'media_{grupo_b_nome}': vals_b.mean(),
            f'n_{grupo_a_nome}': len(vals_a),
            f'n_{grupo_b_nome}': len(vals_b),
            'diferenca': vals_a.mean() - vals_b.mean(),
            'U_stat': stat,
            'p_value': p_value,
            'significativo': p_value < 0.05
        })
    return pd.DataFrame(comparacoes)


def gerar_relatorio_texto(df_resultado, target, output_file):
    """Gera relatório textual completo da análise de erros."""

    vp = df_resultado[df_resultado['tipo'] == 'VP']
    vn = df_resultado[df_resultado['tipo'] == 'VN']
    fp = df_resultado[df_resultado['tipo'] == 'FP']
    fn = df_resultado[df_resultado['tipo'] == 'FN']

    outro_target = 'SAD' if target == 'GAD' else 'GAD'

    with open(output_file, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"  ANÁLISE DETALHADA DE ERROS - {target}\n")
        f.write(f"  XGBoost + SMOTE | 10-Fold Stratified CV\n")
        f.write(f"{'='*70}\n\n")

        # --- Distribuição geral ---
        f.write(f"1. DISTRIBUIÇÃO DAS PREDIÇÕES\n")
        f.write(f"{'-'*50}\n")
        total = len(df_resultado)
        f.write(f"  Total de amostras: {total}\n")
        f.write(f"  Verdadeiros Positivos (VP): {len(vp):>4} ({len(vp)/total*100:.1f}%)\n")
        f.write(f"  Verdadeiros Negativos (VN): {len(vn):>4} ({len(vn)/total*100:.1f}%)\n")
        f.write(f"  Falsos Positivos (FP):      {len(fp):>4} ({len(fp)/total*100:.1f}%)\n")
        f.write(f"  Falsos Negativos (FN):       {len(fn):>4} ({len(fn)/total*100:.1f}%)\n\n")

        # --- Probabilidades preditas ---
        f.write(f"2. DISTRIBUIÇÃO DE PROBABILIDADES PREDITAS\n")
        f.write(f"{'-'*50}\n")
        for nome, grupo in [('VP', vp), ('VN', vn), ('FP', fp), ('FN', fn)]:
            if len(grupo) > 0:
                probas = grupo['y_proba']
                f.write(f"  {nome}: média={probas.mean():.3f}, "
                        f"mediana={probas.median():.3f}, "
                        f"min={probas.min():.3f}, max={probas.max():.3f}\n")
        f.write("\n")

        # --- Análise demográfica ---
        f.write(f"3. PERFIL DEMOGRÁFICO POR TIPO DE CLASSIFICAÇÃO\n")
        f.write(f"{'-'*50}\n\n")

        # Sexo
        f.write(f"  3.1 Distribuição por Sexo\n")
        for nome, grupo in [('VP', vp), ('VN', vn), ('FP', fp), ('FN', fn)]:
            if len(grupo) > 0 and 'Sex' in grupo.columns:
                sex_counts = grupo['Sex'].value_counts()
                total_g = len(grupo)
                m_count = sex_counts.get('M', 0)
                f_count = sex_counts.get('F', 0)
                f.write(f"    {nome} (n={total_g}): M={m_count} ({m_count/total_g*100:.1f}%), "
                        f"F={f_count} ({f_count/total_g*100:.1f}%)\n")
        f.write("\n")

        # Raça
        f.write(f"  3.2 Distribuição por Raça\n")
        for nome, grupo in [('VP', vp), ('VN', vn), ('FP', fp), ('FN', fn)]:
            if len(grupo) > 0 and 'Race' in grupo.columns:
                race_counts = grupo['Race'].value_counts()
                total_g = len(grupo)
                f.write(f"    {nome} (n={total_g}): ")
                partes = [f"Race={r}: {c} ({c/total_g*100:.1f}%)"
                          for r, c in race_counts.items()]
                f.write(", ".join(partes) + "\n")
        f.write("\n")

        # Pobreza
        f.write(f"  3.3 Distribuição por Status de Pobreza\n")
        for nome, grupo in [('VP', vp), ('VN', vn), ('FP', fp), ('FN', fn)]:
            if len(grupo) > 0 and 'Poverty Status' in grupo.columns:
                pov_counts = grupo['Poverty Status'].value_counts()
                total_g = len(grupo)
                pob = pov_counts.get(1, 0)
                nao_pob = pov_counts.get(0, 0)
                f.write(f"    {nome} (n={total_g}): Em pobreza={pob} ({pob/total_g*100:.1f}%), "
                        f"Não={nao_pob} ({nao_pob/total_g*100:.1f}%)\n")
        f.write("\n")

        # Idade
        f.write(f"  3.4 Distribuição por Idade\n")
        for nome, grupo in [('VP', vp), ('VN', vn), ('FP', fp), ('FN', fn)]:
            if len(grupo) > 0 and 'Age' in grupo.columns:
                idades = grupo['Age'].dropna()
                f.write(f"    {nome} (n={len(idades)}): média={idades.mean():.1f}, "
                        f"mediana={idades.median():.0f}, "
                        f"min={idades.min()}, max={idades.max()}\n")
        f.write("\n")

        # --- Comorbidades ---
        f.write(f"4. COMORBIDADES\n")
        f.write(f"{'-'*50}\n")
        f.write(f"  Prevalência de outros diagnósticos por grupo:\n\n")
        comorbidades = [outro_target] + COLUNAS_COMORBIDADES
        comorbidades = [c for c in comorbidades if c in df_resultado.columns]

        f.write(f"  {'Comorbidade':<20} {'VP':>8} {'VN':>8} {'FP':>8} {'FN':>8}\n")
        f.write(f"  {'-'*52}\n")
        for comorbidade in comorbidades:
            linha = f"  {comorbidade:<20}"
            for nome, grupo in [('VP', vp), ('VN', vn), ('FP', fp), ('FN', fn)]:
                if len(grupo) > 0 and comorbidade in grupo.columns:
                    prev = grupo[comorbidade].mean() * 100
                    linha += f" {prev:>7.1f}%"
                else:
                    linha += f" {'N/A':>8}"
            f.write(linha + "\n")
        f.write("\n")

        # --- Variáveis clínicas ---
        f.write(f"5. VARIÁVEIS CLÍNICAS (médias por grupo)\n")
        f.write(f"{'-'*50}\n\n")

        colunas_clinicas_disp = [c for c in COLUNAS_CLINICAS if c in df_resultado.columns]

        f.write(f"  {'Variável':<40} {'VP':>8} {'VN':>8} {'FP':>8} {'FN':>8}\n")
        f.write(f"  {'-'*72}\n")
        for col in colunas_clinicas_disp:
            linha = f"  {col:<40}"
            for nome, grupo in [('VP', vp), ('VN', vn), ('FP', fp), ('FN', fn)]:
                if len(grupo) > 0:
                    val = pd.to_numeric(grupo[col], errors='coerce').mean()
                    if pd.notna(val):
                        linha += f" {val:>8.2f}"
                    else:
                        linha += f" {'N/A':>8}"
                else:
                    linha += f" {'N/A':>8}"
            f.write(linha + "\n")
        f.write("\n")

        # --- Testes estatísticos FN vs VP ---
        f.write(f"6. TESTES ESTATÍSTICOS\n")
        f.write(f"{'-'*50}\n\n")

        # FN vs VP: o que diferencia os positivos que o modelo acerta dos que erra?
        f.write(f"  6.1 FN vs VP (crianças COM {target}: o que diferencia acertos de erros?)\n")
        f.write(f"      Teste Mann-Whitney U (bicaudal, α=0.05)\n\n")

        todas_colunas_teste = colunas_clinicas_disp + ['Age']
        comp_fn_vp = comparar_grupos(df_resultado, 'FN', 'VP', todas_colunas_teste)
        if len(comp_fn_vp) > 0:
            comp_fn_vp = comp_fn_vp.sort_values('p_value')
            f.write(f"  {'Variável':<40} {'Média FN':>10} {'Média VP':>10} {'p-valor':>10} {'Sig.':>5}\n")
            f.write(f"  {'-'*78}\n")
            for _, row in comp_fn_vp.iterrows():
                sig = '*' if row['significativo'] else ''
                f.write(f"  {row['variavel']:<40} {row['media_FN']:>10.2f} {row['media_VP']:>10.2f} "
                        f"{row['p_value']:>10.4f} {sig:>5}\n")
        f.write("\n")

        # FP vs VN: o que diferencia os negativos que o modelo erra dos que acerta?
        f.write(f"  6.2 FP vs VN (crianças SEM {target}: o que diferencia erros de acertos?)\n")
        f.write(f"      Teste Mann-Whitney U (bicaudal, α=0.05)\n\n")

        comp_fp_vn = comparar_grupos(df_resultado, 'FP', 'VN', todas_colunas_teste)
        if len(comp_fp_vn) > 0:
            comp_fp_vn = comp_fp_vn.sort_values('p_value')
            f.write(f"  {'Variável':<40} {'Média FP':>10} {'Média VN':>10} {'p-valor':>10} {'Sig.':>5}\n")
            f.write(f"  {'-'*78}\n")
            for _, row in comp_fp_vn.iterrows():
                sig = '*' if row['significativo'] else ''
                f.write(f"  {row['variavel']:<40} {row['media_FP']:>10.2f} {row['media_VN']:>10.2f} "
                        f"{row['p_value']:>10.4f} {sig:>5}\n")
        f.write("\n")

        # --- Resumo interpretativo ---
        f.write(f"7. RESUMO INTERPRETATIVO\n")
        f.write(f"{'-'*50}\n\n")

        f.write(f"  Falsos Negativos (FN = {len(fn)} crianças COM {target} classificadas como SEM):\n")
        f.write(f"  → Crianças que TÊM o transtorno mas o modelo não detecta.\n")
        f.write(f"  → Erro clinicamente mais grave (paciente não recebe tratamento).\n")
        if len(fn) > 0:
            f.write(f"  → Probabilidade média predita: {fn['y_proba'].mean():.3f} "
                    f"(limiar=0.5, portanto próximas do limiar)\n")
        f.write("\n")

        f.write(f"  Falsos Positivos (FP = {len(fp)} crianças SEM {target} classificadas como COM):\n")
        f.write(f"  → Crianças que NÃO TÊM o transtorno mas o modelo indica que têm.\n")
        f.write(f"  → Erro menos grave (encaminhamento desnecessário, mas sem dano direto).\n")
        if len(fp) > 0:
            f.write(f"  → Probabilidade média predita: {fp['y_proba'].mean():.3f}\n")
        f.write("\n")

    print(f"  Relatório salvo em: {output_file}")
    return comp_fn_vp, comp_fp_vn


def plotar_distribuicao_probabilidades(df_resultado, target, output_file):
    """Plota histograma de probabilidades preditas por tipo de classificação."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cores = {'VP': '#2ecc71', 'VN': '#3498db', 'FP': '#e74c3c', 'FN': '#e67e22'}
    titulos = [f'Crianças COM {target} (VP vs FN)', f'Crianças SEM {target} (VN vs FP)']

    # Painel 1: Positivos reais (VP vs FN)
    for tipo, cor in [('VP', cores['VP']), ('FN', cores['FN'])]:
        grupo = df_resultado[df_resultado['tipo'] == tipo]
        if len(grupo) > 0:
            axes[0].hist(grupo['y_proba'], bins=15, alpha=0.6, color=cor,
                         label=f'{tipo} (n={len(grupo)})', edgecolor='white')
    axes[0].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Limiar (0.5)')
    axes[0].set_xlabel('Probabilidade Predita', fontsize=11)
    axes[0].set_ylabel('Frequência', fontsize=11)
    axes[0].set_title(titulos[0], fontsize=12, fontweight='bold')
    axes[0].legend()

    # Painel 2: Negativos reais (VN vs FP)
    for tipo, cor in [('VN', cores['VN']), ('FP', cores['FP'])]:
        grupo = df_resultado[df_resultado['tipo'] == tipo]
        if len(grupo) > 0:
            axes[1].hist(grupo['y_proba'], bins=15, alpha=0.6, color=cor,
                         label=f'{tipo} (n={len(grupo)})', edgecolor='white')
    axes[1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Limiar (0.5)')
    axes[1].set_xlabel('Probabilidade Predita', fontsize=11)
    axes[1].set_ylabel('Frequência', fontsize=11)
    axes[1].set_title(titulos[1], fontsize=12, fontweight='bold')
    axes[1].legend()

    fig.suptitle(f'Distribuição de Probabilidades Preditas - {target}\nXGBoost + SMOTE | 10-Fold CV',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico de probabilidades salvo em: {output_file}")


def plotar_perfil_comorbidades(df_resultado, target, output_file):
    """Plota gráfico de barras das comorbidades por tipo de classificação."""
    outro_target = 'SAD' if target == 'GAD' else 'GAD'
    comorbidades = [outro_target] + [c for c in COLUNAS_COMORBIDADES if c in df_resultado.columns]

    tipos = ['VP', 'VN', 'FP', 'FN']
    cores = {'VP': '#2ecc71', 'VN': '#3498db', 'FP': '#e74c3c', 'FN': '#e67e22'}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comorbidades))
    largura = 0.8 / len(tipos)
    offsets = [i - (len(tipos) - 1) / 2 for i in range(len(tipos))]

    for i, tipo in enumerate(tipos):
        grupo = df_resultado[df_resultado['tipo'] == tipo]
        if len(grupo) == 0:
            continue
        prevalencias = []
        for comorbidade in comorbidades:
            if comorbidade in grupo.columns:
                prevalencias.append(grupo[comorbidade].mean() * 100)
            else:
                prevalencias.append(0)

        ax.bar(x + offsets[i] * largura, prevalencias, largura,
               label=f'{tipo} (n={len(grupo)})', color=cores[tipo],
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Comorbidade', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prevalência (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Prevalência de Comorbidades por Tipo de Classificação - {target}\n'
                 f'XGBoost + SMOTE | 10-Fold CV',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(comorbidades, fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico de comorbidades salvo em: {output_file}")


def plotar_variaveis_clinicas(df_resultado, target, output_file):
    """Plota boxplot das variáveis clínicas por tipo de classificação."""
    colunas_disp = [c for c in COLUNAS_CLINICAS if c in df_resultado.columns]

    # Selecionar top variáveis mais discriminativas (por variância entre grupos)
    diferencas = []
    for col in colunas_disp:
        medias = []
        for tipo in ['VP', 'VN', 'FP', 'FN']:
            grupo = df_resultado[df_resultado['tipo'] == tipo]
            if len(grupo) > 0:
                medias.append(pd.to_numeric(grupo[col], errors='coerce').mean())
        if medias:
            diferencas.append((col, np.nanstd(medias)))
    diferencas.sort(key=lambda x: x[1], reverse=True)
    top_vars = [d[0] for d in diferencas[:6]]

    if not top_vars:
        return

    cores = {'VP': '#2ecc71', 'VN': '#3498db', 'FP': '#e74c3c', 'FN': '#e67e22'}
    tipos = ['VP', 'FN', 'VN', 'FP']

    n_vars = len(top_vars)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(top_vars):
        ax = axes[idx]
        dados_box = []
        labels = []
        box_colors = []
        for tipo in tipos:
            grupo = df_resultado[df_resultado['tipo'] == tipo]
            if len(grupo) > 0:
                vals = pd.to_numeric(grupo[col], errors='coerce').dropna()
                if len(vals) > 0:
                    dados_box.append(vals.values)
                    labels.append(f'{tipo}\n(n={len(vals)})')
                    box_colors.append(cores[tipo])

        if dados_box:
            bp = ax.boxplot(dados_box, tick_labels=labels, patch_artist=True, widths=0.6)
            for patch, cor in zip(bp['boxes'], box_colors):
                patch.set_facecolor(cor)
                patch.set_alpha(0.6)

        # Truncar nome se muito longo
        nome_curto = col if len(col) <= 35 else col[:32] + '...'
        ax.set_title(nome_curto, fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Variáveis Clínicas por Tipo de Classificação - {target}\n'
                 f'Top 6 variáveis mais discriminativas | XGBoost + SMOTE',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico de variáveis clínicas salvo em: {output_file}")


def executar_analise(target='GAD'):
    """Executa a análise completa de erros para um target."""
    output_path = f'{OUTPUT_PATH}/{target}'
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ANÁLISE DETALHADA DE ERROS - {target}")
    print(f"  XGBoost + SMOTE | 10-Fold Stratified CV")
    print(f"{'='*70}\n")

    print("  [1/5] Coletando predições por amostra...")
    df_resultado, feature_names = coletar_predicoes_por_amostra(target)

    contagem = df_resultado['tipo'].value_counts()
    print(f"         VP={contagem.get('VP', 0)}, VN={contagem.get('VN', 0)}, "
          f"FP={contagem.get('FP', 0)}, FN={contagem.get('FN', 0)}")

    print("\n  [2/5] Gerando relatório textual...")
    comp_fn_vp, comp_fp_vn = gerar_relatorio_texto(
        df_resultado, target, f'{output_path}/analise_erros_{target.lower()}.txt'
    )

    print("\n  [3/5] Plotando distribuição de probabilidades...")
    plotar_distribuicao_probabilidades(
        df_resultado, target, f'{output_path}/distribuicao_probabilidades_{target.lower()}.png'
    )

    print("\n  [4/5] Plotando perfil de comorbidades...")
    plotar_perfil_comorbidades(
        df_resultado, target, f'{output_path}/comorbidades_{target.lower()}.png'
    )

    print("\n  [5/5] Plotando variáveis clínicas...")
    plotar_variaveis_clinicas(
        df_resultado, target, f'{output_path}/variaveis_clinicas_{target.lower()}.png'
    )

    print(f"\n{'='*70}")
    print(f"  Análise de erros completa para {target}!")
    print(f"  Resultados em: {output_path}/")
    print(f"{'='*70}\n")

    return df_resultado


if __name__ == "__main__":
    for target in ['GAD', 'SAD']:
        executar_analise(target)
