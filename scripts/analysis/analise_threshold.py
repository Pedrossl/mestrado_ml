"""
Análise de Threshold de Decisão
================================
Avalia o impacto de diferentes limiares de classificação nas métricas
de desempenho (Sensitivity, Specificity, PPV, NPV, F1, Kappa).

Em contexto clínico, reduzir o threshold aumenta a Sensitivity (detecta
mais casos) à custa de mais falsos positivos. Este script identifica o
threshold ótimo para diferentes critérios (Youden, F1 máximo, Sensitivity
mínima de 70%).

Usa XGBoost + SMOTE com 10-fold Stratified CV.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from scripts.utils import preparar_dados, calcular_metricas_fold, calcular_ic


OUTPUT_PATH = 'output/plots/AnaliseThreshold'


def coletar_probabilidades_cv(target='GAD'):
    """Executa 10-fold CV e retorna y_true e y_proba agregados."""
    df, target_name = preparar_dados(target)

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    y_true_all = np.array([], dtype=int)
    y_proba_all = np.array([])

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

        y_proba = model.predict_proba(X_test)[:, 1]
        y_true_all = np.concatenate([y_true_all, y_test])
        y_proba_all = np.concatenate([y_proba_all, y_proba])

    return y_true_all, y_proba_all


def calcular_metricas_threshold(y_true, y_proba, threshold):
    """Calcula métricas para um threshold específico."""
    y_pred = (y_proba >= threshold).astype(int)
    return calcular_metricas_fold(y_true, y_pred)


def analisar_thresholds(y_true, y_proba, target):
    """Analisa métricas para uma faixa de thresholds."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    resultados = []

    for t in thresholds:
        m = calcular_metricas_threshold(y_true, y_proba, t)
        m['threshold'] = t

        # Youden's J statistic = Sensitivity + Specificity - 100
        m['youden_j'] = m['sensitivity'] + m['specificity'] - 100
        resultados.append(m)

    return resultados


def encontrar_thresholds_otimos(resultados):
    """Encontra thresholds ótimos para diferentes critérios."""
    otimos = {}

    # 1. Youden's J (maximiza Sensitivity + Specificity)
    melhor_youden = max(resultados, key=lambda r: r['youden_j'])
    otimos['Youden (Sens+Spec)'] = melhor_youden

    # 2. F1-Score máximo
    melhor_f1 = max(resultados, key=lambda r: r['f1'])
    otimos['F1 máximo'] = melhor_f1

    # 3. Sensitivity >= 70% com melhor Specificity
    sens70 = [r for r in resultados if r['sensitivity'] >= 70]
    if sens70:
        otimos['Sensitivity ≥ 70%'] = max(sens70, key=lambda r: r['specificity'])

    # 4. Sensitivity >= 80% com melhor Specificity
    sens80 = [r for r in resultados if r['sensitivity'] >= 80]
    if sens80:
        otimos['Sensitivity ≥ 80%'] = max(sens80, key=lambda r: r['specificity'])

    # 5. Default (0.5)
    default = min(resultados, key=lambda r: abs(r['threshold'] - 0.5))
    otimos['Default (0.50)'] = default

    return otimos


def gerar_relatorio(resultados, otimos, target, output_file):
    """Gera relatório textual da análise de threshold."""
    with open(output_file, 'w') as f:
        f.write(f"{'='*75}\n")
        f.write(f"  ANÁLISE DE THRESHOLD DE DECISÃO - {target}\n")
        f.write(f"  XGBoost + SMOTE | 10-Fold Stratified CV\n")
        f.write(f"{'='*75}\n\n")

        # Tabela de thresholds ótimos
        f.write("1. THRESHOLDS ÓTIMOS POR CRITÉRIO\n")
        f.write(f"{'-'*75}\n\n")

        f.write(f"  {'Critério':<25} {'Thresh':>7} {'Sens':>7} {'Spec':>7} "
                f"{'PPV':>7} {'NPV':>7} {'F1':>7} {'Kappa':>7}\n")
        f.write(f"  {'-'*73}\n")

        for nome, m in otimos.items():
            f.write(f"  {nome:<25} {m['threshold']:>7.2f} {m['sensitivity']:>6.1f}% "
                    f"{m['specificity']:>6.1f}% {m['ppv']:>6.1f}% {m['npv']:>6.1f}% "
                    f"{m['f1']:>6.1f}% {m['kappa']:>7.3f}\n")

        f.write(f"\n\n2. COMPARAÇÃO: DEFAULT (0.50) vs YOUDEN\n")
        f.write(f"{'-'*75}\n\n")

        default = otimos['Default (0.50)']
        youden = otimos['Youden (Sens+Spec)']

        f.write(f"  {'Métrica':<20} {'Default (0.50)':>15} {'Youden':>15} {'Diferença':>15}\n")
        f.write(f"  {'-'*65}\n")
        for metrica in ['sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'kappa']:
            v_def = default[metrica]
            v_you = youden[metrica]
            diff = v_you - v_def
            sinal = '+' if diff >= 0 else ''
            if metrica == 'kappa':
                f.write(f"  {metrica.capitalize():<20} {v_def:>14.3f} {v_you:>14.3f} "
                        f"{sinal}{diff:>14.3f}\n")
            else:
                f.write(f"  {metrica.capitalize():<20} {v_def:>13.1f}% {v_you:>13.1f}% "
                        f"{sinal}{diff:>13.1f}%\n")

        f.write(f"\n  Threshold Youden: {youden['threshold']:.2f} "
                f"(vs default 0.50, diferença: {youden['threshold'] - 0.50:+.2f})\n")

        # Análise do impacto clínico
        f.write(f"\n\n3. IMPACTO CLÍNICO DA MUDANÇA DE THRESHOLD\n")
        f.write(f"{'-'*75}\n\n")

        total = default['vn'] + default['fp'] + default['fn'] + default['vp']
        n_positivos = default['fn'] + default['vp']
        n_negativos = default['vn'] + default['fp']

        f.write(f"  Total de amostras: {total}\n")
        f.write(f"  Crianças COM {target}: {n_positivos}\n")
        f.write(f"  Crianças SEM {target}: {n_negativos}\n\n")

        f.write(f"  Com threshold DEFAULT (0.50):\n")
        f.write(f"    Detecta {default['vp']} de {n_positivos} crianças com {target} "
                f"({default['sensitivity']:.1f}%)\n")
        f.write(f"    Falsos alarmes: {default['fp']} de {n_negativos} crianças saudáveis "
                f"({100-default['specificity']:.1f}%)\n\n")

        f.write(f"  Com threshold YOUDEN ({youden['threshold']:.2f}):\n")
        f.write(f"    Detecta {youden['vp']} de {n_positivos} crianças com {target} "
                f"({youden['sensitivity']:.1f}%)\n")
        f.write(f"    Falsos alarmes: {youden['fp']} de {n_negativos} crianças saudáveis "
                f"({100-youden['specificity']:.1f}%)\n\n")

        ganho_vp = youden['vp'] - default['vp']
        custo_fp = youden['fp'] - default['fp']
        f.write(f"  Impacto da mudança:\n")
        f.write(f"    → Detecta {ganho_vp:+d} crianças a mais com {target}\n")
        f.write(f"    → Gera {custo_fp:+d} falsos alarmes adicionais\n")
        if custo_fp > 0 and ganho_vp > 0:
            f.write(f"    → Custo: {custo_fp/ganho_vp:.1f} falsos alarmes por caso "
                    f"verdadeiro adicional detectado\n")

        # Sugestão para uso como screening
        if 'Sensitivity ≥ 70%' in otimos:
            screen = otimos['Sensitivity ≥ 70%']
            f.write(f"\n\n4. SUGESTÃO PARA USO COMO SCREENING\n")
            f.write(f"{'-'*75}\n\n")
            f.write(f"  Para uso como ferramenta de triagem (screening), priorizar\n")
            f.write(f"  alta Sensitivity é fundamental (não perder casos).\n\n")
            f.write(f"  Threshold para Sensitivity ≥ 70%: {screen['threshold']:.2f}\n")
            f.write(f"    Sensitivity: {screen['sensitivity']:.1f}%\n")
            f.write(f"    Specificity: {screen['specificity']:.1f}%\n")
            f.write(f"    PPV: {screen['ppv']:.1f}%\n")
            f.write(f"    NPV: {screen['npv']:.1f}%\n")
            f.write(f"    Detecta {screen['vp']} de {n_positivos} crianças com {target}\n")
            f.write(f"    Falsos alarmes: {screen['fp']} ({100-screen['specificity']:.1f}% "
                    f"dos saudáveis)\n\n")
            f.write(f"  Interpretação: Com este limiar, o modelo funciona como uma\n")
            f.write(f"  rede de segurança - captura a maioria dos casos, aceitando\n")
            f.write(f"  mais encaminhamentos desnecessários em troca de não perder\n")
            f.write(f"  crianças que precisam de atenção.\n")

        # Tabela completa
        f.write(f"\n\n5. TABELA COMPLETA DE THRESHOLDS\n")
        f.write(f"{'-'*75}\n\n")
        f.write(f"  {'Thresh':>7} {'Sens':>7} {'Spec':>7} {'PPV':>7} {'NPV':>7} "
                f"{'F1':>7} {'Kappa':>7} {'VP':>5} {'FP':>5} {'FN':>5} {'VN':>5}\n")
        f.write(f"  {'-'*73}\n")

        for r in resultados:
            t = r['threshold']
            if t % 0.05 < 0.01 or abs(t % 0.05 - 0.05) < 0.01:
                f.write(f"  {t:>7.2f} {r['sensitivity']:>6.1f}% {r['specificity']:>6.1f}% "
                        f"{r['ppv']:>6.1f}% {r['npv']:>6.1f}% {r['f1']:>6.1f}% "
                        f"{r['kappa']:>7.3f} {r['vp']:>5} {r['fp']:>5} {r['fn']:>5} {r['vn']:>5}\n")

    print(f"  Relatório salvo em: {output_file}")


def plotar_metricas_vs_threshold(resultados, otimos, target, output_file):
    """Plota curvas de métricas em função do threshold."""
    thresholds = [r['threshold'] for r in resultados]
    sensitivity = [r['sensitivity'] for r in resultados]
    specificity = [r['specificity'] for r in resultados]
    ppv = [r['ppv'] for r in resultados]
    npv = [r['npv'] for r in resultados]
    f1 = [r['f1'] for r in resultados]
    kappa = [r['kappa'] * 100 for r in resultados]  # escalar para %

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(thresholds, sensitivity, 'r-', lw=2.5, label='Sensitivity')
    ax.plot(thresholds, specificity, 'b-', lw=2.5, label='Specificity')
    ax.plot(thresholds, ppv, 'g--', lw=1.5, label='PPV')
    ax.plot(thresholds, npv, 'm--', lw=1.5, label='NPV')
    ax.plot(thresholds, f1, 'orange', lw=2, label='F1-Score')
    ax.plot(thresholds, kappa, 'brown', lw=1.5, alpha=0.7, label='Kappa (×100)')

    # Marcar thresholds ótimos
    marcadores = {
        'Default (0.50)': ('k', 's', 10),
        'Youden (Sens+Spec)': ('#2ecc71', 'D', 12),
        'F1 máximo': ('#e67e22', '^', 12),
    }
    if 'Sensitivity ≥ 70%' in otimos:
        marcadores['Sensitivity ≥ 70%'] = ('#e74c3c', 'o', 10)

    for nome, (cor, marker, size) in marcadores.items():
        if nome in otimos:
            m = otimos[nome]
            ax.axvline(x=m['threshold'], color=cor, linestyle=':', alpha=0.5)
            ax.plot(m['threshold'], m['sensitivity'], marker=marker, color=cor,
                    markersize=size, zorder=5, label=f'{nome} (t={m["threshold"]:.2f})')

    ax.set_xlabel('Threshold de Decisão', fontsize=12, fontweight='bold')
    ax.set_ylabel('Métrica (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Métricas vs Threshold de Decisão - {target}\n'
                 f'XGBoost + SMOTE | 10-Fold CV',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0, 105])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico métricas vs threshold salvo em: {output_file}")


def plotar_tradeoff_sens_spec(resultados, otimos, target, output_file):
    """Plota trade-off Sensitivity vs Specificity com anotações."""
    sensitivity = [r['sensitivity'] for r in resultados]
    specificity = [r['specificity'] for r in resultados]
    thresholds = [r['threshold'] for r in resultados]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 8))

    # Colorir por threshold
    scatter = ax.scatter(specificity, sensitivity, c=thresholds, cmap='RdYlGn_r',
                         s=20, alpha=0.7, zorder=3)
    plt.colorbar(scatter, ax=ax, label='Threshold', shrink=0.8)

    # Marcar pontos ótimos
    cores_pontos = {
        'Default (0.50)': ('black', 's', 120),
        'Youden (Sens+Spec)': ('#2ecc71', 'D', 140),
        'F1 máximo': ('#e67e22', '^', 140),
    }
    if 'Sensitivity ≥ 70%' in otimos:
        cores_pontos['Sensitivity ≥ 70%'] = ('#e74c3c', 'o', 120)

    for nome, (cor, marker, size) in cores_pontos.items():
        if nome in otimos:
            m = otimos[nome]
            ax.scatter(m['specificity'], m['sensitivity'], c=cor, marker=marker,
                       s=size, edgecolors='black', linewidths=1.5, zorder=5,
                       label=f'{nome}\n  t={m["threshold"]:.2f}, '
                             f'Sens={m["sensitivity"]:.1f}%, Spec={m["specificity"]:.1f}%')

    # Linha diagonal (trade-off perfeito)
    ax.plot([0, 100], [100, 0], 'k--', alpha=0.2)

    # Linha de referência 70% Sensitivity
    ax.axhline(y=70, color='red', linestyle=':', alpha=0.3, label='Sensitivity = 70%')

    ax.set_xlabel('Specificity (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sensitivity (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Trade-off Sensitivity vs Specificity - {target}\n'
                 f'XGBoost + SMOTE | Variação de Threshold',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([0, 105])
    ax.set_ylim([0, 105])
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico trade-off salvo em: {output_file}")


def executar_analise(target='GAD'):
    """Executa análise completa de threshold para um target."""
    output_path = f'{OUTPUT_PATH}/{target}'
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ANÁLISE DE THRESHOLD - {target}")
    print(f"  XGBoost + SMOTE | 10-Fold Stratified CV")
    print(f"{'='*70}\n")

    print("  [1/5] Coletando probabilidades por fold...")
    y_true, y_proba = coletar_probabilidades_cv(target)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    print(f"         {len(y_true)} amostras ({n_pos} positivos, {n_neg} negativos)")

    print("\n  [2/5] Analisando thresholds de 0.05 a 0.95...")
    resultados = analisar_thresholds(y_true, y_proba, target)

    print("\n  [3/5] Identificando thresholds ótimos...")
    otimos = encontrar_thresholds_otimos(resultados)
    for nome, m in otimos.items():
        print(f"         {nome}: t={m['threshold']:.2f} → "
              f"Sens={m['sensitivity']:.1f}%, Spec={m['specificity']:.1f}%, "
              f"F1={m['f1']:.1f}%")

    print("\n  [4/5] Gerando relatório e gráficos...")
    gerar_relatorio(resultados, otimos, target,
                    f'{output_path}/analise_threshold_{target.lower()}.txt')

    plotar_metricas_vs_threshold(resultados, otimos, target,
                                f'{output_path}/metricas_vs_threshold_{target.lower()}.png')

    print("\n  [5/5] Plotando trade-off Sensitivity vs Specificity...")
    plotar_tradeoff_sens_spec(resultados, otimos, target,
                             f'{output_path}/tradeoff_sens_spec_{target.lower()}.png')

    print(f"\n{'='*70}")
    print(f"  Análise de threshold completa para {target}!")
    print(f"  Resultados em: {output_path}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    for target in ['GAD', 'SAD']:
        executar_analise(target)
