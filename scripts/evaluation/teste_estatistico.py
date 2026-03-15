"""
Testes de Significancia Estatistica para Comparacao de Classificadores.

Testes implementados:
  - Wilcoxon signed-rank test: Compara metricas pareadas por fold (nao-parametrico)
  - McNemar's test: Compara predicoes individuais entre classificadores

Comparacoes realizadas:
  1. Entre algoritmos: XGBoost vs SVM (ambos com SMOTE)
  2. Entre tecnicas de balanceamento dentro de cada algoritmo

Saidas:
  - Arquivo TXT com resultados detalhados de cada teste
  - Tabela resumo como imagem PNG
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from scripts.utils import preparar_dados, calcular_metricas_fold


# ============================================================
#  Coleta de predicoes por fold
# ============================================================

def coletar_predicoes_xgboost(X, y, tecnica='smote'):
    """Treina XGBoost e coleta predicoes + metricas por fold.

    Returns:
        predicoes: lista de (test_idx, y_true, y_pred) por fold
        metricas: lista de dicts com metricas por fold
    """
    peso = np.sum(y == 0) / np.sum(y == 1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    predicoes, metricas = [], []

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

        predicoes.append((test_idx, y_test, y_pred))
        metricas.append(calcular_metricas_fold(y_test, y_pred))

    return predicoes, metricas


def coletar_predicoes_svm(X, y, tecnica='smote'):
    """Treina SVM e coleta predicoes + metricas por fold.

    Returns:
        predicoes: lista de (test_idx, y_true, y_pred) por fold
        metricas: lista de dicts com metricas por fold
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    predicoes, metricas = [], []

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
        elif tecnica == 'weighted':
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

        predicoes.append((test_idx, y_test, y_pred))
        metricas.append(calcular_metricas_fold(y_test, y_pred))

    return predicoes, metricas


# ============================================================
#  Testes Estatisticos
# ============================================================

def teste_wilcoxon(metricas_a, metricas_b, metrica='accuracy'):
    """Wilcoxon signed-rank test em metricas pareadas por fold.

    Teste nao-parametrico que compara se a distribuicao de diferencas
    entre os pares e simetrica em torno de zero.

    Args:
        metricas_a: lista de dicts de metricas (modelo A, 10 folds)
        metricas_b: lista de dicts de metricas (modelo B, 10 folds)
        metrica: nome da metrica a comparar

    Returns:
        (p_value, statistic, valores_a, valores_b)
    """
    valores_a = np.array([m[metrica] for m in metricas_a])
    valores_b = np.array([m[metrica] for m in metricas_b])
    diff = valores_a - valores_b

    if np.all(diff == 0):
        return 1.0, 0.0, valores_a, valores_b

    try:
        stat, p_value = stats.wilcoxon(valores_a, valores_b, alternative='two-sided')
    except ValueError:
        # Wilcoxon requires at least 10 non-zero differences ideally
        # Fall back to p=1.0 if test cannot be computed
        return 1.0, 0.0, valores_a, valores_b

    return p_value, stat, valores_a, valores_b


def teste_mcnemar(predicoes_a, predicoes_b, n_total):
    """McNemar's test comparando predicoes individuais de dois classificadores.

    Constroi tabela de contingencia 2x2:
        - n11: ambos corretos
        - n10: A correto, B errado
        - n01: A errado, B correto
        - n00: ambos errados

    Testa H0: P(A correto, B errado) = P(A errado, B correto)
    Usa correcao de continuidade de Edwards.

    Args:
        predicoes_a: lista de (test_idx, y_true, y_pred) do modelo A
        predicoes_b: lista de (test_idx, y_true, y_pred) do modelo B
        n_total: numero total de instancias

    Returns:
        (p_value, chi2_stat, tabela_contingencia)
    """
    correto_a = np.zeros(n_total, dtype=bool)
    correto_b = np.zeros(n_total, dtype=bool)

    for test_idx, y_true, y_pred in predicoes_a:
        for i, idx in enumerate(test_idx):
            correto_a[idx] = (y_true[i] == y_pred[i])

    for test_idx, y_true, y_pred in predicoes_b:
        for i, idx in enumerate(test_idx):
            correto_b[idx] = (y_true[i] == y_pred[i])

    n11 = int(np.sum(correto_a & correto_b))
    n10 = int(np.sum(correto_a & ~correto_b))
    n01 = int(np.sum(~correto_a & correto_b))
    n00 = int(np.sum(~correto_a & ~correto_b))

    tabela = {'n11': n11, 'n10': n10, 'n01': n01, 'n00': n00}

    b, c = n01, n10
    if b + c == 0:
        return 1.0, 0.0, tabela

    # McNemar com correcao de continuidade
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return p_value, chi2, tabela


# ============================================================
#  Formatacao e salvamento de resultados
# ============================================================

def interpretar_p_valor(p):
    """Retorna interpretacao textual do p-valor."""
    if p < 0.001:
        return "*** Altamente significativo (p < 0.001)"
    elif p < 0.01:
        return "**  Muito significativo (p < 0.01)"
    elif p < 0.05:
        return "*   Significativo (p < 0.05)"
    elif p < 0.10:
        return ".   Marginalmente significativo (p < 0.10)"
    else:
        return "    Nao significativo (p >= 0.10)"


def salvar_resultados(resultados, target_name, output_file):
    """Salva todos os resultados dos testes em arquivo texto."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"TESTES DE SIGNIFICANCIA ESTATISTICA - {target_name}\n")
        f.write("=" * 80 + "\n")
        f.write("Nivel de significancia: alpha = 0.05\n")
        f.write("Convencao: * p<0.05  ** p<0.01  *** p<0.001\n\n")

        for secao in resultados:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"{secao['titulo']}\n")
            f.write("-" * 80 + "\n\n")

            if secao['tipo'] == 'wilcoxon':
                f.write("WILCOXON SIGNED-RANK TEST (pareado por fold, nao-parametrico)\n")
                f.write(f"H0: Nao ha diferenca entre {secao['modelo_a']} e {secao['modelo_b']}\n\n")

                f.write(f"{'Metrica':<15} {'Media A':>10} {'Media B':>10} {'Diff':>10} {'Statistic':>12} {'p-valor':>10} {'Resultado':}\n")
                f.write("." * 80 + "\n")

                for r in secao['resultados']:
                    sig = ""
                    if r['p_valor'] < 0.001:
                        sig = "***"
                    elif r['p_valor'] < 0.01:
                        sig = "**"
                    elif r['p_valor'] < 0.05:
                        sig = "*"

                    diff = r['media_a'] - r['media_b']
                    f.write(f"{r['metrica']:<15} {r['media_a']:>10.2f} {r['media_b']:>10.2f} "
                            f"{diff:>+10.2f} {r['statistic']:>12.1f} {r['p_valor']:>10.4f} {sig}\n")

                f.write("\n")
                melhor = secao['modelo_a'] if np.mean([r['media_a'] for r in secao['resultados']]) > np.mean([r['media_b'] for r in secao['resultados']]) else secao['modelo_b']
                sigs = [r for r in secao['resultados'] if r['p_valor'] < 0.05]
                if sigs:
                    f.write(f"Conclusao: Diferenca significativa encontrada em {len(sigs)} metrica(s).\n")
                else:
                    f.write("Conclusao: Nenhuma diferenca estatisticamente significativa encontrada (p >= 0.05).\n")
                    f.write("           Os modelos tem desempenho estatisticamente equivalente.\n")

            elif secao['tipo'] == 'mcnemar':
                f.write("McNEMAR'S TEST (comparacao de predicoes individuais)\n")
                f.write(f"H0: {secao['modelo_a']} e {secao['modelo_b']} cometem os mesmos erros\n\n")

                tab = secao['tabela']
                f.write("Tabela de Contingencia:\n")
                f.write(f"                         {secao['modelo_b']}\n")
                f.write(f"                     Correto    Errado\n")
                f.write(f"  {secao['modelo_a']:<12} Correto  {tab['n11']:>5}      {tab['n10']:>5}\n")
                f.write(f"  {'':12} Errado   {tab['n01']:>5}      {tab['n00']:>5}\n\n")
                f.write(f"  Discordancias: {secao['modelo_a']} correto/{secao['modelo_b']} errado = {tab['n10']}\n")
                f.write(f"                 {secao['modelo_a']} errado/{secao['modelo_b']} correto = {tab['n01']}\n\n")
                f.write(f"  Chi-quadrado (corrigido): {secao['chi2']:.4f}\n")
                f.write(f"  p-valor:                  {secao['p_valor']:.4f}\n")
                f.write(f"  {interpretar_p_valor(secao['p_valor'])}\n\n")

                if secao['p_valor'] < 0.05:
                    if tab['n10'] > tab['n01']:
                        f.write(f"Conclusao: {secao['modelo_a']} acerta significativamente mais que {secao['modelo_b']}.\n")
                    else:
                        f.write(f"Conclusao: {secao['modelo_b']} acerta significativamente mais que {secao['modelo_a']}.\n")
                else:
                    f.write("Conclusao: Os classificadores nao diferem significativamente nos erros cometidos.\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("NOTA METODOLOGICA\n")
        f.write("=" * 80 + "\n\n")
        f.write("Wilcoxon signed-rank test:\n")
        f.write("  Teste nao-parametrico para amostras pareadas. Compara as metricas de\n")
        f.write("  desempenho obtidas nos mesmos 10 folds do cross-validation. Adequado\n")
        f.write("  quando nao se pode assumir normalidade (n=10 folds e pequeno).\n")
        f.write("  Referencia: Demsar (2006). Statistical Comparisons of Classifiers over\n")
        f.write("  Multiple Data Sets. JMLR, 7, 1-30.\n\n")
        f.write("McNemar's test:\n")
        f.write("  Teste para tabelas de contingencia 2x2. Compara se dois classificadores\n")
        f.write("  cometem erros diferentes nas mesmas instancias. Usa correcao de\n")
        f.write("  continuidade de Edwards. Apropriado quando as predicoes sao feitas\n")
        f.write("  nos mesmos dados (10-fold CV garante que cada instancia aparece\n")
        f.write("  exatamente uma vez no teste).\n")
        f.write("  Referencia: McNemar (1947). Note on the sampling error of the\n")
        f.write("  difference between correlated proportions or percentages.\n")
        f.write("  Psychometrika, 12(2), 153-157.\n")

    print(f"  Resultados salvos em: {output_file}")


def plotar_tabela_pvalores(comparacoes, target_name, output_file):
    """Gera tabela resumo de p-valores como imagem PNG."""
    metricas_exibir = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'Kappa']
    metricas_chave = ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']

    nomes_comp = []
    dados_tabela = []

    for comp in comparacoes:
        if comp['tipo'] != 'wilcoxon':
            continue
        nome = f"{comp['modelo_a']}\nvs\n{comp['modelo_b']}"
        nomes_comp.append(nome)

        linha = []
        for chave in metricas_chave:
            for r in comp['resultados']:
                if r['metrica'] == chave:
                    p = r['p_valor']
                    if p < 0.001:
                        linha.append(f"{p:.4f} ***")
                    elif p < 0.01:
                        linha.append(f"{p:.4f} **")
                    elif p < 0.05:
                        linha.append(f"{p:.4f} *")
                    else:
                        linha.append(f"{p:.4f}")
                    break
        dados_tabela.append(linha)

    # Adicionar linha de McNemar para comparacoes entre algoritmos
    for comp in comparacoes:
        if comp['tipo'] == 'mcnemar':
            nome = f"McNemar\n{comp['modelo_a']}\nvs {comp['modelo_b']}"
            nomes_comp.append(nome)
            p = comp['p_valor']
            sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            linha = [f"p={p:.4f}{sig}"] + ["-"] * (len(metricas_exibir) - 1)
            dados_tabela.append(linha)

    if not dados_tabela:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, max(4, len(nomes_comp) * 1.2 + 2)))
    ax.axis('off')

    colunas = ['Comparacao'] + metricas_exibir
    dados_com_nomes = []
    for i, nome in enumerate(nomes_comp):
        dados_com_nomes.append([nome] + dados_tabela[i])

    tabela = ax.table(cellText=dados_com_nomes, colLabels=colunas,
                      cellLoc='center', loc='center',
                      colColours=['#2c3e50'] + ['#34495e'] * len(metricas_exibir))
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1.2, 2.5)

    for j in range(len(colunas)):
        tabela[(0, j)].set_text_props(color='white', fontweight='bold')

    # Colorir celulas por significancia
    for i in range(len(dados_com_nomes)):
        for j in range(1, len(colunas)):
            celula = dados_com_nomes[i][j]
            if '***' in celula:
                tabela[(i + 1, j)].set_facecolor('#c0392b30')
                tabela[(i + 1, j)].set_text_props(fontweight='bold', color='#c0392b')
            elif '**' in celula:
                tabela[(i + 1, j)].set_facecolor('#e67e2230')
                tabela[(i + 1, j)].set_text_props(fontweight='bold', color='#e67e22')
            elif '*' in celula:
                tabela[(i + 1, j)].set_facecolor('#f1c40f30')
                tabela[(i + 1, j)].set_text_props(fontweight='bold', color='#c0820a')
            else:
                tabela[(i + 1, j)].set_facecolor('#27ae6015')

    ax.set_title(f'Testes de Significancia Estatistica - {target_name}\n'
                 f'p-valores (Wilcoxon signed-rank + McNemar) | '
                 f'* p<0.05  ** p<0.01  *** p<0.001',
                 fontsize=13, fontweight='bold', pad=25, y=0.98)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Tabela de p-valores salva em: {output_file}")


# ============================================================
#  Execucao de comparacoes
# ============================================================

def executar_wilcoxon(nome_a, nome_b, metricas_a, metricas_b):
    """Executa Wilcoxon para todas as metricas entre dois modelos."""
    metricas_testar = ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']
    resultados = []

    for metrica in metricas_testar:
        p_val, stat, vals_a, vals_b = teste_wilcoxon(metricas_a, metricas_b, metrica)
        resultados.append({
            'metrica': metrica,
            'p_valor': p_val,
            'statistic': stat,
            'media_a': np.mean(vals_a),
            'media_b': np.mean(vals_b),
        })

    return {
        'tipo': 'wilcoxon',
        'titulo': f"WILCOXON: {nome_a} vs {nome_b}",
        'modelo_a': nome_a,
        'modelo_b': nome_b,
        'resultados': resultados,
    }


def executar_mcnemar(nome_a, nome_b, predicoes_a, predicoes_b, n_total):
    """Executa McNemar entre dois modelos."""
    p_val, chi2, tabela = teste_mcnemar(predicoes_a, predicoes_b, n_total)

    return {
        'tipo': 'mcnemar',
        'titulo': f"McNEMAR: {nome_a} vs {nome_b}",
        'modelo_a': nome_a,
        'modelo_b': nome_b,
        'p_valor': p_val,
        'chi2': chi2,
        'tabela': tabela,
    }


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    tecnicas = [
        ('sem_balanceamento', 'Sem Balanc.'),
        ('weighted', 'Class Weight'),
        ('smote', 'SMOTE'),
        ('undersampling', 'Undersamp.'),
    ]

    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 70)
        print(f"    TESTES DE SIGNIFICANCIA ESTATISTICA - {target}")
        print(f"    Wilcoxon signed-rank + McNemar's test")
        print("#" * 70)

        df, target_name = preparar_dados(target)
        X = df.drop(columns=[target_name]).values
        y = df[target_name].values
        n_total = len(y)

        print(f"\n[DATASET]")
        print(f"  Amostras: {n_total} | Features: {X.shape[1]} | Target: {target_name}")

        # --- Coletar predicoes de todos os modelos ---
        print("\n[TREINAMENTO] Coletando predicoes por fold...")

        preds_xgb = {}
        mets_xgb = {}
        for tecnica_id, tecnica_nome in tecnicas:
            print(f"  XGBoost {tecnica_nome}...", end=" ")
            p, m = coletar_predicoes_xgboost(X, y, tecnica_id)
            preds_xgb[tecnica_id] = p
            mets_xgb[tecnica_id] = m
            print("OK")

        preds_svm = {}
        mets_svm = {}
        for tecnica_id, tecnica_nome in tecnicas:
            print(f"  SVM {tecnica_nome}...", end=" ")
            p, m = coletar_predicoes_svm(X, y, tecnica_id)
            preds_svm[tecnica_id] = p
            mets_svm[tecnica_id] = m
            print("OK")

        # --- Executar testes ---
        print("\n[TESTES ESTATISTICOS]")
        todas_comparacoes = []

        # 1. XGBoost SMOTE vs SVM SMOTE (comparacao principal)
        print("  XGBoost SMOTE vs SVM SMOTE (Wilcoxon)...", end=" ")
        comp = executar_wilcoxon("XGBoost SMOTE", "SVM SMOTE",
                                  mets_xgb['smote'], mets_svm['smote'])
        todas_comparacoes.append(comp)
        print("OK")

        print("  XGBoost SMOTE vs SVM SMOTE (McNemar)...", end=" ")
        comp = executar_mcnemar("XGBoost SMOTE", "SVM SMOTE",
                                 preds_xgb['smote'], preds_svm['smote'], n_total)
        todas_comparacoes.append(comp)
        print("OK")

        # 2. Dentro do XGBoost: SMOTE vs cada outra tecnica
        for tecnica_id, tecnica_nome in tecnicas:
            if tecnica_id == 'smote':
                continue
            print(f"  XGBoost SMOTE vs XGBoost {tecnica_nome} (Wilcoxon)...", end=" ")
            comp = executar_wilcoxon(f"XGB SMOTE", f"XGB {tecnica_nome}",
                                      mets_xgb['smote'], mets_xgb[tecnica_id])
            todas_comparacoes.append(comp)
            print("OK")

        # 3. Dentro do SVM: SMOTE vs cada outra tecnica
        for tecnica_id, tecnica_nome in tecnicas:
            if tecnica_id == 'smote':
                continue
            print(f"  SVM SMOTE vs SVM {tecnica_nome} (Wilcoxon)...", end=" ")
            comp = executar_wilcoxon(f"SVM SMOTE", f"SVM {tecnica_nome}",
                                      mets_svm['smote'], mets_svm[tecnica_id])
            todas_comparacoes.append(comp)
            print("OK")

        # --- Exibir resumo ---
        print("\n" + "=" * 70)
        print(f"  RESUMO - {target}")
        print("=" * 70)

        for comp in todas_comparacoes:
            if comp['tipo'] == 'wilcoxon':
                sigs = [r for r in comp['resultados'] if r['p_valor'] < 0.05]
                status = f"{len(sigs)} metrica(s) com p<0.05" if sigs else "Nao significativo"
                print(f"  Wilcoxon {comp['modelo_a']:>15} vs {comp['modelo_b']:<15} -> {status}")
            elif comp['tipo'] == 'mcnemar':
                p = comp['p_valor']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"  McNemar  {comp['modelo_a']:>15} vs {comp['modelo_b']:<15} -> p={p:.4f} {sig}")

        # --- Salvar ---
        output_path = f'output/plots/Comparativo/{target}'
        os.makedirs(output_path, exist_ok=True)

        salvar_resultados(todas_comparacoes, target_name,
                          f'{output_path}/testes_estatisticos_{target.lower()}.txt')

        plots_path = f'output/plots/Comparativo/{target}'
        plotar_tabela_pvalores(todas_comparacoes, target_name,
                               f'{plots_path}/testes_estatisticos_{target.lower()}.png')

        print("\n" + "#" * 70 + "\n")
