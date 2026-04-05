# =============================================================================
# AVALIAÇÃO DE COMBINAÇÕES: Algoritmo + Técnica de Balanceamento
# N repetições de divisão aleatória 70/30
# Ao final: ranking das melhores combinações por Kappa e por Sensibilidade
# =============================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic, comparativo_modelos

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

ALVO    = 'GAD'        # Mude para 'SAD' se necessário
N_LOOPS = 30           # Número de repetições de cada divisão
SPLITS  = [0.30, 0.20] # Tamanhos do conjunto de teste: 70/30 e 80/20

# =============================================================================
# CARREGAR DADOS
# =============================================================================

df, target_name = preparar_dados(ALVO)

X = df.drop(columns=[target_name]).values
y = df[target_name].values

# Proporção da classe positiva — usada no Class Weighting
n_neg, n_pos = np.bincount(y)
peso_positivo = n_neg / n_pos  # ex: 243/44 ≈ 5.5

print(f"Amostras: {len(X)} | Features: {X.shape[1]} | Alvo: {target_name}")
print(f"Classe 0: {n_neg} | Classe 1: {n_pos} | Peso positivo: {peso_positivo:.1f}\n")

# =============================================================================
# DEFINIR AS COMBINAÇÕES (algoritmo + técnica de balanceamento)
# Cada entrada: (nome_exibição, modelo, técnica_balanceamento)
# técnica: None = sem balanceamento | 'smote' | 'under' | 'weight' (via modelo)
# =============================================================================

combinacoes = [
    # --- Random Forest ---
    ('RF  | Sem balanceamento',  RandomForestClassifier(n_estimators=100),                              None),
    ('RF  | Class Weight',       RandomForestClassifier(n_estimators=100, class_weight='balanced'),     None),
    ('RF  | SMOTE',              RandomForestClassifier(n_estimators=100),                              'smote'),
    ('RF  | Undersampling',      RandomForestClassifier(n_estimators=100),                              'under'),

    # --- SVM ---
    ('SVM | Sem balanceamento',  SVC(),                                                                 None),
    ('SVM | Class Weight',       SVC(class_weight='balanced'),                                          None),
    ('SVM | SMOTE',              SVC(),                                                                 'smote'),
    ('SVM | Undersampling',      SVC(),                                                                 'under'),

    # --- Rede Neural ---
    ('MLP | Sem balanceamento',  MLPClassifier(max_iter=2000),                                          None),
    ('MLP | Class Weight',       MLPClassifier(max_iter=2000),                                          None),  # MLP não tem class_weight nativo, usa SMOTE como proxy
    ('MLP | SMOTE',              MLPClassifier(max_iter=2000),                                          'smote'),
    ('MLP | Undersampling',      MLPClassifier(max_iter=2000),                                          'under'),

    # --- XGBoost ---
    ('XGB | Sem balanceamento',  XGBClassifier(eval_metric='logloss', verbosity=0),                                         None),
    ('XGB | Class Weight',       XGBClassifier(scale_pos_weight=peso_positivo, eval_metric='logloss', verbosity=0),         None),
    ('XGB | SMOTE',              XGBClassifier(eval_metric='logloss', verbosity=0),                                         'smote'),
    ('XGB | Undersampling',      XGBClassifier(eval_metric='logloss', verbosity=0),                                         'under'),
]

# =============================================================================
# LOOP PRINCIPAL — roda para cada proporção de split
# =============================================================================

todos_resultados = {}  # {split_label: {combinacao: metricas_agregadas}}

for test_size in SPLITS:
    split_label = f"{int((1-test_size)*100)}/{int(test_size*100)}"
    print(f"\n{'=' * 70}")
    print(f"Iniciando {N_LOOPS} repetições ({split_label}) com {len(combinacoes)} combinações...")
    print(f"{'=' * 70}\n")

    resultados_folds = {nome: [] for nome, _, _ in combinacoes}

    for i in range(N_LOOPS):

        # Divide aleatoriamente mantendo proporção das classes
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=test_size, stratify=y
        )

        for nome, modelo, tecnica in combinacoes:

            X_tr, y_tr = X_treino.copy(), y_treino.copy()

            # Aplica balanceamento APENAS no treino (nunca no teste)
            if tecnica == 'smote':
                X_tr, y_tr = SMOTE().fit_resample(X_tr, y_tr)
            elif tecnica == 'under':
                X_tr, y_tr = RandomUnderSampler().fit_resample(X_tr, y_tr)

            modelo.fit(X_tr, y_tr)
            y_pred = modelo.predict(X_teste)

            resultados_folds[nome].append(calcular_metricas_fold(y_teste, y_pred))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_LOOPS} iterações concluídas")

    # Agrega métricas do split atual
    resultados_agregados = {
        nome: agregar_metricas_com_ic(resultados_folds[nome])
        for nome, _, _ in combinacoes
    }
    todos_resultados[split_label] = resultados_agregados

    # Salva tabela comparativa deste split
    output_path = f'output/ml_avaliacao/{target_name}/{split_label.replace("/", "_")}'
    comparativo_modelos(resultados_agregados, target_name, f"{split_label} Aleatório — Combinações", output_path)

    # Ranking deste split
    print(f"\n{'=' * 70}")
    print(f"RANKING — {target_name} | Split {split_label}")
    print(f"{'=' * 70}")

    por_kappa = sorted(resultados_agregados.items(), key=lambda x: x[1]['kappa'], reverse=True)
    print(f"\nTop 5 por KAPPA:")
    print(f"  {'Combinação':<35} {'Kappa':>8}  {'Sens':>8}  {'Spec':>8}")
    print(f"  {'-' * 65}")
    for nome, m in por_kappa[:5]:
        print(f"  {nome:<35} {m['kappa']:>7.3f}   {m['sensitivity']:>6.1f}%   {m['specificity']:>6.1f}%")

    por_sens = sorted(resultados_agregados.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
    print(f"\nTop 5 por SENSIBILIDADE:")
    print(f"  {'Combinação':<35} {'Sens':>8}  {'Kappa':>8}  {'Spec':>8}")
    print(f"  {'-' * 65}")
    for nome, m in por_sens[:5]:
        print(f"  {nome:<35} {m['sensitivity']:>6.1f}%   {m['kappa']:>7.3f}   {m['specificity']:>6.1f}%")

    print(f"\n  Vencedor Kappa:        {por_kappa[0][0].strip()}")
    print(f"  Vencedor Sensibilidade: {por_sens[0][0].strip()}")

# =============================================================================
# COMPARATIVO FINAL: 70/30 vs 80/20 — vencedor por Kappa em cada split
# =============================================================================

print(f"\n{'=' * 70}")
print(f"COMPARATIVO FINAL — {target_name}: 70/30 vs 80/20")
print(f"{'=' * 70}")
print(f"\n  {'Combinação':<35}", end="")
for split_label in todos_resultados:
    print(f"  {split_label:>10}", end="")
print()
print(f"  {'-' * 65}")

# Pega todas as combinações únicas
for nome, _, _ in combinacoes:
    print(f"  {nome:<35}", end="")
    for split_label, resultados in todos_resultados.items():
        kappa = resultados[nome]['kappa']
        print(f"  {kappa:>10.3f}", end="")
    print()

print(f"\n{'=' * 70}\n")
