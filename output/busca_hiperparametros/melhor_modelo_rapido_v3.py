# =============================================================================
# Melhor modelo — busca_hiperparametros v2 (modo rapido) — CORRIGIDO v3
# Experimento: SMOTEENN_gmean
# Score Composto √(Sens×F1): 0.4847
# Sensibilidade: 63.5% ± 16.1%
# F1-Score:      37.0% ± 10.6%
# Especificidade: 65.2%
# Kappa:         0.194
# Threshold (F2 otimizado no treino): 0.491
# =============================================================================
#
# CORRECOES EM RELACAO AO v2
# -----------------------------------------------------------------------------
#
# [PROBLEMA] O v2 carregava o CSV bruto com pd.read_csv() e aplicava apenas
#   SimpleImputer, mas os hiperparametros foram descobertos pela busca usando
#   preparar_dados() → carregar_teste_normalizado(), que aplica:
#     - MinMax nas colunas numericas
#     - Number of Siblings convertido para binario (0/1)
#     - Number of Bio. Parents mapeado para 0 / 0.5 / 1
#     - Remocao de colunas (Depression, Family History - Substance Abuse, etc.)
#     - Codificacao de Sex (M→0, F→1)
#     - Drop de linhas com NaN residual
#   Aplicar hiperparametros otimizados em dados nao normalizados distorce as
#   probabilidades e o threshold, degradando os resultados.
#
# [CORRECAO] Substitui o carregamento manual por preparar_dados(), que e a
#   mesma funcao usada durante a busca de hiperparametros. O preprocessamento
#   e agora identico ao do treinamento original.
#   O SimpleImputer foi removido pois preparar_dados() ja faz dropna().
#
# =============================================================================
#
# TECNICAS UTILIZADAS (resumo)
# -----------------------------------------------------------------------------
#
# [1] PREPROCESSAMENTO — pipeline identico ao da busca de hiperparametros
#     Normalização MinMax nas features numericas, binarizacao de siblings,
#     mapeamento de bio parents, codificacao de sex, remocao de colunas
#     nao preditivas. Tudo aplicado antes da CV (dados ja limpos).
#
# [2] BALANCEAMENTO DE CLASSES — SMOTEENN
#     O dataset e altamente desbalanceado (84.7% negativos / 15.3% positivos).
#     SMOTEENN combina duas etapas:
#       - SMOTE: gera amostras sinteticas da classe minoritaria interpolando
#         entre vizinhos reais.
#       - ENN (Edited Nearest Neighbours): remove amostras cuja classificacao
#         pelos k vizinhos diverge da sua classe real, limpando a fronteira.
#     Aplicado APENAS no treino de cada fold (sem contaminacao do teste).
#
# [3] CLASSIFICADOR — XGBoost (XGBClassifier)
#     Gradient Boosting baseado em arvores de decisao com regularizacao.
#     Parametros relevantes:
#       - n_estimators=397: numero de arvores
#       - max_depth=7: profundidade maxima de cada arvore
#       - learning_rate=0.0785: taxa de aprendizado (shrinkage)
#       - subsample=0.7288: fracao de amostras por arvore
#       - colsample_bytree=0.6364: fracao de features por arvore
#       - colsample_bylevel=0.6152: fracao de features por nivel da arvore
#       - gamma=0.0387: ganho minimo para dividir um no (regularizacao)
#       - min_child_weight=7: peso minimo de amostras em um no folha
#       - reg_alpha=0.0056: regularizacao L1 nos pesos das folhas
#       - reg_lambda=0.8776: regularizacao L2 nos pesos das folhas
#       - scale_pos_weight=2.0621: peso extra para a classe positiva
#
# [4] SCORER DE OTIMIZACAO — G-Mean (gmean)
#     G-Mean = sqrt(Sensibilidade x Especificidade). Penaliza modelos que
#     sacrificam uma das classes, forcando equilibrio entre elas.
#
# [5] THRESHOLD OTIMIZADO — F2 no treino
#     O threshold padrao (0.5) e ajustado para maximizar o F2-Score no
#     conjunto de treino de cada fold (sem data leakage).
#     F2 pesa sensibilidade 2x mais que precisao — alinhado com o contexto
#     clinico onde falsos negativos (GAD nao detectado) sao mais graves.
#     Threshold final: 0.491 (media dos 10 folds da busca original).
#
# [6] VALIDACAO CRUZADA — Stratified 10-Fold
#     10 folds mantendo a proporcao de classes em cada fold. SMOTEENN
#     ajustado apenas no treino de cada fold (sem data leakage).
#
# =============================================================================

import sys
import os
import numpy as np
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score, f1_score, cohen_kappa_score,
    confusion_matrix, accuracy_score, precision_score,
)

# Adiciona raiz do projeto ao path para importar scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.utils import preparar_dados

# ---------------------------------------------------------------------------
# Carregamento e preprocessamento — identico ao usado na busca de hiperparametros
# ---------------------------------------------------------------------------
df, TARGET = preparar_dados('GAD')

X = df.drop(columns=[TARGET]).values
y = df[TARGET].values

# ---------------------------------------------------------------------------
# Parametros otimos (mediana dos 10 folds da busca)
# ---------------------------------------------------------------------------
PARAMS = dict(
    colsample_bylevel=0.6152,
    colsample_bytree=0.6364,
    gamma=0.0387,
    learning_rate=0.0785,
    max_depth=7,
    min_child_weight=7,
    n_estimators=397,
    reg_alpha=0.0056,
    reg_lambda=0.8776,
    scale_pos_weight=2.0621,
    subsample=0.7288,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

THRESHOLD = 0.491

# ---------------------------------------------------------------------------
# Validacao cruzada 10-fold estratificada
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
sampler = SMOTEENN(random_state=42)

sens_list, f1_list, spec_list, kappa_list, acc_list, prec_list = [], [], [], [], [], []
tp_list, tn_list, fp_list, fn_list = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_res, y_res = sampler.fit_resample(X_train, y_train)

    model = XGBClassifier(**PARAMS)
    model.fit(X_res, y_res)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    sens = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    sens_list.append(sens)
    f1_list.append(f1)
    spec_list.append(spec)
    kappa_list.append(kappa)
    acc_list.append(acc)
    prec_list.append(prec)
    tp_list.append(tp)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)

    print(f"Fold {fold:2d}/10 | TP={tp:2d}  TN={tn:3d}  FP={fp:2d}  FN={fn:2d} | "
          f"Sens={sens*100:.1f}%  Spec={spec*100:.1f}%  Prec={prec*100:.1f}%  "
          f"Acc={acc*100:.1f}%  F1={f1*100:.1f}%  Kappa={kappa:.3f}")

print()
print("=" * 65)
print("RESUMO (media +- desvio nos 10 folds)")
print("=" * 65)
print(f"Acuracia:      {np.mean(acc_list)*100:.1f}% +- {np.std(acc_list)*100:.1f}%")
print(f"Sensibilidade: {np.mean(sens_list)*100:.1f}% +- {np.std(sens_list)*100:.1f}%")
print(f"Especificidade:{np.mean(spec_list)*100:.1f}% +- {np.std(spec_list)*100:.1f}%")
print(f"Precisao:      {np.mean(prec_list)*100:.1f}% +- {np.std(prec_list)*100:.1f}%")
print(f"F1-Score:      {np.mean(f1_list)*100:.1f}% +- {np.std(f1_list)*100:.1f}%")
print(f"Kappa:         {np.mean(kappa_list):.3f} +- {np.std(kappa_list):.3f}")
print(f"Threshold:     {THRESHOLD}")
print()
print("MATRIZ DE CONFUSAO (soma dos 10 folds)")
print(f"  TP (verdadeiro positivo): {sum(tp_list):4d}  —  casos GAD corretamente detectados")
print(f"  TN (verdadeiro negativo): {sum(tn_list):4d}  —  casos saudaveis corretamente descartados")
print(f"  FP (falso positivo):      {sum(fp_list):4d}  —  saudaveis classificados como GAD")
print(f"  FN (falso negativo):      {sum(fn_list):4d}  —  casos GAD nao detectados (erro critico)")
print()
print("MEDIA POR FOLD")
print(f"  TP: {np.mean(tp_list):.1f} +- {np.std(tp_list):.1f}")
print(f"  TN: {np.mean(tn_list):.1f} +- {np.std(tn_list):.1f}")
print(f"  FP: {np.mean(fp_list):.1f} +- {np.std(fp_list):.1f}")
print(f"  FN: {np.mean(fn_list):.1f} +- {np.std(fn_list):.1f}")
print("=" * 65)
