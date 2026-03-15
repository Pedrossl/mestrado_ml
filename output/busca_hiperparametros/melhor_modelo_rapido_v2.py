# =============================================================================
# Melhor modelo — busca_hiperparametros v2 (modo rapido)
# Experimento: SMOTEENN_gmean
# Score Composto √(Sens×F1): 0.4847
# Sensibilidade: 63.5% ± 16.1%
# F1-Score:      37.0% ± 10.6%
# Especificidade: 65.2%
# Kappa:         0.194
# Threshold (F2 otimizado no treino): 0.491
# =============================================================================
#
# TECNICAS UTILIZADAS (resumo)
# -----------------------------------------------------------------------------
#
# [1] BALANCEAMENTO DE CLASSES — SMOTEENN
#     O dataset é altamente desbalanceado (84.7% negativos / 15.3% positivos).
#     SMOTEENN combina duas etapas:
#       - SMOTE (Synthetic Minority Over-sampling Technique): gera amostras
#         sintéticas da classe minoritária interpolando entre vizinhos reais.
#       - ENN (Edited Nearest Neighbours): remove amostras (de qualquer classe)
#         cuja classificação pelos k vizinhos mais próximos diverge da sua
#         classe real, limpando a fronteira de decisão.
#     Resultado: oversampling da minoria + limpeza de ruído na fronteira.
#     Aplicado APENAS no treino de cada fold (sem contaminação do teste).
#
# [2] CLASSIFICADOR — XGBoost (XGBClassifier)
#     Gradient Boosting baseado em árvores de decisão com regularização.
#     Constrói árvores sequencialmente, cada uma corrigindo os erros da
#     anterior. Parâmetros relevantes:
#       - n_estimators=397: número de árvores
#       - max_depth=7: profundidade máxima de cada árvore
#       - learning_rate=0.0785: taxa de aprendizado (shrinkage)
#       - subsample=0.7288: fração de amostras por árvore (evita overfitting)
#       - colsample_bytree=0.6364: fração de features por árvore
#       - colsample_bylevel=0.6152: fração de features por nível da árvore
#       - gamma=0.0387: ganho mínimo para dividir um nó (regularização)
#       - min_child_weight=7: peso mínimo de amostras em um nó folha
#       - reg_alpha=0.0056: regularização L1 (Lasso) nos pesos das folhas
#       - reg_lambda=0.8776: regularização L2 (Ridge) nos pesos das folhas
#       - scale_pos_weight=2.0621: peso extra para a classe positiva,
#         compensação adicional do desbalanceamento residual pós-SMOTEENN
#
# [3] NORMALIZAÇÃO
#     Neste experimento (NoScaler): nenhuma normalização foi aplicada.
#     XGBoost é baseado em árvores e invariante a escala, portanto não
#     requer normalização das features.
#
# [4] SCORER DE OTIMIZAÇÃO — G-Mean (gmean)
#     Usado na busca de hiperparâmetros (inner CV) para selecionar os
#     melhores parâmetros. G-Mean = √(Sensibilidade × Especificidade).
#     Penaliza modelos que sacrificam uma das classes para maximizar a outra,
#     forçando equilíbrio entre detecção de positivos e negativos.
#
# [5] THRESHOLD OTIMIZADO — F2 no treino
#     O limiar de decisão padrão do XGBoost é 0.5. Aqui ele é ajustado
#     para maximizar o F2-Score no conjunto de treino de cada fold.
#     F2 = (1 + 4) × Precisão × Sensibilidade / (4 × Precisão + Sensibilidade)
#     Pesa a sensibilidade 2x mais que a precisão, alinhado com o contexto
#     clínico onde falsos negativos (criança com GAD não detectada) são
#     mais graves que falsos positivos.
#     Threshold final: 0.491 (média dos 10 folds).
#
# [6] VALIDACAO CRUZADA — Stratified 10-Fold
#     O dataset é dividido em 10 folds mantendo a proporção de classes
#     em cada fold (estratificação). O SMOTEENN é ajustado apenas no
#     treino, nunca vendo os dados de teste (sem data leakage).
#
# =============================================================================

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    recall_score, f1_score, cohen_kappa_score,
    confusion_matrix, accuracy_score, precision_score,
)

# ---------------------------------------------------------------------------
# Carregamento dos dados — ajuste o caminho conforme necessário
# ---------------------------------------------------------------------------
df = pd.read_csv("datasets/mestrado-teste.csv")
TARGET = "GAD"  # ajuste para o nome da coluna alvo

y = df[TARGET].values
features = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
X = features.values

# ---------------------------------------------------------------------------
# Parâmetros ótimos (mediana dos 10 folds da busca)
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
# Validação cruzada 10-fold estratificada
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
sampler = SMOTEENN(random_state=42)
imputer = SimpleImputer(strategy="median")

sens_list, f1_list, spec_list, kappa_list, acc_list, prec_list = [], [], [], [], [], []
tp_list, tn_list, fp_list, fn_list = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

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
print("RESUMO (media ± desvio nos 10 folds)")
print("=" * 65)
print(f"Acuracia:      {np.mean(acc_list)*100:.1f}% ± {np.std(acc_list)*100:.1f}%")
print(f"Sensibilidade: {np.mean(sens_list)*100:.1f}% ± {np.std(sens_list)*100:.1f}%")
print(f"Especificidade:{np.mean(spec_list)*100:.1f}% ± {np.std(spec_list)*100:.1f}%")
print(f"Precisao:      {np.mean(prec_list)*100:.1f}% ± {np.std(prec_list)*100:.1f}%")
print(f"F1-Score:      {np.mean(f1_list)*100:.1f}% ± {np.std(f1_list)*100:.1f}%")
print(f"Kappa:         {np.mean(kappa_list):.3f} ± {np.std(kappa_list):.3f}")
print(f"Threshold:     {THRESHOLD}")
print()
print("MATRIZ DE CONFUSAO (soma dos 10 folds)")
print(f"  TP (verdadeiro positivo): {sum(tp_list):4d}  —  casos GAD corretamente detectados")
print(f"  TN (verdadeiro negativo): {sum(tn_list):4d}  —  casos saudaveis corretamente descartados")
print(f"  FP (falso positivo):      {sum(fp_list):4d}  —  saudaveis classificados como GAD")
print(f"  FN (falso negativo):      {sum(fn_list):4d}  —  casos GAD nao detectados (erro critico)")
print()
print("MEDIA POR FOLD")
print(f"  TP: {np.mean(tp_list):.1f} ± {np.std(tp_list):.1f}")
print(f"  TN: {np.mean(tn_list):.1f} ± {np.std(tn_list):.1f}")
print(f"  FP: {np.mean(fp_list):.1f} ± {np.std(fp_list):.1f}")
print(f"  FN: {np.mean(fn_list):.1f} ± {np.std(fn_list):.1f}")
print("=" * 65)
