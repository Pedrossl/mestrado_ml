"""
Permutation Importance — Mede a queda de Kappa ao embaralhar cada feature.
Usa CV 10-fold com XGBoost + SMOTE (mesmo setup do pipeline).
Features com importância baixa ou negativa são candidatas a remoção.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, cohen_kappa_score
from scripts.utils import preparar_dados

ALVO = 'GAD'
SEED = 42
N_REPEATS = 10

df, target = preparar_dados(ALVO)
feat_names = [c for c in df.columns if c != target]
X = df.drop(columns=[target]).values
y = df[target].values

kappa_scorer = make_scorer(cohen_kappa_score)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

importancias_media = np.zeros(len(feat_names))
importancias_std = np.zeros(len(feat_names))
n_folds = 0

print(f"\nPermutation Importance — {ALVO} | {len(feat_names)} features | CV 10-fold")
print(f"{'=' * 70}\n")

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    X_tr_res, y_tr_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)

    modelo = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=SEED)
    modelo.fit(X_tr_res, y_tr_res)

    result = permutation_importance(
        modelo, X_te, y_te,
        scoring=kappa_scorer,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    importancias_media += result.importances_mean
    importancias_std += result.importances_std
    n_folds += 1
    print(f"  Fold {fold_idx + 1}/10 concluído", flush=True)

importancias_media /= n_folds
importancias_std /= n_folds

ranking = np.argsort(importancias_media)

print(f"\n{'=' * 70}")
print(f"  RANKING — Permutation Importance (Kappa)")
print(f"  Quanto maior, mais importante. Negativo = remover pode ajudar.")
print(f"{'=' * 70}\n")
print(f"  {'#':<4} {'Feature':<40} {'Importância':>12} {'± Std':>10}")
print(f"  {'-' * 68}")

for i, idx in enumerate(ranking):
    nome = feat_names[idx]
    imp = importancias_media[idx]
    std = importancias_std[idx]
    flag = " ← candidata" if imp < 0.005 else ""
    print(f"  {i+1:<4} {nome:<40} {imp:>+12.4f} {std:>10.4f}{flag}")

print(f"\n  Features com importância < 0.005 são candidatas a remoção.")
print(f"  Features com importância negativa podem estar atrapalhando o modelo.\n")
