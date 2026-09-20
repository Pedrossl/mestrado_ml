"""MC v1 — Combinar max_depth=8 + SMOTETomek."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42; N_HARD = 20; TAM = 15; N_SIM = 200


def mc_v1(X, y, sampler, xgb_params):
    base = dict(eval_metric='logloss', verbosity=0, random_state=SEED)
    base.update(xgb_params)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr_r, y_tr_r = sampler.fit_resample(X_tr, y_tr)
    mb = XGBClassifier(**base)
    mb.fit(X_tr_r, y_tr_r)
    margem = np.abs(mb.predict_proba(X_te)[:, 1] - 0.5)
    ih = np.argsort(margem)[:N_HARD]
    X_h, y_h = X_te[ih], y_te[ih]
    res = []
    rng = np.random.default_rng(seed=0)
    for _ in range(N_SIM):
        idx = rng.choice(N_HARD, size=TAM, replace=False)
        X_c = np.vstack([X_tr_r, X_h[idx]])
        y_c = np.concatenate([y_tr_r, y_h[idx]])
        m = XGBClassifier(**base)
        m.fit(X_c, y_c)
        res.append(calcular_metricas_fold(y_h.astype(int), m.predict(X_h).astype(int)))
    return agregar_metricas_com_ic(res)


df, t = preparar_dados('GAD')
X = df.drop(columns=[t]).values
y = df[t].values

configs = [
    ('BASELINE (SMOTE, depth=6)', SMOTE(random_state=SEED), {}),
    ('max_depth=8', SMOTE(random_state=SEED), {'max_depth': 8}),
    ('SMOTETomek', SMOTETomek(random_state=SEED), {}),
    ('SMOTETomek + depth=8', SMOTETomek(random_state=SEED), {'max_depth': 8}),
]

header = f"{'Config':<30} {'Kappa':>8} {'Sens':>8} {'F1':>8} {'Spec':>8} {'sig_k':>7} {'sig_s':>7}"
print(header)
print("-" * 82)

for nome, samp, xgb in configs:
    print(f"  Rodando {nome}...", end=" ", flush=True)
    agg = mc_v1(X, y, samp, xgb)
    print("OK")
    line = f"{nome:<30} {agg['kappa']:>8.4f} {agg['sensitivity']:>7.2f}% {agg['f1']:>7.2f}% {agg['specificity']:>7.2f}% {agg['kappa_std']:>7.4f} {agg['sensitivity_std']:>7.2f}"
    print(line)
