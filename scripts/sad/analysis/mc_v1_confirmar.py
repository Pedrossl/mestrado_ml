"""
Confirma uma config de SAD com 200 simulacoes, com e sem smoothing.

Uso:
    python3 -m scripts.sad.analysis.mc_v1_confirmar --smote 0.2 --depth 5 --n-est 100 --lr 0.2
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42
N_SIM = 200
N_HARD = 20
TAM = 15
EPS = 0.1


def main(smote, depth, n_est, lr):
    df, target = preparar_dados("SAD")
    X = df.drop(columns=[target]).values
    y = df[target].values

    params = dict(eval_metric="logloss", verbosity=0, random_state=SEED, n_jobs=1,
                  max_depth=depth, n_estimators=n_est, learning_rate=lr)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr_r, y_tr_r = SMOTE(random_state=SEED, sampling_strategy=smote).fit_resample(X_tr, y_tr)

    mb = XGBClassifier(**params)
    mb.fit(X_tr_r, y_tr_r)
    margem = np.abs(mb.predict_proba(X_te)[:, 1] - 0.5)
    ih = np.argsort(margem)[:N_HARD]
    X_h, y_h = X_te[ih], y_te[ih]

    res_sem, res_com = [], []
    rng = np.random.default_rng(seed=0)
    for _ in range(N_SIM):
        idx = rng.choice(N_HARD, size=TAM, replace=False)
        X_c = np.vstack([X_tr_r, X_h[idx]])
        y_c = np.concatenate([y_tr_r, y_h[idx]])

        m = XGBClassifier(**params)
        m.fit(X_c, y_c)
        res_sem.append(calcular_metricas_fold(y_h.astype(int), m.predict(X_h).astype(int)))

        marg_h = np.abs(mb.predict_proba(X_h[idx])[:, 1] - 0.5)
        pesos = np.concatenate([np.ones(len(y_tr_r)), EPS + (1 - EPS) * (marg_h / 0.5)])
        mc = XGBClassifier(**params)
        mc.fit(X_c, y_c, sample_weight=pesos)
        res_com.append(calcular_metricas_fold(y_h.astype(int), mc.predict(X_h).astype(int)))

    tag = f"smote={smote} depth={depth} n_est={n_est} lr={lr}"
    print(f"=== {tag} | hard: {int(y_h.sum())} pos | base acerta {int((mb.predict(X_h) == y_h).sum())}/20 ===")
    for nome, agg in [("sem smoothing", agregar_metricas_com_ic(res_sem)),
                      ("com smoothing", agregar_metricas_com_ic(res_com))]:
        print(f"  {nome}: kappa={agg['kappa']:.4f} (sig={agg['kappa_std']:.4f}) "
              f"sens={agg['sensitivity']:.2f}% spec={agg['specificity']:.2f}% "
              f"f1={agg['f1']:.2f}% acc={agg['accuracy']:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smote", type=float, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--n-est", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    a = p.parse_args()
    main(a.smote, a.depth, a.n_est, a.lr)
