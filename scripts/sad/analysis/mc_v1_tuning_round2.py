"""
Monte Carlo v1 — Segunda rodada de tuning para SAD, sobre o baseline de
max_depth=4 ja validado. Testa regularizacao do XGBoost (subsample,
colsample_bytree, min_child_weight, learning_rate), scale_pos_weight,
threshold e balanceamento alternativo (ADASYN, BorderlineSMOTE).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42
ALVO = "SAD"
DEPTH = 4


def mc_v1(X, y, sampler=None, xgb_params=None, thresh=0.5, thresh_from_youden=False):
    if xgb_params is None:
        xgb_params = {}
    base_params = dict(eval_metric="logloss", verbosity=0, random_state=SEED, max_depth=DEPTH)
    base_params.update(xgb_params)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    if sampler is None:
        sampler = SMOTE(random_state=SEED)
    X_tr_r, y_tr_r = sampler.fit_resample(X_tr, y_tr)

    mb = XGBClassifier(**base_params)
    mb.fit(X_tr_r, y_tr_r)

    if thresh_from_youden:
        probas_te = mb.predict_proba(X_te)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_te, probas_te)
        thresh = thresholds[np.argmax(tpr - fpr)]

    margem = np.abs(mb.predict_proba(X_te)[:, 1] - 0.5)
    ih = np.argsort(margem)[:20]
    X_h, y_h = X_te[ih], y_te[ih]

    resultados = []
    rng = np.random.default_rng(seed=0)
    for _ in range(200):
        idx = rng.choice(20, size=15, replace=False)
        X_c = np.vstack([X_tr_r, X_h[idx]])
        y_c = np.concatenate([y_tr_r, y_h[idx]])

        m = XGBClassifier(**base_params)
        m.fit(X_c, y_c)
        y_proba = m.predict_proba(X_h)[:, 1]
        y_pred = (y_proba >= thresh).astype(int)
        resultados.append(calcular_metricas_fold(y_h.astype(int), y_pred))

    return agregar_metricas_com_ic(resultados), thresh


def main():
    df, target = preparar_dados(ALVO)
    X = df.drop(columns=[target]).values
    y = df[target].values

    print(f"\n{'=' * 90}")
    print(f"  MONTE CARLO v1 — Tuning Round 2 (sobre depth={DEPTH}) | {ALVO}")
    print(f"{'=' * 90}\n")

    configs = [
        ("BASELINE (depth=4)", dict()),
        ("subsample=0.8", dict(xgb_params={"subsample": 0.8})),
        ("subsample=0.6", dict(xgb_params={"subsample": 0.6})),
        ("colsample_bytree=0.8", dict(xgb_params={"colsample_bytree": 0.8})),
        ("colsample_bytree=0.6", dict(xgb_params={"colsample_bytree": 0.6})),
        ("min_child_weight=3", dict(xgb_params={"min_child_weight": 3})),
        ("min_child_weight=5", dict(xgb_params={"min_child_weight": 5})),
        ("learning_rate=0.05", dict(xgb_params={"learning_rate": 0.05, "n_estimators": 200})),
        ("learning_rate=0.2", dict(xgb_params={"learning_rate": 0.2})),
        ("reg_alpha=1.0", dict(xgb_params={"reg_alpha": 1.0})),
        ("reg_lambda=3.0", dict(xgb_params={"reg_lambda": 3.0})),
        ("subsample=0.8+colsample=0.8", dict(xgb_params={"subsample": 0.8, "colsample_bytree": 0.8})),
        ("n_estimators=50 (com depth=4)", dict(xgb_params={"n_estimators": 50})),
        ("scale_pos_weight=2", dict(xgb_params={"scale_pos_weight": 2})),
        ("scale_pos_weight=3", dict(xgb_params={"scale_pos_weight": 3})),
        ("Youden threshold", dict(thresh_from_youden=True)),
        ("ADASYN", dict(sampler=ADASYN(random_state=SEED))),
        ("BorderlineSMOTE", dict(sampler=BorderlineSMOTE(random_state=SEED))),
    ]

    print(f"  {'Config':<32} {'Kappa':>8} {'Sens':>8} {'F1':>8} {'Spec':>8} {'sig_k':>7}")
    print(f"  {'-' * 78}")

    resultados = []
    for nome, kwargs in configs:
        print(f"  Rodando {nome}...", end=" ", flush=True)
        try:
            agg, thresh = mc_v1(X, y, **kwargs)
            print("OK")
            extra = f" (t={thresh:.3f})" if kwargs.get("thresh_from_youden") else ""
            print(f"  {nome:<32} {agg['kappa']:>8.4f} {agg['sensitivity']:>7.2f}% "
                  f"{agg['f1']:>7.2f}% {agg['specificity']:>7.2f}% {agg['kappa_std']:>7.4f}{extra}")
            resultados.append((nome, agg))
        except Exception as e:
            print(f"ERRO: {e}")

    kappa_base = resultados[0][1]["kappa"]
    print(f"\n{'=' * 90}")
    print("  RESUMO (delta vs baseline depth=4)")
    print(f"{'=' * 90}\n")
    for nome, agg in sorted(resultados[1:], key=lambda x: x[1]["kappa"], reverse=True):
        delta = agg["kappa"] - kappa_base
        marca = " <-- MELHOR" if delta > 0 else ""
        print(f"  {nome:<32} Kappa={agg['kappa']:.4f} ({delta:+.4f}){marca}")

    print()


if __name__ == "__main__":
    main()
