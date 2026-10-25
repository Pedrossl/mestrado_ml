"""
Monte Carlo v1 — Troca de algoritmo para GAD. Espelha
scripts/sad/analysis/mc_v1_algoritmos.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42
ALVO = "GAD"
N_SIM = 200
N_HARD = 20
TAM = 15


def mc_v1(X, y, make_model):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_tr_r, y_tr_r = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)

    mb = make_model()
    mb.fit(X_tr_r, y_tr_r)
    margem = np.abs(mb.predict_proba(X_te)[:, 1] - 0.5)
    ih = np.argsort(margem)[:N_HARD]
    X_h, y_h = X_te[ih], y_te[ih]
    n_pos = int(y_h.sum())
    acertos = int((mb.predict(X_h) == y_h).sum())

    resultados = []
    rng = np.random.default_rng(seed=0)
    for _ in range(N_SIM):
        idx = rng.choice(N_HARD, size=TAM, replace=False)
        X_c = np.vstack([X_tr_r, X_h[idx]])
        y_c = np.concatenate([y_tr_r, y_h[idx]])
        m = make_model()
        m.fit(X_c, y_c)
        resultados.append(calcular_metricas_fold(y_h.astype(int), m.predict(X_h).astype(int)))

    agg = agregar_metricas_com_ic(resultados)
    return agg, n_pos, acertos


def main():
    df, target = preparar_dados(ALVO)
    X = df.drop(columns=[target]).values
    y = df[target].values

    print(f"\n{'=' * 90}")
    print(f"  MONTE CARLO v1 — Troca de algoritmo | {ALVO} | {X.shape[1]} features")
    print(f"{'=' * 90}\n")

    configs = [
        ("XGBoost depth=8 (baseline)", lambda: __import__("xgboost").XGBClassifier(
            eval_metric="logloss", verbosity=0, random_state=SEED, max_depth=8)),
        ("LightGBM depth=8", lambda: LGBMClassifier(
            random_state=SEED, max_depth=8, verbosity=-1)),
        ("LightGBM depth=-1 (default)", lambda: LGBMClassifier(
            random_state=SEED, verbosity=-1)),
        ("CatBoost depth=8", lambda: CatBoostClassifier(
            random_seed=SEED, depth=8, verbose=False)),
        ("CatBoost depth=6 (default)", lambda: CatBoostClassifier(
            random_seed=SEED, verbose=False)),
        ("RandomForest (default)", lambda: RandomForestClassifier(
            random_state=SEED, n_estimators=100)),
        ("RandomForest sem limite de depth", lambda: RandomForestClassifier(
            random_state=SEED, n_estimators=100, max_depth=None)),
        ("HistGradientBoosting (default)", lambda: HistGradientBoostingClassifier(
            random_state=SEED)),
        ("HistGradientBoosting max_depth=8", lambda: HistGradientBoostingClassifier(
            random_state=SEED, max_depth=8)),
    ]

    print(f"  {'Algoritmo':<32} {'Kappa':>8} {'Sens':>8} {'F1':>8} {'Spec':>8} {'sig_k':>7}")
    print(f"  {'-' * 78}")

    resultados = []
    for nome, make_model in configs:
        print(f"  Rodando {nome}...", end=" ", flush=True)
        try:
            agg, n_pos, acertos = mc_v1(X, y, make_model)
            print(f"OK | Hard: {n_pos} pos | Base: {acertos}/{N_HARD}")
            print(f"  {nome:<32} {agg['kappa']:>8.4f} {agg['sensitivity']:>7.2f}% "
                  f"{agg['f1']:>7.2f}% {agg['specificity']:>7.2f}% {agg['kappa_std']:>7.4f}")
            resultados.append((nome, agg))
        except Exception as e:
            print(f"ERRO: {type(e).__name__}: {e}")

    kappa_base = resultados[0][1]["kappa"]
    print(f"\n{'=' * 90}")
    print("  RESUMO (delta vs XGBoost depth=8)")
    print(f"{'=' * 90}\n")
    for nome, agg in sorted(resultados[1:], key=lambda x: x[1]["kappa"], reverse=True):
        delta = agg["kappa"] - kappa_base
        marca = " <-- MELHOR" if delta > 0 else ""
        print(f"  {nome:<32} Kappa={agg['kappa']:.4f} ({delta:+.4f}){marca}")

    print()


if __name__ == "__main__":
    main()
