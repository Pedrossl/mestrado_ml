"""
Monte Carlo v1 — Cruzamento balanceamento x profundidade para SAD.

A rodada de balanceamento (03_balanceamento) mostrou que SMOTE leve (ratio=0.3)
ou sem balanceamento batem o SMOTE 1.0, porque SAD e pouco desbalanceado e o
oversampling agressivo distorce a distribuicao. Mas max_depth=4 foi tunado sob
SMOTE 1.0 — ao mudar o balanceamento, a profundidade otima pode mudar. Aqui
testamos o produto cartesiano balanceamento x depth para achar a combinacao
real otima.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42
ALVO = "SAD"
N_SIM = 200
N_HARD = 20
TAM = 15

OUTPUT = "output/sad/feature_removal_runs/05_balanc_x_depth"


def mc_v1(X, y, sampler, depth):
    params = dict(eval_metric="logloss", verbosity=0, random_state=SEED, max_depth=depth)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    if sampler is not None:
        X_tr_r, y_tr_r = sampler.fit_resample(X_tr, y_tr)
    else:
        X_tr_r, y_tr_r = X_tr, y_tr

    mb = XGBClassifier(**params)
    mb.fit(X_tr_r, y_tr_r)
    margem = np.abs(mb.predict_proba(X_te)[:, 1] - 0.5)
    ih = np.argsort(margem)[:N_HARD]
    X_h, y_h = X_te[ih], y_te[ih]
    n_pos = int(y_h.sum())

    resultados = []
    rng = np.random.default_rng(seed=0)
    for _ in range(N_SIM):
        idx = rng.choice(N_HARD, size=TAM, replace=False)
        X_c = np.vstack([X_tr_r, X_h[idx]])
        y_c = np.concatenate([y_tr_r, y_h[idx]])
        m = XGBClassifier(**params)
        m.fit(X_c, y_c)
        resultados.append(calcular_metricas_fold(y_h.astype(int), m.predict(X_h).astype(int)))
    return agregar_metricas_com_ic(resultados), n_pos


def make_sampler(nome):
    if nome == "none":
        return None
    if nome == "smote1.0":
        return SMOTE(random_state=SEED)
    if nome == "smote0.3":
        return SMOTE(random_state=SEED, sampling_strategy=0.3)
    if nome == "smote0.2":
        return SMOTE(random_state=SEED, sampling_strategy=0.2)


def main():
    import os
    os.makedirs(OUTPUT, exist_ok=True)

    df, target = preparar_dados(ALVO)
    X = df.drop(columns=[target]).values
    y = df[target].values

    print(f"\n{'=' * 90}")
    print(f"  MONTE CARLO v1 — Balanceamento x Depth | {ALVO} | {X.shape[1]} features")
    print(f"{'=' * 90}\n")

    balanceamentos = ["smote1.0", "smote0.3", "smote0.2", "none"]
    depths = [2, 3, 4, 5, 6, 8]

    print(f"  {'balanc':<10} {'depth':>5} {'Kappa':>9} {'Sens':>8} {'F1':>8} {'Spec':>8} {'sig_k':>7} {'hard+':>6}")
    print(f"  {'-' * 76}")

    resultados = []
    for bal in balanceamentos:
        for d in depths:
            agg, n_pos = mc_v1(X, y, make_sampler(bal), d)
            resultados.append((bal, d, agg, n_pos))
            print(f"  {bal:<10} {d:>5} {agg['kappa']:>9.4f} {agg['sensitivity']:>7.2f}% "
                  f"{agg['f1']:>7.2f}% {agg['specificity']:>7.2f}% {agg['kappa_std']:>7.4f} {n_pos:>6}")

    print(f"\n{'=' * 90}")
    print("  TOP 10 por Kappa")
    print(f"{'=' * 90}\n")
    ordenado = sorted(resultados, key=lambda r: r[2]["kappa"], reverse=True)
    linhas = []
    for bal, d, agg, n_pos in ordenado[:10]:
        linha = (f"  {bal:<10} depth={d}  Kappa={agg['kappa']:.4f}  "
                 f"Sens={agg['sensitivity']:.2f}%  F1={agg['f1']:.2f}%  Spec={agg['specificity']:.2f}%")
        print(linha)
        linhas.append(linha)

    with open(f"{OUTPUT}/balanc_x_depth.txt", "w", encoding="utf-8") as f:
        f.write(f"MONTE CARLO v1 — Balanceamento x Depth | {ALVO} | {X.shape[1]} features\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'balanc':<10} {'depth':>5} {'Kappa':>9} {'Sens':>8} {'F1':>8} {'Spec':>8} {'sig_k':>7} {'hard+':>6}\n")
        for bal, d, agg, n_pos in resultados:
            f.write(f"{bal:<10} {d:>5} {agg['kappa']:>9.4f} {agg['sensitivity']:>7.2f}% "
                    f"{agg['f1']:>7.2f}% {agg['specificity']:>7.2f}% {agg['kappa_std']:>7.4f} {n_pos:>6}\n")
        f.write("\nTOP 10:\n")
        for l in linhas:
            f.write(l.strip() + "\n")

    print(f"\n  Salvo em: {OUTPUT}/balanc_x_depth.txt\n")


if __name__ == "__main__":
    main()
