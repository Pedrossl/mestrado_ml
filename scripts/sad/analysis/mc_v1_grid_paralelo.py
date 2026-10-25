"""
Grid paralelo de hiperparametros para SAD (Monte Carlo v1).

Objetivo: alem de Kappa alto (ja temos 0.8992), aproximar o perfil do GAD:
sensibilidade maior (GAD 93%) e sigma do Kappa menor (GAD 0.114).

Uso (cada worker pega uma fatia do grid):
    python3 -m scripts.sad.analysis.mc_v1_grid_paralelo --worker 0 --n-workers 4 --n-sim 100
Depois:
    python3 -m scripts.sad.analysis.mc_v1_grid_paralelo --merge
"""

import argparse
import glob
import itertools
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42
ALVO = "SAD"
N_HARD = 20
TAM = 15
OUTPUT = "output/sad/feature_removal_runs/06_grid_paralelo"

GRID = {
    # razao natural do SAD no treino ja e 0.193 -> valores abaixo disso sao invalidos.
    # 0.2 gera so ~1 sintetico (na pratica, quase sem balanceamento).
    "smote": [0.2, 0.25, 0.3, 0.4, 0.5],
    "depth": [2, 3, 4, 5],
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2, 0.3],
}


def mc_v1(X, y, smote, depth, n_est, lr, n_sim):
    params = dict(eval_metric="logloss", verbosity=0, random_state=SEED, n_jobs=1,
                  max_depth=depth, n_estimators=n_est, learning_rate=lr)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr_r, y_tr_r = SMOTE(random_state=SEED, sampling_strategy=smote).fit_resample(X_tr, y_tr)

    mb = XGBClassifier(**params)
    mb.fit(X_tr_r, y_tr_r)
    margem = np.abs(mb.predict_proba(X_te)[:, 1] - 0.5)
    ih = np.argsort(margem)[:N_HARD]
    X_h, y_h = X_te[ih], y_te[ih]

    resultados = []
    rng = np.random.default_rng(seed=0)
    for _ in range(n_sim):
        idx = rng.choice(N_HARD, size=TAM, replace=False)
        m = XGBClassifier(**params)
        m.fit(np.vstack([X_tr_r, X_h[idx]]), np.concatenate([y_tr_r, y_h[idx]]))
        resultados.append(calcular_metricas_fold(y_h.astype(int), m.predict(X_h).astype(int)))
    return agregar_metricas_com_ic(resultados), int(y_h.sum())


def rodar_worker(worker, n_workers, n_sim):
    os.makedirs(OUTPUT, exist_ok=True)
    df, target = preparar_dados(ALVO)
    X = df.drop(columns=[target]).values
    y = df[target].values

    combos = list(itertools.product(*GRID.values()))
    meus = combos[worker::n_workers]
    print(f"worker {worker}/{n_workers}: {len(meus)} configs, n_sim={n_sim}", flush=True)

    linhas = []
    for i, (smote, depth, n_est, lr) in enumerate(meus, 1):
        agg, hard_pos = mc_v1(X, y, smote, depth, n_est, lr, n_sim)
        linhas.append({
            "smote": smote, "depth": depth, "n_estimators": n_est, "learning_rate": lr,
            "kappa": agg["kappa"], "kappa_std": agg["kappa_std"],
            "sensitivity": agg["sensitivity"], "specificity": agg["specificity"],
            "f1": agg["f1"], "hard_pos": hard_pos,
        })
        print(f"[w{worker} {i}/{len(meus)}] smote={smote} d={depth} n={n_est} lr={lr} "
              f"kappa={agg['kappa']:.4f} sens={agg['sensitivity']:.1f} sig={agg['kappa_std']:.3f}",
              flush=True)
        pd.DataFrame(linhas).to_csv(f"{OUTPUT}/worker_{worker}.csv", index=False)


def merge():
    arquivos = sorted(glob.glob(f"{OUTPUT}/worker_*.csv"))
    df = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)
    df.to_csv(f"{OUTPUT}/grid_completo.csv", index=False)
    print(f"{len(df)} configs no total\n")

    cols = ["smote", "depth", "n_estimators", "learning_rate", "kappa", "sensitivity",
            "specificity", "f1", "kappa_std", "hard_pos"]
    print("=== TOP 15 por Kappa ===")
    print(df.sort_values("kappa", ascending=False)[cols].head(15).to_string(index=False))

    print("\n=== Perfil GAD-like: Kappa >= 0.88 e sens >= 90, ordenado por sigma ===")
    gad_like = df[(df.kappa >= 0.88) & (df.sensitivity >= 90)].sort_values("kappa_std")
    print(gad_like[cols].head(15).to_string(index=False) if len(gad_like) else "nenhuma")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--n-sim", type=int, default=100)
    p.add_argument("--merge", action="store_true")
    a = p.parse_args()
    if a.merge:
        merge()
    else:
        rodar_worker(a.worker, a.n_workers, a.n_sim)
