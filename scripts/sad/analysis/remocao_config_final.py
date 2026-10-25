"""
SAD — importancia e remocao de variaveis na CONFIG FINAL (SMOTE 0.2, depth=3).

Os testes de remocao anteriores foram feitos com SMOTE 1.0 e depth 4/6. Como o
balanceamento mudou, refazemos aqui. Cada cenario e avaliado de duas formas:
  - MC oficial: Monte Carlo v1 com os hard samples escolhidos pelo proprio modelo
  - Teste completo: os mesmos 58 casos de teste para todos (comparacao justa)

Uso:
  python3 -m scripts.sad.analysis.remocao_config_final importancia
  python3 -m scripts.sad.analysis.remocao_config_final loo --worker 0 --n-workers 4
  python3 -m scripts.sad.analysis.remocao_config_final pares --features "A,B,C"
  python3 -m scripts.sad.analysis.remocao_config_final merge --modo loo
"""

import argparse
import glob
import itertools
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.config import SMOTE_STRATEGY_BY_TARGET, XGB_MAX_DEPTH_BY_TARGET
from scripts.utils import agregar_metricas_com_ic, calcular_metricas_fold, preparar_dados

ALVO = "SAD"
SEED = 42
N_SIM = 200
N_HARD = 20
TAM = 15
SMOTE_RATIO = SMOTE_STRATEGY_BY_TARGET[ALVO]
DEPTH = XGB_MAX_DEPTH_BY_TARGET[ALVO]
OUTPUT = "output/sad/feature_removal_runs/07_remocao_config_final"


def modelo():
    return XGBClassifier(eval_metric="logloss", verbosity=0, random_state=SEED,
                         n_jobs=1, max_depth=DEPTH)


def split(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr_r, y_tr_r = SMOTE(random_state=SEED, sampling_strategy=SMOTE_RATIO).fit_resample(X_tr, y_tr)
    return X_tr_r, y_tr_r, X_te, y_te


def monte_carlo(X_tr_r, y_tr_r, X_av, y_av):
    n = len(y_av)
    tam = int(round(n * TAM / N_HARD))
    res = []
    rng = np.random.default_rng(seed=0)
    for _ in range(N_SIM):
        idx = rng.choice(n, size=tam, replace=False)
        m = modelo()
        m.fit(np.vstack([X_tr_r, X_av[idx]]), np.concatenate([y_tr_r, y_av[idx]]))
        res.append(calcular_metricas_fold(y_av.astype(int), m.predict(X_av).astype(int)))
    return agregar_metricas_com_ic(res)


def avaliar(df, target):
    X = df.drop(columns=[target]).values
    y = df[target].values
    X_tr_r, y_tr_r, X_te, y_te = split(X, y)

    mb = modelo()
    mb.fit(X_tr_r, y_tr_r)
    ih = np.argsort(np.abs(mb.predict_proba(X_te)[:, 1] - 0.5))[:N_HARD]

    oficial = monte_carlo(X_tr_r, y_tr_r, X_te[ih], y_te[ih])
    completo = monte_carlo(X_tr_r, y_tr_r, X_te, y_te)
    return oficial, completo, int(y_te[ih].sum())


def linha(nome, n_feat, oficial, completo, hard_pos):
    return {
        "cenario": nome, "n_features": n_feat, "hard_pos": hard_pos,
        "mc_kappa": oficial["kappa"], "mc_kappa_std": oficial["kappa_std"],
        "mc_sens": oficial["sensitivity"], "mc_sens_std": oficial["sensitivity_std"],
        "mc_spec": oficial["specificity"], "mc_f1": oficial["f1"],
        "tc_kappa": completo["kappa"], "tc_kappa_std": completo["kappa_std"],
        "tc_sens": completo["sensitivity"], "tc_spec": completo["specificity"],
        "tc_f1": completo["f1"],
    }


def rodar_cenarios(cenarios, arquivo):
    os.makedirs(OUTPUT, exist_ok=True)
    df_base, target = preparar_dados(ALVO, aplicar_limpeza_features=False)
    linhas = []
    for nome, drops in cenarios:
        df = df_base.drop(columns=drops) if drops else df_base
        oficial, completo, hard_pos = avaliar(df, target)
        linhas.append(linha(nome, df.shape[1] - 1, oficial, completo, hard_pos))
        print(f"{nome:<45} MC={oficial['kappa']:.4f}±{oficial['kappa_std']:.3f} "
              f"TC={completo['kappa']:.4f}±{completo['kappa_std']:.3f} hard+={hard_pos}", flush=True)
        pd.DataFrame(linhas).to_csv(arquivo, index=False)


def importancia():
    os.makedirs(OUTPUT, exist_ok=True)
    df, target = preparar_dados(ALVO, aplicar_limpeza_features=False)
    feats = [c for c in df.columns if c != target]
    X = df[feats].values
    y = df[target].values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    perm = np.zeros(len(feats))
    gain = np.zeros(len(feats))
    for tr, te in skf.split(X, y):
        X_r, y_r = SMOTE(random_state=SEED, sampling_strategy=SMOTE_RATIO).fit_resample(X[tr], y[tr])
        m = modelo()
        m.fit(X_r, y_r)
        gain += m.feature_importances_
        r = permutation_importance(m, X[te], y[te], scoring=make_scorer(cohen_kappa_score),
                                   n_repeats=10, random_state=SEED)
        perm += r.importances_mean
    perm /= 10
    gain /= 10

    spearman = df.corr(method="spearman")[target].drop(target)
    out = pd.DataFrame({"feature": feats, "permutation_kappa": perm, "gain": gain,
                        "spearman_sad": spearman[feats].values})
    out = out.sort_values("permutation_kappa")
    out.to_csv(f"{OUTPUT}/importancia.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


def merge(modo):
    arquivos = sorted(glob.glob(f"{OUTPUT}/{modo}_*.csv"))
    df = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True).drop_duplicates("cenario")
    base = df[df.cenario == "baseline (17 features)"].iloc[0]
    df["delta_mc"] = df.mc_kappa - base.mc_kappa
    df["delta_tc"] = df.tc_kappa - base.tc_kappa
    df = df.sort_values("delta_tc", ascending=False)
    df.to_csv(f"{OUTPUT}/{modo}_completo.csv", index=False)
    cols = ["cenario", "hard_pos", "mc_kappa", "mc_kappa_std", "delta_mc",
            "tc_kappa", "tc_kappa_std", "delta_tc", "tc_sens"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("modo", choices=["importancia", "loo", "pares", "merge"])
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--features", type=str, default="")
    p.add_argument("--modo-merge", dest="modo_merge", default="loo")
    a = p.parse_args()

    if a.modo == "importancia":
        importancia()
    elif a.modo == "loo":
        df_base, target = preparar_dados(ALVO, aplicar_limpeza_features=False)
        feats = [c for c in df_base.columns if c != target]
        cenarios = [("baseline (17 features)", [])] + [(f"sem {f}", [f]) for f in feats]
        rodar_cenarios(cenarios[a.worker::a.n_workers], f"{OUTPUT}/loo_{a.worker}.csv")
    elif a.modo == "pares":
        feats = [f.strip() for f in a.features.split(",") if f.strip()]
        cenarios = [("baseline (17 features)", [])]
        cenarios += [(f"sem {f1} + {f2}", [f1, f2]) for f1, f2 in itertools.combinations(feats, 2)]
        if len(feats) >= 3:
            cenarios += [(f"sem {' + '.join(c)}", list(c)) for c in itertools.combinations(feats, 3)]
        rodar_cenarios(cenarios[a.worker::a.n_workers], f"{OUTPUT}/pares_{a.worker}.csv")
    else:
        merge(a.modo_merge)
