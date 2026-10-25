"""
Monte Carlo v1 — Features de interacao para SAD.

Em vez de REMOVER ou COMBINAR-substituindo (ja testados, ambos pioraram), aqui
CRIAMOS features novas a partir das que ja existem e ADICIONAMOS por cima das
17 originais. A ideia: dar ao modelo termos de interacao que ele teria
dificuldade de descobrir sozinho com arvores rasas (max_depth=4).

Interacoes testadas (produtos/combinacoes dos preditores mais fortes de SAD
pelo Spearman: Sleep Disturbances, Social Phobia, Sensory Sensitivities,
Type B Stressors, Impairments):
  - Sleep x Social Phobia
  - Sleep x Sensory
  - Social Phobia x Sensory
  - Impairments x Type B Stressors
  - score_severidade = soma dos 5 preditores padronizados
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42
ALVO = "SAD"
DEPTH = 4
N_SIM = 200
N_HARD = 20
TAM = 15

OUTPUT = "output/sad/feature_removal_runs/04_interacoes"

TOP5 = ["Number of Sleep Disturbances", "Social Phobia",
        "Number of Sensory Sensitivities", "Number of Type B Stressors",
        "Number of Impairments"]


def add_interacoes(df, quais):
    d = df.copy()
    if "sleep_x_social" in quais:
        d["sleep_x_social"] = d["Number of Sleep Disturbances"] * d["Social Phobia"]
    if "sleep_x_sensory" in quais:
        d["sleep_x_sensory"] = d["Number of Sleep Disturbances"] * d["Number of Sensory Sensitivities"]
    if "social_x_sensory" in quais:
        d["social_x_sensory"] = d["Social Phobia"] * d["Number of Sensory Sensitivities"]
    if "impair_x_stress" in quais:
        d["impair_x_stress"] = d["Number of Impairments"] * d["Number of Type B Stressors"]
    if "score_severidade" in quais:
        sc = StandardScaler().fit_transform(d[TOP5])
        d["score_severidade"] = sc.sum(axis=1)
    return d


def mc_v1(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_tr_r, y_tr_r = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    params = dict(eval_metric="logloss", verbosity=0, random_state=SEED, max_depth=DEPTH)
    mb = XGBClassifier(**params)
    mb.fit(X_tr_r, y_tr_r)
    margem = np.abs(mb.predict_proba(X_te)[:, 1] - 0.5)
    ih = np.argsort(margem)[:N_HARD]
    X_h, y_h = X_te[ih], y_te[ih]

    resultados = []
    rng = np.random.default_rng(seed=0)
    for _ in range(N_SIM):
        idx = rng.choice(N_HARD, size=TAM, replace=False)
        X_c = np.vstack([X_tr_r, X_h[idx]])
        y_c = np.concatenate([y_tr_r, y_h[idx]])
        m = XGBClassifier(**params)
        m.fit(X_c, y_c)
        resultados.append(calcular_metricas_fold(y_h.astype(int), m.predict(X_h).astype(int)))
    return agregar_metricas_com_ic(resultados)


def main():
    import os
    os.makedirs(OUTPUT, exist_ok=True)

    df_full, target = preparar_dados(ALVO)

    print(f"\n{'=' * 90}")
    print(f"  MONTE CARLO v1 — Features de Interacao | {ALVO} | max_depth={DEPTH}")
    print(f"{'=' * 90}\n")

    todas = ["sleep_x_social", "sleep_x_sensory", "social_x_sensory",
             "impair_x_stress", "score_severidade"]

    cenarios = [("baseline (17 features)", [])]
    for inter in todas:
        cenarios.append((f"+ {inter}", [inter]))
    cenarios.append(("+ todas as 5 interacoes", todas))
    cenarios.append(("+ so os 2 produtos de sleep", ["sleep_x_social", "sleep_x_sensory"]))

    print(f"  {'Cenario':<38} {'Feat':>4} {'Kappa':>8} {'Delta':>8} {'Sens':>8} {'Spec':>8}")
    print(f"  {'-' * 84}")

    resultados = []
    for nome, quais in cenarios:
        df = add_interacoes(df_full, quais) if quais else df_full
        X = df.drop(columns=[target]).values
        y = df[target].values
        agg = mc_v1(X, y)
        resultados.append((nome, X.shape[1], agg))

    kappa_base = resultados[0][2]["kappa"]
    linhas = []
    for nome, n_feat, agg in resultados:
        delta = agg["kappa"] - kappa_base
        delta_str = f"{delta:+.4f}" if nome != "baseline (17 features)" else "-"
        marca = " <-- MELHOR" if delta > 0 else ""
        linha = (f"  {nome:<38} {n_feat:>4} {agg['kappa']:>8.4f} {delta_str:>8} "
                 f"{agg['sensitivity']:>7.2f}% {agg['specificity']:>7.2f}%{marca}")
        print(linha)
        linhas.append(linha)

    with open(f"{OUTPUT}/interacoes.txt", "w", encoding="utf-8") as f:
        f.write(f"MONTE CARLO v1 — Features de Interacao | {ALVO} | max_depth={DEPTH}\n")
        f.write("=" * 90 + "\n\n")
        for l in linhas:
            f.write(l.strip() + "\n")

    print(f"\n  Salvo em: {OUTPUT}/interacoes.txt\n")


if __name__ == "__main__":
    main()
