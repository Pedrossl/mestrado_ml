"""
Monte Carlo v1 — Investigacao de balanceamento para SAD.

SAD e MENOS desbalanceado que GAD (16% vs 15% positivos), entao o problema nao
e falta de positivos. Mesmo assim, o balanceamento aplicado no treino afeta
quais hard samples sao escolhidos e como o modelo retreina. Aqui varremos:
  - sampling_strategy (razao minoritaria alvo): 0.3, 0.5, 0.75, 1.0
  - SMOTE k_neighbors: 3, 5, 7
  - RandomOverSampler (duplica, nao sintetiza)
  - RandomUnderSampler
  - SMOTE + RandomUnderSampler combinados (pipeline)
  - Sem balanceamento + scale_pos_weight (class weighting puro)
  - Sem balanceamento nenhum

Tudo com XGBoost max_depth=4 (otimo ja validado para SAD).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42
ALVO = "SAD"
DEPTH = 4
N_SIM = 200
N_HARD = 20
TAM = 15

OUTPUT = "output/sad/feature_removal_runs/03_balanceamento"


def mc_v1(X, y, sampler, xgb_extra=None):
    xgb_params = dict(eval_metric="logloss", verbosity=0, random_state=SEED, max_depth=DEPTH)
    if xgb_extra:
        xgb_params.update(xgb_extra)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    if sampler is not None:
        X_tr_r, y_tr_r = sampler.fit_resample(X_tr, y_tr)
    else:
        X_tr_r, y_tr_r = X_tr, y_tr

    mb = XGBClassifier(**xgb_params)
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
        m = XGBClassifier(**xgb_params)
        m.fit(X_c, y_c)
        resultados.append(calcular_metricas_fold(y_h.astype(int), m.predict(X_h).astype(int)))

    return agregar_metricas_com_ic(resultados), n_pos, acertos


def main():
    import os
    os.makedirs(OUTPUT, exist_ok=True)

    df, target = preparar_dados(ALVO)
    X = df.drop(columns=[target]).values
    y = df[target].values

    print(f"\n{'=' * 90}")
    print(f"  MONTE CARLO v1 — Balanceamento | {ALVO} | {X.shape[1]} features | max_depth={DEPTH}")
    print(f"{'=' * 90}\n")

    configs = [
        ("SMOTE 1.0 (baseline)", SMOTE(random_state=SEED), None),
        ("SMOTE ratio=0.3", SMOTE(random_state=SEED, sampling_strategy=0.3), None),
        ("SMOTE ratio=0.5", SMOTE(random_state=SEED, sampling_strategy=0.5), None),
        ("SMOTE ratio=0.75", SMOTE(random_state=SEED, sampling_strategy=0.75), None),
        ("SMOTE k_neighbors=3", SMOTE(random_state=SEED, k_neighbors=3), None),
        ("SMOTE k_neighbors=7", SMOTE(random_state=SEED, k_neighbors=7), None),
        ("RandomOverSampler", RandomOverSampler(random_state=SEED), None),
        ("RandomUnderSampler", RandomUnderSampler(random_state=SEED), None),
        ("SMOTE 0.5 + RUS 0.8", ImbPipeline([
            ("over", SMOTE(random_state=SEED, sampling_strategy=0.5)),
            ("under", RandomUnderSampler(random_state=SEED, sampling_strategy=0.8)),
        ]), None),
        ("Sem balanc + scale_pos_weight(auto)", None, {"scale_pos_weight": 5.27}),
        ("Sem balanceamento nenhum", None, None),
    ]

    print(f"  {'Config':<38} {'Kappa':>8} {'Sens':>8} {'F1':>8} {'Spec':>8} {'sig_k':>7}")
    print(f"  {'-' * 84}")

    resultados = []
    for nome, sampler, xgb_extra in configs:
        print(f"  Rodando {nome}...", end=" ", flush=True)
        try:
            agg, n_pos, acertos = mc_v1(X, y, sampler, xgb_extra)
            print(f"OK | Hard: {n_pos}p {acertos}/{N_HARD}")
            print(f"  {nome:<38} {agg['kappa']:>8.4f} {agg['sensitivity']:>7.2f}% "
                  f"{agg['f1']:>7.2f}% {agg['specificity']:>7.2f}% {agg['kappa_std']:>7.4f}")
            resultados.append((nome, agg))
        except Exception as e:
            print(f"ERRO: {type(e).__name__}: {e}")

    kappa_base = resultados[0][1]["kappa"]
    print(f"\n{'=' * 90}")
    print("  RESUMO (delta vs SMOTE 1.0 baseline)")
    print(f"{'=' * 90}\n")
    linhas = []
    for nome, agg in sorted(resultados[1:], key=lambda x: x[1]["kappa"], reverse=True):
        delta = agg["kappa"] - kappa_base
        marca = " <-- MELHOR" if delta > 0 else ""
        linha = f"  {nome:<38} Kappa={agg['kappa']:.4f} ({delta:+.4f}){marca}"
        print(linha)
        linhas.append(linha)

    with open(f"{OUTPUT}/balanceamento.txt", "w", encoding="utf-8") as f:
        f.write(f"MONTE CARLO v1 — Balanceamento | {ALVO} | max_depth={DEPTH}\n")
        f.write("=" * 90 + "\n\n")
        for nome, agg in resultados:
            f.write(f"{nome:<38} Kappa={agg['kappa']:.4f} Sens={agg['sensitivity']:.2f}% "
                    f"F1={agg['f1']:.2f}% Spec={agg['specificity']:.2f}% sig_k={agg['kappa_std']:.4f}\n")
        f.write(f"\nbaseline SMOTE 1.0 = {kappa_base:.4f}\n")

    print(f"\n  Salvo em: {OUTPUT}/balanceamento.txt\n")


if __name__ == "__main__":
    main()
