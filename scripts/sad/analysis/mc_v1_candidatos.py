"""
Monte Carlo v1 — Testa candidatas a remocao para SAD, uma de cada vez.

Candidatas vieram do cruzamento de Spearman + Permutation Importance
(ver docs/sad/FEATURE_SELECTION.md). Cada uma e testada isoladamente contra
o baseline (17 features) para nao repetir o erro cometido em GAD de testar
varias remocoes ao mesmo tempo (isso muda a composicao da amostra e confunde
a comparacao).

So promove uma candidata para FEATURE_DROP_COLUMNS_BY_TARGET["SAD"] se o
Monte Carlo v1 melhorar (nao so a CV/Permutation Importance).
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic
from scripts.config import XGB_MAX_DEPTH_BY_TARGET

ALVO = "SAD"
SEED = 42
N_SIMULACOES = 200
N_HARD = 20
TAMANHO_SORTEIO = 15

OUTPUT = "output/sad/feature_removal_runs/01_candidatos"

CANDIDATAS = [
    ("baseline (17 features)", []),
    ("sem ODD", ["ODD"]),
    ("sem Family History", ["Family History - Psychiatric Diagnosis"]),
    ("sem Number of Bio. Parents", ["Number of Bio. Parents"]),
    ("sem Sex", ["Sex"]),
    ("sem Number of Impairments", ["Number of Impairments"]),
]


def mc_v1(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_tr_res, y_tr_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)

    params = dict(eval_metric="logloss", verbosity=0, random_state=SEED,
                  max_depth=XGB_MAX_DEPTH_BY_TARGET[ALVO])
    modelo_base = XGBClassifier(**params)
    modelo_base.fit(X_tr_res, y_tr_res)

    margem = np.abs(modelo_base.predict_proba(X_te)[:, 1] - 0.5)
    idx_hard = np.argsort(margem)[:N_HARD]
    X_hard, y_hard = X_te[idx_hard], y_te[idx_hard]
    n_pos_hard = int(y_hard.sum())
    acertos_base = int((modelo_base.predict(X_hard) == y_hard).sum())

    resultados = []
    rng = np.random.default_rng(seed=0)
    for _ in range(N_SIMULACOES):
        idx = rng.choice(N_HARD, size=TAMANHO_SORTEIO, replace=False)
        X_comb = np.vstack([X_tr_res, X_hard[idx]])
        y_comb = np.concatenate([y_tr_res, y_hard[idx]])

        modelo = XGBClassifier(**params)
        modelo.fit(X_comb, y_comb)
        y_pred = modelo.predict(X_hard)
        resultados.append(calcular_metricas_fold(y_hard.astype(int), y_pred.astype(int)))

    agg = agregar_metricas_com_ic(resultados)
    return agg, n_pos_hard, acertos_base


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"  MONTE CARLO v1 — Candidatas a remocao | {ALVO}")
    print(f"  {N_SIMULACOES} simulacoes | Sorteio {TAMANHO_SORTEIO}/{N_HARD} hard samples")
    print(f"  max_depth={XGB_MAX_DEPTH_BY_TARGET[ALVO]} (default XGBoost - SAD ainda nao foi tunado)")
    print(f"{'=' * 80}")

    # IMPORTANTE: usa sempre as mesmas linhas do baseline (sem novo dropna a
    # cada remocao). Se uma feature removida tiver valores faltantes (ex:
    # Poverty Status em SAD), reaplicar preparar_dados() com essa feature fora
    # reincorpora linhas antes descartadas, mudando a amostra e invalidando a
    # comparacao (mesmo bug encontrado e corrigido para GAD — ver
    # docs/gad/FEATURE_SELECTION.md, secao 1).
    df_base, target = preparar_dados(ALVO, aplicar_limpeza_features=False)

    resultados = {}
    for nome, drops in CANDIDATAS:
        df = df_base.drop(columns=drops) if drops else df_base
        X = df.drop(columns=[target]).values
        y = df[target].values

        print(f"\n  Rodando [{nome}] ({X.shape[1]} features)...", end=" ", flush=True)
        agg, n_pos, acertos = mc_v1(X, y)
        resultados[nome] = (agg, n_pos, acertos, X.shape[1])
        print(f"OK | Hard: {n_pos} pos | Base: {acertos}/{N_HARD}")

    metricas = ["accuracy", "sensitivity", "specificity", "f1", "kappa"]

    print(f"\n{'=' * 80}")
    print("  RESUMO (delta vs baseline)")
    print(f"{'=' * 80}\n")

    kappa_base = resultados["baseline (17 features)"][0]["kappa"]

    print(f"  {'Cenario':<32} {'Feat':>4} {'Kappa':>10} {'Delta':>8} {'Sens':>8} {'F1':>8} {'Spec':>8}")
    print(f"  {'-' * 82}")
    for nome, (agg, n_pos, acertos, n_feat) in resultados.items():
        delta = agg["kappa"] - kappa_base
        delta_str = f"{delta:+.4f}" if nome != "baseline (17 features)" else "—"
        print(f"  {nome:<32} {n_feat:>4} {agg['kappa']:>10.4f} {delta_str:>8} "
              f"{agg['sensitivity']:>7.2f}% {agg['f1']:>7.2f}% {agg['specificity']:>7.2f}%")

    with open(f"{OUTPUT}/comparativo_candidatos.txt", "w", encoding="utf-8") as f:
        f.write(f"MONTE CARLO v1 — Candidatas a remocao | {ALVO}\n")
        f.write(f"{N_SIMULACOES} simulacoes | Sorteio {TAMANHO_SORTEIO}/{N_HARD} | "
                f"max_depth={XGB_MAX_DEPTH_BY_TARGET[ALVO]}\n")
        f.write("=" * 80 + "\n\n")

        for nome, (agg, n_pos, acertos, n_feat) in resultados.items():
            f.write(f"[{nome}]\n")
            f.write(f"  {n_feat} features | Hard: {n_pos} pos | Base: {acertos}/{N_HARD}\n")
            for m in metricas:
                v, ic, std = agg[m], agg[m + "_ic"], agg[m + "_std"]
                if m == "kappa":
                    f.write(f"  {m:<14} {v:.4f} +/- {ic:.4f} (sigma={std:.4f})\n")
                else:
                    f.write(f"  {m:<14} {v:.2f} +/- {ic:.2f} (sigma={std:.2f})\n")
            f.write("\n")

        f.write("Resumo (delta vs baseline):\n")
        f.write(f"{'Cenario':<32} {'Feat':>4} {'Kappa':>10} {'Delta':>8} {'Sens':>8} {'F1':>8} {'Spec':>8}\n")
        f.write("-" * 82 + "\n")
        for nome, (agg, n_pos, acertos, n_feat) in resultados.items():
            delta = agg["kappa"] - kappa_base
            delta_str = f"{delta:+.4f}" if nome != "baseline (17 features)" else "-"
            f.write(f"{nome:<32} {n_feat:>4} {agg['kappa']:>10.4f} {delta_str:>8} "
                    f"{agg['sensitivity']:>7.2f}% {agg['f1']:>7.2f}% {agg['specificity']:>7.2f}%\n")

    print(f"\n  Salvo em: {OUTPUT}/comparativo_candidatos.txt\n")


if __name__ == "__main__":
    main()
