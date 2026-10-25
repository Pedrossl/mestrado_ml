"""
Monte Carlo v1 — Engenharia de features para SAD.

Ao inves de REMOVER features de blocos correlacionados (o que ja testamos e
sempre piorou, ver docs/sad/FEATURE_SELECTION.md), esta abordagem COMBINA cada
bloco correlacionado (identificado via Spearman) numa unica feature composta
(media dos valores padronizados), preservando o sinal dos varios membros do
bloco em vez de descartar parte dele.

Blocos usados (Spearman >= 0.40 entre si, ver docs/sad/FEATURE_SELECTION.md):
  - bloco_comportamental: ODD, CD, Number of Impairments, Frequency Irritable Mood,
    Frequency Temper Tantrums
  - bloco_social: Social Phobia, Number of Sensory Sensitivities
  - bloco_familia: Race, Number of Bio. Parents
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

BLOCOS = {
    "bloco_comportamental": ["ODD", "CD", "Number of Impairments",
                              "Frequency Irritable Mood", "Frequency Temper Tantrums"],
    "bloco_social": ["Social Phobia", "Number of Sensory Sensitivities"],
    "bloco_familia": ["Race", "Number of Bio. Parents"],
}


def compor_features(df, blocos_a_usar):
    """Substitui cada bloco por uma feature composta (media padronizada)."""
    df_novo = df.copy()
    scaler = StandardScaler()
    for nome_bloco, membros in blocos_a_usar.items():
        membros_validos = [m for m in membros if m in df_novo.columns]
        if len(membros_validos) < 2:
            continue
        padronizado = scaler.fit_transform(df_novo[membros_validos])
        df_novo[nome_bloco] = padronizado.mean(axis=1)
        df_novo = df_novo.drop(columns=membros_validos)
    return df_novo


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
    df_full, target = preparar_dados(ALVO, aplicar_limpeza_features=False)

    print(f"\n{'=' * 90}")
    print(f"  MONTE CARLO v1 — Engenharia de Features (blocos compostos) | {ALVO}")
    print(f"  max_depth={DEPTH}")
    print(f"{'=' * 90}\n")

    cenarios = [
        ("baseline (17 features cruas)", {}),
        ("so bloco_comportamental composto", {"bloco_comportamental": BLOCOS["bloco_comportamental"]}),
        ("so bloco_social composto", {"bloco_social": BLOCOS["bloco_social"]}),
        ("so bloco_familia composto", {"bloco_familia": BLOCOS["bloco_familia"]}),
        ("todos os 3 blocos compostos", BLOCOS),
    ]

    resultados = []
    for nome, blocos in cenarios:
        df = compor_features(df_full, blocos) if blocos else df_full
        X = df.drop(columns=[target]).values
        y = df[target].values

        print(f"  Rodando [{nome}] ({X.shape[1]} features)...", end=" ", flush=True)
        agg = mc_v1(X, y)
        print(f"OK | Kappa={agg['kappa']:.4f}")
        resultados.append((nome, X.shape[1], agg))

    kappa_base = resultados[0][2]["kappa"]
    print(f"\n{'=' * 90}")
    print("  RESUMO (delta vs baseline)")
    print(f"{'=' * 90}\n")
    print(f"  {'Cenario':<40} {'Feat':>4} {'Kappa':>8} {'Delta':>8} {'Sens':>8} {'Spec':>8}")
    print(f"  {'-' * 78}")
    for nome, n_feat, agg in resultados:
        delta = agg["kappa"] - kappa_base
        delta_str = f"{delta:+.4f}" if nome != "baseline (17 features cruas)" else "-"
        marca = " <-- MELHOR" if delta > 0 else ""
        print(f"  {nome:<40} {n_feat:>4} {agg['kappa']:>8.4f} {delta_str:>8} "
              f"{agg['sensitivity']:>7.2f}% {agg['specificity']:>7.2f}%{marca}")

    print()


if __name__ == "__main__":
    main()
