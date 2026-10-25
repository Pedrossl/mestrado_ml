"""
Monte Carlo v1 — Varredura leave-one-out para SAD.

Remove cada uma das 17 features, uma de cada vez, e compara o Kappa contra o
baseline completo. Mais abrangente que mc_v1_candidatos.py (que so testava as
5 candidatas apontadas por Spearman/Permutation Importance) — aqui testamos
TODAS, porque em SAD os sinais de CV foram inconsistentes (12/17 features com
Permutation Importance negativa) e a remocao das 5 candidatas prioritarias
piorou o Kappa, entao vale checar se alguma das 12 restantes ajuda de verdade.
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

OUTPUT = "output/sad/feature_removal_runs/02_leave_one_out"


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

    df_full, target = preparar_dados(ALVO, aplicar_limpeza_features=False)
    todas_features = [c for c in df_full.columns if c != target]

    cenarios = [("baseline (17 features)", [])]
    for feat in todas_features:
        cenarios.append((f"sem {feat}", [feat]))

    print(f"\n{'=' * 80}")
    print(f"  MONTE CARLO v1 — Leave-one-out | {ALVO}")
    print(f"  {N_SIMULACOES} simulacoes | Sorteio {TAMANHO_SORTEIO}/{N_HARD} hard samples")
    print(f"  max_depth={XGB_MAX_DEPTH_BY_TARGET[ALVO]} (default XGBoost)")
    print(f"  {len(cenarios)} cenarios (baseline + {len(todas_features)} remocoes individuais)")
    print(f"{'=' * 80}")

    # IMPORTANTE: usa sempre as mesmas linhas do baseline (sem novo dropna a
    # cada remocao). Se a feature removida tiver valores faltantes (ex:
    # Poverty Status em SAD), reaplicar preparar_dados() com essa feature fora
    # reincorpora linhas antes descartadas, mudando a amostra e invalidando a
    # comparacao (mesmo bug encontrado e corrigido para GAD — ver
    # docs/gad/FEATURE_SELECTION.md, secao 1). A primeira tentativa deste
    # script tinha esse bug: "sem Poverty Status" parecia melhorar o Kappa,
    # mas era so o efeito de ir de 287 para 306 amostras.
    resultados = {}
    for nome, drops in cenarios:
        df = df_full.drop(columns=drops) if drops else df_full
        X = df.drop(columns=[target]).values
        y = df[target].values

        print(f"\n  Rodando [{nome}] ({X.shape[1]} features)...", end=" ", flush=True)
        agg, n_pos, acertos = mc_v1(X, y)
        resultados[nome] = (agg, n_pos, acertos, X.shape[1])
        print(f"OK | Hard: {n_pos} pos | Base: {acertos}/{N_HARD} | Kappa={agg['kappa']:.4f}")

    kappa_base = resultados["baseline (17 features)"][0]["kappa"]

    linhas_resumo = []
    for nome, (agg, n_pos, acertos, n_feat) in resultados.items():
        delta = agg["kappa"] - kappa_base
        linhas_resumo.append((nome, n_feat, agg, delta))

    linhas_resumo_ordenado = sorted(
        [l for l in linhas_resumo if l[0] != "baseline (17 features)"],
        key=lambda x: x[3], reverse=True
    )
    linhas_resumo_ordenado = [linhas_resumo[0]] + linhas_resumo_ordenado

    print(f"\n{'=' * 80}")
    print("  RESUMO — ordenado por delta de Kappa (melhor primeiro)")
    print(f"{'=' * 80}\n")
    print(f"  {'Cenario':<45} {'Feat':>4} {'Kappa':>10} {'Delta':>9} {'Sens':>8} {'Spec':>8}")
    print(f"  {'-' * 90}")
    for nome, n_feat, agg, delta in linhas_resumo_ordenado:
        delta_str = f"{delta:+.4f}" if nome != "baseline (17 features)" else "—"
        marca = " <-- MELHOROU" if delta > 0 else ""
        print(f"  {nome:<45} {n_feat:>4} {agg['kappa']:>10.4f} {delta_str:>9} "
              f"{agg['sensitivity']:>7.2f}% {agg['specificity']:>7.2f}%{marca}")

    melhorou = [l for l in linhas_resumo_ordenado if l[0] != "baseline (17 features)" and l[3] > 0]

    with open(f"{OUTPUT}/leave_one_out.txt", "w", encoding="utf-8") as f:
        f.write(f"MONTE CARLO v1 — Leave-one-out | {ALVO}\n")
        f.write(f"{N_SIMULACOES} simulacoes | Sorteio {TAMANHO_SORTEIO}/{N_HARD} | "
                f"max_depth={XGB_MAX_DEPTH_BY_TARGET[ALVO]}\n")
        f.write("=" * 80 + "\n\n")

        metricas = ["accuracy", "sensitivity", "specificity", "f1", "kappa"]
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

        f.write("Resumo (ordenado por delta de Kappa, melhor primeiro):\n")
        f.write(f"{'Cenario':<45} {'Feat':>4} {'Kappa':>10} {'Delta':>9} {'Sens':>8} {'Spec':>8}\n")
        f.write("-" * 90 + "\n")
        for nome, n_feat, agg, delta in linhas_resumo_ordenado:
            delta_str = f"{delta:+.4f}" if nome != "baseline (17 features)" else "-"
            marca = " <-- MELHOROU" if delta > 0 else ""
            f.write(f"{nome:<45} {n_feat:>4} {agg['kappa']:>10.4f} {delta_str:>9} "
                    f"{agg['sensitivity']:>7.2f}% {agg['specificity']:>7.2f}%{marca}\n")

    print(f"\n  Salvo em: {OUTPUT}/leave_one_out.txt")
    if melhorou:
        print(f"\n  {len(melhorou)} feature(s) cuja remocao MELHOROU o Kappa:")
        for nome, n_feat, agg, delta in melhorou:
            print(f"    {nome}: {delta:+.4f}")
    else:
        print("\n  Nenhuma remocao individual melhorou o Kappa.")
    print()


if __name__ == "__main__":
    main()
