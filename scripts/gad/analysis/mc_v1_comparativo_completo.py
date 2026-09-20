"""
Monte Carlo v1 — Comparativo justo: 17 feat (287 amostras) vs 12 feat (306 amostras).
Também testa 17 feat sem CD (16 feat, 287 amostras) para isolar o efeito.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

N_SIMULACOES = 200
TAMANHO_SORTEIO = 15
N_HARD = 20
ALVO = 'GAD'
SEED = 42

OUTPUT = 'output/feature_removal_runs/05_comparativo_completo'
os.makedirs(OUTPUT, exist_ok=True)


def pipeline_monte_carlo(X, y, label):
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    smote = SMOTE(random_state=SEED)
    X_tr_res, y_tr_res = smote.fit_resample(X_treino, y_treino)

    modelo_base = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=SEED)
    modelo_base.fit(X_tr_res, y_tr_res)

    probas_teste = modelo_base.predict_proba(X_teste)[:, 1]
    margem = np.abs(probas_teste - 0.5)
    idx_hard = np.argsort(margem)[:N_HARD]

    X_hard = X_teste[idx_hard]
    y_hard = y_teste[idx_hard]

    n_pos_hard = int(y_hard.sum())
    acertos_base = int((modelo_base.predict(X_hard) == y_hard).sum())

    resultados = []
    rng = np.random.default_rng(seed=0)

    for _ in range(N_SIMULACOES):
        idx = rng.choice(N_HARD, size=TAMANHO_SORTEIO, replace=False)
        X_sim = X_hard[idx]
        y_sim = y_hard[idx]

        X_comb = np.vstack([X_tr_res, X_sim])
        y_comb = np.concatenate([y_tr_res, y_sim])

        modelo = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=SEED)
        modelo.fit(X_comb, y_comb)

        y_pred = modelo.predict(X_hard)
        resultados.append(calcular_metricas_fold(y_hard.astype(int), y_pred.astype(int)))

    agg = agregar_metricas_com_ic(resultados)
    return agg, n_pos_hard, acertos_base


def main():
    # 17 features, 287 amostras (original da dissertação)
    df_17, target = preparar_dados(ALVO, aplicar_limpeza_features=False)
    feat_17 = [c for c in df_17.columns if c != target]
    X_17 = df_17.drop(columns=[target]).values
    y_17 = df_17[target].values

    # 16 features (17 sem CD), 287 amostras
    cd_idx = feat_17.index('CD')
    X_16 = np.delete(X_17, cd_idx, axis=1)

    # 13 features, 306 amostras (após remoção das 4 do config)
    df_13, _ = preparar_dados(ALVO, aplicar_limpeza_features=True, features_remover=[
        "Poverty Status", "Number of Siblings",
        "Family History - Psychiatric Diagnosis", "Number of Bio. Parents",
    ])
    X_13 = df_13.drop(columns=[target]).values
    y_13 = df_13[target].values

    # 12 features (13 sem CD), 306 amostras (atual)
    feat_13 = [c for c in df_13.columns if c != target]
    cd_idx_13 = feat_13.index('CD')
    X_12 = np.delete(X_13, cd_idx_13, axis=1)

    cenarios = [
        ("17 feat / 287 amostras (original)", X_17, y_17),
        ("16 feat / 287 (sem CD)", X_16, y_17),
        ("13 feat / 306 (sem 4 feat)", X_13, y_13),
        ("12 feat / 306 (sem 4 + CD)", X_12, y_13),
    ]

    print(f"\n{'=' * 80}")
    print(f"  MONTE CARLO v1 — Comparativo Completo | {ALVO}")
    print(f"  {N_SIMULACOES} simulações | Sorteio {TAMANHO_SORTEIO}/{N_HARD} hard samples")
    print(f"{'=' * 80}")

    resultados = {}
    for nome, X, y in cenarios:
        print(f"\n  Rodando [{nome}]...", end=" ", flush=True)
        agg, n_pos, acertos = pipeline_monte_carlo(X, y, nome)
        resultados[nome] = (agg, n_pos, acertos, X.shape[1], len(X))
        print(f"OK | {X.shape[1]} feat, {len(X)} amostras | Hard: {n_pos} pos | Base: {acertos}/{N_HARD}")

    metricas = ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']

    print(f"\n{'=' * 80}")
    print(f"  RESULTADOS")
    print(f"{'=' * 80}\n")

    for nome in resultados:
        agg, n_pos, acertos, n_feat, n_amostras = resultados[nome]
        print(f"  [{nome}]")
        print(f"    {n_feat} features | {n_amostras} amostras | Hard: {n_pos} pos | Base: {acertos}/{N_HARD}")
        for m in metricas:
            v = agg[m]
            ic = agg[m + '_ic']
            std = agg[m + '_std']
            if m == 'kappa':
                print(f"    {m:<14} {v:.4f} ± {ic:.4f} (σ={std:.4f})")
            else:
                print(f"    {m:<14} {v:.2f} ± {ic:.2f} (σ={std:.2f})")
        print()

    # Tabela resumo
    print(f"  {'Cenário':<35} {'Feat':>4} {'N':>5} {'Kappa':>8} {'Sens':>8} {'F1':>8} {'Spec':>8}")
    print(f"  {'-' * 80}")
    for nome in resultados:
        agg, _, _, n_feat, n_amostras = resultados[nome]
        print(f"  {nome:<35} {n_feat:>4} {n_amostras:>5} {agg['kappa']:>8.4f} {agg['sensitivity']:>7.2f}% {agg['f1']:>7.2f}% {agg['specificity']:>7.2f}%")

    # Salvar
    with open(f'{OUTPUT}/comparativo_completo.txt', 'w') as f:
        f.write(f"MONTE CARLO v1 — Comparativo Completo | {ALVO}\n")
        f.write(f"{N_SIMULACOES} simulações | Sorteio {TAMANHO_SORTEIO}/{N_HARD}\n")
        f.write(f"{'=' * 80}\n\n")

        for nome in resultados:
            agg, n_pos, acertos, n_feat, n_amostras = resultados[nome]
            f.write(f"[{nome}]\n")
            f.write(f"  {n_feat} features | {n_amostras} amostras | Hard: {n_pos} pos | Base: {acertos}/{N_HARD}\n")
            for m in metricas:
                v = agg[m]
                ic = agg[m + '_ic']
                std = agg[m + '_std']
                if m == 'kappa':
                    f.write(f"  {m:<14} {v:.4f} ± {ic:.4f} (σ={std:.4f})\n")
                else:
                    f.write(f"  {m:<14} {v:.2f} ± {ic:.2f} (σ={std:.2f})\n")
            f.write(f"\n")

        f.write(f"\nResumo:\n")
        f.write(f"{'Cenário':<35} {'Feat':>4} {'N':>5} {'Kappa':>8} {'Sens':>8} {'F1':>8} {'Spec':>8}\n")
        f.write(f"{'-' * 80}\n")
        for nome in resultados:
            agg, _, _, n_feat, n_amostras = resultados[nome]
            f.write(f"{nome:<35} {n_feat:>4} {n_amostras:>5} {agg['kappa']:>8.4f} {agg['sensitivity']:>7.2f}% {agg['f1']:>7.2f}% {agg['specificity']:>7.2f}%\n")

    print(f"\n  Salvo em: {OUTPUT}/comparativo_completo.txt\n")


if __name__ == '__main__':
    main()
