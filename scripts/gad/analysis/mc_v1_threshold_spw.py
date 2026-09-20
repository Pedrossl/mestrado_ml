"""
Monte Carlo v1 — Teste com scale_pos_weight=3 + Youden threshold.
Compara: default vs spw=3 (thresh 0.50) vs spw=3 (Youden) vs spw=1 (Youden).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

N_SIM = 200
TAM = 15
N_HARD = 20
SEED = 42


def mc_v1(X, y, spw=1, use_youden=False):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_tr_r, y_tr_r = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)

    params = dict(eval_metric='logloss', verbosity=0, random_state=SEED, scale_pos_weight=spw)
    mb = XGBClassifier(**params)
    mb.fit(X_tr_r, y_tr_r)

    # Threshold
    if use_youden:
        probas_tr = mb.predict_proba(X_te)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_te, probas_tr)
        youden_j = tpr - fpr
        thresh = thresholds[np.argmax(youden_j)]
    else:
        thresh = 0.50

    # Hard samples (selecionados com threshold padrao 0.50 para manter comparabilidade)
    probas_teste = mb.predict_proba(X_te)[:, 1]
    margem = np.abs(probas_teste - 0.5)
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

        m = XGBClassifier(**params)
        m.fit(X_c, y_c)

        y_proba = m.predict_proba(X_h)[:, 1]
        y_pred = (y_proba >= thresh).astype(int)
        resultados.append(calcular_metricas_fold(y_h.astype(int), y_pred))

    agg = agregar_metricas_com_ic(resultados)
    return agg, n_pos, acertos, thresh


def main():
    df, target = preparar_dados('GAD')
    X = df.drop(columns=[target]).values
    y = df[target].values
    n_feat = X.shape[1]

    cenarios = [
        ("Default (spw=1, t=0.50)", 1, False),
        ("spw=3, t=0.50", 3, False),
        ("spw=1 + Youden", 1, True),
        ("spw=3 + Youden", 3, True),
        ("spw=5 + Youden", 5, True),
    ]

    print(f"\n{'=' * 80}")
    print(f"  MONTE CARLO v1 — Threshold + scale_pos_weight | {n_feat} features | GAD")
    print(f"  {N_SIM} simulacoes | Sorteio {TAM}/{N_HARD} hard samples")
    print(f"{'=' * 80}")

    metricas = ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']
    resultados = {}

    for nome, spw, youden in cenarios:
        print(f"\n  Rodando [{nome}]...", end=" ", flush=True)
        agg, n_pos, acertos, thresh = mc_v1(X, y, spw=spw, use_youden=youden)
        resultados[nome] = (agg, n_pos, acertos, thresh)
        print(f"OK | thresh={thresh:.4f} | Hard: {n_pos} pos | Base: {acertos}/{N_HARD}")

    print(f"\n{'=' * 80}")
    print(f"  RESULTADOS")
    print(f"{'=' * 80}\n")

    print(f"  {'Cenario':<28} {'Thresh':>7} {'Kappa':>10} {'Sens':>10} {'F1':>10} {'Spec':>10} {'sigma_k':>8}")
    print(f"  {'-' * 88}")

    for nome in resultados:
        agg, _, _, thresh = resultados[nome]
        print(f"  {nome:<28} {thresh:>7.3f} {agg['kappa']:>10.4f} {agg['sensitivity']:>9.2f}% {agg['f1']:>9.2f}% {agg['specificity']:>9.2f}% {agg['kappa_std']:>8.4f}")

    print()


if __name__ == '__main__':
    main()
