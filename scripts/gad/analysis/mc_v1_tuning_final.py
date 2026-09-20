"""
Monte Carlo v1 — Tuning final: N_HARD, TAMANHO_SORTEIO, SMOTE+limpeza, depth.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42


def mc_v1(X, y, n_hard=20, tam_sorteio=15, n_sim=200, sampler=None,
          re_smote=False, xgb_params=None):
    if xgb_params is None:
        xgb_params = {}
    base_params = dict(eval_metric='logloss', verbosity=0, random_state=SEED)
    base_params.update(xgb_params)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    if sampler is None:
        sampler = SMOTE(random_state=SEED)
    X_tr_r, y_tr_r = sampler.fit_resample(X_tr, y_tr)

    mb = XGBClassifier(**base_params)
    mb.fit(X_tr_r, y_tr_r)
    margem = np.abs(mb.predict_proba(X_te)[:, 1] - 0.5)
    ih = np.argsort(margem)[:n_hard]
    X_h, y_h = X_te[ih], y_te[ih]

    resultados = []
    rng = np.random.default_rng(seed=0)

    for _ in range(n_sim):
        idx = rng.choice(n_hard, size=tam_sorteio, replace=False)
        X_c = np.vstack([X_tr_r, X_h[idx]])
        y_c = np.concatenate([y_tr_r, y_h[idx]])

        if re_smote:
            try:
                sm2 = SMOTE(random_state=SEED, k_neighbors=min(3, int(y_c.sum()) - 1))
                X_c, y_c = sm2.fit_resample(X_c, y_c)
            except Exception:
                pass

        m = XGBClassifier(**base_params)
        m.fit(X_c, y_c)
        y_pred = m.predict(X_h)
        resultados.append(calcular_metricas_fold(y_h.astype(int), y_pred.astype(int)))

    return agregar_metricas_com_ic(resultados)


def main():
    df, target = preparar_dados('GAD')
    X = df.drop(columns=[target]).values
    y = df[target].values

    print(f"\n{'=' * 90}")
    print(f"  MONTE CARLO v1 — Tuning Final | 15 features | GAD")
    print(f"{'=' * 90}\n")

    configs = [
        # Baseline
        ("BASELINE (20h, 15s, 200sim)", dict()),

        # N_HARD
        ("N_HARD=15, sorteio=10", dict(n_hard=15, tam_sorteio=10)),
        ("N_HARD=25, sorteio=18", dict(n_hard=25, tam_sorteio=18)),
        ("N_HARD=30, sorteio=22", dict(n_hard=30, tam_sorteio=22)),

        # TAMANHO_SORTEIO
        ("N_HARD=20, sorteio=10", dict(tam_sorteio=10)),
        ("N_HARD=20, sorteio=18", dict(tam_sorteio=18)),

        # Limpeza
        ("SMOTETomek", dict(sampler=SMOTETomek(random_state=SEED))),
        ("SMOTEENN", dict(sampler=SMOTEENN(random_state=SEED))),

        # Re-SMOTE
        ("Re-SMOTE apos hard", dict(re_smote=True)),

        # Profundidade
        ("max_depth=8", dict(xgb_params={'max_depth': 8})),
        ("max_depth=10", dict(xgb_params={'max_depth': 10})),
        ("n_estimators=200", dict(xgb_params={'n_estimators': 200})),
        ("depth=8 + n_est=200", dict(xgb_params={'max_depth': 8, 'n_estimators': 200})),

        # Mais simulacoes
        ("500 simulacoes", dict(n_sim=500)),
    ]

    print(f"  {'Config':<32} {'Kappa':>8} {'Sens':>8} {'F1':>8} {'Spec':>8} {'sig_k':>7} {'sig_s':>7}")
    print(f"  {'-' * 85}")

    for nome, kwargs in configs:
        print(f"  Rodando {nome}...", end=" ", flush=True)
        try:
            agg = mc_v1(X, y, **kwargs)
            print("OK")
            print(f"  {nome:<32} {agg['kappa']:>8.4f} {agg['sensitivity']:>7.2f}% {agg['f1']:>7.2f}% {agg['specificity']:>7.2f}% {agg['kappa_std']:>7.4f} {agg['sensitivity_std']:>7.2f}")
        except Exception as e:
            print(f"ERRO: {e}")

    print()


if __name__ == '__main__':
    main()
