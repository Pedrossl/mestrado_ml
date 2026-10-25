"""
Triagem RAPIDA via CV 10-fold para SAD — usar ANTES do Monte Carlo v1.

CV 10-fold com XGBoost+SMOTE (max_depth=4, ja validado) e ~20x mais barato que
o Monte Carlo v1 (200 simulacoes x2 com/sem smoothing). Serve como filtro:
so promove ao Monte Carlo (caro) as combinacoes que parecerem promissoras
aqui. CV e Monte Carlo podem discordar (ja vimos isso com GAD e com SAD:
Poverty Status "parecia" bom e piorou muito no MC) — entao NADA aqui deve
virar decisao final sem passar pelo Monte Carlo depois.

Testa:
  1. Baseline e cada feature individual (sanity check vs leave-one-out ja feito)
  2. Pares das features menos prejudiciais individualmente
  3. Subconjuntos maiores guiados por Spearman (agrupamentos correlacionados)
  4. RFECV (Recursive Feature Elimination) - busca sistematica or sklearn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, cohen_kappa_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados, calcular_metricas_fold, agregar_metricas_com_ic

SEED = 42
ALVO = "SAD"
DEPTH = 4
N_SPLITS = 10


def cv_kappa(X, y):
    """CV 10-fold rapido: retorna Kappa medio +/- IC, sem Monte Carlo."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    resultados = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr_r, y_tr_r = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
        m = XGBClassifier(eval_metric="logloss", verbosity=0, random_state=SEED, max_depth=DEPTH)
        m.fit(X_tr_r, y_tr_r)
        y_pred = m.predict(X_te)
        resultados.append(calcular_metricas_fold(y_te.astype(int), y_pred.astype(int)))
    return agregar_metricas_com_ic(resultados)


def main():
    df_full, target = preparar_dados(ALVO, aplicar_limpeza_features=False)
    todas_features = [c for c in df_full.columns if c != target]
    X_full = df_full.drop(columns=[target]).values
    y = df_full[target].values

    print(f"\n{'=' * 90}")
    print(f"  TRIAGEM CV 10-fold (rapida) | {ALVO} | max_depth={DEPTH}")
    print(f"{'=' * 90}\n")

    # ------------------------------------------------------------------
    # 1. Baseline + leave-one-out (sanity check rapido)
    # ------------------------------------------------------------------
    print("--- 1. Baseline + leave-one-out (CV) ---\n")
    agg_base = cv_kappa(X_full, y)
    kappa_base = agg_base["kappa"]
    print(f"  {'baseline (17 features)':<40} Kappa={kappa_base:.4f}")

    resultados_loo = {}
    for feat in todas_features:
        df = df_full.drop(columns=[feat])
        X = df.drop(columns=[target]).values
        agg = cv_kappa(X, y)
        resultados_loo[feat] = agg["kappa"]
        delta = agg["kappa"] - kappa_base
        marca = " <-- melhora na CV" if delta > 0 else ""
        print(f"  sem {feat:<36} Kappa={agg['kappa']:.4f} ({delta:+.4f}){marca}")

    # Ranking das menos prejudiciais (maior Kappa ao remover)
    ranking = sorted(resultados_loo.items(), key=lambda x: x[1], reverse=True)
    top5_menos_prejudiciais = [f for f, k in ranking[:5]]

    print(f"\n  Top 5 menos prejudiciais na CV: {top5_menos_prejudiciais}")

    # ------------------------------------------------------------------
    # 2. Pares entre as 5 menos prejudiciais
    # ------------------------------------------------------------------
    print(f"\n--- 2. Pares entre as 5 menos prejudiciais (CV) ---\n")
    from itertools import combinations
    resultados_pares = {}
    for f1, f2 in combinations(top5_menos_prejudiciais, 2):
        df = df_full.drop(columns=[f1, f2])
        X = df.drop(columns=[target]).values
        agg = cv_kappa(X, y)
        delta = agg["kappa"] - kappa_base
        resultados_pares[(f1, f2)] = agg["kappa"]
        marca = " <-- melhora na CV" if delta > 0 else ""
        print(f"  sem {f1} + {f2:<25} Kappa={agg['kappa']:.4f} ({delta:+.4f}){marca}")

    # ------------------------------------------------------------------
    # 3. Subconjuntos por agrupamento Spearman (blocos correlacionados)
    #    ver docs/sad/FEATURE_SELECTION.md etapa 1
    # ------------------------------------------------------------------
    print(f"\n--- 3. Subconjuntos guiados por Spearman ---\n")
    grupos = {
        "sem bloco ODD (ODD+CD+Impairments+FreqIrritable, mantendo 1)":
            ["ODD", "CD", "Number of Impairments"],
        "sem bloco Social Phobia (SocialPhobia+SensorySens+Impairments, mantendo 1)":
            ["Number of Sensory Sensitivities", "Number of Impairments"],
        "sem demograficas fracas (Sex+Race+Age+BioParents+Poverty+Siblings+ADHD+FamHist)":
            ["Sex", "Race", "Age", "Number of Bio. Parents", "Poverty Status",
             "Number of Siblings", "ADHD", "Family History - Psychiatric Diagnosis"],
        "so top-9 por Spearman (mantem so as com |rho|>=0.15)":
            ["Number of Siblings", "Race", "Poverty Status", "Number of Bio. Parents",
             "Family History - Psychiatric Diagnosis", "Sex", "ADHD", "Age"],
    }
    for nome, drops in grupos.items():
        drops_validos = [d for d in drops if d in df_full.columns]
        df = df_full.drop(columns=drops_validos)
        X = df.drop(columns=[target]).values
        agg = cv_kappa(X, y)
        delta = agg["kappa"] - kappa_base
        marca = " <-- melhora na CV" if delta > 0 else ""
        print(f"  {nome:<65} ({X.shape[1]}f) Kappa={agg['kappa']:.4f} ({delta:+.4f}){marca}")

    # ------------------------------------------------------------------
    # 4. RFECV - busca sistematica do sklearn
    # ------------------------------------------------------------------
    print(f"\n--- 4. RFECV (Recursive Feature Elimination, sklearn) ---\n")
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_full, y)
    kappa_scorer = make_scorer(cohen_kappa_score)
    estimator = XGBClassifier(eval_metric="logloss", verbosity=0, random_state=SEED, max_depth=DEPTH)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rfecv = RFECV(estimator=estimator, step=1, cv=skf, scoring=kappa_scorer, min_features_to_select=5)
    rfecv.fit(X_res, y_res)

    n_selecionadas = rfecv.n_features_
    selecionadas = [f for f, s in zip(todas_features, rfecv.support_) if s]
    removidas_rfecv = [f for f in todas_features if f not in selecionadas]

    print(f"  RFECV escolheu {n_selecionadas} features (de 17)")
    print(f"  Removeria: {removidas_rfecv if removidas_rfecv else '(nenhuma)'}")

    if removidas_rfecv:
        df = df_full.drop(columns=removidas_rfecv)
        X = df.drop(columns=[target]).values
        agg = cv_kappa(X, y)
        delta = agg["kappa"] - kappa_base
        marca = " <-- melhora na CV" if delta > 0 else ""
        print(f"  Config RFECV ({n_selecionadas}f) Kappa={agg['kappa']:.4f} ({delta:+.4f}){marca}")

    print(f"\n{'=' * 90}")
    print("  FIM DA TRIAGEM — promova ao Monte Carlo v1 so o que melhorou aqui")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
