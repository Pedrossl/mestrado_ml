"""
Curva ROC + AUC para o conjunto GAD, com banda de IC 95% (não ±1σ).

Motivação:
  As figuras originais (curva_roc.py) desenham a banda como ±1 desvio-padrão,
  enquanto o texto/tabela da dissertação reportam o intervalo de confiança de
  95% (IC 95% = t * s / sqrt(n)). Este script regenera as figuras do GAD de
  forma COERENTE com a tabela:

    - banda sombreada  = IC 95% da TPR
    - legenda          = AUC médio ± IC 95%
    - curvas por fold  = desenhadas ao fundo (linhas finas), tornando visíveis
                         os folds desfavoráveis com AUC <= 0,50

Reproduz exatamente os mesmos folds/modelos de curva_roc.py
(StratifiedKFold 10-fold, random_state=42), logo os números batem com a tabela:
  XGBoost + SMOTE (GAD) = 0,722  (IC 95% = ±0,100)
  SVM + SMOTE     (GAD) = 0,678  (IC 95% = ±0,069)

Execução (a partir da raiz do repositório):
  PYTHONPATH=. python scripts/analysis/curva_roc_gad_ic95.py
"""

import os
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from scripts.utils import preparar_dados

N_SPLITS = 10
RANDOM_STATE = 42
CONFIANCA = 0.95


# ============================================================
#  Coleta genérica de curvas ROC por fold (reproduz curva_roc.py)
# ============================================================

def coletar_folds(X, y, algo, tecnica):
    """Coleta curvas ROC por fold para um (algoritmo, técnica de balanceamento).

    Reproduz fielmente a ordem de scaling/reamostragem usada em curva_roc.py,
    garantindo AUCs idênticos aos da tabela.

    Args:
        algo: 'xgb' ou 'svm'
        tecnica: 'sem', 'weighted', 'smote', 'undersampling'

    Returns:
        dict com mean_fpr, mean_tpr, std_tpr, tprs (lista por fold) e aucs.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    mean_fpr = np.linspace(0, 1, 100)
    peso = np.sum(y == 0) / np.sum(y == 1)
    tprs, aucs = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if algo == "xgb":
            if tecnica == "smote":
                X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
            elif tecnica == "undersampling":
                X_train, y_train = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

            kwargs = dict(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric="logloss", verbosity=0,
            )
            if tecnica == "weighted":
                kwargs["scale_pos_weight"] = peso
            model = XGBClassifier(**kwargs)
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]

        elif algo == "svm":
            # Ordem de scaling/reamostragem idêntica a curva_roc.py:
            if tecnica == "undersampling":
                X_train, y_train = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            else:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                if tecnica == "smote":
                    X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

            svc_kwargs = dict(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
            if tecnica == "weighted":
                svc_kwargs["class_weight"] = "balanced"
            model = SVC(**svc_kwargs)
            model.fit(X_train, y_train)
            y_score = model.decision_function(X_test)
        else:
            raise ValueError(f"algoritmo desconhecido: {algo}")

        fpr, tpr, _ = roc_curve(y_test, y_score)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": tprs.std(axis=0, ddof=1),
        "tprs": tprs,
        "aucs": aucs,
    }


def ic95(valores):
    """Média, desvio (ddof=1) e meia-largura do IC 95% (t de Student)."""
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    media = valores.mean()
    desvio = valores.std(ddof=1)
    ic = stats.t.ppf((1 + CONFIANCA) / 2, n - 1) * desvio / np.sqrt(n)
    return media, desvio, ic


# ============================================================
#  Plotagem com banda de IC 95%
# ============================================================

def plotar_roc_ic95(roc_dict, titulo, output_file, cores, mostrar_folds=True):
    """Plota curvas ROC com banda de IC 95% e curvas por fold ao fundo."""
    n = N_SPLITS
    t_val = stats.t.ppf((1 + CONFIANCA) / 2, n - 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    anotacoes = []
    for i, (nome, d) in enumerate(roc_dict.items()):
        cor = cores[i % len(cores)]
        media, desvio, ic = ic95(d["aucs"])

        # Curvas individuais por fold (evidenciam folds ruins com AUC <= 0,50)
        if mostrar_folds:
            for tpr in d["tprs"]:
                ax.plot(d["mean_fpr"], tpr, color=cor, lw=0.7, alpha=0.18)

        # Curva média
        ax.plot(d["mean_fpr"], d["mean_tpr"], color=cor, lw=2.5,
                label=f"{nome} (AUC = {media:.3f} ± {ic:.3f} | IC 95%)")

        # Banda de IC 95% da TPR
        ic_tpr = t_val * d["std_tpr"] / np.sqrt(n)
        lower = np.clip(d["mean_tpr"] - ic_tpr, 0, 1)
        upper = np.clip(d["mean_tpr"] + ic_tpr, 0, 1)
        ax.fill_between(d["mean_fpr"], lower, upper, color=cor, alpha=0.18)

        pior = min(d["aucs"])
        anotacoes.append(f"{nome}: pior fold AUC = {pior:.2f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Aleatório (AUC = 0.500)")

    # Caixa com o pior fold de cada modelo (torna explícito o AUC <= 0,50)
    ax.text(0.98, 0.30, "\n".join(anotacoes), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=12, fontweight="bold")
    ax.set_title(f"Curva ROC - {titulo} (GAD)\n10-Fold Stratified CV com IC 95%",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_aspect("equal")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Figura salva em: {output_file}")


def salvar_txt(roc_dict, titulo, output_file):
    """Salva AUC médio, desvio e IC 95% + AUC por fold."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"AUC (IC 95%) - {titulo} - GAD\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Modelo':<25} {'AUC Media':>12} {'Desvio':>10} {'IC 95%':>12}\n")
        f.write("-" * 60 + "\n")
        for nome, d in roc_dict.items():
            media, desvio, ic = ic95(d["aucs"])
            f.write(f"{nome:<25} {media:>12.4f} {desvio:>10.4f} {ic:>12.4f}\n")
        f.write("\n\nAUC por fold:\n")
        f.write("-" * 60 + "\n")
        for nome, d in roc_dict.items():
            f.write(f"\n{nome}:\n")
            for i, a in enumerate(d["aucs"]):
                marca = "   <-- AUC <= 0,50" if a <= 0.50 else ""
                f.write(f"  Fold {i+1:2d}: {a:.4f}{marca}\n")
    print(f"  Métricas salvas em: {output_file}")


# ============================================================
#  Main - somente GAD
# ============================================================

if __name__ == "__main__":
    cores_tecnicas = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    cores_algoritmos = ["#e67e22", "#27ae60"]

    print("#" * 70)
    print("   CURVA ROC + AUC (IC 95%) - GAD")
    print("#" * 70)

    df, target_name = preparar_dados("GAD")
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    print(f"\n[DATASET] Amostras: {df.shape[0]} | Features: {df.shape[1]-1} | Target: {target_name}")

    tecnicas = [("Sem Balanceamento", "sem"), ("Class Weighting", "weighted"),
                ("SMOTE", "smote"), ("Undersampling", "undersampling")]

    # ---- XGBoost (4 técnicas) ----
    print("\n[XGBoost] coletando folds...")
    roc_xgb = {nome: coletar_folds(X, y, "xgb", tid) for nome, tid in tecnicas}
    plotar_roc_ic95(roc_xgb, "XGBoost",
                    "output/plots/XGBoost/GAD/plots/roc_xgboost_gad_ic95.png", cores_tecnicas)
    salvar_txt(roc_xgb, "XGBoost", "output/plots/XGBoost/GAD/xgboost_gad_auc_ic95.txt")

    # ---- SVM (4 técnicas) ----
    print("\n[SVM] coletando folds...")
    roc_svm = {nome: coletar_folds(X, y, "svm", tid) for nome, tid in tecnicas}
    plotar_roc_ic95(roc_svm, "SVM",
                    "output/plots/SVM/GAD/plots/roc_svm_gad_ic95.png", cores_tecnicas)
    salvar_txt(roc_svm, "SVM", "output/plots/SVM/GAD/svm_gad_auc_ic95.txt")

    # ---- Comparativo (XGBoost vs SVM, ambos SMOTE) ----
    print("\n[Comparativo] XGBoost + SMOTE vs SVM + SMOTE...")
    roc_comp = {"XGBoost + SMOTE": roc_xgb["SMOTE"], "SVM + SMOTE": roc_svm["SMOTE"]}
    plotar_roc_ic95(roc_comp, "Comparativo de Algoritmos",
                    "output/plots/Comparativo/GAD/roc_comparativo_gad_ic95.png", cores_algoritmos)
    salvar_txt(roc_comp, "Comparativo", "output/plots/Comparativo/GAD/comparativo_gad_auc_ic95.txt")

    # ---- Resumo no terminal ----
    print("\n" + "=" * 55)
    print("  RESUMO AUC (IC 95%) - GAD")
    print("=" * 55)
    for grupo, roc in [("XGBoost", roc_xgb), ("SVM", roc_svm)]:
        for nome, d in roc.items():
            media, desvio, ic = ic95(d["aucs"])
            print(f"  {grupo:<8} {nome:<20} AUC = {media:.3f} ± {ic:.3f}")
    print("=" * 55)
    print("\nConcluído. Novas figuras com sufixo *_ic95.png\n")
