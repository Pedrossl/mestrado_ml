"""
Experimento de ablacao de features.

Compara o baseline completo com o conjunto limpo definido em scripts/config.py.
O objetivo e mostrar exatamente quais features sairam e como as metricas mudaram.
"""

from pathlib import Path

from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from scripts.config import (
    FEATURE_DROP_COLUMNS,
    FEATURE_DROP_RATIONALE,
    N_SPLITS,
    OUTPUT_DIR,
    RANDOM_STATE,
)
from scripts.utils import agregar_metricas_com_ic, calcular_metricas_fold, preparar_dados


OUTPUT_PATH = OUTPUT_DIR / "feature_ablation"


def treinar_xgboost_smote(df, target_name):
    """Treina XGBoost com SMOTE dentro de cada fold."""
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metricas_folds = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        metricas_folds.append(calcular_metricas_fold(y_test, y_pred))

    return agregar_metricas_com_ic(metricas_folds)


def montar_linha(nome, df, metricas):
    return {
        "cenario": nome,
        "n_amostras": df.shape[0],
        "n_features": df.shape[1] - 1,
        "accuracy": metricas["accuracy"],
        "accuracy_ic": metricas["accuracy_ic"],
        "sensitivity": metricas["sensitivity"],
        "sensitivity_ic": metricas["sensitivity_ic"],
        "specificity": metricas["specificity"],
        "specificity_ic": metricas["specificity_ic"],
        "f1": metricas["f1"],
        "f1_ic": metricas["f1_ic"],
        "kappa": metricas["kappa"],
        "kappa_ic": metricas["kappa_ic"],
    }


def formatar_metrica(linha, nome):
    return f"{linha[nome]:.2f} +/- {linha[f'{nome}_ic']:.2f}"


def salvar_relatorio(target, linhas, features_antes, features_depois):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_PATH / f"comparativo_ablation_{target.lower()}.csv"
    txt_path = OUTPUT_PATH / f"comparativo_ablation_{target.lower()}.txt"

    pd.DataFrame(linhas).to_csv(csv_path, index=False)

    baseline = linhas[0]
    limpo = linhas[2]

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"ABLACAO DE FEATURES - {target.upper()}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Modelo: XGBoost + SMOTE com 10-fold CV\n")
        f.write("Objetivo: comparar baseline completo vs conjunto limpo.\n\n")

        f.write("FEATURES REMOVIDAS\n")
        f.write("-" * 80 + "\n")
        for feature in FEATURE_DROP_COLUMNS:
            f.write(f"- {feature}: {FEATURE_DROP_RATIONALE.get(feature, '')}\n")

        f.write("\nFEATURES ANTES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total: {len(features_antes)}\n")
        f.write(", ".join(features_antes) + "\n")

        f.write("\nFEATURES DEPOIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total: {len(features_depois)}\n")
        f.write(", ".join(features_depois) + "\n")

        f.write("\nCOMPARATIVO\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Cenario':<22} {'Amostras':>8} {'Features':>8} {'Sensibilidade':>18} {'Especificidade':>18} {'F1':>14} {'Kappa':>14}\n")
        f.write("-" * 80 + "\n")
        for linha in linhas:
            f.write(
                f"{linha['cenario']:<22} "
                f"{linha['n_amostras']:>8} "
                f"{linha['n_features']:>8} "
                f"{formatar_metrica(linha, 'sensitivity'):>18} "
                f"{formatar_metrica(linha, 'specificity'):>18} "
                f"{formatar_metrica(linha, 'f1'):>14} "
                f"{linha['kappa']:.4f} +/- {linha['kappa_ic']:.4f}\n"
            )

        f.write("\nDELTA LIMPO_MESMAS_LINHAS - BASELINE\n")
        f.write("-" * 80 + "\n")
        limpo_mesmas_linhas = linhas[1]
        for metrica in ["accuracy", "sensitivity", "specificity", "f1", "kappa"]:
            delta = limpo_mesmas_linhas[metrica] - baseline[metrica]
            f.write(f"{metrica}: {delta:+.4f}\n")

        f.write("\nDELTA LIMPO_COM_MAIS_AMOSTRAS - BASELINE\n")
        f.write("-" * 80 + "\n")
        for metrica in ["accuracy", "sensitivity", "specificity", "f1", "kappa"]:
            delta = limpo[metrica] - baseline[metrica]
            f.write(f"{metrica}: {delta:+.4f}\n")

    return csv_path, txt_path


def executar_ablation(target="GAD"):
    df_baseline, target_name = preparar_dados(target, aplicar_limpeza_features=False)
    df_limpo, _ = preparar_dados(target, aplicar_limpeza_features=True)
    df_limpo_mesmas_linhas = df_limpo.loc[df_baseline.index]

    metricas_baseline = treinar_xgboost_smote(df_baseline, target_name)
    metricas_limpo_mesmas_linhas = treinar_xgboost_smote(df_limpo_mesmas_linhas, target_name)
    metricas_limpo = treinar_xgboost_smote(df_limpo, target_name)

    linhas = [
        montar_linha("baseline", df_baseline, metricas_baseline),
        montar_linha("sem_4_mesmas_linhas", df_limpo_mesmas_linhas, metricas_limpo_mesmas_linhas),
        montar_linha("sem_4_mais_amostras", df_limpo, metricas_limpo),
    ]

    features_antes = df_baseline.drop(columns=[target_name]).columns.tolist()
    features_depois = df_limpo.drop(columns=[target_name]).columns.tolist()
    csv_path, txt_path = salvar_relatorio(target, linhas, features_antes, features_depois)

    print(f"Relatorio salvo em: {txt_path}")
    print(f"CSV salvo em: {csv_path}")
    return linhas


if __name__ == "__main__":
    executar_ablation("GAD")
