"""
Varredura de cenarios de remocao de features.

Roda varias hipoteses com embasamento em Spearman, VIF, importancia do XGBoost
e criterio de fairness/sensibilidade. Salva os resultados em uma pasta separada
para comparacao incremental das rodadas de limpeza.
"""

from pathlib import Path
import warnings

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from scripts.config import N_SPLITS, OUTPUT_DIR, RANDOM_STATE
from scripts.utils import calcular_ic, calcular_metricas_fold, preparar_dados


warnings.filterwarnings("ignore", category=FutureWarning)

RUN_ID = "02_sweep_feature_candidates"
OUTPUT_PATH = OUTPUT_DIR / "feature_removal_runs" / RUN_ID

MONTE_CARLO_SIMULACOES = 200
MONTE_CARLO_HARD_SAMPLES = 20
MONTE_CARLO_TAMANHO_SORTEIO = 15
SMOOTH_EPSILON = 0.1


SCENARIOS = [
    {
        "id": "baseline",
        "drops": [],
        "rationale": "Todas as features atuais; referencia para comparacao.",
    },
    {
        "id": "sem_poverty_status",
        "drops": ["Poverty Status"],
        "rationale": "Menor correlacao com GAD e redundancia socioeconomica.",
    },
    {
        "id": "sem_number_siblings",
        "drops": ["Number of Siblings"],
        "rationale": "Correlacao quase nula com GAD e baixa importancia.",
    },
    {
        "id": "sem_family_history_psych",
        "drops": ["Family History - Psychiatric Diagnosis"],
        "rationale": "Correlacao fraca com GAD e baixa importancia.",
    },
    {
        "id": "sem_bio_parents",
        "drops": ["Number of Bio. Parents"],
        "rationale": "Correlacao fraca com GAD, VIF moderado e redundancia com Race/Poverty.",
    },
    {
        "id": "sem_4_fracas_redundantes",
        "drops": [
            "Poverty Status",
            "Number of Siblings",
            "Family History - Psychiatric Diagnosis",
            "Number of Bio. Parents",
        ],
        "rationale": "Primeira limpeza segura: baixa relacao com GAD e/ou redundancia.",
    },
    {
        "id": "sem_status_race_bio",
        "drops": ["Poverty Status", "Race", "Number of Bio. Parents"],
        "rationale": "Bloco socioeconomico/demografico correlacionado e sensivel.",
    },
    {
        "id": "sem_features_sensiveis",
        "drops": ["Race", "Sex", "Poverty Status", "Number of Bio. Parents"],
        "rationale": "Teste de fairness: remove variaveis sensiveis ou proxy demografico.",
    },
    {
        "id": "sem_cd",
        "drops": ["CD"],
        "rationale": "CD e redundante com ODD; ODD tem maior sinal com GAD.",
    },
    {
        "id": "sem_odd",
        "drops": ["ODD"],
        "rationale": "Alternativa inversa do par CD/ODD para validar qual carrega mais sinal.",
    },
    {
        "id": "sem_tantrums",
        "drops": ["Frequency Temper Tantrums"],
        "rationale": "Redundante com Frequency Irritable Mood e ODD.",
    },
    {
        "id": "sem_irritable_mood",
        "drops": ["Frequency Irritable Mood"],
        "rationale": "Alternativa inversa do par irritabilidade/tantrums.",
    },
    {
        "id": "sem_social_phobia",
        "drops": ["Social Phobia"],
        "rationale": "Alternativa do par Social Phobia/Sensory Sensitivities.",
    },
    {
        "id": "sem_sensory",
        "drops": ["Number of Sensory Sensitivities"],
        "rationale": "Redundancia moderada com Social Phobia; testar perda de sinal.",
    },
    {
        "id": "sem_sleep",
        "drops": ["Number of Sleep Disturbances"],
        "rationale": "VIF moderado; testar se multicolinearidade pesa mais que sinal clinico.",
    },
    {
        "id": "sem_age",
        "drops": ["Age"],
        "rationale": "Maior VIF; testar custo de remover feature importante.",
    },
    {
        "id": "sem_4_mais_cd",
        "drops": [
            "Poverty Status",
            "Number of Siblings",
            "Family History - Psychiatric Diagnosis",
            "Number of Bio. Parents",
            "CD",
        ],
        "rationale": "Primeira limpeza + remocao do par redundante CD/ODD.",
    },
    {
        "id": "sem_4_mais_tantrums",
        "drops": [
            "Poverty Status",
            "Number of Siblings",
            "Family History - Psychiatric Diagnosis",
            "Number of Bio. Parents",
            "Frequency Temper Tantrums",
        ],
        "rationale": "Primeira limpeza + remocao de tantrums por redundancia com irritabilidade.",
    },
    {
        "id": "sem_4_cd_tantrums",
        "drops": [
            "Poverty Status",
            "Number of Siblings",
            "Family History - Psychiatric Diagnosis",
            "Number of Bio. Parents",
            "CD",
            "Frequency Temper Tantrums",
        ],
        "rationale": "Limpeza intermediaria: baixa relevancia + dois blocos correlacionados.",
    },
    {
        "id": "sem_4_cd_tantrums_sleep",
        "drops": [
            "Poverty Status",
            "Number of Siblings",
            "Family History - Psychiatric Diagnosis",
            "Number of Bio. Parents",
            "CD",
            "Frequency Temper Tantrums",
            "Number of Sleep Disturbances",
        ],
        "rationale": "Limpeza mais agressiva incluindo feature com VIF moderado.",
    },
    {
        "id": "sem_4_e_sensiveis",
        "drops": [
            "Poverty Status",
            "Number of Siblings",
            "Family History - Psychiatric Diagnosis",
            "Number of Bio. Parents",
            "Race",
            "Sex",
        ],
        "rationale": "Primeira limpeza + retirada de variaveis sensiveis remanescentes.",
    },
    {
        "id": "sem_10_agressivo",
        "drops": [
            "Poverty Status",
            "Number of Siblings",
            "Family History - Psychiatric Diagnosis",
            "Number of Bio. Parents",
            "CD",
            "Frequency Temper Tantrums",
            "Number of Sleep Disturbances",
            "Race",
            "Sex",
            "ADHD",
        ],
        "rationale": "Teste extremo para medir quanto sinal se perde com uma limpeza grande.",
    },
]


def treinar_xgboost_smote_cv(df, target_name):
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    metricas_folds = []
    aucs = []

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
            n_jobs=1,
        )
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metricas_folds.append(calcular_metricas_fold(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_proba))

    return agregar_metricas_com_auc(metricas_folds, aucs)


def agregar_metricas_com_auc(metricas_folds, aucs):
    nomes = ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "kappa"]
    resultado = {
        "vn": sum(m["vn"] for m in metricas_folds),
        "fp": sum(m["fp"] for m in metricas_folds),
        "fn": sum(m["fn"] for m in metricas_folds),
        "vp": sum(m["vp"] for m in metricas_folds),
    }

    for nome in nomes:
        valores = [m[nome] for m in metricas_folds]
        media, desvio, ic = calcular_ic(valores)
        resultado[nome] = media
        resultado[f"{nome}_std"] = desvio
        resultado[f"{nome}_ic"] = ic

    media, desvio, ic = calcular_ic(aucs)
    resultado["auc"] = media
    resultado["auc_std"] = desvio
    resultado["auc_ic"] = ic
    return resultado


def preparar_cenario(target, drops, baseline_index=None):
    df, target_name = preparar_dados(
        target,
        aplicar_limpeza_features=bool(drops),
        features_remover=drops,
    )
    if baseline_index is not None:
        df = df.loc[baseline_index]
    return df, target_name


def linha_cv(scenario, modo, df, metricas):
    return {
        "scenario": scenario["id"],
        "modo": modo,
        "n_drops": len(scenario["drops"]),
        "drops": "; ".join(scenario["drops"]),
        "rationale": scenario["rationale"],
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
        "auc": metricas["auc"],
        "auc_ic": metricas["auc_ic"],
    }


def rodar_cv_sweep(target="GAD"):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    df_baseline, target_name = preparar_cenario(target, [])
    baseline_index = df_baseline.index

    linhas = []
    for scenario in SCENARIOS:
        print(f"[CV] {scenario['id']}")
        df_full, _ = preparar_cenario(target, scenario["drops"])
        metricas_full = treinar_xgboost_smote_cv(df_full, target_name)
        linhas.append(linha_cv(scenario, "amostras_validas", df_full, metricas_full))

        if scenario["id"] != "baseline":
            df_same, _ = preparar_cenario(target, scenario["drops"], baseline_index)
            metricas_same = treinar_xgboost_smote_cv(df_same, target_name)
            linhas.append(linha_cv(scenario, "mesmas_linhas_baseline", df_same, metricas_same))

    df_resultados = pd.DataFrame(linhas)
    df_resultados.to_csv(OUTPUT_PATH / "cv_sweep_resultados.csv", index=False)
    salvar_resumo_cv(df_resultados)
    return df_resultados


def salvar_resumo_cv(df_resultados):
    baseline = df_resultados[df_resultados["scenario"].eq("baseline")].iloc[0]
    df = df_resultados.copy()
    for metrica in ["accuracy", "sensitivity", "specificity", "f1", "kappa", "auc"]:
        df[f"delta_{metrica}"] = df[metrica] - baseline[metrica]

    principais = df[df["modo"].eq("mesmas_linhas_baseline")].copy()
    principais = principais.sort_values(["delta_kappa", "delta_f1", "delta_sensitivity"], ascending=False)
    principais.to_csv(OUTPUT_PATH / "cv_sweep_ranking_mesmas_linhas.csv", index=False)

    with (OUTPUT_PATH / "cv_sweep_resumo.md").open("w", encoding="utf-8") as f:
        f.write("# Sweep de remocao de features\n\n")
        f.write("Modelo: XGBoost + SMOTE com 10-fold CV.\n\n")
        f.write("## Baseline\n\n")
        f.write(
            f"- Amostras: {int(baseline['n_amostras'])}\n"
            f"- Features: {int(baseline['n_features'])}\n"
            f"- Sensibilidade: {baseline['sensitivity']:.2f} +/- {baseline['sensitivity_ic']:.2f}\n"
            f"- Especificidade: {baseline['specificity']:.2f} +/- {baseline['specificity_ic']:.2f}\n"
            f"- F1: {baseline['f1']:.2f} +/- {baseline['f1_ic']:.2f}\n"
            f"- Kappa: {baseline['kappa']:.4f} +/- {baseline['kappa_ic']:.4f}\n"
            f"- AUC: {baseline['auc']:.4f} +/- {baseline['auc_ic']:.4f}\n\n"
        )

        f.write("## Ranking controlado pelas mesmas linhas do baseline\n\n")
        f.write("| Cenario | Drops | Sens. | F1 | Kappa | AUC | Delta Kappa | Racional |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for _, row in principais.iterrows():
            f.write(
                f"| {row['scenario']} | {int(row['n_drops'])} | "
                f"{row['sensitivity']:.2f} | {row['f1']:.2f} | "
                f"{row['kappa']:.4f} | {row['auc']:.4f} | "
                f"{row['delta_kappa']:+.4f} | {row['rationale']} |\n"
            )


def rodar_monte_carlo_v1_para_cenario(target, scenario, baseline_index=None, modo="amostras_validas"):
    df, target_name = preparar_cenario(target, scenario["drops"])
    if baseline_index is not None:
        df = df.loc[baseline_index]
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_tr_res, y_tr_res = smote.fit_resample(X_treino, y_treino)

    modelo_base = XGBClassifier(eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE, n_jobs=1)
    modelo_base.fit(X_tr_res, y_tr_res)

    probas = modelo_base.predict_proba(X_teste)[:, 1]
    y_pred = modelo_base.predict(X_teste)
    margem = np.abs(probas - 0.5)
    idx_hard = np.argsort(margem)[:MONTE_CARLO_HARD_SAMPLES]
    X_hard = X_teste[idx_hard]
    y_hard = y_teste[idx_hard]

    def rodar(usar_smoothing):
        resultados = []
        rng = np.random.default_rng(seed=0)
        for _ in range(MONTE_CARLO_SIMULACOES):
            idx = rng.choice(len(X_hard), size=MONTE_CARLO_TAMANHO_SORTEIO, replace=False)
            X_sim = X_hard[idx]
            y_sim = y_hard[idx]

            X_combinado = np.vstack([X_tr_res, X_sim])
            y_combinado = np.concatenate([y_tr_res, y_sim])

            pesos_treino = np.ones(len(y_tr_res))
            if usar_smoothing:
                probas_hard = modelo_base.predict_proba(X_sim)[:, 1]
                margem_hard = np.abs(probas_hard - 0.5)
                pesos_hard = SMOOTH_EPSILON + (1 - SMOOTH_EPSILON) * (margem_hard / 0.5)
            else:
                pesos_hard = np.ones(len(y_sim))

            pesos = np.concatenate([pesos_treino, pesos_hard])
            modelo = XGBClassifier(eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE, n_jobs=1)
            modelo.fit(X_combinado, y_combinado, sample_weight=pesos)
            pred = modelo.predict(X_hard)
            resultados.append(calcular_metricas_fold(y_hard.astype(int), pred.astype(int)))
        return agregar_metricas_com_auc(resultados, [0.5] * len(resultados))

    sem = rodar(False)
    com = rodar(True)
    return {
        "scenario": scenario["id"],
        "modo": modo,
        "n_drops": len(scenario["drops"]),
        "drops": "; ".join(scenario["drops"]),
        "rationale": scenario["rationale"],
        "n_amostras": df.shape[0],
        "n_features": df.shape[1] - 1,
        "hard_positivos": int(y_hard.sum()),
        "hard_negativos": int(len(y_hard) - y_hard.sum()),
        "hard_acertos_base": int((y_hard == y_pred[idx_hard]).sum()),
        "hard_prob_media": float(probas[idx_hard].mean()),
        "sem_accuracy": sem["accuracy"],
        "sem_sensitivity": sem["sensitivity"],
        "sem_specificity": sem["specificity"],
        "sem_f1": sem["f1"],
        "sem_kappa": sem["kappa"],
        "com_accuracy": com["accuracy"],
        "com_sensitivity": com["sensitivity"],
        "com_specificity": com["specificity"],
        "com_f1": com["f1"],
        "com_kappa": com["kappa"],
    }


def rodar_monte_carlo_sweep(target="GAD"):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    df_baseline, _ = preparar_cenario(target, [])
    baseline_index = df_baseline.index
    ids = [
        "baseline",
        "sem_4_fracas_redundantes",
        "sem_poverty_status",
        "sem_number_siblings",
        "sem_family_history_psych",
        "sem_bio_parents",
        "sem_features_sensiveis",
        "sem_cd",
        "sem_odd",
        "sem_tantrums",
        "sem_irritable_mood",
        "sem_social_phobia",
        "sem_sensory",
        "sem_sleep",
        "sem_age",
        "sem_4_mais_cd",
        "sem_4_mais_tantrums",
        "sem_4_cd_tantrums",
    ]
    scenarios = [s for s in SCENARIOS if s["id"] in ids]
    linhas = []
    for scenario in scenarios:
        print(f"[Monte Carlo v1] {scenario['id']} | amostras_validas")
        linhas.append(rodar_monte_carlo_v1_para_cenario(target, scenario))
        if scenario["id"] != "baseline":
            print(f"[Monte Carlo v1] {scenario['id']} | mesmas_linhas_baseline")
            linhas.append(
                rodar_monte_carlo_v1_para_cenario(
                    target,
                    scenario,
                    baseline_index=baseline_index,
                    modo="mesmas_linhas_baseline",
                )
            )

    df = pd.DataFrame(linhas)
    df.to_csv(OUTPUT_PATH / "monte_carlo_v1_sweep_resultados.csv", index=False)
    salvar_resumo_monte_carlo(df)
    return df


def salvar_resumo_monte_carlo(df):
    baseline = df[df["scenario"].eq("baseline")].iloc[0]
    df_rank = df.copy()
    for metrica in ["accuracy", "sensitivity", "specificity", "f1", "kappa"]:
        df_rank[f"delta_sem_{metrica}"] = df_rank[f"sem_{metrica}"] - baseline[f"sem_{metrica}"]
        df_rank[f"delta_com_{metrica}"] = df_rank[f"com_{metrica}"] - baseline[f"com_{metrica}"]

    df_rank = df_rank.sort_values(["delta_sem_kappa", "delta_sem_f1"], ascending=False)
    df_rank.to_csv(OUTPUT_PATH / "monte_carlo_v1_ranking.csv", index=False)

    df_controlado = df_rank[df_rank["modo"].eq("mesmas_linhas_baseline")].copy()
    df_controlado.to_csv(OUTPUT_PATH / "monte_carlo_v1_ranking_mesmas_linhas.csv", index=False)

    with (OUTPUT_PATH / "monte_carlo_v1_resumo.md").open("w", encoding="utf-8") as f:
        f.write("# Monte Carlo v1 por cenario de remocao\n\n")
        f.write("Cada cenario refaz split 80/20, seleciona 20 hard samples e roda 200 simulacoes.\n\n")
        f.write("## Ranking com amostras validas por cenario\n\n")
        f.write("| Cenario | Drops | Amostras | Hard + | Acc sem | Sens sem | F1 sem | Kappa sem | Delta Kappa | Racional |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for _, row in df_rank[df_rank["modo"].eq("amostras_validas")].iterrows():
            f.write(
                f"| {row['scenario']} | {int(row['n_drops'])} | "
                f"{int(row['n_amostras'])} | "
                f"{int(row['hard_positivos'])} | {row['sem_accuracy']:.2f} | "
                f"{row['sem_sensitivity']:.2f} | {row['sem_f1']:.2f} | "
                f"{row['sem_kappa']:.4f} | {row['delta_sem_kappa']:+.4f} | "
                f"{row['rationale']} |\n"
            )

        f.write("\n## Ranking controlado pelas mesmas linhas do baseline\n\n")
        f.write("| Cenario | Drops | Amostras | Hard + | Acc sem | Sens sem | F1 sem | Kappa sem | Delta Kappa | Racional |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for _, row in df_controlado.iterrows():
            f.write(
                f"| {row['scenario']} | {int(row['n_drops'])} | "
                f"{int(row['n_amostras'])} | "
                f"{int(row['hard_positivos'])} | {row['sem_accuracy']:.2f} | "
                f"{row['sem_sensitivity']:.2f} | {row['sem_f1']:.2f} | "
                f"{row['sem_kappa']:.4f} | {row['delta_sem_kappa']:+.4f} | "
                f"{row['rationale']} |\n"
            )


def main():
    rodar_cv_sweep("GAD")
    rodar_monte_carlo_sweep("GAD")
    print(f"Resultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
