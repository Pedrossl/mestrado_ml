"""
Maximizar Sensibilidade e F1-Score para GAD — Análise Abrangente
=================================================================

Objetivo: testar os métodos mais validados da literatura para melhorar
sensibilidade (recall da classe positiva) e F1-score em classificação
binária com dados severamente desbalanceados (85%/15%), contexto clínico.

Cada método é rastreável a uma referência bibliográfica.
Todos usam 10-fold Stratified CV (sem data leakage).
Output: output/sensibilidade_gad/

Referência geral (survey canônico):
  He & Garcia (2009). Learning from imbalanced data.
  IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.

Autor: Dissertação de Mestrado — Fevereiro 2026
"""

import numpy as np
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold, GridSearchCV, TunedThresholdClassifierCV
from sklearn.metrics import (
    confusion_matrix, roc_curve, fbeta_score,
    precision_recall_curve, make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier, DMatrix
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    BalancedBaggingClassifier,
    RUSBoostClassifier,
)
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline


import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    preparar_dados, calcular_ic, calcular_metricas_fold,
    agregar_metricas_com_ic,
)

warnings.filterwarnings('ignore')

# ─── Constantes ────────────────────────────────────────────────────────────────

TARGET = 'GAD'
OUTPUT_DIR = 'output/sensibilidade_gad'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'
FOLDS_DIR = f'{OUTPUT_DIR}/resultados_individuais'
N_FOLDS = 10
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FOLDS_DIR, exist_ok=True)

# ─── Funções auxiliares ────────────────────────────────────────────────────────

def _base_xgb(scale_pos_weight=None):
    params = dict(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, use_label_encoder=False,
        eval_metric='logloss', verbosity=0,
    )
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight
    return XGBClassifier(**params)


def _encontrar_threshold_youdenj(y_true, y_prob):
    """Threshold que maximiza Sensitivity + Specificity - 1 (Youden's J)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx])


def _encontrar_threshold_fbeta(y_true, y_prob, beta=2.0):
    """Threshold que maximiza F-beta (beta>1 prioriza recall/sensitivity)."""
    melhor_t, melhor_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if score > melhor_f:
            melhor_f, melhor_t = score, t
    return float(melhor_t)


def _encontrar_threshold_pr_curve(y_true, y_prob):
    """Threshold no ponto de máximo F1 na curva Precision-Recall."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    if len(thresholds) == 0:
        return 0.5
    idx = np.argmax(f1[:-1])  # thresholds tem len - 1
    return float(thresholds[idx])


def _rodar_cv_com_sampler(sampler, scale_pos_weight=None, threshold_fn=None):
    """
    Executa 10-fold CV com um sampler de oversampling/hybrid.

    threshold_fn: se fornecida, é uma função (y_true, y_prob) -> threshold
                  calculada nos dados de TREINO de cada fold (sem leakage).
    """
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    dist = df[target_name].value_counts()
    spw = scale_pos_weight if scale_pos_weight is not None else dist[0] / dist[1] if scale_pos_weight == 'auto' else None

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = []
    thresholds_usados = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Aplicar resampling apenas no treino
        if sampler is not None:
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

        model = _base_xgb(scale_pos_weight=spw)
        model.fit(X_tr, y_tr)

        if threshold_fn is not None:
            y_prob_tr = model.predict_proba(X_tr)[:, 1]
            t = threshold_fn(y_tr, y_prob_tr)
            thresholds_usados.append(t)
            y_pred = (model.predict_proba(X_te)[:, 1] >= t).astype(int)
        else:
            y_pred = model.predict(X_te)

        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas), thresholds_usados


def _rodar_cv_ensemble(EstimatorClass, **kwargs):
    """Executa 10-fold CV com um estimador ensemble do imblearn."""
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = EstimatorClass(random_state=RANDOM_STATE, **kwargs)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas)


def _salvar_resultado_individual(nome, codigo, referencia, descricao, metricas, thresholds=None):
    """Salva resultado individual de um método em arquivo texto."""
    fname = f'{FOLDS_DIR}/{codigo}.txt'
    m = metricas
    with open(fname, 'w') as f:
        f.write(f"MÉTODO: {nome}\n")
        f.write(f"Código: {codigo}\n")
        f.write(f"Referência: {referencia}\n")
        f.write(f"Descrição: {descricao}\n")
        f.write("=" * 60 + "\n\n")
        f.write("MÉTRICAS (média ± IC 95% — 10-fold Stratified CV)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:     {m['accuracy']:.2f}% ± {m['accuracy_ic']:.2f}%\n")
        f.write(f"Sensitivity:  {m['sensitivity']:.2f}% ± {m['sensitivity_ic']:.2f}%\n")
        f.write(f"Specificity:  {m['specificity']:.2f}% ± {m['specificity_ic']:.2f}%\n")
        f.write(f"PPV:          {m['ppv']:.2f}% ± {m['ppv_ic']:.2f}%\n")
        f.write(f"NPV:          {m['npv']:.2f}% ± {m['npv_ic']:.2f}%\n")
        f.write(f"F1-Score:     {m['f1']:.2f}% ± {m['f1_ic']:.2f}%\n")
        f.write(f"Kappa:        {m['kappa']:.4f} ± {m['kappa_ic']:.4f}\n\n")
        f.write("MATRIZ DE CONFUSÃO AGREGADA\n")
        f.write("-" * 60 + "\n")
        f.write(f"  VN={m['vn']:>4}  FP={m['fp']:>4}\n")
        f.write(f"  FN={m['fn']:>4}  VP={m['vp']:>4}\n")
        if thresholds:
            f.write(f"\nTHRESHOLD (determinado no treino de cada fold)\n")
            f.write(f"  Média: {np.mean(thresholds):.3f} | Mín: {min(thresholds):.3f} | Máx: {max(thresholds):.3f}\n")
            f.write(f"  Por fold: {[f'{t:.2f}' for t in thresholds]}\n")


def _print_metodo(nome, metricas, thresholds=None):
    m = metricas
    sens_str = f"{m['sensitivity']:.1f}% ± {m['sensitivity_ic']:.1f}%"
    f1_str = f"{m['f1']:.1f}% ± {m['f1_ic']:.1f}%"
    spec_str = f"{m['specificity']:.1f}%"
    kappa_str = f"{m['kappa']:.3f}"
    t_str = f" | threshold≈{np.mean(thresholds):.2f}" if thresholds else ""
    print(f"  {nome:<40}  Sens={sens_str:<18}  F1={f1_str:<18}  Spec={spec_str:<8}  Kappa={kappa_str}{t_str}")


# ─── DEFINIÇÃO DE CADA MÉTODO ──────────────────────────────────────────────────

METODOS = OrderedDict()
"""
Cada entrada: {
    'nome': str,
    'referencia': str,
    'descricao': str,
    'fn': callable -> (metricas_dict, thresholds_list_or_None)
}
"""

# ── Categoria 1: Oversampling Padrão (baseline) ────────────────────────────────

def _m01_smote():
    m, t = _rodar_cv_com_sampler(SMOTE(random_state=RANDOM_STATE))
    return m, None

METODOS['M01_SMOTE'] = {
    'nome': 'SMOTE (baseline)',
    'referencia': 'Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR 16, 321-357.',
    'descricao': 'Gera amostras sintéticas da classe minoritária interpolando linearmente entre um ponto real e k vizinhos mais próximos.',
    'fn': _m01_smote,
}

# ── Categoria 2: Oversampling Adaptativo ──────────────────────────────────────

def _m02_adasyn():
    m, t = _rodar_cv_com_sampler(ADASYN(random_state=RANDOM_STATE))
    return m, None

METODOS['M02_ADASYN'] = {
    'nome': 'ADASYN',
    'referencia': 'He et al. (2008). ADASYN: Adaptive Synthetic Sampling. IEEE IJCNN, 1322-1328.',
    'descricao': 'Gera mais sintéticos onde o classificador erra mais (regiões de fronteira de alta densidade negativa), adaptando a distribuição.',
    'fn': _m02_adasyn,
}

def _m03_borderline_smote():
    m, t = _rodar_cv_com_sampler(BorderlineSMOTE(random_state=RANDOM_STATE, kind='borderline-1'))
    return m, None

METODOS['M03_BorderlineSMOTE'] = {
    'nome': 'BorderlineSMOTE',
    'referencia': 'Han et al. (2005). Borderline-SMOTE: A new over-sampling method. Advances in Intelligent Computing, 878-887.',
    'descricao': 'Aplica SMOTE apenas nas amostras minoritárias "perigosas" (na fronteira), ignorando amostras seguras, gerando sintéticos mais informativos.',
    'fn': _m03_borderline_smote,
}

# ── Categoria 3: Híbrido Over + Undersampling ─────────────────────────────────

def _m04_smoteenn():
    m, t = _rodar_cv_com_sampler(SMOTEENN(random_state=RANDOM_STATE))
    return m, None

METODOS['M04_SMOTEENN'] = {
    'nome': 'SMOTEENN',
    'referencia': 'Batista et al. (2004). A study of several methods for balancing ML training data. ACM SIGKDD Explorations 6(1), 20-29.',
    'descricao': 'SMOTE para oversample + ENN (Edited Nearest Neighbours) para limpar amostras ambíguas de ambas as classes, resultando em fronteira mais limpa.',
    'fn': _m04_smoteenn,
}

def _m05_smotetomek():
    m, t = _rodar_cv_com_sampler(SMOTETomek(random_state=RANDOM_STATE))
    return m, None

METODOS['M05_SMOTETomek'] = {
    'nome': 'SMOTETomek',
    'referencia': 'Batista et al. (2004). Idem acima. SMOTETomek: SMOTE + remoção de Tomek Links.',
    'descricao': 'SMOTE para oversample + remove Tomek Links (pares inter-classe que são vizinhos mais próximos), limpando a fronteira de decisão.',
    'fn': _m05_smotetomek,
}

# ── Categoria 4: Cost-Sensitive (Pesos de Classe) ─────────────────────────────

def _m06_spw_alto():
    df, target_name = preparar_dados(TARGET)
    dist = df[target_name].value_counts()
    spw_natural = dist[0] / dist[1]
    spw = spw_natural * 2  # ~2x o peso natural
    m, t = _rodar_cv_com_sampler(None, scale_pos_weight=spw)
    return m, None

METODOS['M06_SPW_Alto'] = {
    'nome': 'scale_pos_weight 2× natural',
    'referencia': 'Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. ACM SIGKDD, 785-794.',
    'descricao': 'scale_pos_weight = 2× a razão natural neg/pos (~11x), penalizando ainda mais erros na classe positiva durante o boosting.',
    'fn': _m06_spw_alto,
}

def _m07_spw_extremo():
    m, t = _rodar_cv_com_sampler(None, scale_pos_weight=20.0)
    return m, None

METODOS['M07_SPW_Extremo'] = {
    'nome': 'scale_pos_weight=20 (extremo)',
    'referencia': 'Elkan (2001). The foundations of cost-sensitive learning. IJCAI 17(1), 973-978.',
    'descricao': 'Penaliza falsos negativos 20x mais que falsos positivos na função de perda do XGBoost, forçando atenção máxima à classe positiva.',
    'fn': _m07_spw_extremo,
}

# ── Categoria 5: Threshold Optimization ──────────────────────────────────────

def _m08_threshold_youdenj():
    m, t = _rodar_cv_com_sampler(
        SMOTE(random_state=RANDOM_STATE),
        threshold_fn=_encontrar_threshold_youdenj,
    )
    return m, t

METODOS['M08_Threshold_YoudenJ'] = {
    'nome': 'SMOTE + Threshold Youden\'s J',
    'referencia': 'Youden (1950). Index for rating diagnostic tests. Cancer 3(1), 32-35. Survey: Fluss et al. (2005). Biometrical Journal 47(4), 458-472.',
    'descricao': 'Threshold t* que maximiza Sensitivity + Specificity - 1 (ponto mais alto da curva ROC acima da diagonal), determinado nos dados de treino de cada fold.',
    'fn': _m08_threshold_youdenj,
}

def _m09_threshold_f2():
    m, t = _rodar_cv_com_sampler(
        SMOTE(random_state=RANDOM_STATE),
        threshold_fn=lambda y_true, y_prob: _encontrar_threshold_fbeta(y_true, y_prob, beta=2.0),
    )
    return m, t

METODOS['M09_Threshold_F2'] = {
    'nome': 'SMOTE + Threshold F2-score (β=2)',
    'referencia': 'Van Rijsbergen (1979). Information Retrieval. Sokolova & Lapalme (2009). Information Processing & Management 45(4), 427-437.',
    'descricao': 'Threshold que maximiza F-beta com β=2: penaliza falsos negativos 4× mais que falsos positivos, priorizando sensibilidade sobre precisão.',
    'fn': _m09_threshold_f2,
}

def _m10_threshold_pr_curve():
    m, t = _rodar_cv_com_sampler(
        SMOTE(random_state=RANDOM_STATE),
        threshold_fn=_encontrar_threshold_pr_curve,
    )
    return m, t

METODOS['M10_Threshold_PRCurve'] = {
    'nome': 'SMOTE + Threshold PR-Curve (max F1)',
    'referencia': 'Davis & Goadrich (2006). The relationship between Precision-Recall and ROC curves. ICML, 233-240.',
    'descricao': 'Threshold no ponto de máximo F1 da curva Precision-Recall (mais informativa que ROC em datasets desbalanceados por ignorar os TN abundantes).',
    'fn': _m10_threshold_pr_curve,
}

# ── Categoria 6: Ensemble para Dados Desbalanceados ──────────────────────────

def _m11_balanced_rf():
    m = _rodar_cv_ensemble(
        BalancedRandomForestClassifier,
        n_estimators=100,
        sampling_strategy='auto',
    )
    return m, None

METODOS['M11_BalancedRF'] = {
    'nome': 'Balanced Random Forest',
    'referencia': 'Chen, Liaw & Breiman (2004). Using Random Forest to Learn Imbalanced Data. UCB Technical Report 666.',
    'descricao': 'Modifica o Random Forest para realizar undersampling balanceado em cada árvore individualmente, preservando diversidade sem descartar dados globalmente.',
    'fn': _m11_balanced_rf,
}

def _m12_easy_ensemble():
    m = _rodar_cv_ensemble(
        EasyEnsembleClassifier,
        n_estimators=20,
    )
    return m, None

METODOS['M12_EasyEnsemble'] = {
    'nome': 'EasyEnsemble',
    'referencia': 'Liu, Wu & Zhou (2009). Exploratory undersampling for class-imbalance learning. IEEE TSMC 39(2), 539-550.',
    'descricao': 'Múltiplos AdaBoosts treinados em subconjuntos balanceados via undersampling aleatório, agregados por votação — reduz variância do undersampling simples.',
    'fn': _m12_easy_ensemble,
}

def _m13_rusboost():
    m = _rodar_cv_ensemble(
        RUSBoostClassifier,
        n_estimators=100,
        sampling_strategy='auto',
    )
    return m, None

METODOS['M13_RUSBoost'] = {
    'nome': 'RUSBoost',
    'referencia': 'Seiffert et al. (2010). RUSBoost: A Hybrid Approach to Alleviating Class Imbalance. IEEE TSMC 40(1), 185-197.',
    'descricao': 'Undersampling aleatório integrado diretamente no loop de boosting do AdaBoost: em cada iteração, re-balanceia antes de treinar o classificador fraco.',
    'fn': _m13_rusboost,
}

def _m14_balanced_bagging():
    m = _rodar_cv_ensemble(
        BalancedBaggingClassifier,
        estimator=None,  # usa DecisionTreeClassifier por padrão
        n_estimators=100,
        sampling_strategy='auto',
    )
    return m, None

METODOS['M14_BalancedBagging'] = {
    'nome': 'Balanced Bagging',
    'referencia': 'Wallace et al. (2011). Class imbalance, redux. IEEE ICDM. Implementação: imblearn docs.',
    'descricao': 'Bagging clássico com resampling balanceado em cada bootstrap, permitindo que qualquer classificador base se beneficie do ensemble com dados balanceados.',
    'fn': _m14_balanced_bagging,
}

# ── Categoria 7: Combinações Potentes ─────────────────────────────────────────

def _m15_smoteenn_spw_f2():
    """Combinação: SMOTEENN + scale_pos_weight alto + threshold F2."""
    df, target_name = preparar_dados(TARGET)
    dist = df[target_name].value_counts()
    spw = dist[0] / dist[1]  # peso natural

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = []
    thresholds = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        sampler = SMOTEENN(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)

        model = _base_xgb(scale_pos_weight=spw)
        model.fit(X_tr_res, y_tr_res)

        y_prob_tr = model.predict_proba(X_tr_res)[:, 1]
        t = _encontrar_threshold_fbeta(y_tr_res, y_prob_tr, beta=2.0)
        thresholds.append(t)

        y_pred = (model.predict_proba(X_te)[:, 1] >= t).astype(int)
        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas), thresholds

METODOS['M15_SMOTEENN_SPW_F2'] = {
    'nome': 'SMOTEENN + scale_pos_weight + Threshold F2 (combo)',
    'referencia': 'Combinação de Batista 2004 (SMOTEENN) + Chen 2016 (XGBoost SPW) + Van Rijsbergen 1979 (F-beta).',
    'descricao': 'Combina as três abordagens: resampling híbrido (SMOTEENN), custo assimétrico no treino (SPW), e limiar de decisão otimizado para sensibilidade (F2).',
    'fn': _m15_smoteenn_spw_f2,
}

# ── Categoria 8: Focal Loss ────────────────────────────────────────────────────

def _focal_loss_obj(predt, dtrain, gamma=2.0, alpha=0.25):
    """
    Objetivo de Focal Loss para XGBoost.

    Referência: Lin et al. (2017). Focal loss for dense object detection.
    IEEE ICCV, 2980-2988.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    onde p_t = p se y=1, (1-p) se y=0; alpha_t = alpha se y=1, (1-alpha) se y=0.

    Gradiente: aproximação prática (ignora d/dp[(1-p_t)^gamma] por estabilidade
    numérica — simplificação universalmente usada em implementações de produção).
    """
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-predt))
    p_t = np.where(y == 1, p, 1 - p)
    alpha_t = np.where(y == 1, alpha, 1 - alpha)
    fl_weight = alpha_t * (1 - p_t) ** gamma
    grad = fl_weight * (p - y)
    hess = fl_weight * p * (1 - p)
    return grad, hess


def _m16_focal_loss():
    """XGBoost com Focal Loss como objective customizado."""
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        dtrain = DMatrix(X_tr_res, label=y_tr_res)
        dtest = DMatrix(X_te, label=y_te)

        params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'seed': RANDOM_STATE,
            'verbosity': 0,
        }
        booster = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', verbosity=0,
        )
        # Treinar com focal loss via objective customizado
        import xgboost as xgb
        # alpha=0.75: 3× mais peso para a classe positiva (minoritária) vs negativa
        booster_raw = xgb.train(
            {'max_depth': 5, 'learning_rate': 0.1, 'seed': RANDOM_STATE,
             'verbosity': 0, 'nthread': -1},
            dtrain,
            num_boost_round=100,
            obj=lambda predt, dtrain: _focal_loss_obj(predt, dtrain, gamma=2.0, alpha=0.75),
            verbose_eval=False,
        )
        y_prob = booster_raw.predict(dtest)
        y_prob = 1.0 / (1.0 + np.exp(-y_prob))  # sigmoid (saída é logit)
        y_pred = (y_prob >= 0.5).astype(int)
        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas), None


METODOS['M16_FocalLoss'] = {
    'nome': 'Focal Loss (SMOTE + XGBoost)',
    'referencia': 'Lin, Goyal, Girshick, He & Dollar (2017). Focal Loss for Dense Object Detection. IEEE ICCV, 2980-2988. [>20.000 citações]',
    'descricao': 'Modifica a cross-entropy com fator (1-p)^γ que reduz a contribuição de exemplos fáceis (bem classificados), focando o treino nos casos difíceis da classe positiva.',
    'fn': _m16_focal_loss,
}

# ── Categoria 9: TunedThresholdClassifierCV (sklearn ≥ 1.5) ───────────────────

def _m17_tuned_threshold_cv():
    """
    TunedThresholdClassifierCV: wrapper sklearn que otimiza automaticamente
    o threshold via cross-validation interna, sem tocar no modelo treinado.

    Referência: sklearn 1.5 docs + Largeron et al. (2023).
    Threshold tuning for imbalanced classification.
    """
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = []
    thresholds = []

    scorer_f2 = make_scorer(fbeta_score, beta=2, zero_division=0)

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # SMOTE no treino
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        base_clf = _base_xgb()
        tuned_clf = TunedThresholdClassifierCV(
            estimator=base_clf,
            scoring=scorer_f2,
            cv=5,
            random_state=RANDOM_STATE,
        )
        tuned_clf.fit(X_tr_res, y_tr_res)
        thresholds.append(float(tuned_clf.best_threshold_))

        y_pred = tuned_clf.predict(X_te)
        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas), thresholds


METODOS['M17_TunedThresholdCV'] = {
    'nome': 'TunedThresholdClassifierCV (F2, sklearn 1.5)',
    'referencia': 'scikit-learn >= 1.5 (2024). TunedThresholdClassifierCV. Documentação oficial sklearn. Baseado em Provost & Fawcett (1997). Analysis and Visualization of Classifier Performance.',
    'descricao': 'Wrapper sklearn que re-calibra automaticamente o threshold de decisão via cross-validation interna (5-fold), otimizando F2-score sem alterar o modelo base.',
    'fn': _m17_tuned_threshold_cv,
}

# ── Categoria 10: GridSearch com scoring=recall ────────────────────────────────

def _m18_gridsearch_recall():
    """
    GridSearch com nested CV otimizando diretamente para recall (sensibilidade).

    Inner CV (5-fold): seleciona hiperparâmetros que maximizam recall
    Outer CV (10-fold): avalia o modelo no conjunto de teste

    Referência: Bergstra & Bengio (2012). Random Search for Hyper-Parameter
    Optimization. JMLR 13, 281-305.
    """
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    dist = df[target_name].value_counts()

    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metricas = []

    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__scale_pos_weight': [
            dist[0] / dist[1],          # natural
            (dist[0] / dist[1]) * 2,    # 2x
            20.0,                        # extremo
        ],
    }

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipeline = ImbPipeline([
            ('sampler', SMOTE(random_state=RANDOM_STATE)),
            ('clf', XGBClassifier(
                learning_rate=0.1, random_state=RANDOM_STATE,
                use_label_encoder=False, eval_metric='logloss', verbosity=0,
            )),
        ])

        gs = GridSearchCV(
            pipeline, param_grid,
            cv=inner_cv,
            scoring='recall',
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)
        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas), None


METODOS['M18_GridSearch_Recall'] = {
    'nome': 'GridSearch Nested CV (scoring=recall)',
    'referencia': 'Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization. JMLR 13, 281-305. + Nested CV: Varma & Simon (2006). Bias in error estimation. BMC Bioinformatics 7, 91.',
    'descricao': 'GridSearch com nested CV (outer 10-fold, inner 5-fold) que otimiza hiperparâmetros do XGBoost+SMOTE diretamente para maximizar recall (sensibilidade).',
    'fn': _m18_gridsearch_recall,
}

# ── Categoria 11: Estratégias Avançadas (Calibração, HistGBM, Stacking) ────────

def _m19_calibrated_xgb_pr():
    """M19: SMOTE + Calibrated XGBoost + PR/F2 Threshold."""
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    dist = df[target_name].value_counts()
    spw = dist[0] / dist[1]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = []
    thresholds = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        base_model = _base_xgb(scale_pos_weight=spw)
        # Wrap com CalibratedClassifierCV isotônico (melhor para non-sigmoid distorções)
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_model.fit(X_tr_res, y_tr_res)

        y_prob_tr = calibrated_model.predict_proba(X_tr_res)[:, 1]
        t = _encontrar_threshold_fbeta(y_tr_res, y_prob_tr, beta=2.0)
        thresholds.append(t)

        y_pred = (calibrated_model.predict_proba(X_te)[:, 1] >= t).astype(int)
        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas), thresholds

METODOS['M19_Calibrated_XGB'] = {
    'nome': 'Calibrated XGBoost + Threshold F2',
    'referencia': 'Zadrozny & Elkan (2002). Transforming classifier scores into accurate multiclass probability estimates. KDD. E Platt (1999).',
    'descricao': 'Calibra as probabilidades geradas pelo XGBoost (distorcidas pelo SMOTE) usando regressão isotônica antes de otimizar o limiar de decisão F2.',
    'fn': _m19_calibrated_xgb_pr,
}

def _m20_histgbm_f2():
    """M20: SMOTE + HistGradientBoosting + Threshold F2."""
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = []
    thresholds = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        model = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=100,
            max_depth=5, random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        model.fit(X_tr_res, y_tr_res)

        y_prob_tr = model.predict_proba(X_tr_res)[:, 1]
        t = _encontrar_threshold_fbeta(y_tr_res, y_prob_tr, beta=2.0)
        thresholds.append(t)

        y_pred = (model.predict_proba(X_te)[:, 1] >= t).astype(int)
        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas), thresholds

METODOS['M20_HistGBM'] = {
    'nome': 'HistGradientBoosting + Threshold F2',
    'referencia': 'Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS (Inspiração do Sklearn HistGBM).',
    'descricao': 'Usa HistGradientBoosting (equivalente nativo do Sklearn ao LightGBM) que agrupa features em bins, suportando class_weight e otimizando F2.',
    'fn': _m20_histgbm_f2,
}

def _m21_stacking_ensemble():
    """M21: Stacking (KNN, AdaBoost, XGB) + Logistic Reg + Threshold F2."""
    df, target_name = preparar_dados(TARGET)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    dist = df[target_name].value_counts()
    spw = dist[0] / dist[1]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = []
    thresholds = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        estimators = [
            ('knn', KNeighborsClassifier(n_neighbors=3)),
            ('ada', AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE)),
            ('xgb', _base_xgb(scale_pos_weight=spw))
        ]
        meta_model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)

        stacking = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=3)
        stacking.fit(X_tr_res, y_tr_res)

        y_prob_tr = stacking.predict_proba(X_tr_res)[:, 1]
        t = _encontrar_threshold_fbeta(y_tr_res, y_prob_tr, beta=2.0)
        thresholds.append(t)

        y_pred = (stacking.predict_proba(X_te)[:, 1] >= t).astype(int)
        metricas.append(calcular_metricas_fold(y_te, y_pred))

    return agregar_metricas_com_ic(metricas), thresholds

METODOS['M21_Stacking'] = {
    'nome': 'Stacking (KNN+Ada+XGB) + MetaLR + F2',
    'referencia': 'Wolpert (1992). Stacked generalization. Neural networks, 5(2), 241-259.',
    'descricao': 'Stacking heterogêneo combinando modelos base (KNN, AdaBoost, XGBoost) e um meta-modelo (Regressão Logística balanceada) para maximizar recall inter-modelos.',
    'fn': _m21_stacking_ensemble,
}



# ─── EXECUÇÃO PRINCIPAL ────────────────────────────────────────────────────────

def executar_todos_os_metodos():
    print("\n" + "=" * 80)
    print("   MAXIMIZAR SENSIBILIDADE E F1 — XGBoost para GAD")
    print("   21 métodos validados pela literatura | 10-fold Stratified CV")
    print("=" * 80)

    # Informações do dataset
    df, target_name = preparar_dados(TARGET)
    dist = df[target_name].value_counts()
    print(f"\n  Dataset: {df.shape[0]} amostras | {df.shape[1]-1} features | Target: {target_name}")
    print(f"  Classe 0 (sem GAD): {dist[0]} ({dist[0]/len(df)*100:.1f}%)")
    print(f"  Classe 1 (com GAD): {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"  Razão neg/pos: {dist[0]/dist[1]:.1f}x\n")

    categorias = {
        'Categoria 1 — Oversampling padrão': ['M01_SMOTE'],
        'Categoria 2 — Oversampling adaptativo': ['M02_ADASYN', 'M03_BorderlineSMOTE'],
        'Categoria 3 — Híbrido Over+Under': ['M04_SMOTEENN', 'M05_SMOTETomek'],
        'Categoria 4 — Cost-Sensitive': ['M06_SPW_Alto', 'M07_SPW_Extremo'],
        'Categoria 5 — Threshold Optimization': ['M08_Threshold_YoudenJ', 'M09_Threshold_F2', 'M10_Threshold_PRCurve'],
        'Categoria 6 — Ensemble Imbalanced': ['M11_BalancedRF', 'M12_EasyEnsemble', 'M13_RUSBoost', 'M14_BalancedBagging'],
        'Categoria 7 — Combinação': ['M15_SMOTEENN_SPW_F2'],
        'Categoria 8 — Focal Loss': ['M16_FocalLoss'],
        'Categoria 9 — TunedThresholdClassifierCV': ['M17_TunedThresholdCV'],
        'Categoria 10 — GridSearch com scoring=recall': ['M18_GridSearch_Recall'],
        'Categoria 11 — Estratégias Avançadas (Calibração, HistGBM, Stacking)': ['M19_Calibrated_XGB', 'M20_HistGBM', 'M21_Stacking'],
    }

    resultados_completos = {}

    for cat_nome, codigos in categorias.items():
        print(f"\n  {'─' * 70}")
        print(f"  {cat_nome}")
        print(f"  {'─' * 70}")

        for codigo in codigos:
            metodo = METODOS[codigo]
            print(f"  [{codigo}] Executando...", end=" ", flush=True)
            try:
                m, t = metodo['fn']()
                print("OK")
                _print_metodo(metodo['nome'], m, t)
                _salvar_resultado_individual(
                    metodo['nome'], codigo,
                    metodo['referencia'], metodo['descricao'],
                    m, t,
                )
                resultados_completos[codigo] = {
                    'nome': metodo['nome'],
                    'metricas': m,
                    'thresholds': t,
                    'referencia': metodo['referencia'],
                    'descricao': metodo['descricao'],
                }
            except Exception as e:
                print(f"ERRO: {e}")
                import traceback; traceback.print_exc()

    return resultados_completos


# ─── RELATÓRIO COMPARATIVO ─────────────────────────────────────────────────────

def gerar_relatorio(resultados):
    """Gera arquivo comparativo ordenado por sensibilidade."""

    # Ordenar por sensibilidade (decrescente)
    ordenados = sorted(
        resultados.items(),
        key=lambda x: x[1]['metricas']['sensitivity'],
        reverse=True,
    )

    relatorio_path = f'{OUTPUT_DIR}/comparativo_sensibilidade_gad.txt'
    with open(relatorio_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("  COMPARATIVO DE MÉTODOS PARA MAXIMIZAR SENSIBILIDADE — XGBoost para GAD\n")
        f.write("  Dissertação de Mestrado | Fevereiro 2026\n")
        f.write("  10-fold Stratified CV | Ordenado por Sensibilidade (decrescente)\n")
        f.write("=" * 100 + "\n\n")

        # Tabela principal
        header = f"{'Rank':<5} {'Código':<28} {'Sensitivity':>16} {'F1-Score':>16} {'Specificity':>14} {'PPV':>10} {'Kappa':>10}"
        f.write(header + "\n")
        f.write("-" * 100 + "\n")

        for rank, (codigo, dados) in enumerate(ordenados, 1):
            m = dados['metricas']
            t_str = ""
            if dados['thresholds']:
                t_str = f" [t≈{np.mean(dados['thresholds']):.2f}]"
            sens = f"{m['sensitivity']:.1f}±{m['sensitivity_ic']:.1f}%"
            f1   = f"{m['f1']:.1f}±{m['f1_ic']:.1f}%"
            spec = f"{m['specificity']:.1f}%"
            ppv  = f"{m['ppv']:.1f}%"
            kappa= f"{m['kappa']:.3f}"
            f.write(f"{rank:<5} {codigo+t_str:<28} {sens:>16} {f1:>16} {spec:>14} {ppv:>10} {kappa:>10}\n")

        f.write("-" * 100 + "\n\n")

        # Seção detalhada por método
        f.write("=" * 100 + "\n")
        f.write("  DETALHAMENTO POR MÉTODO\n")
        f.write("=" * 100 + "\n\n")

        for rank, (codigo, dados) in enumerate(ordenados, 1):
            m = dados['metricas']
            f.write(f"[{rank}] {codigo}: {dados['nome']}\n")
            f.write(f"     Ref: {dados['referencia']}\n")
            f.write(f"     O que faz: {dados['descricao']}\n")
            f.write(f"     Sensitivity:  {m['sensitivity']:.2f}% ± {m['sensitivity_ic']:.2f}%\n")
            f.write(f"     F1-Score:     {m['f1']:.2f}% ± {m['f1_ic']:.2f}%\n")
            f.write(f"     Specificity:  {m['specificity']:.2f}% ± {m['specificity_ic']:.2f}%\n")
            f.write(f"     PPV:          {m['ppv']:.2f}% ± {m['ppv_ic']:.2f}%\n")
            f.write(f"     Kappa:        {m['kappa']:.4f} ± {m['kappa_ic']:.4f}\n")
            f.write(f"     Matriz:  VN={m['vn']}  FP={m['fp']}  FN={m['fn']}  VP={m['vp']}\n")
            if dados['thresholds']:
                f.write(f"     Threshold: média={np.mean(dados['thresholds']):.3f} (min={min(dados['thresholds']):.3f}, max={max(dados['thresholds']):.3f})\n")
            f.write("\n")

        # Análise de trade-off
        f.write("=" * 100 + "\n")
        f.write("  ANÁLISE DE TRADE-OFF: SENSIBILIDADE × ESPECIFICIDADE × F1\n")
        f.write("=" * 100 + "\n\n")
        f.write("  Top 5 por Sensibilidade:\n")
        for rank, (codigo, dados) in enumerate(ordenados[:5], 1):
            m = dados['metricas']
            f.write(f"    {rank}. {dados['nome']:<40}  Sens={m['sensitivity']:.1f}%  Spec={m['specificity']:.1f}%  F1={m['f1']:.1f}%\n")

        ordenados_f1 = sorted(resultados.items(), key=lambda x: x[1]['metricas']['f1'], reverse=True)
        f.write("\n  Top 5 por F1-Score:\n")
        for rank, (codigo, dados) in enumerate(ordenados_f1[:5], 1):
            m = dados['metricas']
            f.write(f"    {rank}. {dados['nome']:<40}  F1={m['f1']:.1f}%  Sens={m['sensitivity']:.1f}%  Spec={m['specificity']:.1f}%\n")

        ordenados_kappa = sorted(resultados.items(), key=lambda x: x[1]['metricas']['kappa'], reverse=True)
        f.write("\n  Top 5 por Kappa:\n")
        for rank, (codigo, dados) in enumerate(ordenados_kappa[:5], 1):
            m = dados['metricas']
            f.write(f"    {rank}. {dados['nome']:<40}  Kappa={m['kappa']:.3f}  Sens={m['sensitivity']:.1f}%  F1={m['f1']:.1f}%\n")

    print(f"\n  Relatório salvo em: {relatorio_path}")
    return relatorio_path, ordenados


# ─── GRÁFICOS ─────────────────────────────────────────────────────────────────

def gerar_graficos(resultados, ordenados):
    """Gera gráficos de comparação: sensibilidade, F1, scatter."""

    codigos_labels = [dados['nome'].replace(' (baseline)', '').replace(' (combo)', '') for _, dados in ordenados]
    sensitividades = [dados['metricas']['sensitivity'] for _, dados in ordenados]
    f1_scores      = [dados['metricas']['f1']          for _, dados in ordenados]
    especificidades= [dados['metricas']['specificity'] for _, dados in ordenados]
    sensitivity_ics= [dados['metricas']['sensitivity_ic'] for _, dados in ordenados]
    f1_ics         = [dados['metricas']['f1_ic']          for _, dados in ordenados]

    plt.style.use('seaborn-v0_8-whitegrid')
    cores = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(ordenados)))

    # ── Gráfico 1: Sensibilidade por método ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(7, len(ordenados) * 0.55)))
    y_pos = np.arange(len(ordenados))
    cores_inv = cores[::-1]  # maior sensibilidade = mais verde (no topo)
    barras = ax.barh(y_pos, sensitividades[::-1], xerr=sensitivity_ics[::-1],
                     color=cores, edgecolor='white', linewidth=0.5,
                     capsize=4, error_kw={'elinewidth': 1.5})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(codigos_labels[::-1], fontsize=9)
    ax.set_xlabel('Sensibilidade (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sensibilidade por Método — XGBoost para GAD\n15 métodos da literatura | 10-fold CV (com IC 95%)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Limite mínimo clínico (50%)')
    ax.axvline(x=34.5, color='gray', linestyle=':', alpha=0.5, label='SMOTE baseline (34.5%)')
    ax.legend(fontsize=9)
    for barra, val in zip(barras, sensitividades[::-1]):
        ax.text(barra.get_width() + 1, barra.get_y() + barra.get_height() / 2,
                f'{val:.1f}%', ha='left', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(0, max(sensitividades) + 15)
    plt.tight_layout()
    path1 = f'{PLOTS_DIR}/sensibilidade_por_metodo.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico salvo em: {path1}")

    # ── Gráfico 2: F1-Score por método ──────────────────────────────────────
    ordenados_f1 = sorted(resultados.items(), key=lambda x: x[1]['metricas']['f1'], reverse=True)
    nomes_f1  = [d['nome'].replace(' (baseline)', '').replace(' (combo)', '') for _, d in ordenados_f1]
    vals_f1   = [d['metricas']['f1']    for _, d in ordenados_f1]
    ics_f1    = [d['metricas']['f1_ic'] for _, d in ordenados_f1]
    cores_f1 = plt.cm.RdYlBu(np.linspace(0.15, 0.85, len(ordenados_f1)))

    fig, ax = plt.subplots(figsize=(14, max(7, len(ordenados_f1) * 0.55)))
    y_pos = np.arange(len(ordenados_f1))
    barras = ax.barh(y_pos, vals_f1[::-1], xerr=ics_f1[::-1],
                     color=cores_f1, edgecolor='white', linewidth=0.5,
                     capsize=4, error_kw={'elinewidth': 1.5})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nomes_f1[::-1], fontsize=9)
    ax.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score por Método — XGBoost para GAD\n15 métodos da literatura | 10-fold CV (com IC 95%)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.axvline(x=41.1, color='gray', linestyle=':', alpha=0.5, label='SMOTE baseline (41.1%)')
    ax.legend(fontsize=9)
    for barra, val in zip(barras, vals_f1[::-1]):
        ax.text(barra.get_width() + 0.5, barra.get_y() + barra.get_height() / 2,
                f'{val:.1f}%', ha='left', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(0, max(vals_f1) + 12)
    plt.tight_layout()
    path2 = f'{PLOTS_DIR}/f1_por_metodo.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico salvo em: {path2}")

    # ── Gráfico 3: Scatter Sensibilidade × Especificidade ───────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter_cores = plt.cm.tab20(np.linspace(0, 1, len(resultados)))

    for i, (codigo, dados) in enumerate(resultados.items()):
        m = dados['metricas']
        ax.scatter(m['specificity'], m['sensitivity'],
                   s=120, color=scatter_cores[i], zorder=5,
                   label=f"{codigo}: {dados['nome'][:25]}")
        ax.annotate(codigo.split('_')[0],
                    (m['specificity'], m['sensitivity']),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7, fontweight='bold')

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.4, label='Sens=50% (limiar clínico)')
    ax.axvline(x=50, color='blue', linestyle='--', alpha=0.4, label='Spec=50%')
    ax.set_xlabel('Especificidade (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sensibilidade (%)', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off Sensibilidade × Especificidade\nCada ponto = 1 método (XGBoost, GAD)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(20, 105)
    ax.set_ylim(0, 100)

    # Quadrante ideal (alta Sens, alta Spec)
    ax.axhspan(50, 100, xmin=(50-20)/85, alpha=0.05, color='green', label='Quadrante ideal')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    path3 = f'{PLOTS_DIR}/scatter_sens_spec.png'
    plt.savefig(path3, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico salvo em: {path3}")

    # ── Gráfico 4: Bubble chart Sens × F1 (tamanho = Kappa) ─────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (codigo, dados) in enumerate(resultados.items()):
        m = dados['metricas']
        size = max(50, m['kappa'] * 2000)
        ax.scatter(m['sensitivity'], m['f1'],
                   s=size, color=scatter_cores[i], alpha=0.7, zorder=5)
        ax.annotate(codigo,
                    (m['sensitivity'], m['f1']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7)

    ax.set_xlabel('Sensibilidade (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sensibilidade × F1-Score (tamanho do círculo ∝ Kappa)\nXGBoost para GAD — 15 métodos',
                 fontsize=13, fontweight='bold', pad=15)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path4 = f'{PLOTS_DIR}/bubble_sens_f1.png'
    plt.savefig(path4, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico salvo em: {path4}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    resultados = executar_todos_os_metodos()

    print("\n" + "=" * 80)
    print("  GERANDO RELATÓRIO E GRÁFICOS...")
    print("=" * 80 + "\n")

    relatorio_path, ordenados = gerar_relatorio(resultados)
    gerar_graficos(resultados, ordenados)

    print("\n" + "=" * 80)
    print("  RESUMO FINAL — TOP 5 POR SENSIBILIDADE (18 métodos)")
    print("=" * 80)
    for rank, (codigo, dados) in enumerate(ordenados[:5], 1):
        m = dados['metricas']
        print(f"  {rank}. {dados['nome']:<42}  Sens={m['sensitivity']:.1f}%  F1={m['f1']:.1f}%  Kappa={m['kappa']:.3f}")

    print(f"\n  Todos os resultados em: {OUTPUT_DIR}/")
    print(f"  Relatório principal:   {relatorio_path}")
    print(f"  Gráficos:              {PLOTS_DIR}/\n")
