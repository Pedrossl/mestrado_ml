import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scripts.utils import preparar_dados, calcular_metricas_fold

SEED = 42

df, target = preparar_dados('GAD')
feat = [c for c in df.columns if c != target]
X = df.drop(columns=[target]).values
y = df[target].values

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

all_proba = []
all_true = []

for tr, te in cv.split(X, y):
    X_tr, y_tr = SMOTE(random_state=SEED).fit_resample(X[tr], y[tr])
    m = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=SEED)
    m.fit(X_tr, y_tr)
    all_proba.extend(m.predict_proba(X[te])[:, 1])
    all_true.extend(y[te])

all_proba = np.array(all_proba)
all_true = np.array(all_true)

fpr, tpr, thresholds = roc_curve(all_true, all_proba)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_thresh = thresholds[best_idx]

print(f'Threshold otimo (Youden J): {best_thresh:.4f}')
print(f'Youden J maximo: {youden_j[best_idx]:.4f}')
print(f'AUC: {roc_auc_score(all_true, all_proba):.4f}')
print()

threshs = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, round(best_thresh, 2)]
threshs = sorted(set(threshs), reverse=True)

header = f"{'Threshold':>10} {'Sens':>8} {'Spec':>8} {'F1':>8} {'Kappa':>8} {'FN':>5} {'FP':>5}"
print(header)
print("-" * 60)

n_pos = int(all_true.sum())
n_neg = len(all_true) - n_pos

for t in threshs:
    y_pred = (all_proba >= t).astype(int)
    met = calcular_metricas_fold(all_true.astype(int), y_pred.astype(int))
    fn = int(((all_true == 1) & (y_pred == 0)).sum())
    fp = int(((all_true == 0) & (y_pred == 1)).sum())
    marker = ' <-- Youden' if abs(t - best_thresh) < 0.015 else ''
    print(f'{t:>10.2f} {met["sensitivity"]:>7.1f}% {met["specificity"]:>7.1f}% {met["f1"]:>7.1f}% {met["kappa"]:>8.4f} {fn:>5} {fp:>5}{marker}')

print(f'\nTotal: {n_pos} positivos, {n_neg} negativos, {len(all_true)} amostras')

# Agora testar com scale_pos_weight
print('\n' + '=' * 60)
print('  COM scale_pos_weight')
print('=' * 60 + '\n')

for spw in [1, 3, 5, 7, 10]:
    metricas_folds = []
    aucs = []
    for tr, te in cv.split(X, y):
        X_tr, y_tr = SMOTE(random_state=SEED).fit_resample(X[tr], y[tr])
        m = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=SEED, scale_pos_weight=spw)
        m.fit(X_tr, y_tr)
        yp = m.predict(X[te])
        ypr = m.predict_proba(X[te])[:, 1]
        metricas_folds.append(calcular_metricas_fold(y[te].astype(int), yp.astype(int)))
        aucs.append(roc_auc_score(y[te], ypr))

    res = {}
    for k in ['accuracy', 'sensitivity', 'specificity', 'f1', 'kappa']:
        vals = [f[k] for f in metricas_folds]
        res[k] = np.mean(vals)
    res['auc'] = np.mean(aucs)

    print(f'  scale_pos_weight={spw:<3}  Kappa={res["kappa"]:.4f}  Sens={res["sensitivity"]:.1f}%  Spec={res["specificity"]:.1f}%  F1={res["f1"]:.1f}%  AUC={res["auc"]:.4f}')

# Combinacao: scale_pos_weight + threshold otimizado
print('\n' + '=' * 60)
print('  COMBINACAO: scale_pos_weight + Youden threshold')
print('=' * 60 + '\n')

for spw in [1, 3, 5, 7]:
    all_p = []
    all_t = []
    for tr, te in cv.split(X, y):
        X_tr, y_tr = SMOTE(random_state=SEED).fit_resample(X[tr], y[tr])
        m = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=SEED, scale_pos_weight=spw)
        m.fit(X_tr, y_tr)
        all_p.extend(m.predict_proba(X[te])[:, 1])
        all_t.extend(y[te])

    all_p = np.array(all_p)
    all_t = np.array(all_t)

    fpr2, tpr2, th2 = roc_curve(all_t, all_p)
    yj2 = tpr2 - fpr2
    bi2 = np.argmax(yj2)
    bt2 = th2[bi2]

    y_pred = (all_p >= bt2).astype(int)
    met = calcular_metricas_fold(all_t.astype(int), y_pred.astype(int))
    fn = int(((all_t == 1) & (y_pred == 0)).sum())
    fp = int(((all_t == 0) & (y_pred == 1)).sum())

    print(f'  spw={spw:<3} thresh={bt2:.3f}  Kappa={met["kappa"]:.4f}  Sens={met["sensitivity"]:.1f}%  Spec={met["specificity"]:.1f}%  F1={met["f1"]:.1f}%  FN={fn}  FP={fp}')
