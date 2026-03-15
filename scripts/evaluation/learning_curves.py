"""
Learning Curves para diagnostico de overfitting/underfitting.

Gera curvas de aprendizado para XGBoost e SVM com as 4 tecnicas
de balanceamento. Mostra como o desempenho muda com o tamanho
do conjunto de treino.
"""

import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from scripts.utils import preparar_dados, plotar_learning_curve


def gerar_learning_curves(algoritmo, target='GAD'):
    """Gera learning curves para um algoritmo com diferentes tecnicas de balanceamento.

    Args:
        algoritmo: 'xgboost' ou 'svm'
        target: 'GAD' ou 'SAD'
    """
    df, target_name = preparar_dados(target)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    dist = df[target_name].value_counts()
    peso_classe = dist[0] / dist[1]

    nome_algo = 'XGBoost' if algoritmo == 'xgboost' else 'SVM'
    output_path = f'output/plots/LearningCurves/{nome_algo}/{target.upper()}'
    os.makedirs(output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  LEARNING CURVES - {nome_algo} ({target})")
    print("=" * 70)
    print(f"\n  Amostras: {df.shape[0]} | Features: {df.shape[1] - 1}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_sizes_frac = np.linspace(0.1, 1.0, 10)

    tecnicas = [
        ('Sem Balanceamento', 'none'),
        ('Class Weighting', 'weighted'),
        ('SMOTE', 'smote'),
        ('Undersampling', 'undersampling'),
    ]

    for nome_tecnica, bal in tecnicas:
        print(f"\n  --- {nome_tecnica} ---")

        if algoritmo == 'xgboost':
            clf_params = dict(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
            if bal == 'weighted':
                clf_params['scale_pos_weight'] = peso_classe

            steps = []
            if bal == 'smote':
                steps.append(('sampler', SMOTE(random_state=42, k_neighbors=3)))
            elif bal == 'undersampling':
                steps.append(('sampler', RandomUnderSampler(random_state=42)))
            steps.append(('clf', XGBClassifier(**clf_params)))
        else:
            clf_params = dict(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            if bal == 'weighted':
                clf_params['class_weight'] = 'balanced'

            steps = [('scaler', StandardScaler())]
            if bal == 'smote':
                steps.append(('sampler', SMOTE(random_state=42, k_neighbors=3)))
            elif bal == 'undersampling':
                steps.append(('sampler', RandomUnderSampler(random_state=42)))
            steps.append(('clf', SVC(**clf_params)))

        pipeline = ImbPipeline(steps)

        print(f"    Calculando learning curve...", end=" ")
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, X, y,
            cv=cv,
            train_sizes=train_sizes_frac,
            scoring='f1',
            n_jobs=-1
        )
        print("OK")

        sufixo = bal if bal != 'none' else 'baseline'
        plot_file = f'{output_path}/learning_curve_{target.lower()}_{sufixo}.png'
        plotar_learning_curve(
            train_sizes, train_scores, val_scores,
            f'Learning Curve - {nome_algo} {nome_tecnica} ({target})\n10-Fold Stratified CV | Scoring: F1',
            plot_file
        )

        # Salvar metricas em txt
        txt_file = f'{output_path}/learning_curve_{target.lower()}_{sufixo}.txt'
        with open(txt_file, 'w') as f:
            f.write(f"Learning Curve - {nome_algo} {nome_tecnica} - {target}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Tam. Treino':>12} {'F1 Treino':>12} {'F1 Valid.':>12} {'Gap':>10}\n")
            f.write("-" * 50 + "\n")
            for i in range(len(train_sizes)):
                t_mean = np.mean(train_scores[i])
                v_mean = np.mean(val_scores[i])
                gap = t_mean - v_mean
                f.write(f"{train_sizes[i]:>12d} {t_mean:>12.4f} {v_mean:>12.4f} {gap:>10.4f}\n")

            f.write(f"\nDiagnostico:\n")
            final_train = np.mean(train_scores[-1])
            final_val = np.mean(val_scores[-1])
            gap = final_train - final_val

            if gap > 0.15:
                f.write(f"  OVERFITTING: Gap treino-validacao alto ({gap:.3f})\n")
                f.write(f"  Sugestao: Mais dados, regularizacao ou modelo mais simples\n")
            elif final_val < 0.3:
                f.write(f"  UNDERFITTING: F1 validacao baixo ({final_val:.3f})\n")
                f.write(f"  Sugestao: Modelo mais complexo ou features melhores\n")
            else:
                f.write(f"  OK: Gap={gap:.3f}, F1 validacao={final_val:.3f}\n")

            # Verificar se mais dados ajudariam
            if len(train_sizes) >= 3:
                melhoria = np.mean(val_scores[-1]) - np.mean(val_scores[-3])
                if melhoria > 0.02:
                    f.write(f"  MAIS DADOS PODEM AJUDAR: Curva ainda subindo (+{melhoria:.3f})\n")
                else:
                    f.write(f"  PLATEAU: Curva estabilizou (melhoria={melhoria:.3f})\n")

        print(f"    Metricas salvas em: {txt_file}")


if __name__ == "__main__":
    for target in ['GAD', 'SAD']:
        print("\n" + "#" * 70)
        print(f"{'':>20}LEARNING CURVES - {target}")
        print("#" * 70)

        gerar_learning_curves('xgboost', target)
        gerar_learning_curves('svm', target)
