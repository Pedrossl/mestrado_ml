# =============================================================================
# PASSO 2 — Identificar os 20 "Hard Samples"
# Casos de difícil classificação: menor margem de confiança no conjunto de teste
# Usa XGBoost + SMOTE (melhor modelo do projeto) treinado nos 80% de treino
# =============================================================================

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils import preparar_dados

ALVO   = 'GAD'
N_HARD = 20       # Quantos hard samples selecionar

OUTPUT = 'output/experimento_hard_samples'
os.makedirs(OUTPUT, exist_ok=True)

# Carrega split salvo pelo passo 1
X_treino = np.load(f'{OUTPUT}/X_treino.npy')
X_teste  = np.load(f'{OUTPUT}/X_teste.npy')
y_treino = np.load(f'{OUTPUT}/y_treino.npy')
y_teste  = np.load(f'{OUTPUT}/y_teste.npy')

# Carrega nomes das features para salvar junto
df, target_name = preparar_dados(ALVO)
feature_names = df.drop(columns=[target_name]).columns.tolist()

# Treina XGBoost + SMOTE nos 80%
n_neg, n_pos = np.bincount(y_treino.astype(int))
smote = SMOTE(random_state=42)
X_tr_res, y_tr_res = smote.fit_resample(X_treino, y_treino)

modelo = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
modelo.fit(X_tr_res, y_tr_res)

# Obtém probabilidade de ser positivo para cada amostra do teste
probas = modelo.predict_proba(X_teste)[:, 1]  # probabilidade da classe 1
y_pred = modelo.predict(X_teste)

# "Margem de confiança" = distância até 0.5 (threshold de decisão)
# Quanto menor a margem, mais incerto o modelo está = hard sample
margem = np.abs(probas - 0.5)

# Ordena pelo menor margem (mais incerto primeiro)
idx_ordenado = np.argsort(margem)
idx_hard     = idx_ordenado[:N_HARD]  # os 20 mais incertos

# Monta DataFrame dos hard samples
df_hard = pd.DataFrame(X_teste[idx_hard], columns=feature_names)
df_hard['y_real']      = y_teste[idx_hard].astype(int)
df_hard['y_pred']      = y_pred[idx_hard].astype(int)
df_hard['prob_pos']    = probas[idx_hard].round(4)
df_hard['margem']      = margem[idx_hard].round(4)
df_hard['acertou']     = (df_hard['y_real'] == df_hard['y_pred']).astype(int)

# Salva
df_hard.to_csv(f'{OUTPUT}/hard_samples.csv', index=False)
np.save(f'{OUTPUT}/X_hard.npy', X_teste[idx_hard])
np.save(f'{OUTPUT}/y_hard.npy', y_teste[idx_hard])

# Exibe resumo
print(f"\n{'=' * 60}")
print(f"  HARD SAMPLES — {N_HARD} casos mais incertos ({target_name})")
print(f"{'=' * 60}")
print(f"\n  Total no teste: {len(X_teste)} amostras")
print(f"  Hard samples selecionados: {N_HARD}")
print(f"\n  Composição dos hard samples:")
print(f"    Positivos reais (com {target_name}): {df_hard['y_real'].sum()}")
print(f"    Negativos reais (sem {target_name}): {(df_hard['y_real'] == 0).sum()}")
print(f"    Acertos dentro dos hard: {df_hard['acertou'].sum()}/{N_HARD}")
print(f"\n  Probabilidades (prob de ter {target_name}):")
print(f"    Min: {df_hard['prob_pos'].min():.3f} | Max: {df_hard['prob_pos'].max():.3f} | Média: {df_hard['prob_pos'].mean():.3f}")
print(f"\n  Arquivo salvo: {OUTPUT}/hard_samples.csv")
print(f"{'=' * 60}\n")
