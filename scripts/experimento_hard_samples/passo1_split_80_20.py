# =============================================================================
# PASSO 1 — Split 80/20 e validação da distribuição das classes
# =============================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from scripts.utils import preparar_dados

ALVO = 'GAD'

df, target_name = preparar_dados(ALVO)
X = df.drop(columns=[target_name]).values
y = df[target_name].values

# Divide 80% treino / 20% teste mantendo proporção das classes
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Valida distribuição em cada split
def distribuicao(y_split, nome):
    n_pos = y_split.sum()
    n_neg = len(y_split) - n_pos
    print(f"  {nome}: {len(y_split)} amostras | Positivos: {n_pos} ({n_pos/len(y_split)*100:.1f}%) | Negativos: {n_neg} ({n_neg/len(y_split)*100:.1f}%)")

print(f"\nSplit 80/20 — Alvo: {target_name}")
distribuicao(y,        "Total  ")
distribuicao(y_treino, "Treino ")
distribuicao(y_teste,  "Teste  ")

print("\n✓ Distribuição validada. Split salvo para os próximos passos.")

import os
os.makedirs('output/experimento_hard_samples', exist_ok=True)
np.save('output/experimento_hard_samples/X_treino.npy', X_treino)
np.save('output/experimento_hard_samples/X_teste.npy',  X_teste)
np.save('output/experimento_hard_samples/y_treino.npy', y_treino)
np.save('output/experimento_hard_samples/y_teste.npy',  y_teste)
print("Arquivos salvos em output/experimento_hard_samples/")
