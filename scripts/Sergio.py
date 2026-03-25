
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scripts.preprocessing.normalizacao import carregar_teste_normalizado

# Carregar e normalizar dataset
BASE_DIR = Path(__file__).resolve().parent.parent
df = carregar_teste_normalizado(caminho=str(BASE_DIR / "datasets" / "mestrado-teste.csv"))

# Codificar coluna Sex (F=0, M=1)
df["Sex"] = df["Sex"].map({"F": 0, "M": 1})

# Definir target e features
TARGET = "GAD"
COLUNAS_EXCLUIR = ["Subject", "GAD", "GAD Probabiliy - Gamma", "SAD Probability - Gamma", "Sample Weight"]

y = df[TARGET]
X = df.drop(columns=COLUNAS_EXCLUIR, errors="ignore")

tamanho = X.shape
print(tamanho)

# Calcular informação mútua (baseada em entropia)
mi = mutual_info_classif(X, y)

# Organizar resultados
mi_series = pd.Series(mi, index=X.columns)
mi_series = mi_series.sort_values(ascending=False)

print("Top 10 atributos mais informativos:")
print(mi_series.head(10))

# Selecionar os 5 melhores atributos
top_features = mi_series.head(5).index
X_selected = X[top_features]

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, random_state=42
)

# Treinar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)
print("\nAcurácia usando apenas 5 atributos:", accuracy_score(y_test, y_pred))
