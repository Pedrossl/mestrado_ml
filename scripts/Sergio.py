
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

tamanho =  X.shape
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
