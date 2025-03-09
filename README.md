# Atividade-04
Criado para depositar o código desta atividade, referente a questão 01.

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Criando dados fictícios
dados = {
    "Combustivel": ["Gasolina", "Diesel", "Etanol", "Gasolina", "Diesel", "Etanol", "Gasolina", "Diesel"],
    "Idade": [5, 3, 8, 2, 7, 4, 6, 1],
    "Quilometragem": [50000, 30000, 80000, 20000, 70000, 40000, 60000, 10000],
    "Preco": [30000, 35000, 20000, 40000, 25000, 32000, 28000, 45000]
}

# Convertendo para DataFrame
df = pd.DataFrame(dados)

# Separando variáveis dependentes e independentes
X = df.drop(columns=["Preco"])
y = df["Preco"]

# Definindo colunas categóricas e numéricas
colunas_categoricas = ["Combustivel"]
colunas_numericas = ["Idade", "Quilometragem"]

# Criando os transformadores
transformador_categorico = OneHotEncoder()
transformador_numerico = StandardScaler()

# Criando o ColumnTransformer
preprocessador = ColumnTransformer(
    transformers=[
        ("cat", transformador_categorico, colunas_categoricas),
        ("num", transformador_numerico, colunas_numericas)
    ]
)

# Criando o pipeline
pipeline = Pipeline([
    ("preprocessador", preprocessador),
    ("modelo", LinearRegression())
])

# Dividindo os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
pipeline.fit(X_treino, y_treino)

# Fazendo previsões
y_pred = pipeline.predict(X_teste)

# Avaliando o modelo
mse = mean_squared_error(y_teste, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse}")
