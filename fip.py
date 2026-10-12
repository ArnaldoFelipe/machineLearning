# %%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# 1. Gerando uma ABT sintética com 1.000 motos usadas
np.random.seed(42)
anos = np.random.randint(2015, 2026, 1000)
km = (2026 - anos) * 8000 + np.random.randint(-3000, 5000, 1000)
km = np.where(km < 0, 0, km) # Impedir quilometragem negativa

preco = 18000 - ((2026 - anos) * 600) - (km / 1000) * 30 + np.random.randint(-500, 500, 1000)

df_motos = pd.DataFrame({
    'ano': anos, 
    'quilometragem': km, 
    'preco_venda': preco
})

X = df_motos[['ano', 'quilometragem']]
y = df_motos['preco_venda']

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

modelo_padrao = DecisionTreeRegressor(random_state=42)
modelo_padrao.fit(X_treino, y_treino)

previsoes_padrao = modelo_padrao.predict(X_teste)
erro_padrao = mean_absolute_error(y_teste, previsoes_padrao)

# ====================================================================
# 4. MODELO 2: O TUNADO (Apertando os parafusos)
# max_depth=5: A árvore só pode ter 5 níveis de profundidade.
# min_samples_leaf=15: Cada regra final precisa englobar no mínimo 15 motos.
# Isso impede que ela crie regras viciadas para motos muito específicas.
# ====================================================================
modelo_tunado = DecisionTreeRegressor(  min_samples_leaf=15, random_state=42)
modelo_tunado.fit(X_treino, y_treino)

previsoes_tunado = modelo_tunado.predict(X_teste)
erro_tunado = mean_absolute_error(y_teste, previsoes_tunado)

print(f"Erro do Modelo Padrão: Errou os preços em R$ {erro_padrao:.2f} (em média)")
print(f"Erro do Modelo Tunado: Errou os preços em R$ {erro_tunado:.2f} (em média)")

print("\nArquivo 'motor_precificacao_motos.pkl' salvo com sucesso para a API!")
# %%
