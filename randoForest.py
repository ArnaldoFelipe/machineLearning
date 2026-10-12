#%%

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(42)
qtd = 5000

dados = {
    'valor_compra': np.random.exponential(scale=200, size=qtd), # Maioria compras baixas, algumas altíssimas
    'idade_conta_dias': np.random.randint(1, 1500, size=qtd), # Tempo de conta do usuário
    'qtd_tentativas_falhas': np.random.choice([0, 1, 2, 3, 4], size=qtd, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
    'distancia_km_ip': np.random.exponential(scale=50, size=qtd) # Distância entre o IP e o endereço de entrega
}

df_transacoes = pd.DataFrame(dados)


# 2. Engenharia de Features Oculta (Criando a regra da Fraude para simular o mundo real)
# No mundo real, você já teria a coluna "fraude_confirmada". Aqui vamos simular:
# Compras altas em contas novas, com IP distante ou erros de senha têm alta chance de fraude.
df_transacoes['fraude'] = np.where(
    ((df_transacoes['valor_compra'] > 500) & (df_transacoes['idade_conta_dias'] < 30)) | 
    (df_transacoes['qtd_tentativas_falhas'] >= 2) |
    (df_transacoes['distancia_km_ip'] > 300), 
    1, 0
)

X = df_transacoes.drop(columns=['fraude'])
y = df_transacoes['fraude']

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

modelo = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)

print(f"Acurácia do Motor Antifraude: {acuracia * 100:.2f}%\n")
print("Relatório detalhado (Olhe para a linha do '1', que são as fraudes):")
print(classification_report(y_teste, previsoes))

joblib.dump(modelo, 'motor_antifraude_rf.pkl')
print("\n[OK] Arquivo 'motor_antifraude_rf.pkl' salvo com sucesso!")
# %%
