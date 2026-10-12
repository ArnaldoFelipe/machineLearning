#%%

import pandas as pd

dados = {
    'idade': [25, 45, 30, 50, 23, 40],
    'renda_mensal': [2000, 8000, 4500, 9000, 1500, 7500],
    'valor_emprestimo': [10000, 5000, 15000, 8000, 20000, 6000],
    'historico_calote': [0, 0, 0, 0, 1, 0], # 0 = Não, 1 = Sim
    'aprovado': [0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(dados)
df.head()
# %%

X = df.drop(columns=['aprovado'])

y = df['aprovado']

print("Features (X):")
print(X)
print("Gabarito (y):")
print(y)
# %%

from sklearn import tree

modelo = tree.DecisionTreeClassifier(random_state=42)
modelo.fit(X, y)

# %%
cliente_novo = pd.DataFrame({
    'idade': [28],
    'renda_mensal': [3000],
    'valor_emprestimo': [5000],
    'historico_calote': [0]
})

previsao = modelo.predict(cliente_novo)

if previsao[0] == 1:
    print("\nRESULTADO: Empréstimo APROVADO! ✅")
else:
    print("\nRESULTADO: Empréstimo RECUSADO! ❌")
# %%

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# Desenhando a árvore de decisão
plot_tree(modelo, 
          feature_names=X.columns, 
          class_names=['Recusado', 'Aprovado'], 
          filled=True, 
          rounded=True)

plt.show()

# %%


# %%
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Gerando a nossa massa de dados (1.000 clientes fictícios)
# Simulamos 4 features (ex: idade, renda, valor pedido, score) e o target (0 ou 1)
X_falso, y_falso = make_classification(
    n_samples=1000,     # Total de linhas
    n_features=4,       # Total de colunas (características)
    n_classes=2,        # 2 respostas possíveis: 0 (Recusado) ou 1 (Aprovado)
    random_state=42     # Garante que os dados aleatórios sejam sempre os mesmos
)

# 2. O GRANDE CORTE: Separando Treino (80%) e Teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_falso, y_falso, 
    test_size=0.20,     # Corta 20% exclusivamente para a prova final
    random_state=42
)

print(f"Total de clientes enviados para TREINO: {len(X_treino)}")
print(f"Total de clientes separados para TESTE: {len(X_teste)}\n")

# 3. O Treinamento (O aluno estudando o histórico)
modelo_real = DecisionTreeClassifier(random_state=42)

# ATENÇÃO AQUI: Passamos APENAS os dados de treino para o .fit()
modelo_real.fit(X_treino, y_treino) 
print("Modelo treinado com sucesso nos 800 clientes!\n")

# 4. A Prova Final (Inferência em Lote)
# Pedimos para a árvore prever os 200 clientes que ela nunca viu na vida
previsoes_teste = modelo_real.predict(X_teste)

# 5. O momento da verdade: Calculando a nota (Acurácia)
# Comparamos as respostas da árvore (previsoes) com o gabarito escondido (y_teste)
nota = accuracy_score(y_teste, previsoes_teste)

print(f"RESULTADO: O modelo acertou {nota * 100:.2f}% dos clientes inéditos!")
# %%


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Nossa Tabela Bruta (Ainda não é a ABT perfeita)
dados = {
    'idade': [25, 45, 30, 50, 23, 40, 25, 50], # Adicionamos um 25 e um 50 no final
    'renda_mensal': [2000, 8000, 3000, 10000, 5000, 6000, 15000, 2000],
    'valor_emprestimo': [10000, 4000, 9000, 2000, 10000, 3000, 1000, 20000],
    'aprovado': [0, 1, 0, 1, 0, 1, 1, 0]  # O jovem final aprovou(1), o senhor reprovou(0)
}

df = pd.DataFrame(dados)
print("--- TABELA ANTES DA ENGENHARIA DE FEATURES ---")
print(df)

# =================================================================
# 2. ENGENHARIA DE FEATURES (A MÁGICA)
# Criamos uma coluna nova calculando quantas vezes o empréstimo 
# é maior que a renda mensal. (Ex: Empréstimo 10k / Renda 2k = 5.0)
# =================================================================
df['comp_renda'] = df['valor_emprestimo'] / df['renda_mensal']

print("\n--- TABELA DEPOIS (Nossa ABT pronta) ---")
print(df[['renda_mensal', 'valor_emprestimo', 'comp_renda', 'aprovado']])

# 3. Separando X e y
X = df.drop(columns=['aprovado'])
y = df['aprovado']

# 4. Treinando o modelo
modelo_inteligente = DecisionTreeClassifier(random_state=42)
modelo_inteligente.fit(X, y)

# 5. Desenhando a nova mente do algoritmo
plt.figure(figsize=(8, 5))
plot_tree(modelo_inteligente, 
          feature_names=X.columns, 
          class_names=['Recusado', 'Aprovado'], 
          filled=True, 
          rounded=True)
plt.show()
# %%
