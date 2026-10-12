# %%

import pandas as pd 

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()
# %%

from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']]
y = df[['nota']]

reg = linear_model.LinearRegression()
reg.fit(X, y)

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)

predict_arvore_full = arvore_full.predict(X.drop_duplicates())
# %%

predict_reg = reg.predict(X.drop_duplicates())
# %%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação cerveja vs notas")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict_reg)
plt.plot(X.drop_duplicates()['cerveja'],
predict_arvore_full)

# %%
