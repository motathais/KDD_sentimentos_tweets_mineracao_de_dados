import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('./projeto_mineracao/data/raw/sentiment_tweets.csv', encoding='latin-1', header=None)

# Nomear as colunas
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Visualizar algumas linhas de forma bonita
print("\nAmostra do dataset:")
print(df[['target', 'text']].head())

# Verificar distribuição dos sentimentos
print("\nDistribuição dos sentimentos:")
print(df['target'].value_counts())

# Mapear os valores para facilitar leitura
df['sentiment'] = df['target'].map({0: 'Negativo', 4: 'Positivo'})

# Plotar o gráfico de contagem
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Distribuição dos Sentimentos')
plt.xlabel('Sentimento')
plt.ylabel('Quantidade de Tweets')
plt.show()

