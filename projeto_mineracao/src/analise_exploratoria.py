# Importando bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o dataset
df = pd.read_csv('./projeto_mineracao/data/raw/sentiment_tweets.csv', encoding='latin-1', header=None)

# Nomeando as colunas
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# ========================
# 🔎 Análise Inicial
# ========================

print("🔸 Primeiras linhas do dataset:")
print(df.head())

print("\n🔸 Informações gerais sobre o dataset:")
print(df.info())

print("\n🔸 Verificando valores nulos:")
print(df.isnull().sum())

print("\n🔸 Distribuição das classes:")
print(df['target'].value_counts())

# ========================
# 📊 Visualização das Classes
# ========================

plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df)
plt.title('Distribuição das Classes (Sentimentos)')
plt.xlabel('Sentimento (0 = Negativo, 4 = Positivo)')
plt.ylabel('Quantidade de Tweets')
plt.xticks([0, 1], ['Negativo', 'Positivo'])  # Caso só existam 0 e 4, pode usar plt.xticks([0,4], ['Negativo', 'Positivo'])
plt.show()

# ========================
# 📝 Análise do Tamanho dos Tweets
# ========================

# Criando coluna com tamanho dos tweets
df['text_length'] = df['text'].apply(len)

print("\n🔸 Estatísticas sobre o tamanho dos tweets:")
print(df['text_length'].describe())

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='text_length', data=df)
plt.title('Distribuição do Tamanho dos Tweets por Classe')
plt.xlabel('Sentimento (0 = Negativo, 4 = Positivo)')
plt.ylabel('Tamanho do Tweet')
plt.xticks([0, 1], ['Negativo', 'Positivo'])  
plt.show()

# ========================
# 🗂️ Checagem de Duplicatas
# ========================

print("\n🔸 Total de registros duplicados (considerando o texto):", df.duplicated(subset='text').sum())

# ========================
# ✅ Fim da Análise Exploratória
# ========================

print("\n✅ Análise exploratória concluída!")
