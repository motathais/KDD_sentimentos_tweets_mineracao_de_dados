import pandas as pd
import re

# Carregando dataset limpo da análise exploratória
df = pd.read_csv('./projeto_mineracao/data/raw/sentiment_tweets.csv', encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# ========================
# 🔍 1. Remover duplicatas
# ========================
print(f"🔸 Registros antes de remover duplicatas: {len(df)}")
df = df.drop_duplicates(subset='text')
print(f"🔸 Registros após remover duplicatas: {len(df)}")

# ========================
# ❌ 2. Tratar valores nulos
# ========================
print("\n🔸 Valores ausentes por coluna:")
print(df.isnull().sum())

# Se houvesse nulos, poderíamos remover ou tratar:
# df = df.dropna()

# ========================
# 🧹 3. Limpeza dos textos
# ========================

def limpar_texto(text):
    # Remover URLs
    text = re.sub(r'http\S+', '', text)
    # Remover menções (@usuario)
    text = re.sub(r'@\w+', '', text)
    # Remover números
    text = re.sub(r'\d+', '', text)
    # Remover caracteres especiais e pontuações
    text = re.sub(r'[^\w\s]', '', text)
    # Converter para minúsculo
    text = text.lower()
    # Remover espaços extras
    text = text.strip()
    return text

# Aplicando a limpeza no dataset
df['clean_text'] = df['text'].apply(limpar_texto)

# Mostrando exemplos de como ficou
print("\n🔸 Exemplo de tweets antes e depois da limpeza:")
for i in range(3):
    print(f"\nOriginal: {df.iloc[i]['text']}")
    print(f"Limpo: {df.iloc[i]['clean_text']}")

# ========================
# 💾 Salvando dataset limpo
# ========================
df.to_csv('./projeto_mineracao/data/processed/sentiment_tweets_clean.csv', index=False, encoding='utf-8-sig')

print("\n✅ Limpeza concluída e arquivo salvo em: data/processed/sentiment_tweets_clean.csv")
