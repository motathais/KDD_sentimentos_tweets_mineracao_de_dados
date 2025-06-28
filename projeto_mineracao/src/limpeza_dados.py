import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ========================
# 📥 Baixar recursos do NLTK
# ========================
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ========================
# 📂 Carregar dataset bruto
# ========================
df = pd.read_csv('./projeto_mineracao/data/raw/sentiment_tweets.csv', encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# ========================
# 🔍 Remover duplicatas
# ========================
print(f"🔸 Registros antes de remover duplicatas: {len(df)}")
df = df.drop_duplicates(subset='text')
print(f"🔸 Registros após remover duplicatas: {len(df)}")

# ========================
# ❌ Tratar valores nulos
# ========================
print("\n🔸 Valores ausentes por coluna:")
print(df.isnull().sum())
df = df.dropna(subset=['text'])  # Garante que não há texto vazio

# ========================
# 🧹 Função de limpeza avançada
# ========================
def limpar_texto(text):
    # Remover URLs
    text = re.sub(r'http\S+', '', text)
    # Remover menções (@usuario)
    text = re.sub(r'@\w+', '', text)
    # Remover números
    text = re.sub(r'\d+', '', text)
    # Remover caracteres especiais e pontuação
    text = re.sub(r'[^\w\s]', '', text)
    # Converter para minúsculo
    text = text.lower()
    # Tokenizar
    tokens = text.split()
    # Remover stopwords e lematizar
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Unir de volta
    clean_text = ' '.join(tokens)
    return clean_text.strip()

# Aplicar limpeza
df['clean_text'] = df['text'].apply(limpar_texto)

# ========================
# 🔍 Conferir exemplos
# ========================
print("\n🔸 Exemplo de tweets antes e depois da limpeza:")
for i in range(3):
    print(f"\nOriginal: {df.iloc[i]['text']}")
    print(f"Limpo: {df.iloc[i]['clean_text']}")

# ========================
# 💾 Salvar dataset processado
# ========================
df.to_csv('./projeto_mineracao/data/processed/sentiment_tweets_clean.csv', index=False, encoding='utf-8-sig')

print("\n✅ Limpeza avançada concluída e arquivo salvo em: data/processed/sentiment_tweets_clean.csv")

