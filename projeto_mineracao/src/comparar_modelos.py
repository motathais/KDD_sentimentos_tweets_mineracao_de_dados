import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ========================
# 📥 Carregar dados
# ========================
df = pd.read_csv('./projeto_mineracao/data/processed/sentiment_tweets_clean.csv')

# Filtrar e mapear rótulos
df = df[df['target'].isin([0, 4])]
df['target'] = df['target'].map({0: 0, 4: 1})

# Limpar valores nulos ou vazios
df['clean_text'] = df['clean_text'].fillna('')
df = df[df['clean_text'].str.strip() != '']

# ========================
# ✂️ Separar treino/teste
# ========================
X = df['clean_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vetorização
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ========================
# 🤖 Função de avaliação
# ========================
def avaliar_modelo(nome, modelo):
    modelo.fit(X_train_vec, y_train)
    y_pred = modelo.predict(X_test_vec)
    print(f"\n🔍 Modelo: {nome}")
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"]))

# ========================
# 🚀 Aplicar os modelos
# ========================
avaliar_modelo("Naive Bayes", MultinomialNB())
avaliar_modelo("Regressão Logística", LogisticRegression(max_iter=1000))
avaliar_modelo("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))  

