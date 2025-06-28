import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ========================
# 📥 1. Carregar dataset
# ========================
df = pd.read_csv('./projeto_mineracao/data/processed/sentiment_tweets_clean.csv')

# ========================
# 🧼 2. Preparar os dados
# ========================

# Filtrar apenas classes 0 (Negativo) e 4 (Positivo)
df = df[df['target'].isin([0, 4])]
df['target'] = df['target'].map({0: 0, 4: 1})  # Reclassifica: 0 = Negativo, 1 = Positivo

# Garantir que a coluna clean_text não tenha valores nulos ou vazios
df['clean_text'] = df['clean_text'].fillna('')
df = df[df['clean_text'].str.strip() != '']

# ========================
# ✂️ 3. Separar treino e teste
# ========================
X = df['clean_text']
y = df['target']

# Usar stratify para balancear as classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# 🧠 4. Vetorização (Bag of Words)
# ========================
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ========================
# 🤖 5. Treinar o modelo Naive Bayes
# ========================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ========================
# 📈 6. Fazer previsões
# ========================
y_pred = model.predict(X_test_vec)

# ========================
# 🧪 7. Avaliação
# ========================
print("\n🔍 Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"]))
print("✅ Acurácia:", accuracy_score(y_test, y_pred))

# ========================
# 📊 8. Matriz de Confusão
# ========================
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negativo", "Positivo"],
            yticklabels=["Negativo", "Positivo"])
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
