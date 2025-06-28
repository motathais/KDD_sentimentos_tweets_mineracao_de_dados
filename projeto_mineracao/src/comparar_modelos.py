import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

# ========================
# 📥 Carregar dados
# ========================
df = pd.read_csv('./projeto_mineracao/data/processed/sentiment_tweets_clean.csv')

# Labels binários
df = df[df['target'].isin([0, 4])]
df['target'] = df['target'].map({0: 0, 4: 1})

# Garantir textos não nulos
df['clean_text'] = df['clean_text'].fillna('')
df = df[df['clean_text'].str.strip() != '']

X = df['clean_text']
y = df['target']

# ========================
# ✂️ Treino/Teste
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# ⚙️ Pipelines: Logistic & Naive Bayes
# ========================
pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', MultinomialNB())
])

# ========================
# 🚀 Avaliar modelos
# ========================
for name, pipe in [('Logistic Regression', pipe_logreg), ('Naive Bayes', pipe_nb)]:
    print(f"\n🔍 Modelo: {name}")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"]))
    
    # Cross-validation (5-fold)
    cv_scores = cross_val_score(pipe, X, y, cv=5)
    print(f"Cross-val Mean Acc: {cv_scores.mean():.4f}")

# ========================
# 💾 Salvar Logistic como modelo final
# ========================
pipe_logreg.fit(X, y)
joblib.dump(pipe_logreg, './projeto_mineracao/modelos/logistic_regression_model.joblib')
print("\n✅ Modelo Logistic Regression salvo em: modelos/logistic_regression_model.joblib")
