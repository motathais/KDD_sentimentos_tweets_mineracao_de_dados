import joblib

# ========================
# 📂 Carregar modelo salvo
# ========================
model = joblib.load('./projeto_mineracao/modelos/logistic_regression_model.joblib')

# ========================
# 📝 Função para testar frase nova
# ========================
def testar_frase(frase):
    pred = model.predict([frase])[0]
    prob = model.predict_proba([frase])[0]
    sentimento = 'Positivo' if pred == 1 else 'Negativo'
    print(f"🔹 Frase: {frase}")edg
    print(f"➡️ Previsão: {sentimento} ({prob[pred]*100:.2f}% de confiança)")

# ========================
# ✅ Teste
# ========================
testar_frase("I love this movie, it was amazing!")
testar_frase("I hate waiting in traffic.")
