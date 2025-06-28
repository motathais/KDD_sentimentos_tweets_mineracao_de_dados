# 📊 Análise de Sentimentos com Mineração de Dados

Este projeto realiza uma **análise de sentimentos** em tweets usando o dataset **Sentiment140**, aplicando técnicas de **limpeza de dados**, **vetorização TF-IDF** e **classificação supervisionada**.

---

## 📁 Estrutura do Projeto

projeto_mineracao/
├── data/
│ ├── raw/ # Dataset original (.csv)
│ ├── processed/ # Dataset limpo
├── modelos/ # Modelos treinados salvos (.joblib)
├── notebooks/ # Análises exploratórias (opcional)
├── src/ # Scripts Python
│ ├── comparar_modelos.py # Treino e comparação de modelos
│ ├── teste_frase.py # Teste de novas frases
├── README.md
├── requirements.txt


---

## 📌 Objetivo

- Explorar, limpar e preparar o dataset de tweets.
- Comparar algoritmos de classificação de sentimentos (**Naive Bayes** e **Regressão Logística**).
- Validar os modelos com **k-fold cross-validation**.
- Salvar o melhor modelo treinado.
- Fazer testes em **novas frases**.

---

## 📦 Pré-requisitos

- Python 3.9+  
- Virtualenv (opcional, mas recomendado)

---

## ⚙️ Configuração do Ambiente

1️⃣ Clone o repositório ou copie os arquivos:

git clone <URL_DO_SEU_REPOSITORIO>
cd projeto_mineracao

2️⃣ Crie um ambiente virtual:


python -m venv venv

3️⃣ Ative o ambiente virtual:

- Windows:

venv\Scripts\activate

- Mac/Linux:

source venv/bin/activate

4️⃣ Instale as dependências:


pip install -r requirements.txt

---
## 📥 Dataset
O dataset Sentiment140 está salvo em data/raw/sentiment_tweets.csv.

Após limpeza, o arquivo limpo é salvo em data/processed/sentiment_tweets_clean.csv.

---
## 🚀 Como Rodar

1️⃣ Comparar Modelos

Execute o script que:

Treina Naive Bayes e Logistic Regression

Valida com k-fold

Salva o modelo Logistic como .joblib

python src/comparar_modelos.py


2️⃣ Testar Novas Frases

python src/teste_frase.py

---
📊 Resultados
Cross-validation

Acurácia

Matriz de confusão

Relatório de classificação

---
✅ Tecnologias Usadas
Python

Pandas

Scikit-learn

NLTK

Joblib

Seaborn / Matplotlib



✨ Autor
Thais Marques Mota
Emerson Rodrigo Lopes

