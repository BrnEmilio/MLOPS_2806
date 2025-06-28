import streamlit as st
import joblib
import os 

st.title("Classificador de sentimentos")

texto = st.text_input("Digite um tweet:")

if os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib"):
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    if st.button("Analisar"):
        if texto.strip():
            vetor = vectorizer.transform([texto])
            pred = model.predict(vetor)[0]
            st.write(f"Setimento: {pred}")
        else:
            st.warning("Por favor, insira um texto para analise.")
else:
    st.error("Modelo ou vetor não foram encontrados. Certifique-se que os arquivos model e vectorizer estão na raiz do projeto.")

