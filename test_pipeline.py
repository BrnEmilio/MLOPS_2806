import pandas as pd
import joblib
import os 
from sklearn.metrics import accuracy_score

model_path = "model.joblib"
vectorizer_path = "vectorizer.joblib"

# Lista de cenários de teste: (texto, classe_esperada)
test_cases = [
    # Curto
    ("Adorei o produto de investimentos no mercado de capitais", "positivo"),
    ("O serviço de instalação internet doi péssimo. Não recomendo", "negativo"),
    ("Foi ok","neutro"),
    # Médio
    ("Entrega dentro do esperado, ok.", "neutro"),
    ("Adorei este serviço", "positivo"),
    ("Não gostei.", "negativo"),
    # Longo
    ("Recebi o produto conforme o anunciado, dentro do prazo e sem nenhum problema, foi uma experiência normal, sem destaques.", "neutro"),
    ("Super satisfeito: atendimento ótimo, entrega rápida, qualidade acima do esperado, voltarei a comprar com certeza!", "positivo"),
    ("Além da entrega atrasada, o produto não funcionou como deveria e o suporte não resolveu, muito insatisfeito.", "negativo")
]

def test_model_files_exist():
    assert os.path.exists("./model.joblib"), "Modelo não encontrado"
    assert os.path.exists("./vectorizer.joblib"), "Vetor não encontrado"

def test_vectorizer_output_shape():
    vectorizer = joblib.load("./vectorizer.joblib")
    sample = ["Este é um ótimo produto!"]
    vetor = vectorizer.transform(sample)
    assert vetor.shape[0] == 1, "Vetor retornou de forma incorreta"

def test_vectorizer_output_shape_neutro():
    vectorizer = joblib.load("./vectorizer.joblib")
    sample = ["Entrega normal sem problemas."]
    vetor = vectorizer.transform(sample)
    print(vetor)
    assert vetor.shape[0] == 1, "Vetor retornou de forma incorreta"

def test_model_prediction_labels():
    model = joblib.load("./model.joblib")
    vectorizer = joblib.load("./vectorizer.joblib")
    sample = ["O serviço foi péssimo"]
    vetor = vectorizer.transform(sample)
    pred = model.predict(vetor)[0]
    assert pred in ["positivo", "negativo"], f"Rótulo inesperado {pred}"

def test_data_validation():
    df = pd.read_csv("./data/tweets_limpo.csv")
    assert "text" in df.columns and "label" in df.columns
    assert df["text"].notnull().all()
    assert df["label"].isin(["positivo","negativo","neutro"]).all()

def test_fairness_by_text_length():
    model_path = "./model.joblib"
    vectorizer_path = "./vectorizer.joblib"
    data_path = "./data/tweets_limpo.csv"

    assert os.path.exists(model_path), "Modelo não encontrado"
    assert os.path.exists(vectorizer_path), "Vetor não encontrado"
    assert os.path.exists(data_path), "Dataset não encontrado"

    model = joblib.load(model_path)
    vectorize = joblib.load(vectorizer_path)
    df = pd.read_csv(data_path)

    df["text_len"] = df["text"].apply(len)
    df["len_catogory"] = pd.cut(df["text_len"], bins=[0,50,150,1000], labels=["curto","medio","longo"])
    vetor = vectorize.transform(df["text"])
    y_true = df["label"]
    y_pred = model.predict(vetor)

    results = {}
    for cat in df["len_catogory"].unique():
        subset = df[df["len_catogory"] == cat]
        if not subset.empty:
            X_sub = vectorize.transform(subset["text"])
            y_sub_true = subset["label"]
            y_sub_pred = model.predict(X_sub)
            acc = accuracy_score(y_sub_true, y_sub_pred)
            results[str(cat)] = acc
    
    acc_values = list(results.values())
    max_diff = max(acc_values) - min(acc_values)
    print(f"Acurácia por grupos de tamanho: {results}")
    assert max_diff < 0.2, f"Diferença de acurácia entre grupos muito alta! ({max_diff:.2f})"

def test_sentiment_classification():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    for text, expected in test_cases:
        pred = model.predict(vectorizer.transform([text]))[0]
        print(f"Texto: {text}\nPrevisto: {pred} | Esperado: {expected}\n")
        assert pred == expected