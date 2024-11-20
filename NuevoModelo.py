import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# Cargar y procesar datos
datos = pd.read_csv("nuevo.csv", names=["texto", "categoria"])

# Preprocesamiento con SpaCy
nlp = spacy.load("es_core_news_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

datos['texto_procesado'] = datos['texto'].apply(preprocess_text)

# Vectorización de texto
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 3))
X = vectorizer.fit_transform(datos['texto_procesado'])

# One-Hot Encoding de categorías
y = pd.get_dummies(datos['categoria'])

# Convertir y a NumPy para compatibilidad con RandomOverSampler
y_numpy = y.values  # Convertir etiquetas a matriz NumPy

# Balancear el conjunto de datos
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y_numpy)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Modelo supervisado mejorado
modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

# Entrenar Word2Vec para expansión semántica
sentences = [text.split() for text in datos['texto_procesado']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Predicciones y evaluación
y_pred = modelo.predict(X_test)
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred, zero_division=0))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Exact Match Accuracy:", accuracy_score(y_test, y_pred))

# Guardar modelo, vectorizador y categorías
with open("modelo_habilidades.pkl", "wb") as f:
    pickle.dump(modelo, f)
with open("vectorizador.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("categorias.pkl", "wb") as f:
    pickle.dump(y.columns.tolist(), f)
with open("w2v_model.pkl", "wb") as f:
    pickle.dump(w2v_model, f)

# Función para predicción con expansión semántica
def predict_with_semantics(text, modelo, vectorizer, categorias, w2v_model):
    # Preprocesar texto
    preprocessed_text = preprocess_text(text)

    # Expansión semántica
    expanded_words = []
    for word in preprocessed_text.split():
        try:
            similar_words = [w for w, _ in w2v_model.wv.most_similar(word, topn=3)]
            expanded_words.extend(similar_words)
        except KeyError:
            pass
    expanded_text = ' '.join(preprocessed_text.split() + expanded_words)

    # Vectorizar el texto expandido
    vectorized_text = vectorizer.transform([expanded_text])

    # Predicción supervisada
    predictions = modelo.predict(vectorized_text)[0]

    # Convertir a categorías
    detected_categories = [categorias[i] for i, val in enumerate(predictions) if val > 0]

    return detected_categories

# Probar predicción con un texto de ejemplo
example_text = "Se implementaron diseños de interfaces responsivas con Bootstrap y pruebas de usabilidad para garantizar la mejor experiencia de usuario."
with open("modelo_habilidades.pkl", "rb") as f:
    loaded_model = pickle.load(f)
with open("vectorizador.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)
with open("categorias.pkl", "rb") as f:
    loaded_categories = pickle.load(f)
with open("w2v_model.pkl", "rb") as f:
    loaded_w2v_model = pickle.load(f)

predicted_categories = predict_with_semantics(example_text, loaded_model, loaded_vectorizer, loaded_categories, loaded_w2v_model)
print("Categorías detectadas:", predicted_categories)