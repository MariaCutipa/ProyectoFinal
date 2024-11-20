import fitz  # PyMuPDF
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient

# Cargar el modelo, vectorizador y categorías
with open("modelo_habilidades.pkl", "rb") as f:
    modelo = pickle.load(f)
with open("vectorizador.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("categorias.pkl", "rb") as f:
    categorias = pickle.load(f)

# Conectar a MongoDB
client = MongoClient("mongodb://mongo:27017/")
db = client["mi_base_de_datos"]  # Nombre de la base de datos
coleccion = db["usuarios"]  # Nombre de la colección

# Cargar el modelo de lenguaje SpaCy
nlp = spacy.load("es_core_news_sm")


# Preprocesar texto completo
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)


# Leer y clasificar texto completo del PDF
def classify_pdf(file_path):
    # Leer el PDF completo como texto
    with fitz.open(file_path) as pdf:
        full_text = ""
        for page in pdf:
            full_text += page.get_text()

    # Preprocesar todo el texto del PDF
    preprocessed_text = preprocess_text(full_text)

    # Vectorizar el texto preprocesado
    vectorized_text = vectorizer.transform([preprocessed_text])

    # Hacer predicción
    predictions = modelo.predict(vectorized_text)

    # Identificar categorías detectadas
    habilidades_detectadas = [categorias[i] for i, val in enumerate(predictions[0]) if val > 0]

    return habilidades_detectadas


def guardar_habilidades(usuario_id, nombre, email, habilidades):
    usuario = coleccion.find_one({"_id": usuario_id})
    if not usuario:
        usuario = {
            "_id": usuario_id,
            "nombre": nombre,
            "email": email,
            "habilidades": []
        }

    for habilidad in habilidades:
        habilidad_existente = next((h for h in usuario["habilidades"] if h["nombre"] == habilidad), None)
        if habilidad_existente:
            habilidad_existente["puntaje"] += 1
        else:
            usuario["habilidades"].append({"nombre": habilidad, "puntaje": 1})

    coleccion.replace_one({"_id": usuario_id}, usuario, upsert=True)



# Mostrar el vocabulario para depuración (opcional)
vocabulario = vectorizer.get_feature_names_out()

# Procesar un archivo PDF
pdf_path = "avance2.pdf"  # Reemplaza con tu archivo PDF
usuario_id = "usuarioId123"  # ID del usuario
nombre = "Juan Pérez"  # Nombre del usuario
email = "juan.perez@example.com"  # Email del usuario
habilidades = classify_pdf(pdf_path)
guardar_habilidades(usuario_id, nombre, email, habilidades)
print("Habilidades detectadas:", habilidades)
