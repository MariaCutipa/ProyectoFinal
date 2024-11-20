# Imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos de tu proyecto al contenedor
COPY . /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Descargar el modelo de SpaCy durante la construcción
RUN python -m spacy download es_core_news_sm

# Exponer el puerto 5000 (o el que tu app utilice)
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
