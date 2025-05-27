# Imagen base
FROM python:3.10

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos al contenedor
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto donde correrá FastAPI
EXPOSE 8000

# Comando para correr el servidor
CMD ["uvicorn", "api_localizacion:app", "--host", "0.0.0.0", "--port", "8000"]
