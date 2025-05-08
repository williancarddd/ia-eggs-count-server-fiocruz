# Use uma imagem oficial Python
FROM python:3.10-slim

# Instala dependências do sistema para OpenCV e outras libs
RUN apt-get update && apt-get install -y \
  libglib2.0-0 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos do projeto para o container
COPY . /app

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt


# Comando para iniciar o servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
