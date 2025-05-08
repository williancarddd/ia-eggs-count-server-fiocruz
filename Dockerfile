# ==============================
# Stage 1: Build dependencies
# ==============================
FROM python:3.10-slim AS build

WORKDIR /app

# Instala dependências do sistema para compilação e OpenCV
RUN apt-get update && apt-get install -y \
  build-essential gcc libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
  && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependências no diretório temporário
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copia o código para o container
COPY . .

# ==============================
# Stage 2: Runtime
# ==============================
FROM python:3.10-slim AS runtime

WORKDIR /app

# Instala somente libs necessárias para rodar OpenCV (mínimo possível)
RUN apt-get update && apt-get install -y \
  libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
  && rm -rf /var/lib/apt/lists/*

# Copia pacotes instalados da Stage 1
COPY --from=build /install /usr/local

# Copia o código fonte
COPY --from=build /app /app

# Comando para iniciar o servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
