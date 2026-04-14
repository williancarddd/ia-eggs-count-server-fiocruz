from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List, Dict
from ultralytics import YOLO
from skimage import exposure
import cv2
import numpy as np
import uvicorn
import logging
import platform
import psutil
import hashlib
import time
from urllib.request import Request as UrlRequest, urlopen
from urllib.parse import urlparse

# Configurar logger
logger = logging.getLogger("egg-counter-api")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configurações iniciais
DEFAULT_SQUARE_SIZE = 254
MODEL_PATH = "./best-train2.onnx"

# Carrega o modelo YOLO
model = YOLO(MODEL_PATH)
logger.info(f"Modelo YOLO carregado: {MODEL_PATH}")

app = FastAPI()


def response_422(message: str):
    logger.warning(f"422 detect_objects: {message}")
    return JSONResponse(
        status_code=422,
        content={"error": message}
    )

def normalize_square(square: np.ndarray) -> np.ndarray:
    """Aplica o mesmo filtro de normalização (gamma correction)."""
    return exposure.adjust_gamma(square, gamma=1.5)

def predict_on_square(square: np.ndarray) -> List[Dict[str, int]]:
    """
    Executa a inferência em um square e retorna lista de bounding boxes (x1, y1, x2, y2).
    """
    results = model(square, verbose=False)
    boxes = results[0].boxes
    bounding_boxes = []
    if boxes is not None and boxes.xyxy is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bounding_boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
    return bounding_boxes

@app.post("/detect")
async def detect_objects(
    request: Request,
    square_size: int = Query(DEFAULT_SQUARE_SIZE, description="Tamanho do square em pixels"),
):
    content_type = request.headers.get("content-type", "")
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Request detect_objects from={client_host} content_type={content_type} square_size={square_size}")
    incoming_filename = "remote-image.jpg"

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception as exc:
            logger.warning(f"Falha ao ler JSON: {exc}")
            return response_422("Payload JSON inválido.")

        logger.info(f"Payload JSON keys={list(payload.keys())}")
        image_url = payload.get("imageUrl")
        if not image_url:
            return response_422("Campo imageUrl é obrigatório quando enviar JSON.")

        parsed_url = urlparse(image_url)
        if parsed_url.scheme not in ("http", "https"):
            return response_422("imageUrl deve usar protocolo http/https.")

        logger.info(f"Baixando imageUrl host={parsed_url.netloc} path={parsed_url.path}")

        try:
            req = UrlRequest(image_url, headers={"User-Agent": "ia-eggs-count-server/1.0"})
            with urlopen(req, timeout=30) as response:
                image_bytes = response.read()
                incoming_mime = response.headers.get("Content-Type", "image/jpeg")
                logger.info(
                    f"Download concluído status={response.status} bytes={len(image_bytes)} mime={incoming_mime}"
                )
            incoming_filename = parsed_url.path.split("/")[-1] or incoming_filename
        except Exception as exc:
            logger.warning(f"Erro no download da imageUrl: {exc}")
            return response_422("Não foi possível baixar a imagem da URL informada.")
    else:
        form_data = await request.form()
        logger.info(f"Payload form keys={list(form_data.keys())}")
        file = form_data.get("file")

        if file is None or not hasattr(file, "read"):
            return response_422("Envie file multipart ou imageUrl em JSON.")

        incoming_filename = getattr(file, "filename", incoming_filename)
        incoming_mime = getattr(file, "content_type", "application/octet-stream")
        image_bytes = await file.read()
        logger.info(f"Arquivo multipart recebido filename={incoming_filename} bytes={len(image_bytes)} mime={incoming_mime}")

    logger.info(f"Recebendo arquivo: {incoming_filename}, square_size={square_size}")

    start_request = datetime.utcnow()
    time_start_total = time.time()

    # Validação da extensão
    if not incoming_filename.lower().endswith((".jpg", ".jpeg", ".jpge", ".png")):
        logger.warning(f"Arquivo inválido recebido: {incoming_filename}")
        return JSONResponse(
            status_code=400,
            content={"error": "Formato de arquivo inválido. Use jpg, jpeg, jpge ou png."}
        )

    # Lê e decodifica a imagem
    time_start_read = time.time()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    time_end_read = time.time()

    if image is None:
        logger.error(f"Falha ao decodificar imagem: {incoming_filename}")
        return JSONResponse(
            status_code=400,
            content={"error": "Não foi possível decodificar a imagem."}
        )

    img_height, img_width, channels = image.shape
    logger.info(f"Imagem carregada: {img_width}x{img_height}px, canais={channels}")

    padded_height = ((img_height + square_size - 1) // square_size) * square_size
    padded_width = ((img_width + square_size - 1) // square_size) * square_size

    # Padding para não perder nenhum pedaço
    padded_image = np.zeros((padded_height, padded_width, channels), dtype=image.dtype)
    padded_image[:img_height, :img_width, :] = image

    all_bounding_boxes: List[Dict[str, int]] = []

    # Inferência por square
    logger.info("Iniciando inferência por square...")
    time_start_inference = time.time()
    for y in range(0, padded_height, square_size):
        for x in range(0, padded_width, square_size):
            square = padded_image[y:y + square_size, x:x + square_size]
            processed_square = normalize_square(square)
            bounding_boxes = predict_on_square(processed_square)

            # Ajusta para coordenadas absolutas
            for bbox in bounding_boxes:
                abs_bbox = {
                    "x1": bbox["x1"] + x,
                    "y1": bbox["y1"] + y,
                    "x2": bbox["x2"] + x,
                    "y2": bbox["y2"] + y
                }
                all_bounding_boxes.append(abs_bbox)
    time_end_inference = time.time()

    total_objects = len(all_bounding_boxes)
    total_squares = (padded_height // square_size) * (padded_width // square_size)
    avg_objects = total_objects / total_squares if total_squares > 0 else 0

    logger.info(f"Total de objetos detectados: {total_objects}")
    logger.info(f"Inferência concluída em {round((time_end_inference - time_start_inference) * 1000, 2)}ms")

    # Sistema e arquivo info
    file_hash_md5 = hashlib.md5(image_bytes).hexdigest()
    mem_info = psutil.virtual_memory()

    end_request = datetime.utcnow()
    time_end_total = time.time()

    logger.info(f"Processamento completo em {round((time_end_total - time_start_total) * 1000, 2)}ms")

    result = {
        "startTime": start_request.isoformat(),
        "endTime": end_request.isoformat(),
        "timing": {
            "readTimeMs": round((time_end_read - time_start_read) * 1000, 2),
            "inferenceTimeMs": round((time_end_inference - time_start_inference) * 1000, 2),
            "totalTimeMs": round((time_end_total - time_start_total) * 1000, 2),
        },
        "image": {
            "filename": incoming_filename,
            "fileSize": len(image_bytes),
            "dimensions": {
                "width": img_width,
                "height": img_height
            },
            "mimeType": incoming_mime,
            "hashMD5": file_hash_md5
        },
        "inferenceStats": {
            "totalObjects": total_objects,
            "totalSquares": total_squares,
            "averageObjectsPerSquare": round(avg_objects, 2),
        },
        "system": {
            "host": platform.node(),
            "cpu": platform.processor(),
            "numThreads": psutil.cpu_count(logical=True),
            "totalRAM_MB": mem_info.total // (1024 * 1024),
            "usedRAM_MB": mem_info.used // (1024 * 1024)
        },
        "model": {
            "versionYolo": YOLO._version,
            "modelPath": MODEL_PATH,
            "pythonVersion": platform.python_version()
        },
        "parameters": {
            "squareSize": square_size
        },
        "objects": all_bounding_boxes,
        "totalObjects": total_objects,
    }

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
