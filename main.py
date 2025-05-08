from fastapi import FastAPI, File, UploadFile, Query
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
    file: UploadFile = File(...),
    square_size: int = Query(DEFAULT_SQUARE_SIZE, description="Tamanho do square em pixels"),
):
    logger.info(f"Recebendo arquivo: {file.filename}, square_size={square_size}")

    start_request = datetime.utcnow()
    time_start_total = time.time()

    # Validação da extensão
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".jpge")):
        logger.warning(f"Arquivo inválido recebido: {file.filename}")
        return JSONResponse(
            status_code=400,
            content={"error": "Formato de arquivo inválido. Use jpg, jpeg ou jpge."}
        )

    # Lê e decodifica a imagem
    time_start_read = time.time()
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    time_end_read = time.time()

    if image is None:
        logger.error(f"Falha ao decodificar imagem: {file.filename}")
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
            "filename": file.filename,
            "fileSize": len(image_bytes),
            "dimensions": {
                "width": img_width,
                "height": img_height
            },
            "mimeType": file.content_type,
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
