import base64
import io
import os
import time
import urllib.request
import cv2
import numpy as np
from PIL import Image

import fastapi
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from google import genai
from google.genai import types

app = FastAPI()

# 初始化 Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6KPO6OzhHIH-XCFPFhTzNdbXmHwI-W3EHbbzzT-vngsyw")
client = genai.Client(api_key=API_KEY) if API_KEY else None

# 自動下載並初始化 MediaPipe Face Landmarker
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        MODEL_PATH
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.3,
    output_face_blendshapes=True
)
face_detector = vision.FaceLandmarker.create_from_options(options)

@app.get("/")
def read_root():
    return {"status": "AI Driver Assistant Backend is Running"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    fatigue_count = 0
    
    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data.split(",")[1] if "," in data else data)
            image = Image.open(io.BytesIO(img_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)
            result = face_detector.detect_for_video(mp_image, timestamp_ms)
            
            response_data = {"status": "normal", "message": "監控中", "alert_text": ""}
            
            if result.face_landmarks:
                pass

            await websocket.send_json(response_data)
            
    except WebSocketDisconnect:
        print("連線已中斷")
