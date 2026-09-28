import base64
import io
import os
import time
import urllib.request
import random
import cv2
import numpy as np
from PIL import Image

import fastapi
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from google import genai
import torch
from transformers import pipeline

app = FastAPI()

# --- 🧠 設定 Gemini ---
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY) if API_KEY else None

# --- 👁️ 載入 MediaPipe ---
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

# --- 🤖 測試：載入重型深度學習視覺模型 ---
print("⏳ 正在嘗試載入 HuggingFace 重型表情模型...")
try:
    emotion_classifier = pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection",
        device=-1
    )
    print("✅ 重型表情模型載入成功！")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")

@app.get("/")
def read_root():
    return {"status": "AI 駕駛助理運行中（重型模型測試版）"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data.split(",")[1] if "," in data else data)
            image = Image.open(io.BytesIO(img_bytes))
            
            # 測試重型模型推論
            res = emotion_classifier(image)
            top_emotion = res[0]['label']
            
            await websocket.send_json({"status": "ok", "emotion": top_emotion})
    except WebSocketDisconnect:
        print("連線已中斷")
