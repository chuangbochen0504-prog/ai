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
from google.genai import types

app = FastAPI()

# --- 🧠 設定 Gemini 大腦 ---
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY) if API_KEY else None
MODEL_NAME = "gemini-2.5-flash"

# --- 👁️ 載入 MediaPipe 視覺引擎 ---
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

# 性格風格庫 (來自舊版)
STYLES = ["毒舌但關心的管家", "活潑搞笑的副駕駛", "冷靜幽默的老朋友", "熱血的駕駛教練"]

def generate_ai_suggestion(fatigue_count: int, history_list: list) -> str:
    """調用 Gemini 生成動態提醒詞 (來自舊版語法)"""
    if not client:
        return "檢測到閉眼疲勞，請注意駕駛安全！"
    
    chosen_style = random.choice(STYLES)
    history_text = "、".join(history_list[-4:]) if history_list else "無"
    
    prompt = f"""你是一位智慧隨車管家。
目前偵測到駕駛狀態為：【閉眼打瞌睡/重度疲勞】（疲勞次數：{fatigue_count} 次）。
請以「{chosen_style}」的口吻，臨場發揮說一句行車提醒。

嚴格遵守規則：
1. 字數必須在 8 到 12 字以內（適合 2 秒語音播報）。
2. 絕對禁止重複或使用類似這些句子：[{history_text}]。
3. 請直接輸出說話內容，不要附帶引號或解釋。"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=1.2,
                max_output_tokens=30
            )
        )
        text = response.text.strip().replace("\n", "").replace('"', '').replace('「', '').replace('」', '')
        return text if text else "注意安全，稍微休息一下吧！"
    except Exception as e:
        print(f"Gemini 生成失敗: {e}")
        return "檢測到閉眼疲勞，請注意駕駛安全！"

@app.get("/")
def read_root():
    return {"status": "AI駕駛助理正在運行"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 狀態變數 (來自舊版邏輯)
    sleep_frames = 0
    fatigue_count = 0
    last_fatigue_time = 0
    recent_suggestions = []

    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data.split(",")[1] if "," in data else data)
            image = Image.open(io.BytesIO(img_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # MediaPipe 偵測
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)
            result = face_detector.detect_for_video(mp_image, timestamp_ms)
            
            current_time = time.time()
            response_data = {
                "status": "normal",
                "message": "🌿 [環境平靜]",
                "alert_text": "",
                "fatigue_count": fatigue_count
            }
            
            if result.face_blendshapes and len(result.face_blendshapes) > 0:
                blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
                
                # 計算雙眼閉合度
                eye_blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
                eye_blink_right = blendshapes.get("eyeBlinkRight", 0.0)
                avg_blink = (eye_blink_left + eye_blink_right) / 2.0
                
                # 閉眼判斷 (EAR / Blendshape)
                if avg_blink > 0.5:
                    sleep_frames += 1
                else:
                    sleep_frames = 0
                
                # 連續閉眼超過 3 幀，且距離上次觸發超過 3 秒
                if sleep_frames >= 3 and (current_time - last_fatigue_time > 3.0):
                    fatigue_count += 1
                    last_fatigue_time = current_time
                    
                    # 呼叫 Gemini 生成專屬台詞
                    alert_msg = generate_ai_suggestion(fatigue_count, recent_suggestions)
                    recent_suggestions.append(alert_msg)
                    recent_suggestions = recent_suggestions[-6:]
                    
                    response_data = {
                        "status": "warning",
                        "message": f"😴 [狀態：想睡覺] 累計 {fatigue_count} 次",
                        "alert_text": alert_msg,
                        "fatigue_count": fatigue_count
                    }

            await websocket.send_json(response_data)
            
    except WebSocketDisconnect:
        print("連線已中斷")
