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
import torch
from transformers import pipeline

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

# --- 🤖 載入 HuggingFace 重型表情模型 ---
print("⏳ 正在載入 HuggingFace 表情分類模型...")
emotion_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    device=-1
)
print("✅ 表情分類模型載入完成！")

# 融合大腦分析邏輯 (來自 test1.txt)
def analyze_emotion(pil_image, shapes_dict):
    try:
        outputs = emotion_classifier(pil_image)
        dl_top_emotion = outputs[0]['label']
        dl_top_score = outputs[0]['score']
    except Exception as e:
        return "normal", f"深度學習異常: {str(e)[:15]}"

    smile = max(shapes_dict.get('mouthSmileLeft', 0), shapes_dict.get('mouthSmileRight', 0))
    brow_down = max(shapes_dict.get('browDownLeft', 0), shapes_dict.get('browDownRight', 0))
    mouth_frown = max(shapes_dict.get('mouthFrownLeft', 0), shapes_dict.get('mouthFrownRight', 0))

    final_tag = "normal"
    if dl_top_emotion == "happy" and dl_top_score > 0.40:
        final_tag = "happiness"
    elif smile > 0.35:
        final_tag = "happiness"
    elif dl_top_emotion == "angry":
        if brow_down < 0.25 and mouth_frown < 0.25:
            final_tag = "normal"
        else:
            final_tag = "anger"
    elif dl_top_emotion in ["sad", "fear", "surprise"]:
        if brow_down > 0.40:
            final_tag = "anger"
        else:
            final_tag = "normal"

    detail_msg = f"AI: {dl_top_emotion}({dl_top_score:.2f}) | 微笑: {smile:.2f} 皺眉: {brow_down:.2f}"
    return final_tag, detail_msg

# 性格風格庫 (來自 test1.txt)
STYLES = ["毒舌但關心的管家", "活潑搞笑的副駕駛", "冷靜幽默的老朋友", "熱血的駕駛教練"]

def generate_ai_suggestion(tag: str, count: int, history_list: list) -> str:
    if not client:
        return "請注意駕駛安全！"
    
    tag_zh = {"happiness": "心情愉快", "anger": "路怒/生氣/暴躁", "fatigue": "閉眼打瞌睡/重度疲勞"}.get(tag, tag)
    chosen_style = random.choice(STYLES)
    history_text = "、".join(history_list[-4:]) if history_list else "無"
    
    prompt = f"""你是一位智慧隨車管家。
目前偵測到駕駛狀態為：【{tag_zh}】（疲勞次數：{count} 次）。
請以「{chosen_style}」的口吻，臨場發揮說一句行車提醒。

嚴格遵守規則：
1. 字數必須在 8 到 12 字以內（適合 2 秒語音播報）。
2. 絕對禁止重複或使用類似這些句子：[{history_text}]。
3. 請直接輸出說話內容，不要附帶引號或解釋。"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=1.2, max_output_tokens=30)
        )
        text = response.text.strip().replace("\n", "").replace('"', '').replace('「', '').replace('」', '')
        return text if text else "行車請注意安全！"
    except Exception as e:
        print(f"Gemini 生成失敗: {e}")
        return "行車請注意安全！"

@app.get("/")
def read_root():
    return {"status": "AI駕駛助理正在運行（舊版多模態全功能重現版）"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    sleep_frames = 0
    fatigue_count = 0
    last_fatigue_time = 0
    lock_until = 0
    recent_suggestions = []

    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data.split(",")[1] if "," in data else data)
            image = Image.open(io.BytesIO(img_bytes))
            frame_np = np.array(image)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            current_time = time.time()
            
            # MediaPipe 偵測
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)
            timestamp_ms = int(current_time * 1000)
            result = face_detector.detect_for_video(mp_image, timestamp_ms)
            
            vision_tag = "normal"
            vision_detail = "面部平靜"
            
            if result.face_blendshapes and len(result.face_blendshapes) > 0:
                blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
                
                # 計算眼睛閉合度
                eye_blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
                eye_blink_right = blendshapes.get("eyeBlinkRight", 0.0)
                avg_blink = (eye_blink_left + eye_blink_right) / 2.0
                
                if avg_blink > 0.5:
                    sleep_frames += 1
                    if sleep_frames >= 3:
                        vision_tag = "fatigue"
                        vision_detail = f"😴 眼睛閉合 (Blink: {avg_blink:.2f})"
                else:
                    sleep_frames = 0
                    # 呼叫 HuggingFace 表情分類器
                    vision_tag, vision_detail = analyze_emotion(image, blendshapes)
            
            # 決策層與提示詞生成
            response_data = {
                "status": "normal",
                "message": "🌿 [環境平靜]",
                "alert_text": "",
                "detail": vision_detail
            }
            
            is_fatigue_alert = (vision_tag == "fatigue" and (current_time - last_fatigue_time > 3.0))
            is_normal_trigger = (vision_tag in ["happiness", "anger"] and current_time > lock_until)
            
            if is_fatigue_alert or is_normal_trigger:
                if vision_tag == "fatigue":
                    fatigue_count += 1
                    last_fatigue_time = current_time
                
                lock_until = current_time + 4.0
                
                alert_msg = generate_ai_suggestion(vision_tag, fatigue_count, recent_suggestions)
                recent_suggestions.append(alert_msg)
                recent_suggestions = recent_suggestions[-6:]
                
                response_data = {
                    "status": "warning",
                    "message": f"🚨 狀態：{vision_tag.upper()}",
                    "alert_text": alert_msg,
                    "detail": vision_detail
                }

            await websocket.send_json(response_data)
            
    except WebSocketDisconnect:
        print("連線已中斷")
