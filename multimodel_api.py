import base64
import cv2
import numpy as np
import torch
import joblib
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
from inference import TransformerAutoencoder, analyze_frame

# ==============================
# ⚙️ FastAPI & CORS Setup
# ==============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# ⚙️ Device & Model Config
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXERCISE_MODELS = {
    "squat": {
        "model_path": "models/squat_model.pth",
        "scaler_path": "models/pose_scaler.pkl",
        "threshold": 0.07,
        "window_size": 30,
        "num_features": 74,
        "seq_len": 30,
    }
}

# ==============================
# 🧠 Load All Models
# ==============================
for name, cfg in EXERCISE_MODELS.items():
    try:
        model = TransformerAutoencoder(cfg["num_features"], cfg["seq_len"])
        model.load_state_dict(torch.load(cfg["model_path"], map_location=DEVICE))
        model.to(DEVICE).eval()
        scaler = joblib.load(cfg["scaler_path"])
        EXERCISE_MODELS[name]["model"] = model
        EXERCISE_MODELS[name]["scaler"] = scaler
        print(f"✅ Loaded model '{name}' successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load model '{name}': {e}")

# ==============================
# 🏠 Root Route
# ==============================
@app.get("/")
async def root():
    return {"message": "Backend running successfully 🚀"}


# ==============================
# 🔄 WebSocket Endpoint
# ==============================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket connected")

    websocket_active = True  # flag to prevent double close

    try:
        # ===== Model Selection =====
        init = await websocket.receive_text()
        cfg = json.loads(init)
        model_name = cfg.get("model", "squat")

        if model_name not in EXERCISE_MODELS:
            await websocket.send_json({"error": f"Model '{model_name}' not found"})
            return

        model_cfg = EXERCISE_MODELS[model_name]
        model = model_cfg["model"]
        scaler = model_cfg["scaler"]
        threshold = model_cfg["threshold"]
        buffer = deque(maxlen=model_cfg["window_size"])

        print(f"🎯 Using model: {model_name}")

        # ===== Main Loop =====
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)
            frame_data = payload.get("frame")

            if not frame_data:
                await websocket.send_json({"error": "No frame data received"})
                continue

            # Remove base64 header if exists
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]

            try:
                frame_bytes = base64.b64decode(frame_data)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"error": "Invalid frame"})
                    continue

                # Analyze frame
                result = analyze_frame(
                    frame,
                    model=model,
                    scaler=scaler,
                    threshold=threshold,
                    device=DEVICE,
                    buffer=buffer,
                    window_size=model_cfg["window_size"],
                    num_features=model_cfg["num_features"],
                )

                await websocket.send_json(result)

            except Exception as e:
                await websocket.send_json({"error": f"Processing error: {str(e)}"})

    except WebSocketDisconnect:
        print("⚠️ WebSocket disconnected by client")
        websocket_active = False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        if websocket_active:
            try:
                await websocket.close()
            except Exception:
                pass
        print("🔴 WebSocket closed safely ✅")
