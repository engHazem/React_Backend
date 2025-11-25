import base64
import cv2
import numpy as np
import torch
import joblib
import json
import importlib    # <── NEW
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

# ==============================
# ⚙️ FastAPI & CORS Setup
# ==============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"
print(f"🔧 Running on: {DEVICE}")

# ==============================
# ⚙️ Model Configurations
# ==============================
EXERCISE_MODELS = {
    "squat": {
        "model_path": "models/squat/squat_model.pth",
        "scaler_path": "models/squat/squat_pose_scaler.pkl",
        "inference_module": "models.squat.inference",
        "threshold": 0.032,
        "window_size": 10,
        "num_features": 22,
        "seq_len": 10,
        "rep_state": {
            'rep_counter': 0,
            'prev_angle': None,
            'prev_phase': None,
            'phase': "S1",
            'viable_rep': True,
            'Bottom_ROM_error': False
        },
        "loaded": False  # <-- REQUIRED
    },

    "push_ups": {
        "model_path": "models/pushup/pushup_model.pth",
        "scaler_path": "models/pushup/pushup_pose_scaler.pkl",
        "inference_module": "models.pushup.inference",
        "threshold": 0.11,
        "window_size": 10,
        "num_features": 40,
        "seq_len": 10,
        "rep_state": {
            'rep_counter': 0,
            'prev_angle': None,
            'prev_phase': None,
            'phase': "P1",
            'viable_rep': True,
            'Top_ROM_error': False,
            'Bottom_ROM_error': False
        },
        "loaded": False
    },

    "lateral_raises": {
        "model_path": "models/lateral_raises/lateral_raises_model.pth",
        "scaler_path": "models/lateral_raises/lateral_raises_pose_scaler.pkl",
        "inference_module": "models.lateral_raises.inference",
        "threshold": 0.84,
        "window_size": 10,
        "num_features": 54,
        "seq_len": 10,
        "rep_state": {
            'rep_counter': 0,
            'prev_angle': None,
            'prev_phase': None,
            'phase': "LR1",
            'viable_rep': True,
            'Top_ROM_error': False,
            'Bottom_ROM_error': False
        },
        "loaded": False
    },

    "biceps_curl": {
        "model_path": "models/biceps_curl/biceps_curl_model.pth",
        "scaler_path": "models/biceps_curl/biceps_pose_scaler.pkl",
        "inference_module": "models.biceps_curl.inference",
        "threshold": 0.017,
        "window_size": 10,
        "num_features": 26,
        "seq_len": 10,
        "rep_state": {
            'rep_counter': 0,
            'prev_angle': None,
            'prev_phase': None,
            'phase': "B1",
            'viable_rep': True,
            'Top_ROM_error': False,
            'Bottom_ROM_error': False
        },
        "loaded": False
    }
}


# ==============================
# 🧠 Load All Models (CPU ONLY)
# ==============================

def load_model_if_needed(model_cfg):
    if model_cfg["loaded"]:
        return  # already loaded

    print(f"⏳ Lazy loading model: {model_cfg}")

    # 1. Load dynamic inference module
    inference = importlib.import_module(model_cfg["inference_module"])
    model_cfg["analyze_frame"] = inference.analyze_frame
    TransformerAutoencoder = inference.TransformerAutoencoder

    # 2. Load the model
    model = TransformerAutoencoder(model_cfg["num_features"], model_cfg["seq_len"])
    model.load_state_dict(torch.load(model_cfg["model_path"], map_location="cpu"))
    model.eval()
    model.to("cpu")

    # 3. Load scaler
    scaler = joblib.load(model_cfg["scaler_path"])

    # 4. Store
    model_cfg["model"] = model
    model_cfg["scaler"] = scaler
    model_cfg["loaded"] = True  # mark as loaded

    print(f"✅ Model loaded successfully: {model_cfg}")

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

    websocket_active = True

    try:
        init = await websocket.receive_text()
        cfg = json.loads(init)
        model_name = cfg.get("model")

        if model_name not in EXERCISE_MODELS:
            await websocket.send_json({"error": f"Model '{model_name}' not found"})
            return

        model_cfg = EXERCISE_MODELS[model_name]
       # 🔥 Load the model ONLY when needed
        load_model_if_needed(model_cfg)

        model = model_cfg["model"]
        scaler = model_cfg["scaler"]

        threshold = model_cfg["threshold"]
        analyze_frame = model_cfg["analyze_frame"]
        buffer = deque(maxlen=model_cfg["window_size"])

        print(f"🎯 Using model: {model_name}")

        #  dynamic rep_state based on model
        rep_state = model_cfg["rep_state"].copy()
        
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)
            frame_data = payload.get("frame")

            if not frame_data:
                await websocket.send_json({"error": "No frame data received"})
                continue

            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]

            try:
                frame_bytes = base64.b64decode(frame_data)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"error": "Invalid frame"})
                    continue

                # 🔥 dynamic inference per model
                result = analyze_frame(
                    frame,
                    model=model,
                    scaler=scaler,
                    threshold=threshold,
                    device="cpu",
                    buffer=buffer,
                    window_size=model_cfg["window_size"],
                    num_features=model_cfg["num_features"],
                    rep_state=rep_state if "rep_state" in model_cfg else None
                )

                await websocket.send_json(result)

            except Exception as e:
                await websocket.send_json({"error": f"Processing error: {str(e)}"})

    except WebSocketDisconnect:
        print("⚠️ WebSocket disconnected")
        websocket_active = False

    finally:
        if websocket_active:
            try: await websocket.close()
            except: pass
        print("🔴 WebSocket closed safely")