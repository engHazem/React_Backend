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
        "window_size": 30,
        "num_features": 22,
        "seq_len": 30,
        "rep_state": {
        'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "S1",
        'viable_rep': True,
        'Bottom_ROM_error': False
    }
    },
    "push_ups": {
        "model_path": "models/pushup/pushup_model.pth",
        "scaler_path": "models/pushup/pushup_pose_scaler.pkl",
        "inference_module": "models.pushup.inference",
        "threshold": 0.11,
        "window_size": 30,
        "num_features": 40,
        "seq_len": 30,
        "rep_state": { 'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "P1",
        'viable_rep': True,
        'Top_ROM_error': False,
        'Bottom_ROM_error': False
    }
    },
    "lateral_raises": {
        "model_path": "models/lateral_raises/lateral_raises_model.pth",
        "scaler_path": "models/lateral_raises/lateral_raises_pose_scaler.pkl",
        "inference_module": "models.lateral_raises.inference",
        "threshold": 0.84,
        "window_size": 30,
        "num_features": 54,
        "seq_len": 30,
        "rep_state": { 'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "LR1",
        'viable_rep': True,
        'Top_ROM_error': False,
        'Bottom_ROM_error': False
    }
    },
    "biceps_curl": {
        "model_path": "models/biceps_curl/biceps_curl_model.pth",
        "scaler_path": "models/biceps_curl/pushup_pose_scaler.pkl",
        "inference_module": "models.biceps_curl.inference",
        "threshold": 0.017,
        "window_size": 30,
        "num_features": 26,
        "seq_len": 30,
        "rep_state": {
        'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "B1",
        'viable_rep': True,
        'Top_ROM_error': False,
        'Bottom_ROM_error': False
        }
    }
}

# ==============================
# 🧠 Load All Models (CPU ONLY)
# ==============================

for name, cfg in EXERCISE_MODELS.items():
    try:
        # Load dynamic inference module
        inference = importlib.import_module(cfg["inference_module"])
        TransformerAutoencoder = inference.TransformerAutoencoder
        EXERCISE_MODELS[name]["analyze_frame"] = inference.analyze_frame

        # Load model
        model = TransformerAutoencoder(cfg["num_features"], cfg["seq_len"])
        model.load_state_dict(torch.load(cfg["model_path"], map_location="cpu"))
        model.to("cpu").eval()

        scaler = joblib.load(cfg["scaler_path"])

        EXERCISE_MODELS[name]["model"] = model
        EXERCISE_MODELS[name]["scaler"] = scaler

        print(f"✅ Loaded model '{name}' + module {cfg['inference_module']}")

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

    websocket_active = True

    try:
        init = await websocket.receive_text()
        cfg = json.loads(init)
        model_name = cfg.get("model")

        if model_name not in EXERCISE_MODELS:
            await websocket.send_json({"error": f"Model '{model_name}' not found"})
            return

        model_cfg = EXERCISE_MODELS[model_name]
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
