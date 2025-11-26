# multimodel_api.py  — ONNX + LAZY LOADING + BINARY WS
import base64
import cv2
import numpy as np
import onnxruntime as ort
import joblib
import json
import importlib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"
print(f"🔧 Running ONNX Runtime on: {DEVICE}")

# Your model registry — same as before but with "loaded" flag
EXERCISE_MODELS = {
    "squat": {
        "model_path": "models/squat/squat_model.onnx",
        "scaler_path": "models/squat/squat_pose_scaler.pkl",
        "inference_module": "models.squat.inference",
        "threshold": 0.032,
        "window_size": 10,
        "num_features": 22,
        "seq_len": 10,
        "rep_state": { 'rep_counter': 0, 'prev_angle': None, 'prev_phase': None, 'phase': "S1", 'viable_rep': True, 'Bottom_ROM_error': False },
        "loaded": False
    },
    "push_ups": {
        "model_path": "models/pushup/pushup_model.onnx",
        "scaler_path": "models/pushup/pushup_pose_scaler.pkl",
        "inference_module": "models.pushup.inference",
        "threshold": 0.11,
        "window_size": 10,
        "num_features": 40,
        "seq_len": 10,
        "rep_state": { 'rep_counter': 0, 'prev_angle': None, 'prev_phase': None, 'phase': "P1", 'viable_rep': True, 'Top_ROM_error': False, 'Bottom_ROM_error': False },
        "loaded": False
    },
    "lateral_raises": {
        "model_path": "models/lateral_raises/lateral_raises_model.onnx",
        "scaler_path": "models/lateral_raises/lateral_raises_pose_scaler.pkl",
        "inference_module": "models.lateral_raises.inference",
        "threshold": 0.84,
        "window_size": 10,
        "num_features": 54,
        "seq_len": 10,
        "rep_state": { 'rep_counter': 0, 'prev_angle': None, 'prev_phase': None, 'phase': "LR1", 'viable_rep': True, 'Top_ROM_error': False, 'Bottom_ROM_error': False },
        "loaded": False
    },
    "biceps_curl": {
        "model_path": "models/biceps_curl/biceps_curl_model.onnx",
        "scaler_path": "models/biceps_curl/biceps_pose_scaler.pkl",
        "inference_module": "models.biceps_curl.inference",
        "threshold": 0.017,
        "window_size": 10,
        "num_features": 26,
        "seq_len": 10,
        "rep_state": { 'rep_counter': 0, 'prev_angle': None, 'prev_phase': None, 'phase': "B1", 'viable_rep': True, 'Top_ROM_error': False, 'Bottom_ROM_error': False },
        "loaded": False
    }
}

def load_model_if_needed(model_name: str):
    cfg = EXERCISE_MODELS[model_name]
    if cfg.get("loaded", False):
        return cfg

    print(f"⏳ Loading model lazily: {model_name} ...")
    try:
        inference = importlib.import_module(cfg["inference_module"])
        cfg["analyze_frame"] = inference.analyze_frame

        # ONNX session options - optimization and CPU tuning
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        session = ort.InferenceSession(cfg["model_path"], sess_options=sess_opts, providers=["CPUExecutionProvider"])
        cfg["session"] = session

        scaler = joblib.load(cfg["scaler_path"])
        cfg["scaler"] = scaler

        cfg["loaded"] = True
        print(f"✅ Model '{model_name}' loaded!")
    except Exception as e:
        print(f"❌ Failed to load model '{model_name}': {e}")
    return cfg

@app.get("/")
async def root():
    return {"message": "Backend running with ONNX + LAZY LOADING + BINARY WS 🚀"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket connected")
    websocket_active = True

    try:
        # Expect first message to be a JSON text with model selection { "model": "push_ups" }
        init_msg = await websocket.receive_text()
        try:
            cfg_json = json.loads(init_msg)
        except Exception as e:
            await websocket.send_json({"error": "Expected initial JSON text (e.g. {\"model\":\"push_ups\"})"})
            return

        model_name = cfg_json.get("model")
        if model_name not in EXERCISE_MODELS:
            await websocket.send_json({"error": f"Model '{model_name}' not found"})
            return

        # Lazy load model
        model_cfg = load_model_if_needed(model_name)
        if not model_cfg.get("loaded", False):
            await websocket.send_json({"error": f"Failed to load model '{model_name}' on server"})
            return

        session = model_cfg["session"]
        scaler = model_cfg["scaler"]
        threshold = model_cfg["threshold"]
        analyze_frame = model_cfg["analyze_frame"]
        buffer = deque(maxlen=model_cfg["window_size"])
        rep_state = model_cfg["rep_state"].copy()

        print(f"🎯 USING MODEL: {model_name}")

        # Main loop: accept binary frames or occasional text control messages
        while True:
            msg = await websocket.receive()  # returns dict with 'type' and either 'text' or 'bytes'
            # handle disconnect if client closed
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            if "bytes" in msg and msg["bytes"] is not None:
                # Binary frame received — treat as compressed image bytes (JPEG/PNG)
                frame_bytes = msg["bytes"]
                try:
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        await websocket.send_json({"error": "Invalid frame bytes"})
                        continue

                    # Run model inference (you r existing analyze_frame expects session, scaler, threshold, buffer, window_size, num_features, rep_state)
                    result = analyze_frame(
                        frame=frame,
                        session=session,
                        scaler=scaler,
                        threshold=threshold,
                        buffer=buffer,
                        window_size=model_cfg["window_size"],
                        num_features=model_cfg["num_features"],
                        rep_state=rep_state
                    )
                    # send JSON response
                    await websocket.send_json(result)

                except Exception as e:
                    await websocket.send_json({"error": f"Processing error: {str(e)}"})

            elif "text" in msg and msg["text"] is not None:
                # Control or JSON text (e.g., "reset" or { "command": "reset" })
                text = msg["text"]
                try:
                    payload = json.loads(text)
                except:
                    payload = {"command": text}

                cmd = payload.get("command")
                if cmd == "reset":
                    # reset rep_state / buffer
                    buffer.clear()
                    rep_state = model_cfg["rep_state"].copy()
                    await websocket.send_json({"status": "reset_ok", "rep_state": rep_state})
                elif cmd == "close":
                    await websocket.send_json({"status": "closing"})
                    break
                else:
                    # unknown text message -> echo / ignore
                    await websocket.send_json({"info": "text_received", "payload": payload})

    except WebSocketDisconnect:
        print("⚠️ WebSocket disconnected")
        websocket_active = False

    finally:
        if websocket_active:
            try:
                await websocket.close()
            except:
                pass
        print("🔴 WebSocket closed")
