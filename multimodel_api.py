# multimodel_api.py  — ONNX + LAZY LOADING + BINARY WS + AUTO GPU
import base64
import cv2
import numpy as np
import onnxruntime as ort
import joblib
import json
import importlib
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
from typing import List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 🔥 Auto-detect best ONNX provider
# ---------------------------
def get_preferred_provider_order() -> List[str]:

    available = ort.get_available_providers()
    print("🔍 Available ONNX providers:", available)

    order = []
    # Prefer NVIDIA CUDA
    if "CUDAExecutionProvider" in available:
        order.append("CUDAExecutionProvider")
    # AMD ROCm
    if "ROCMExecutionProvider" in available:
        order.append("ROCMExecutionProvider")
    # Apple CoreML (for Apple Silicon acceleration)
    if "CoreMLExecutionProvider" in available:
        order.append("CoreMLExecutionProvider")
    # TensorRT might be available on some builds
    if "TensorrtExecutionProvider" in available:
        order.append("TensorrtExecutionProvider")
    # Always include CPU as last fallback
    if "CPUExecutionProvider" in available:
        order.append("CPUExecutionProvider")
    else:
        # Should always exist, but just in case
        order.append("CPUExecutionProvider")

    return order

PREFERRED_PROVIDERS = get_preferred_provider_order()
print(f"🧭 Preferred providers order: {PREFERRED_PROVIDERS}")

# ---------------------------
# Model registry + lazy flags
# ---------------------------
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

# ---------------------------
# Helper: create ONNX session with provider fallback
# ---------------------------
def create_session_with_fallback(model_path: str, sess_options: ort.SessionOptions = None):

    last_exception = None
    sess_opts = sess_options if sess_options is not None else ort.SessionOptions()

    for provider in PREFERRED_PROVIDERS:
        try:
            # Try to create session with this single provider; let ONNX Runtime decide fallback internally
            print(f"⏳ Attempting InferenceSession with provider: {provider}")
            # Use provider list with provider first and CPU as fallback
            providers_try = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers_try)
            used_providers = session.get_providers()
            print(f"✅ Loaded session for {model_path} with providers: {used_providers}")
            return session, provider
        except Exception as e:
            last_exception = e
            print(f"⚠️ Failed to create session with provider {provider}: {e}")
            # continue trying next provider

    # If we get here, all attempts failed — raise the last exception
    print("❌ All provider attempts failed. Raising last exception.")
    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError("Failed to create ONNX Runtime session for unknown reasons.")

# ---------------------------
# Lazy loader
# ---------------------------
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
        # Use EXTENDED or ALL depending on your ONNX Runtime version
        try:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        except Exception:
            # In case older/newer API differences
            pass

        # Create session with provider fallback
        try:
            session, used_provider = create_session_with_fallback(cfg["model_path"], sess_options=sess_opts)
            cfg["session"] = session
            cfg["used_provider"] = used_provider
        except Exception as e:
            # If session creation failed entirely, rethrow / track
            print(f"❌ Failed to create ONNX session for model '{model_name}': {e}")
            print(traceback.format_exc())
            cfg["loaded"] = False
            return cfg

        # Load scaler (joblib)
        try:
            scaler = joblib.load(cfg["scaler_path"])
            cfg["scaler"] = scaler
        except Exception as e:
            print(f"⚠️ Failed to load scaler for '{model_name}': {e}")
            cfg["scaler"] = None

        cfg["loaded"] = True
        print(f"✅ Model '{model_name}' loaded (provider: {cfg.get('used_provider')})!")
    except Exception as e:
        print(f"❌ Failed to load model '{model_name}': {e}")
        print(traceback.format_exc())
    return cfg

# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Backend running with ONNX + LAZY LOADING + BINARY WS 🚀"}

# ---------------------------
# WebSocket endpoint (binary frames + json control)
# ---------------------------
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

        session = model_cfg.get("session")
        scaler = model_cfg.get("scaler")
        threshold = model_cfg.get("threshold")
        analyze_frame = model_cfg.get("analyze_frame")
        buffer = deque(maxlen=model_cfg["window_size"])
        rep_state = model_cfg["rep_state"].copy()

        print(f"🎯 USING MODEL: {model_name} (provider: {model_cfg.get('used_provider')})")

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

                    # Run model inference (your existing analyze_frame expects session, scaler, threshold, buffer, window_size, num_features, rep_state)
                    try:
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
                    except Exception as e:
                        # If model inference fails, send a useful error
                        tb = traceback.format_exc()
                        await websocket.send_json({"error": f"Inference error: {str(e)}", "trace": tb})
                        continue

                    # send JSON response
                    await websocket.send_json(result)

                except Exception as e:
                    tb = traceback.format_exc()
                    await websocket.send_json({"error": f"Processing error: {str(e)}", "trace": tb})

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

    except Exception as e:
        print("❌ Unhandled exception in websocket loop:", e)
        print(traceback.format_exc())
        try:
            await websocket.send_json({"error": "server_error", "detail": str(e)})
        except:
            pass

    finally:
        if websocket_active:
            try:
                await websocket.close()
            except:
                pass
        print("🔴 WebSocket closed")
