# multimodel_api.py
import base64
import cv2
import numpy as np
import json
import importlib
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import joblib
import math
import os

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔧 Starting backend (ONNX runtime)")

# -----------------------------
# Model configurations (ONNX paths + scalers)
# Make sure .onnx files exist at these paths
# -----------------------------
EXERCISE_MODELS = {
    "squat": {
        "onnx_path": "models/squat/squat_model.onnx",
        "scaler_path": "models/squat/squat_pose_scaler.pkl",
        "threshold": 0.032,
        "window_size": 10,
        "num_features": 22,
        "seq_len": 10,
        "rep_state": {
            "rep_counter": 0,
            "prev_angle": None,
            "prev_phase": None,
            "phase": "S1",
            "viable_rep": True,
            "Bottom_ROM_error": False,
            "Top_ROM_error": False,
        },
        "loaded": False,
    },
    "push_ups": {
        "onnx_path": "models/pushup/pushup_model.onnx",
        "scaler_path": "models/pushup/pushup_pose_scaler.pkl",
        "threshold": 0.11,
        "window_size": 10,
        "num_features": 40,
        "seq_len": 10,
        "rep_state": {
            "rep_counter": 0,
            "prev_angle": None,
            "prev_phase": None,
            "phase": "P1",
            "viable_rep": True,
            "Bottom_ROM_error": False,
            "Top_ROM_error": False,
        },
        "loaded": False,
    },
    "lateral_raises": {
        "onnx_path": "models/lateral_raises/lateral_raises_model.onnx",
        "scaler_path": "models/lateral_raises/lateral_raises_pose_scaler.pkl",
        "threshold": 0.84,
        "window_size": 10,
        "num_features": 54,
        "seq_len": 10,
        "rep_state": {
            "rep_counter": 0,
            "prev_angle": None,
            "prev_phase": None,
            "phase": "LR1",
            "viable_rep": True,
            "Bottom_ROM_error": False,
            "Top_ROM_error": False,
        },
        "loaded": False,
    },
    "biceps_curl": {
        "onnx_path": "models/biceps_curl/biceps_curl_model.onnx",
        "scaler_path": "models/biceps_curl/biceps_pose_scaler.pkl",
        "threshold": 0.017,
        "window_size": 10,
        "num_features": 26,
        "seq_len": 10,
        "rep_state": {
            "rep_counter": 0,
            "prev_angle": None,
            "prev_phase": None,
            "phase": "B1",
            "viable_rep": True,
            "Bottom_ROM_error": False,
            "Top_ROM_error": False,
        },
        "loaded": False,
    },
}

# -----------------------------
# MediaPipe setup (global)
# -----------------------------
import mediapipe as mp

mp_pose = mp.solutions.pose
# instantiate a single Pose object (thread-safety: onnx serving is single-threaded per connection; this is fine)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Landmark mapping used by feature extractor
LANDMARK_INDICES = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
}

LEFT_LANDMARKS = [
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
    "LEFT_PINKY",
    "LEFT_INDEX",
    "LEFT_THUMB",
    "LEFT_HIP",
]

RIGHT_LANDMARKS = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "RIGHT_PINKY",
    "RIGHT_INDEX",
    "RIGHT_THUMB",
    "RIGHT_HIP",
]


# -----------------------------
# Helper math functions
# -----------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def extract_angles_from_landmarks(landmarks_dict):
    left_shoulder = [landmarks_dict["LEFT_SHOULDER_x"], landmarks_dict["LEFT_SHOULDER_y"]]
    left_elbow = [landmarks_dict["LEFT_ELBOW_x"], landmarks_dict["LEFT_ELBOW_y"]]
    left_wrist = [landmarks_dict["LEFT_WRIST_x"], landmarks_dict["LEFT_WRIST_y"]]
    left_hip = [landmarks_dict["LEFT_HIP_x"], landmarks_dict["LEFT_HIP_y"]]

    right_shoulder = [landmarks_dict["RIGHT_SHOULDER_x"], landmarks_dict["RIGHT_SHOULDER_y"]]
    right_elbow = [landmarks_dict["RIGHT_ELBOW_x"], landmarks_dict["RIGHT_ELBOW_y"]]
    right_wrist = [landmarks_dict["RIGHT_WRIST_x"], landmarks_dict["RIGHT_WRIST_y"]]
    right_hip = [landmarks_dict["RIGHT_HIP_x"], landmarks_dict["RIGHT_HIP_y"]]

    LeftElbow_vis = landmarks_dict.get("LEFT_ELBOW_visibility", 0)
    RightElbow_vis = landmarks_dict.get("RIGHT_ELBOW_visibility", 0)
    active_arm = "left" if LeftElbow_vis > RightElbow_vis else "right"

    vertical_point_left = [left_shoulder[0], left_shoulder[1] - 1]
    vertical_point_right = [right_shoulder[0], right_shoulder[1] - 1]

    if active_arm == "left":
        elbow_fexion_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        torso_lean_angle = calculate_angle(left_hip, left_shoulder, vertical_point_left) if left_hip[0] else 0.0
        upper_arm_torso_angle = calculate_angle(left_hip, left_shoulder, left_elbow) if left_hip[0] else 0.0
        left_index = [left_wrist[0] + (left_wrist[0] - left_elbow[0]), left_wrist[1] + (left_wrist[1] - left_elbow[1])]
        wrist_angle = calculate_angle(left_elbow, left_wrist, left_index)
        forearm_vertical_angle = calculate_angle(vertical_point_left, left_elbow, left_wrist)
    else:
        elbow_fexion_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        torso_lean_angle = calculate_angle(right_hip, right_shoulder, vertical_point_right) if right_hip[0] else 0.0
        upper_arm_torso_angle = calculate_angle(right_hip, right_shoulder, right_elbow) if right_hip[0] else 0.0
        right_index = [right_wrist[0] + (right_wrist[0] - right_elbow[0]), right_wrist[1] + (right_wrist[1] - right_elbow[1])]
        wrist_angle = calculate_angle(right_elbow, right_wrist, right_index)
        forearm_vertical_angle = calculate_angle(vertical_point_right, right_elbow, right_wrist)

    return [
        active_arm,
        elbow_fexion_angle,
        forearm_vertical_angle,
        wrist_angle,
        upper_arm_torso_angle,
        torso_lean_angle,
    ]


# -----------------------------
# Lazy loader: load ONNX session + scaler on demand
# -----------------------------
def make_session(onnx_path):
    # Tune session options if needed
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 1) // 2)
    # so.inter_op_num_threads = 1
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    return sess


def load_model_if_needed(model_cfg):
    if model_cfg.get("loaded"):
        return

    onnx_path = model_cfg.get("onnx_path")
    scaler_path = model_cfg.get("scaler_path")

    if not onnx_path or not scaler_path:
        raise RuntimeError("Model configuration missing onnx_path or scaler_path")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler (pkl) not found: {scaler_path}")

    print(f"⏳ Loading ONNX model: {onnx_path}")
    session = make_session(onnx_path)
    scaler = joblib.load(scaler_path)

    model_cfg["session"] = session
    model_cfg["scaler"] = scaler
    model_cfg["loaded"] = True
    print("✅ ONNX session + scaler loaded")


# -----------------------------
# Core analyze logic (ONNX version)
# -----------------------------
def analyze_frame_onnx(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):
    """
    Extract features from frame, maintain buffer, run ONNX session when buffer full,
    update rep_state, and return result dict.
    """

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    # Extract landmark dict
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = float(lm.x)
        landmarks_dict[f"{name}_y"] = float(lm.y)
        landmarks_dict[f"{name}_visibility"] = float(lm.visibility)

    # Build features
    angles = extract_angles_from_landmarks(landmarks_dict)
    active_arm = angles[0]
    elbow_flexion_angle = angles[1]

    feature_vector = []
    # add numeric angles (skip active_arm string)
    feature_vector.extend([float(x) for x in angles[1:]])  # 5 angles

    selection = LEFT_LANDMARKS if active_arm == "left" else RIGHT_LANDMARKS
    for name in selection:
        feature_vector.extend(
            [
                float(landmarks_dict.get(f"{name}_x", 0.0)),
                float(landmarks_dict.get(f"{name}_y", 0.0)),
                float(landmarks_dict.get(f"{name}_visibility", 0.0)),
            ]
        )

    # Validate feature length
    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        # If mismatch, pad with zeros and push (avoid crash)
        print(f"Warning: Expected {num_features} features, got {len(feature_vector)}")
        buffer.append([0.0] * num_features)

    # --- Rep counting (same logic you had) ---
    angle = elbow_flexion_angle
    if angle is not None:
        if rep_state.get("prev_angle") is None:
            rep_state["prev_angle"] = angle

        # Phase detection
        prev_angle = rep_state.get("prev_angle", angle)
        prev_phase = rep_state.get("prev_phase")
        if angle >= 160 and prev_angle < 160:
            rep_state["Top_ROM_error"] = False
            rep_state["Bottom_ROM_error"] = False
            rep_state["phase"] = "B1"
        elif angle <= 60:
            rep_state["phase"] = "B3"
        elif angle < prev_angle and angle < 160 and angle > 60:
            rep_state["phase"] = "B2"
        elif angle > prev_angle and angle < 160 and angle > 60:
            rep_state["phase"] = "B4"

        # ROM checks
        if prev_phase is not None:
            if rep_state["phase"] == "B2" and prev_phase == "B4":
                rep_state["viable_rep"] = False
                rep_state["Bottom_ROM_error"] = True
        if rep_state["phase"] == "B4" and prev_phase == "B2":
            rep_state["viable_rep"] = False
            rep_state["Top_ROM_error"] = True

        # Rep detection
        if prev_phase == "B4" and rep_state["phase"] == "B1":
            if rep_state.get("viable_rep", True):
                rep_state["rep_counter"] = rep_state.get("rep_counter", 0) + 1
            rep_state["viable_rep"] = True

        rep_state["prev_phase"] = rep_state["phase"]
        rep_state["prev_angle"] = angle

    # Run ONNX model when buffer is full
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32)  # shape (window_size, features)
        try:
            scaled = scaler.transform(window)  # shape (window_size, features)
        except Exception as e:
            return {"form_status": f"Scaler Error: {e}", "rep_state": rep_state}

        # ONNX expects shape (batch, seq_len, features)
        onnx_input = scaled[np.newaxis, :, :].astype(np.float32)

        # Run ONNX
        ort_inputs = {session.get_inputs()[0].name: onnx_input}
        ort_outs = session.run(None, ort_inputs)
        recon = ort_outs[0]  # shape (1, seq_len, features)

        # compute reconstruction error (mean squared error)
        err = float(np.mean((onnx_input - recon) ** 2))

        # Decide form status
        if err > threshold:
            rep_state["viable_rep"] = False
            status = "POOR FORM!"
        elif rep_state.get("Bottom_ROM_error"):
            status = "Extend your arms more!"
        elif rep_state.get("Top_ROM_error"):
            status = "Contract your arms more!"
        else:
            status = "Good Form"

        return {
            "form_status": status,
            "reconstruction_error": err,
            "rep_state": rep_state,
        }

    return {"form_status": "Analyzing...", "rep_state": rep_state}


# -----------------------------
# Root route
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Backend running successfully 🚀"}


# -----------------------------
# WebSocket endpoint (keeps base64 logic)
# -----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket connected")

    try:
        # first message: expect JSON { "model": "biceps_curl" }
        init_msg = await websocket.receive_text()
        cfg = json.loads(init_msg)
        model_name = cfg.get("model")
        if not model_name or model_name not in EXERCISE_MODELS:
            await websocket.send_json({"error": "Model not found"})
            return

        model_cfg = EXERCISE_MODELS[model_name]
        load_model_if_needed(model_cfg)

        session = model_cfg["session"]
        scaler = model_cfg["scaler"]
        threshold = model_cfg["threshold"]
        window_size = model_cfg["window_size"]
        num_features = model_cfg["num_features"]

        buffer = deque(maxlen=window_size)
        rep_state = model_cfg["rep_state"].copy()

        print(f"🎯 Using model: {model_name}")

        # main loop: receive base64 JSON messages { "frame": "data:image/jpeg;base64,..." }
        while True:
            msg = await websocket.receive_text()
            try:
                payload = json.loads(msg)
            except Exception:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            frame_data = payload.get("frame")
            if not frame_data:
                await websocket.send_json({"error": "No frame provided"})
                continue

            # strip data uri header if present
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]

            try:
                frame_bytes = base64.b64decode(frame_data)
            except Exception:
                await websocket.send_json({"error": "Invalid base64 frame"})
                continue

            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "Invalid image bytes"})
                continue

            # Run analysis using ONNX session
            try:
                result = analyze_frame_onnx(
                    frame,
                    session,
                    scaler,
                    threshold,
                    buffer,
                    window_size,
                    num_features,
                    rep_state,
                )
            except Exception as e:
                await websocket.send_json({"error": f"Processing error: {str(e)}"})
                continue

            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("⚠️ WebSocket disconnected")
    finally:
        print("🔴 WebSocket closed safely")
