import cv2
import numpy as np
import mediapipe as mp

# ===============================
# 📐 Angle Calculation
# ===============================
def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_val = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ===============================
# Landmark configuration
# ===============================
LEFT_LANDMARKS = [
    'LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE',
    'LEFT_ANKLE', 'LEFT_FOOT_INDEX', 'LEFT_HEEL'
]

RIGHT_LANDMARKS = [
    'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE',
    'RIGHT_ANKLE', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL'
]

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
}


# ===============================
# MediaPipe Pose
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


# ===============================
# Angle Extraction
# ===============================
def extract_angles_from_landmarks(landmarks_dict):
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    left_hip = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']]
    left_knee = [landmarks_dict['LEFT_KNEE_x'], landmarks_dict['LEFT_KNEE_y']]
    left_ankle = [landmarks_dict['LEFT_ANKLE_x'], landmarks_dict['LEFT_ANKLE_y']]
    left_foot_index = [landmarks_dict['LEFT_FOOT_INDEX_x'], landmarks_dict['LEFT_FOOT_INDEX_y']]

    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    right_hip = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']]
    right_knee = [landmarks_dict['RIGHT_KNEE_x'], landmarks_dict['RIGHT_KNEE_y']]
    right_ankle = [landmarks_dict['RIGHT_ANKLE_x'], landmarks_dict['RIGHT_ANKLE_y']]
    right_foot_index = [landmarks_dict['RIGHT_FOOT_INDEX_x'], landmarks_dict['RIGHT_FOOT_INDEX_y']]

    left_vis = landmarks_dict.get("LEFT_KNEE_visibility", 0.0)
    right_vis = landmarks_dict.get("RIGHT_KNEE_visibility", 0.0)
    active_side = "left" if left_vis > right_vis else "right"

    vertical_left = [left_hip[0], left_hip[1] - 1.0]
    vertical_right = [right_hip[0], right_hip[1] - 1.0]

    if active_side == "left":
        knee_angle  = calculate_angle(left_hip, left_knee, left_ankle)
        hip_angle   = calculate_angle(left_shoulder, left_hip, left_knee)
        torso_angle = calculate_angle(left_shoulder, left_hip, vertical_left)
        ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
    else:
        knee_angle  = calculate_angle(right_hip, right_knee, right_ankle)
        hip_angle   = calculate_angle(right_shoulder, right_hip, right_knee)
        torso_angle = calculate_angle(right_shoulder, right_hip, vertical_right)
        ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)

    return [
        active_side,
        knee_angle,
        hip_angle,
        torso_angle,
        ankle_angle
    ]


# ===============================
# analyze_frame (ONNX)
# ===============================
def analyze_frame(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    # Extract landmarks
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = float(lm.x)
        landmarks_dict[f"{name}_y"] = float(lm.y)
        landmarks_dict[f"{name}_visibility"] = float(lm.visibility)

    # Angles
    angles = extract_angles_from_landmarks(landmarks_dict)
    active_side = angles[0]
    knee_angle = angles[1]

    # Features
    feature_vector = [float(x) for x in angles[1:]]

    selection = LEFT_LANDMARKS if active_side == "left" else RIGHT_LANDMARKS
    for name in selection:
        feature_vector.extend([
            landmarks_dict.get(f"{name}_x", 0.0),
            landmarks_dict.get(f"{name}_y", 0.0),
            landmarks_dict.get(f"{name}_visibility", 0.0)
        ])

    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        buffer.append([0.0] * num_features)

    # =========================
    # REP LOGIC (same + good-form flag)
    # =========================
    rep_state.setdefault("prev_angle", None)
    rep_state.setdefault("prev_phase", None)
    rep_state.setdefault("phase", "S1")
    rep_state.setdefault("viable_rep", True)
    rep_state.setdefault("Bottom_ROM_error", False)
    rep_state.setdefault("good_form_flag", False)
    rep_state.setdefault("rep_counter", 0)

    angle = knee_angle

    if angle is not None:
        if rep_state["prev_angle"] is None:
            rep_state["prev_angle"] = angle

        prev_angle = rep_state["prev_angle"]
        prev_phase = rep_state["prev_phase"]

        if angle >= 160 and prev_angle < 160:
            rep_state["Bottom_ROM_error"] = False
            rep_state["phase"] = "S1"
        elif angle <= 90:
            rep_state["phase"] = "S3"
        elif angle < prev_angle and 90 < angle < 160:
            rep_state["phase"] = "S2"
        elif angle > prev_angle and 90 < angle < 160:
            rep_state["phase"] = "S4"

        if prev_phase == "S4" and rep_state["phase"] == "S2":
            rep_state["Bottom_ROM_error"] = True
            rep_state["viable_rep"] = False

        # REP COUNT — ONLY IF GOOD FORM
        if prev_phase == "S4" and rep_state["phase"] == "S1":
            if rep_state["good_form_flag"]:
                rep_state["rep_counter"] += 1

            rep_state["good_form_flag"] = False
            rep_state["viable_rep"] = True

        rep_state["prev_phase"] = rep_state["phase"]
        rep_state["prev_angle"] = angle

    # =========================
    # ONNX INFERENCE
    # =========================
    if len(buffer) >= window_size:

        window = np.array(list(buffer), dtype=np.float32)

        try:
            scaled = scaler.transform(window)
        except Exception as e:
            return {"form_status": f"Scaler Error: {e}", "rep_state": rep_state}

        onnx_input = scaled[np.newaxis, :, :].astype(np.float32)
        input_name = session.get_inputs()[0].name
        recon = session.run(None, {input_name: onnx_input})[0]

        err = float(np.mean((onnx_input - recon) ** 2))

        # Form classification
        if err > threshold:
            status = "BAD FORM!"
            rep_state["good_form_flag"] = False

        elif rep_state["Bottom_ROM_error"]:
            status = "Not Going Low Enough!"
            rep_state["good_form_flag"] = False

        elif angles[3] > 50:  
            status = "Don't Arch Your Back!"
            rep_state["good_form_flag"] = False

        else:
            status = "Good Form"
            rep_state["good_form_flag"] = True

        return {
            "form_status": status,
            "rep_state": rep_state
        }

    return {"form_status": "Analyzing...", "rep_state": rep_state}


# ===============================
# rep_state template
# ===============================
def reset_rep_counter():
    return {
        "rep_counter": 0,
        "prev_angle": None,
        "prev_phase": None,
        "phase": "S1",
        "viable_rep": True,
        "Bottom_ROM_error": False,
        "good_form_flag": False
    }
