import cv2
import numpy as np
import mediapipe as mp
import joblib
import onnxruntime as ort

# ===============================
# 📐 Angle Calculation
# ===============================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ===============================
# 📍 MediaPipe Landmarks
# ===============================
LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24
}

LEFT_LANDMARKS = [
    'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
    'LEFT_PINKY', 'LEFT_INDEX', 'LEFT_THUMB', 'LEFT_HIP'
]

RIGHT_LANDMARKS = [
    'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
    'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB', 'RIGHT_HIP'
]

# ===============================
# 🧠 Extract Angles
# ===============================
def extract_angles_from_landmarks(landmarks_dict):
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    left_elbow = [landmarks_dict['LEFT_ELBOW_x'], landmarks_dict['LEFT_ELBOW_y']]
    left_wrist = [landmarks_dict['LEFT_WRIST_x'], landmarks_dict['LEFT_WRIST_y']]
    left_hip = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']]

    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    right_elbow = [landmarks_dict['RIGHT_ELBOW_x'], landmarks_dict['RIGHT_ELBOW_y']]
    right_wrist = [landmarks_dict['RIGHT_WRIST_x'], landmarks_dict['RIGHT_WRIST_y']]
    right_hip = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']]

    LeftElbow_vis = landmarks_dict["LEFT_ELBOW_visibility"]
    RightElbow_vis = landmarks_dict["RIGHT_ELBOW_visibility"]
    active_arm = "left" if LeftElbow_vis > RightElbow_vis else "right"

    vertical_point_left = [left_shoulder[0], left_shoulder[1] - 1]
    vertical_point_right = [right_shoulder[0], right_shoulder[1] - 1]

    if active_arm == "left":
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        torso_lean = calculate_angle(left_hip, left_shoulder, vertical_point_left)
        upper_arm = calculate_angle(left_hip, left_shoulder, left_elbow)
        index_point = [
            left_wrist[0] + (left_wrist[0] - left_elbow[0]),
            left_wrist[1] + (left_wrist[1] - left_elbow[1])
        ]
        wrist_angle = calculate_angle(left_elbow, left_wrist, index_point)
        forearm_vertical = calculate_angle(vertical_point_left, left_elbow, left_wrist)

    else:
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        torso_lean = calculate_angle(right_hip, right_shoulder, vertical_point_right)
        upper_arm = calculate_angle(right_hip, right_shoulder, right_elbow)
        index_point = [
            right_wrist[0] + (right_wrist[0] - right_elbow[0]),
            right_wrist[1] + (right_wrist[1] - right_elbow[1])
        ]
        wrist_angle = calculate_angle(right_elbow, right_wrist, index_point)
        forearm_vertical = calculate_angle(vertical_point_right, right_elbow, right_wrist)

    return [
        active_arm,
        elbow_angle,
        forearm_vertical,
        wrist_angle,
        upper_arm,
        torso_lean,
    ]


# ===============================
# 📸 MediaPipe Pose
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# ===============================
# 🔥 ONNX Inference Function
# ===============================
def analyze_frame(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):

    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    # ---------------------------
    # Extract landmarks
    # ---------------------------
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = float(lm.x)
        landmarks_dict[f"{name}_y"] = float(lm.y)
        landmarks_dict[f"{name}_visibility"] = float(lm.visibility)

    # ---------------------------
    # Build feature vector
    # ---------------------------
    angles = extract_angles_from_landmarks(landmarks_dict)
    active_arm = angles[0]
    elbow_angle = angles[1]

    feature_vector = []
    feature_vector.extend(angles[1:])  # add 5 angles

    selection = LEFT_LANDMARKS if active_arm == "left" else RIGHT_LANDMARKS

    for name in selection:
        feature_vector.extend([
            landmarks_dict[f"{name}_x"],
            landmarks_dict[f"{name}_y"],
            landmarks_dict[f"{name}_visibility"]
        ])

    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        buffer.append([0.0] * num_features)

    # ---------------------------
    # Rep Counting Logic
    # ---------------------------
    angle = elbow_angle

    if rep_state["prev_angle"] is None:
        rep_state["prev_angle"] = angle

    prev_angle = rep_state["prev_angle"]
    prev_phase = rep_state["prev_phase"]

    if angle >= 160 and prev_angle < 160:
        rep_state["Top_ROM_error"] = False
        rep_state["Bottom_ROM_error"] = False
        rep_state["phase"] = "B1"
    elif angle <= 60:
        rep_state["phase"] = "B3"
    elif angle < prev_angle:
        rep_state["phase"] = "B2"
    else:
        rep_state["phase"] = "B4"

    # ROM checks
    if prev_phase == "B4" and rep_state["phase"] == "B2":
        rep_state["viable_rep"] = False
        rep_state["Bottom_ROM_error"] = True

    if prev_phase == "B2" and rep_state["phase"] == "B4":
        rep_state["viable_rep"] = False
        rep_state["Top_ROM_error"] = True

    # Count rep
    if prev_phase == "B4" and rep_state["phase"] == "B1":
        if rep_state["viable_rep"]:
            rep_state["rep_counter"] += 1
        rep_state["viable_rep"] = True

    rep_state["prev_phase"] = rep_state["phase"]
    rep_state["prev_angle"] = angle

    # ---------------------------
    # Run ONNX inference
    # ---------------------------
    if len(buffer) >= window_size:
        window = np.array(buffer, dtype=np.float32)
        scaled = scaler.transform(window)
        onnx_input = scaled[np.newaxis, :, :].astype(np.float32)

        ort_inputs = {session.get_inputs()[0].name: onnx_input}
        output = session.run(None, ort_inputs)[0]

        err = float(np.mean((onnx_input - output) ** 2))

        # Form evaluation
        if err > threshold:
            rep_state["viable_rep"] = False
            status = "POOR FORM!"
        elif rep_state["Bottom_ROM_error"]:
            status = "Extend your arms more!"
        elif rep_state["Top_ROM_error"]:
            status = "Contract your arms more!"
        else:
            status = "Good Form"

        return {
            "form_status": status,
            "reconstruction_error": err,
            "rep_state": rep_state
        }

    return {"form_status": "Analyzing...", "rep_state": rep_state}


# ===============================
# Reset Rep Counter
# ===============================
def reset_rep_counter():
    return {
        "rep_counter": 0,
        "prev_angle": None,
        "prev_phase": None,
        "phase": "B1",
        "viable_rep": True,
        "Top_ROM_error": False,
        "Bottom_ROM_error": False
    }
