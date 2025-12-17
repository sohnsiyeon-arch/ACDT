import os
# [설정] 텐서플로우/미디어파이프 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
from ultralytics import YOLO
import requests  # 소켓 대신 사용
import time      # 0.5초 딜레이용

# ==========================================
# 1. 페이지 설정 및 제목
# ==========================================
st.set_page_config(layout="wide", page_title="AI Behavior Monitor")

st.title("AI Behavior & Phone Detection Monitor")
st.markdown("XGBoost와 YOLO를 결합하여 실시간 행동 및 스마트폰 사용을 감지합니다.")

# ==========================================
# 2. 사이드바 설정 (서버 연결 & 모델)
# ==========================================
st.sidebar.header("Connection Settings")
default_ip = "localhost" 
server_ip = st.sidebar.text_input("Server IP Address", value=default_ip)
student_name = st.sidebar.text_input("Student Name", value="Student 1")

# student1.py 파일의 2. 사이드바 설정 부분에 추가

# 이전 이름을 저장할 세션 상태 변수
if 'last_name' not in st.session_state:
    st.session_state.last_name = None

# 서버 주소 설정 (CLEAR용)
CLEAR_URL = f'http://{server_ip}:8000/monitor'

# 이름 변경 감지 및 이전 데이터 삭제 로직
if st.session_state.last_name and st.session_state.last_name != student_name:
    # 이전 이름이 있고, 현재 이름과 다르면
    try:
        old_name = st.session_state.last_name
        # 서버에 DELETE 요청!
        requests.delete(f"{CLEAR_URL}/{old_name}", timeout=0.1)
        st.sidebar.success(f" 이전 데이터 ({old_name}) 삭제 완료")
    except:
        st.sidebar.error(" 이전 데이터 삭제 실패 (서버 연결 확인)")

# 현재 이름을 다음 실행을 위해 저장
st.session_state.last_name = student_name

# 서버 주소 설정
SERVER_URL = f'http://{server_ip}:8000/update'
st.sidebar.header("Model Settings")
SEQ_LEN = 10
confidence_threshold = st.sidebar.slider("YOLO 감지 임계값", 0.1, 0.9, 0.4)

# ==========================================
# 3. 모델 및 리소스 로드 (사용자 요청 코드 유지)
# ==========================================
@st.cache_resource
def load_all_models():
    # 경로 안전 장치
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'model')
    
    # 우선 model 폴더 안을 찾고, 없으면 현재 폴더를 찾음
    if os.path.exists(os.path.join(model_dir, "XGBoost_full35.pkl")):
        base_path = model_dir
    else:
        base_path = current_dir

    # 1. XGBoost 모델 로드
    try:
        model = joblib.load(os.path.join(base_path, "XGBoost_full35.pkl"))
        le = joblib.load(os.path.join(base_path, "label_encoder.pkl"))
        print("XGBoost(Full) & LabelEncoder2 로드 완료")
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None

    # 2. YOLO 로드
    try:
        yolo_model = YOLO('yolov8n.pt')
        print("YOLO 로드 완료")
    except Exception as e:
        st.error(f"YOLO 모델 로드 실패: {e}")
        return None, None, None

    return model, le, yolo_model

model, le, yolo_model = load_all_models()

# ==========================================
# 4. MediaPipe 및 특징 추출 함수!!! 각각 함수 지정해줘야 함. (v3 Full35)
# ==========================================
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

def get_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360.0 - angle
    return float(angle)

def _safe_std(points):
    if len(points) <= 1: return 0.0
    return float(np.linalg.norm(np.std(points, axis=0)))

# *args 추가하여 에러 방지
def calculate_features_v3(buffer_pose, buffer_face, buffer_hands, *args):
    if not buffer_pose:
        return {k: 0.0 for k in ["scale", "gaze_y", "gaze_x", "eye_open", "head_pitch", "head_roll", "mouth_ratio",
            "neck_flexion", "body_lean_x", "body_lean_y", "arm_angle", "hand_depth", "wrist_rot", "hand_face_d", 
            "pinch_dist", "finger_spd", "hand_jitter", "hand_sym", "motion_en", "motion_jerk",
            "rel_wrist_L_x", "rel_wrist_L_y", "rel_wrist_R_x", "rel_wrist_R_y", "rel_elbow_L_x", "rel_elbow_L_y", 
            "rel_elbow_R_x", "rel_elbow_R_y", "rel_sh_L_y", "rel_sh_R_y", "rel_eye_L_x", "rel_eye_R_x",
            "rel_wrist_L_z", "rel_wrist_R_z", "rel_elbow_L_z", "rel_elbow_R_z"]}

    curr_p = buffer_pose[-1]; curr_f = buffer_face[-1] if buffer_face else None
    curr_h = buffer_hands[-1] if buffer_hands else None
    feats = {}; eps = 1e-6

    # (1) Scale
    nose_xy = np.array([curr_p[0].x, curr_p[0].y])
    sh_l_xy = np.array([curr_p[11].x, curr_p[11].y]); sh_r_xy = np.array([curr_p[12].x, curr_p[12].y])
    scale = float(np.linalg.norm(sh_l_xy - sh_r_xy)) + eps
    feats["scale"] = scale

    # (2) Face
    feats.update({"gaze_y":0.5, "gaze_x":0.5, "eye_open":0.0, "head_pitch":0.0, "head_roll":0.0, "mouth_ratio":0.0})
    if curr_f:
        pupil = curr_f[468]; upper_lid = curr_f[159]; lower_lid = curr_f[145]
        eye_h = get_dist([upper_lid.x, upper_lid.y], [lower_lid.x, lower_lid.y]) + eps
        inner_eye = curr_f[133]; outer_eye = curr_f[33]
        eye_w = get_dist([inner_eye.x, inner_eye.y], [outer_eye.x, outer_eye.y]) + eps
        feats["gaze_y"] = (lower_lid.y - pupil.y) / eye_h
        feats["gaze_x"] = (pupil.x - inner_eye.x) / eye_w
        feats["eye_open"] = eye_h / scale
        ear_mid = (np.array([curr_p[7].x, curr_p[7].y]) + np.array([curr_p[8].x, curr_p[8].y])) / 2
        feats["head_pitch"] = (nose_xy[1] - ear_mid[1]) / scale
        feats["head_roll"] = float(np.arctan2(curr_p[8].y - curr_p[7].y, curr_p[8].x - curr_p[7].x))
        mouth_h = get_dist([curr_f[13].x, curr_f[13].y], [curr_f[14].x, curr_f[14].y]) + eps
        mouth_w = get_dist([curr_f[61].x, curr_f[61].y], [curr_f[291].x, curr_f[291].y]) + eps
        feats["mouth_ratio"] = mouth_h / mouth_w

    # (3) Body
    sh_center = (sh_l_xy + sh_r_xy) / 2
    hip_center = (np.array([curr_p[23].x, curr_p[23].y]) + np.array([curr_p[24].x, curr_p[24].y])) / 2
    feats["neck_flexion"] = (sh_center[1] - nose_xy[1]) / scale
    feats["body_lean_x"] = (sh_center[0] - hip_center[0]) / scale
    feats["body_lean_y"] = (hip_center[1] - sh_center[1]) / scale
    la = calculate_angle([curr_p[11].x, curr_p[11].y], [curr_p[13].x, curr_p[13].y], [curr_p[15].x, curr_p[15].y])
    ra = calculate_angle([curr_p[12].x, curr_p[12].y], [curr_p[14].x, curr_p[14].y], [curr_p[16].x, curr_p[16].y])
    feats["arm_angle"] = (la + ra) / 2.0

    # (4) Hands
    feats.update({"hand_depth":0.0, "wrist_rot":0.0, "hand_face_d":1.0, "pinch_dist":1.0, "finger_spd":0.0, "hand_jitter":0.0})
    feats["hand_sym"] = abs(get_dist([curr_p[15].x, curr_p[15].y], nose_xy) - get_dist([curr_p[16].x, curr_p[16].y], nose_xy)) / scale
    if curr_h and len(curr_h) > 0:
        main_hand = curr_h[0]
        feats["pinch_dist"] = get_dist([main_hand[4].x, main_hand[4].y], [main_hand[8].x, main_hand[8].y]) / scale
        feats["hand_face_d"] = get_dist([main_hand[0].x, main_hand[0].y], nose_xy) / scale
        feats["wrist_rot"] = float(main_hand[4].z - main_hand[20].z)
        
    sh_z = (curr_p[11].z + curr_p[12].z) / 2
    feats["hand_depth"] = (sh_z - curr_p[16].z) * 10.0

    # (5) Motion
    nose_traj = np.array([[p[0].x, p[0].y] for p in buffer_pose])
    if len(nose_traj) > 1:
        diffs = np.linalg.norm(np.diff(nose_traj, axis=0), axis=1)
        feats["motion_en"] = float(np.sum(diffs))
        if len(diffs) >= 2: feats["motion_jerk"] = float(np.sum(np.abs(np.diff(diffs))))
        else: feats["motion_jerk"] = 0.0
    else: feats["motion_en"] = 0.0; feats["motion_jerk"] = 0.0

    # (6) Relative Coords
    for name, idx in [("wrist_L", 15), ("wrist_R", 16), ("elbow_L", 13), ("elbow_R", 14)]:
        feats[f"rel_{name}_x"] = (curr_p[idx].x - nose_xy[0]) / scale
        feats[f"rel_{name}_y"] = (curr_p[idx].y - nose_xy[1]) / scale
        feats[f"rel_{name}_z"] = (curr_p[idx].z - curr_p[0].z) * 10.0
    
    feats["rel_sh_L_y"] = (sh_l_xy[1] - nose_xy[1]) / scale
    feats["rel_sh_R_y"] = (sh_r_xy[1] - nose_xy[1]) / scale
    feats["rel_eye_L_x"] = (curr_p[2].x - nose_xy[0]) / scale
    feats["rel_eye_R_x"] = (curr_p[5].x - nose_xy[0]) / scale

    return feats

# ==========================================
# 5. 웹캠 스트리밍 및 예측
# ==========================================
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Live Camera")
    frame_window = st.image([])
with col2:
    st.subheader("Analysis")
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    phone_status_placeholder = st.empty()

run = st.checkbox("Start Camera")

if run and model is not None:
    cap = cv2.VideoCapture(0)
    
    # 버퍼 초기화
    buf_pose = deque(maxlen=SEQ_LEN)
    buf_face = deque(maxlen=SEQ_LEN)
    buf_hands = deque(maxlen=SEQ_LEN)

    # Full35 피처 리스트 (순서 중요!)
    FEATURE_ORDER = model.feature_names_in_ 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        h, w, _ = frame.shape
        
        # 1. MediaPipe 처리
        rp = pose.process(image)
        rf = face_mesh.process(image)
        rh = hands.process(image)
        
        image.flags.writeable = True
        
        # 2. YOLO Phone Detection
        is_phone_detected = False
        if yolo_model:
            results = yolo_model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == 67 and conf > confidence_threshold: # 67: cell phone
                        is_phone_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"PHONE {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # 3. 데이터 수집 및 예측
        status_text = "Analyzing..."
        conf_text = "0%"
        
        if rp.pose_landmarks:
            # 랜드마크 그리기 (선택)
            mp.solutions.drawing_utils.draw_landmarks(frame, rp.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            buf_pose.append(rp.pose_landmarks.landmark)
            buf_face.append(rf.multi_face_landmarks[0].landmark if rf.multi_face_landmarks else None)
            
            current_hands_list = []
            if rh.multi_hand_landmarks:
                for h_land in rh.multi_hand_landmarks:
                    current_hands_list.append(h_land.landmark)
            buf_hands.append(current_hands_list)

            if len(buf_pose) == SEQ_LEN:
                try:
                    # 특징 추출
                    features = calculate_features_v3(buf_pose, buf_face, buf_hands)
                    
                    # XGBoost 예측을 위해 DataFrame 변환 (순서 중요!)
                    input_data = pd.DataFrame([features])
                    if set(FEATURE_ORDER).issubset(input_data.columns):
                        input_data = input_data[FEATURE_ORDER]
                    
                    # 예측
                    pred_prob = model.predict_proba(input_data)[0]
                    pred_idx = np.argmax(pred_prob)
                    confidence = pred_prob[pred_idx] * 100
                    pred_label = le.inverse_transform([pred_idx])[0]
                    
                    # [상세 확률 계산] -> 딕셔너리로 만듦
                    all_probs = {}
                    for i, class_name in enumerate(le.classes_):
                        all_probs[class_name] = round(float(pred_prob[i] * 100), 1)

                    # 룰베이스 보정 (YOLO)
                    if is_phone_detected:
                        pred_label = "phone"
                        confidence = 99.9
                        all_probs['phone'] = 99.9 # 강제 주입

                    status_text = pred_label.upper()
                    conf_text = f"{confidence:.1f}%"
                    
                    # 화면 표시
                    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
                    color = (0, 255, 0) if confidence > 80 else (0, 255, 255)
                    cv2.putText(frame, f"{status_text} ({conf_text})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # ---------------------------------------------------------
                    # [서버 전송] 상세 확률(all_probs) 포함하여 전송
                    # ---------------------------------------------------------
                    now = time.time()
                    if 'last_send_time' not in st.session_state:
                        st.session_state.last_send_time = 0
                    
                    if now - st.session_state.last_send_time > 0.5:
                        try:
                            payload = {
                                "name": student_name,
                                "status": pred_label, # 소문자 (서버 로직에 맞춤)
                                "prob": conf_text,
                                "detail_probs": all_probs # 딕셔너리 전송
                            }
                            # timeout=0.1 : 서버가 느려도 내 화면은 멈추지 않게 함
                            requests.post(SERVER_URL, json=payload, timeout=0.1)
                            st.session_state.last_send_time = now
                        except:
                            pass # 서버 꺼져있어도 에러 안 띄움

                except Exception as e:
                    print(f"Prediction Error: {e}")

        # UI 업데이트
        prediction_placeholder.metric("Current Status", status_text)
        confidence_placeholder.metric("Confidence", conf_text)
        
        if is_phone_detected:
            phone_status_placeholder.error("Phone Detected!")
        else:
            phone_status_placeholder.success("No Phone Detected")

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:

    st.write("Click 'Start Camera' to begin monitoring.")
