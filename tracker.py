import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Последний результат сохраняем здесь — callback асинхронный
latest_result = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='holistic_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result
)

def draw_points(frame, landmarks, color, radius=3):
    if not landmarks:
        return
    h, w = frame.shape[:2]
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), radius, color, -1)

def draw_connections(frame, landmarks, connections, color):
    if not landmarks or not connections:
        return
    h, w = frame.shape[:2]
    for start_idx, end_idx in connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        x1 = int(landmarks[start_idx].x * w)
        y1 = int(landmarks[start_idx].y * h)
        x2 = int(landmarks[end_idx].x * w)
        y2 = int(landmarks[end_idx].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, 1)

# Связи для тела (основные суставы)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28)
]

# Связи для пальцев руки
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

cap = cv2.VideoCapture(0)

with HolisticLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Отправляем кадр в модель
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # Рисуем последний известный результат
        if latest_result:
            # Тело — зелёный
            draw_connections(frame, latest_result.pose_landmarks,
                             POSE_CONNECTIONS, (0, 255, 0))
            draw_points(frame, latest_result.pose_landmarks,
                        (0, 255, 0), radius=4)

            # Левая рука — синий
            draw_connections(frame, latest_result.left_hand_landmarks,
                             HAND_CONNECTIONS, (255, 100, 0))
            draw_points(frame, latest_result.left_hand_landmarks,
                        (255, 100, 0), radius=3)

            # Правая рука — оранжевый
            draw_connections(frame, latest_result.right_hand_landmarks,
                             HAND_CONNECTIONS, (0, 100, 255))
            draw_points(frame, latest_result.right_hand_landmarks,
                        (0, 100, 255), radius=3)

            # Лицо — белый (только точки, связей слишком много)
            draw_points(frame, latest_result.face_landmarks,
                        (255, 255, 255), radius=1)

        cv2.imshow('Body Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()