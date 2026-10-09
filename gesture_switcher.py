import cv2
import mediapipe as mp
import pyautogui
import time

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SWIPE_THRESHOLD  = 0.25  # минимальное смещение руки по X для взмаха
HISTORY_SIZE     = 12    # сколько кадров храним для анализа
COOLDOWN         = 1.2   # секунд между переключениями

latest_result       = None
hand_history        = []
last_switch_time    = 0
last_swipe          = None
swipe_display_until = 0


def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def get_hand_center(result):
    landmarks = result.right_hand_landmarks or result.left_hand_landmarks
    if not landmarks:
        return None, None
    cx = sum(lm.x for lm in landmarks) / len(landmarks)
    cy = sum(lm.y for lm in landmarks) / len(landmarks)
    return cx, cy


def detect_swipe(history):
    if len(history) < HISTORY_SIZE:
        return None
    delta = history[-1] - history[0]
    if delta > SWIPE_THRESHOLD:
        return 'right'
    if delta < -SWIPE_THRESHOLD:
        return 'left'
    return None


def switch_window(direction):
    if direction == 'right':
        pyautogui.hotkey('alt', 'tab')
    else:
        pyautogui.hotkey('alt', 'shift', 'tab')


def draw_ui(frame, history, swipe, show_swipe):
    h, w = frame.shape[:2]

    # Полоска позиции руки внизу
    cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), (40, 40, 40), -1)
    if history:
        x = int(history[-1] * w)
        cv2.circle(frame, (x, h - 20), 7, (0, 220, 220), -1)

    # Стрелка при взмахе
    if show_swipe and swipe:
        mid = (w // 2, h // 2)
        if swipe == 'right':
            cv2.arrowedLine(frame, (mid[0] - 80, mid[1]), (mid[0] + 80, mid[1]),
                            (0, 255, 0), 5, tipLength=0.35)
            cv2.putText(frame, 'NEXT', (mid[0] - 35, mid[1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        else:
            cv2.arrowedLine(frame, (mid[0] + 80, mid[1]), (mid[0] - 80, mid[1]),
                            (0, 255, 0), 5, tipLength=0.35)
            cv2.putText(frame, 'PREV', (mid[0] - 35, mid[1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.putText(frame, 'Swipe to switch windows  |  Q - quit',
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='holistic_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result
)

cap = cv2.VideoCapture(0)

with HolisticLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        now = time.time()

        if latest_result:
            cx, cy = get_hand_center(latest_result)

            if cx is not None:
                hand_history.append(cx)
                if len(hand_history) > HISTORY_SIZE:
                    hand_history.pop(0)

                h, w = frame.shape[:2]
                cv2.circle(frame, (int(cx * w), int(cy * h)), 10, (0, 220, 220), -1)
            else:
                hand_history.clear()

            if now - last_switch_time > COOLDOWN:
                swipe = detect_swipe(hand_history)
                if swipe:
                    switch_window(swipe)
                    last_switch_time = now
                    last_swipe = swipe
                    swipe_display_until = now + 0.6
                    hand_history.clear()

        draw_ui(frame, hand_history, last_swipe, now < swipe_display_until)

        cv2.imshow('Gesture Switcher', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()