# 🖐️ Gesture Window Switcher

Switch between open windows using hand gestures - no keyboard, no mouse.

## Demo

Swipe your hand **left** or **right** in front of the webcam to cycle through open applications, just like `Alt+Tab`.

```
← swipe left   →  previous window
→ swipe right  →  next window
```

## How it works

The app uses **MediaPipe Holistic Landmarker** to track hand position in real time. Each frame, the X-coordinate of the hand center is recorded into a rolling history buffer. When the hand travels more than 25% of the frame width within 12 frames, it registers as a swipe and triggers `Alt+Tab` or `Alt+Shift+Tab` via `pyautogui`.

```
Webcam frame
    ↓
MediaPipe (hand landmarks)
    ↓
Hand center X → history buffer [x0, x1, ..., x11]
    ↓
delta = x11 - x0
    ↓
delta > 0.25  →  Alt+Tab        (next)
delta < -0.25 →  Alt+Shift+Tab  (prev)
```

A 1.2-second cooldown prevents accidental double-triggers.

## Project Structure

```
gesture-window-switcher/
├── gesture_switcher.py       # main app
├── tracker.py                # full body landmark visualizer
├── test_camera.py            # camera sanity check
├── holistic_landmarker.task  # MediaPipe model (download separately)
└── README.md
```

## Setup

**1. Install dependencies**
```bash
pip install mediapipe opencv-python pyautogui pygetwindow
```

**2. Download the model**

Download `holistic_landmarker.task` from the link below and place it in the project folder:
```
https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

**3. Run**
```bash
python gesture_switcher.py
```

Open a few windows beforehand, then swipe your hand in front of the webcam.

## Configuration

All tuning parameters are at the top of `gesture_switcher.py`:

| Parameter | Default | Description |
|---|---|---|
| `SWIPE_THRESHOLD` | `0.25` | Minimum hand travel (as fraction of frame width) to register a swipe. Lower = more sensitive. |
| `HISTORY_SIZE` | `12` | Number of frames used to measure movement. Lower = faster detection. |
| `COOLDOWN` | `1.2` | Seconds between switches. Prevents double-triggers. |

## Requirements

- Python 3.10+
- Webcam
- Windows (uses `Alt+Tab` - macOS/Linux use different shortcuts)

## Tech stack

`Python` · `MediaPipe` · `OpenCV` · `pyautogui`
