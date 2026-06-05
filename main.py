import cv2
import mediapipe as mp
import webbrowser
import time
import math
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

last_action_time = 0


def fingers_up(landmarks):
    fingers = []

    # Thumb
    if landmarks[4][0] > landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    tips = [8, 12, 16, 20]

    for tip in tips:
        if landmarks[tip][1] < landmarks[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    h, w, c = frame.shape

    if detection_result.hand_landmarks:

        for hand_landmarks in detection_result.hand_landmarks:

            landmarks = []

            for lm in hand_landmarks:

                x = int(lm.x * w)
                y = int(lm.y * h)

                landmarks.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            if len(landmarks) < 21:
                continue

            thumb_x, thumb_y = landmarks[4]
            index_x, index_y = landmarks[8]

            distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )

            ok_distance = distance

            finger_states = fingers_up(landmarks)

            current_time = time.time()

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 0, 255),
                3
            )

            # 👌 Screenshot
            print("Distance:", ok_distance)
            print("Fingers:", finger_states)
            if (
                ok_distance < 20
                and finger_states[2] == 1
                and finger_states[3] == 1
                and finger_states[4] == 1
            ):

                cv2.putText(
                    frame,
                    "Taking Screenshot",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

                if current_time - last_action_time > 5:

                    filename = f"screenshot_{int(time.time())}.png"

                    pyautogui.screenshot().save(filename)

                    print(f"Saved: {filename}")

                    last_action_time = current_time

            # 👍 YouTube
            elif finger_states == [1, 0, 0, 0, 0]:

                cv2.putText(
                    frame,
                    "Opening YouTube",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                if current_time - last_action_time > 5:

                    webbrowser.open(
                        "https://www.youtube.com"
                    )

                    last_action_time = current_time

            # ✋ ChatGPT
            elif finger_states == [1, 1, 1, 1, 1]:

                cv2.putText(
                    frame,
                    "Opening ChatGPT",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

                if current_time - last_action_time > 5:

                    webbrowser.open(
                        "https://chat.openai.com"
                    )

                    last_action_time = current_time

            # ✊ Google
            elif finger_states == [0, 0, 0, 0, 0]:

                cv2.putText(
                    frame,
                    "Opening Google",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                if current_time - last_action_time > 5:

                    webbrowser.open(
                        "https://www.google.com"
                    )

                    last_action_time = current_time

    cv2.imshow(
        "Gesture Website Controller",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()