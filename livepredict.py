
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from tensorflow import keras

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading GRU model...")
model = keras.models.load_model("isl_gru_model.h5")
print("Model loaded successfully.")

# ============================================================
# LOAD ACTIONS
# ============================================================

with open("actions.pickle", "rb") as f:
    actions = pickle.load(f)

print("\nActions:")
for i, action in enumerate(actions):
    print(i, "->", action)

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    raise SystemExit

# ============================================================
# VARIABLES
# ============================================================

sequence = []

SEQUENCE_LENGTH = 30
THRESHOLD = 0.70
RESET_FRAMES = 15
STABLE_FRAMES = 3
DISPLAY_DURATION = 2.0

frame_number = 0
no_hand_frames = 0

prediction = ""
prediction_time = 0.0

stable_candidate = None
stable_count = 0

# Prevent repeated output for the same continuous sign
last_recognized_sign = None

# ============================================================
# LIVE PREDICTION
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int(frame_number * 1000 / 30)
        frame_number += 1

        result = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        # Empty hand arrays
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        hands_detected = len(result.hand_landmarks) > 0

        if hands_detected:
            no_hand_frames = 0
        else:
            no_hand_frames += 1

        # ====================================================
        # EXTRACT LANDMARKS AND DRAW HANDS
        # ====================================================

        height, width, _ = frame.shape

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        for hand_id, hand in enumerate(result.hand_landmarks):

            coordinates = []
            points = []

            for landmark in hand:

                coordinates.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            for start, end in connections:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            coordinates = np.array(
                coordinates,
                dtype=np.float32
            )

            if len(coordinates) != 63:
                continue

            if hand_id == 0:
                left_hand = coordinates
            elif hand_id == 1:
                right_hand = coordinates

        # ====================================================
        # CONTINUOUS SLIDING WINDOW
        # ====================================================

        if hands_detected:

            frame_features = np.concatenate([
                left_hand,
                right_hand
            ])

            sequence.append(frame_features)

            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

        # ====================================================
        # RESET AFTER HANDS DISAPPEAR
        # ====================================================

        if no_hand_frames >= RESET_FRAMES:

            sequence.clear()

            stable_candidate = None
            stable_count = 0
            last_recognized_sign = None

        # ====================================================
        # CONTINUOUS MODEL PREDICTION
        # ====================================================

        current_time = time.monotonic()

        if hands_detected and len(sequence) == SEQUENCE_LENGTH:

            input_data = np.array(
                sequence,
                dtype=np.float32
            )

            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            probabilities = model.predict(
                input_data,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            # Add this with your other variables, before the while loop
            last_debug_time = 0

            # Replace the existing Top predictions print block with:
            if time.monotonic() - last_debug_time >= 1:
                top3 = sorted(
                    zip(actions, probabilities),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                print("Top predictions:", [
                    (name, round(float(prob), 3))
                    for name, prob in top3
                ])

                last_debug_time = time.monotonic()

            if confidence >= THRESHOLD:

                current_candidate = actions[predicted_index]

                if current_candidate == stable_candidate:
                    stable_count += 1
                else:
                    stable_candidate = current_candidate
                    stable_count = 1

                # Recognize a new sign after stable predictions
                if (
                    stable_count >= STABLE_FRAMES
                    and current_candidate != last_recognized_sign
                    and current_time >= prediction_time
                ):

                    prediction = current_candidate
                    prediction_time = current_time + DISPLAY_DURATION
                    last_recognized_sign = current_candidate

                    print(
                        f"Prediction: {prediction} | "
                        f"Confidence: {confidence * 100:.2f}%"
                    )

            else:
                stable_candidate = None
                stable_count = 0

        # ====================================================
        # CLEAR OUTPUT AFTER 2 SECONDS
        # ====================================================

        if current_time >= prediction_time:
            prediction = ""

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (750, 115),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "Prediction:",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            prediction if prediction else "---",
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        if not hands_detected:
            status = "Show your hand"
        elif len(sequence) < SEQUENCE_LENGTH:
            status = f"Collecting: {len(sequence)}/{SEQUENCE_LENGTH}"
        else:
            status = "Continuous prediction"

        cv2.putText(
            frame,
            status,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("ISL Live Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nLive recognition stopped.")
