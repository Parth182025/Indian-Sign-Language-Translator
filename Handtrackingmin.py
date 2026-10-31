import cv2
import mediapipe as mp
import csv
import os


# -----------------------------
# MediaPipe setup
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Choose sign
# -----------------------------

sign_name = input("Enter sign name: ").strip().lower()

# Create folder for this sign
folder = os.path.join("dataset", sign_name)
os.makedirs(folder, exist_ok=True)


# -----------------------------
# Find next sequence number
# -----------------------------

existing_files = [
    f for f in os.listdir(folder)
    if f.startswith("sequence_") and f.endswith(".csv")
]

sequence_number = len(existing_files) + 1


# -----------------------------
# Open camera
# -----------------------------

cap = cv2.VideoCapture(0)


# -----------------------------
# Counters
# -----------------------------

# This counter NEVER resets during the camera session.
# MediaPipe needs timestamps to continuously increase.
frame_number = 0

# This counter belongs only to the current recording.
# It can reset for every new sequence.
sequence_frame = 0


# -----------------------------
# Recording variables
# -----------------------------

recording = False
sequence_data = []


# -----------------------------
# MediaPipe detector
# -----------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera")
            break


        # -----------------------------
        # Convert BGR → RGB
        # -----------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )


        # -----------------------------
        # MediaPipe timestamp
        # -----------------------------

        timestamp_ms = int(frame_number * 1000 / 30)


        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        height, width, _ = frame.shape


        # -----------------------------
        # Process detected hands
        # -----------------------------

        for hand_id, hand in enumerate(result.hand_landmarks):

            points = []


            for landmark_id, landmark in enumerate(hand):

                # Pixel coordinates for drawing
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))


                # Draw landmark
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


                # -----------------------------
                # SAVE DATA ONLY WHILE RECORDING
                # -----------------------------

                if recording:

                    sequence_data.append([
                        sequence_frame,
                        hand_id,
                        landmark_id,
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])


            # -----------------------------
            # Draw hand connections
            # -----------------------------

            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]


            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )


        # -----------------------------
        # Recording status
        # -----------------------------

        if recording:

            cv2.putText(
                frame,
                "RECORDING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Press SPACE to record",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


        # -----------------------------
        # Show sign name
        # -----------------------------

        cv2.putText(
            frame,
            "Sign: " + sign_name,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # -----------------------------
        # Show camera
        # -----------------------------

        cv2.imshow(
            "ISL Data Collection",
            frame
        )


        # -----------------------------
        # Keyboard controls
        # -----------------------------

        key = cv2.waitKey(1) & 0xFF


        # SPACE → start/stop recording
        if key == 32:

            if not recording:

                # -----------------------------
                # START RECORDING
                # -----------------------------

                recording = True
                sequence_data = []

                # Reset ONLY the sequence frame counter
                sequence_frame = 0

                print(
                    "Recording sequence",
                    sequence_number
                )


            else:

                # -----------------------------
                # STOP RECORDING
                # -----------------------------

                recording = False


                # File name
                filename = os.path.join(
                    folder,
                    f"sequence_{sequence_number:02d}.csv"
                )


                # -----------------------------
                # Save CSV
                # -----------------------------

                with open(
                    filename,
                    "w",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        "frame",
                        "hand",
                        "landmark",
                        "x",
                        "y",
                        "z"
                    ])

                    writer.writerows(sequence_data)


                print(
                    "Saved:",
                    filename
                )


                sequence_number += 1
                sequence_data = []


        # Q → quit
        elif key == ord("q"):

            break


        # -----------------------------
        # Increase counters
        # -----------------------------

        # MediaPipe timestamp counter
        # ALWAYS increases.
        frame_number += 1


        # Sequence counter
        # Only increases while recording.
        if recording:
            sequence_frame += 1


# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()