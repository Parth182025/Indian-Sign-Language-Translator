import os
import time
import pickle
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras


class ISLRecognizer:
    def __init__(self, model_path="isl_gru_model.h5",
                 actions_path="actions.pickle",
                 task_path="hand_landmarker.task"):
        for path in (model_path, actions_path, task_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {path}. Put it beside the Python scripts."
                )

        self.model = keras.models.load_model(model_path)
        with open(actions_path, "rb") as f:
            self.actions = pickle.load(f)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=task_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.sequence = deque(maxlen=30)
        self.frame_number = 0
        self.no_hand_frames = 0
        self.stable_candidate = None
        self.stable_count = 0
        self.last_recognized_sign = None
        self.prediction = ""
        self.confidence = 0.0
        self.prediction_until = 0.0
        self.threshold = 0.70
        self.stable_frames = 3
        self.reset_frames = 15
        self.last_probabilities = None

        self.connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17)
        ]

    def process(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(self.frame_number * 1000 / 30)
        self.frame_number += 1
        result = self.landmarker.detect_for_video(image, timestamp)

        left = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)
        h, w = frame.shape[:2]
        for hand_id, hand in enumerate(result.hand_landmarks):
            coords = []
            points = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
                points.append((int(lm.x*w), int(lm.y*h)))
            if draw:
                for x, y in points:
                    cv2.circle(frame, (x,y), 3, (30, 170, 95), -1)
                for a, b in self.connections:
                    cv2.line(frame, points[a], points[b], (30,170,95), 2)
            arr = np.asarray(coords, dtype=np.float32)
            if len(arr) == 63:
                if hand_id == 0:
                    left = arr
                elif hand_id == 1:
                    right = arr

        hands = len(result.hand_landmarks) > 0
        if hands:
            self.no_hand_frames = 0
            self.sequence.append(np.concatenate([left, right]))
        else:
            self.no_hand_frames += 1

        if self.no_hand_frames >= self.reset_frames:
            self.sequence.clear()
            self.stable_candidate = None
            self.stable_count = 0
            self.last_recognized_sign = None

        now = time.monotonic()
        if hands and len(self.sequence) == 30:
            probs = self.model.predict(
                np.expand_dims(np.asarray(self.sequence, dtype=np.float32), axis=0),
                verbose=0
            )[0]
            self.last_probabilities = probs
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            candidate = self.actions[idx]
            if conf >= self.threshold:
                if candidate == self.stable_candidate:
                    self.stable_count += 1
                else:
                    self.stable_candidate = candidate
                    self.stable_count = 1
                if (self.stable_count >= self.stable_frames
                    and candidate != self.last_recognized_sign
                    and now >= self.prediction_until):
                    self.prediction = candidate
                    self.confidence = conf
                    self.prediction_until = now + 2.0
                    self.last_recognized_sign = candidate
            else:
                self.stable_candidate = None
                self.stable_count = 0

        if now >= self.prediction_until:
            self.prediction = ""

        if not hands:
            status = "Show your hands to the camera"
        elif len(self.sequence) < 30:
            status = f"Collecting frames: {len(self.sequence)}/30"
        else:
            status = "Recognizing sign"

        return {
            "frame": frame,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "status": status,
            "hands_detected": hands,
            "frames_collected": len(self.sequence),
            "actions": list(self.actions),
        }

    def close(self):
        self.landmarker.close()
