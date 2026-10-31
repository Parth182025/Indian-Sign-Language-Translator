import os
import glob
import numpy as np
import pandas as pd
import pickle


DATASET_PATH = "dataset"

CLASSES = [
    "good afternoon",
    "good morning",
    "hi",
    "how are you",
    "i am fine",
    "namaste",
    "please",
    "sorry",
    "thank you",
    "welcome"
]

SEQUENCE_LENGTH = 30


X = []
y = []


def load_csv(csv_path):

    df = pd.read_csv(csv_path)

    required_columns = [
        "frame",
        "hand",
        "landmark",
        "x",
        "y",
        "z"
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                f"Missing column '{column}' in {csv_path}"
            )

            return None


    frames = []


    # Get all frame numbers
    frame_numbers = sorted(
        df["frame"].unique()
    )


    for frame_number in frame_numbers:

        frame_data = df[
            df["frame"] == frame_number
        ]


        # -----------------------------------------
        # Create empty left and right hand
        # -----------------------------------------

        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)


        # -----------------------------------------
        # Process each detected hand
        # -----------------------------------------

        for hand_id in sorted(
            frame_data["hand"].unique()
        ):

            hand_data = frame_data[
                frame_data["hand"] == hand_id
            ]

            hand_data = hand_data.sort_values(
                "landmark"
            )


            # We need exactly 21 landmarks
            if len(hand_data) != 21:

                continue


            coordinates = hand_data[
                ["x", "y", "z"]
            ].values


            if coordinates.shape != (21, 3):

                continue


            vector = coordinates.flatten()


            # -----------------------------------------
            # Store hand
            # -----------------------------------------

            if hand_id == 0:

                left_hand = vector

            elif hand_id == 1:

                right_hand = vector


        # -----------------------------------------
        # Combine both hands
        # -----------------------------------------

        frame_vector = np.concatenate(
            [left_hand, right_hand]
        )


        # 63 + 63 = 126
        if frame_vector.shape != (126,):

            return None


        frames.append(frame_vector)


    if len(frames) == 0:

        return None


    return np.array(
        frames,
        dtype=np.float32
    )


# ==================================================
# PROCESS DATASET
# ==================================================

print("\n========== PREPARING DATA ==========\n")


for class_index, class_name in enumerate(CLASSES):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )


    csv_files = sorted(
        glob.glob(
            os.path.join(
                class_path,
                "*.csv"
            )
        )
    )


    print(
        f"\n{class_name}"
    )

    print(
        f"CSV files found: {len(csv_files)}"
    )


    usable = 0


    for csv_file in csv_files:

        sequence = load_csv(
            csv_file
        )


        if sequence is None:

            print(
                f"  SKIPPED: "
                f"{os.path.basename(csv_file)}"
            )

            continue


        number_of_frames = len(
            sequence
        )


        # Need at least 30 frames
        if number_of_frames < SEQUENCE_LENGTH:

            print(
                f"  SKIPPED: "
                f"{os.path.basename(csv_file)} "
                f"({number_of_frames} frames)"
            )

            continue


        # -----------------------------------------
        # Select 30 frames across entire sequence
        # -----------------------------------------

        start = (number_of_frames - SEQUENCE_LENGTH) // 2

        sequence_30 = sequence[
            start: start + SEQUENCE_LENGTH
        ]


        if sequence_30.shape != (
            30,
            126
        ):

            print(
                "Unexpected shape:",
                sequence_30.shape
            )

            continue


        X.append(
            sequence_30
        )

        y.append(
            class_index
        )

        usable += 1


    print(
        f"Usable sequences: {usable}"
    )


# ==================================================
# CONVERT TO NUMPY
# ==================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


# ==================================================
# FINAL RESULT
# ==================================================

print(
    "\n=========================================="
)

print(
    "FINAL DATASET"
)

print(
    "=========================================="
)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "Number of samples:",
    len(X)
)

print(
    "Number of classes:",
    len(CLASSES)
)


print(
    "\nSamples per class:"
)


for i, class_name in enumerate(CLASSES):

    count = np.sum(
        y == i
    )

    print(
        f"{i:2d}  "
        f"{class_name:20s} -> {count}"
    )


# ==================================================
# SAVE
# ==================================================

np.save(
    "X.npy",
    X
)

np.save(
    "y.npy",
    y
)


with open(
    "actions.pickle",
    "wb"
) as f:

    pickle.dump(
        CLASSES,
        f
    )


print(
    "\n=========================================="
)

print(
    "SAVED SUCCESSFULLY"
)

print(
    "=========================================="
)

print("X.npy")
print("y.npy")
print("actions.pickle")