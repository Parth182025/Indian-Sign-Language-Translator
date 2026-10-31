import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split


# ==================================================
# LOAD DATA
# ==================================================

X = np.load("X.npy")
y = np.load("y.npy")

with open("actions.pickle", "rb") as f:
    actions = pickle.load(f)


print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", actions)


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==================================================
# BUILD GRU MODEL
# ==================================================

model = Sequential([

    GRU(
        128,
        return_sequences=True,
        input_shape=(30, 126)
    ),

    Dropout(0.3),

    GRU(
        64
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        len(actions),
        activation="softmax"
    )
])


# ==================================================
# COMPILE
# ==================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ==================================================
# TRAIN
# ==================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    callbacks=[early_stopping]
)


# ==================================================
# EVALUATE
# ==================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
from sklearn.metrics import classification_report

y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(
    y_test, y_pred, target_names=actions
))

print("\n====================================")
print("FINAL TEST RESULT")
print("====================================")

print(
    f"Test accuracy: {accuracy * 100:.2f}%"
)


# ==================================================
# SAVE MODEL
# ==================================================

model.save(
    "isl_gru_model.h5"
)

print("\nModel saved as:")
print("isl_gru_model.h5")