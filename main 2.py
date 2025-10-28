import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "digit_recognizer.h5"

def get_model():
    if os.path.exists(MODEL_PATH):
        print("[INFO] Loading saved model...")
        return tf.keras.models.load_model(MODEL_PATH)

    print("[INFO] Training new model...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)
    return model

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    inverted = cv2.bitwise_not(resized)
    normalized = inverted / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped


def run_live_prediction(model):
    cap = cv2.VideoCapture(0)

    print("[INFO] Press 'p' to predict, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[100:300, 100:300]
        cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)
        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            processed = preprocess(roi)
            prediction = np.argmax(model.predict(processed))
            print(f"[Prediction] You wrote: {prediction}")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = get_model()
    run_live_prediction(model)