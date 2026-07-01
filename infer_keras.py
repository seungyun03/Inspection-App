import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

INPUT_MODE = "image"  # "image" or "webcam"
MODEL_PATH = "./weights/leather_model.keras"
TEST_IMAGE_PATH = "./weights/000.png"
INPUT_IMG_SIZE = (224, 224)
CLASSES = ["normal", "defect"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[1] Model loaded: {MODEL_PATH}")
    return model


def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, pil_img):
    arr = preprocess(pil_img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


def show_result(pil_img, label, prob):
    print("=" * 40)
    print(f"Prediction: {label}")
    print(f"Defect probability: {prob:.1%}")
    print(f"Normal probability: {(1 - prob):.1%}")
    print("=" * 40)

    plt.figure(figsize=(4, 4))
    plt.imshow(pil_img.resize(INPUT_IMG_SIZE))
    color = "red" if label == "defect" else "green"
    plt.title(f"{label}  (defect prob: {prob:.1%})", color=color, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_image_from_file():
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE_PATH}")
    print(f"[2] Test image loaded: {TEST_IMAGE_PATH}")
    return Image.open(TEST_IMAGE_PATH).convert("RGB")


def get_image_from_webcam():
    import cv2

    win_name = "webcam (SPACE: capture / ESC: cancel)"
    cap = cv2.VideoCapture(0)
    captured = None

    try:
        if not cap.isOpened():
            raise RuntimeError("Webcam is not available.")

        print("[2] Webcam started. SPACE = capture, ESC = cancel")
        cv2.namedWindow(win_name)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                captured = frame
                break
            if key == 27:
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

    if captured is None:
        raise RuntimeError("Capture cancelled.")

    rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main():
    model = load_model()

    if INPUT_MODE == "image":
        pil_img = get_image_from_file()
    elif INPUT_MODE == "webcam":
        pil_img = get_image_from_webcam()
    else:
        raise ValueError("INPUT_MODE must be 'image' or 'webcam'.")

    print("[3] Running inference...")
    label, prob = predict(model, pil_img)

    print("[4] Result")
    show_result(pil_img, label, prob)


if __name__ == "__main__":
    main()
