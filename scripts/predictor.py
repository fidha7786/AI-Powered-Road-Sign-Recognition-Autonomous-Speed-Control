import os
import numpy as np
import cv2
import tensorflow as tf

 
MODEL_PATH = os.path.join("models", "best_model.h5")

# load model once
print(f"Loading model from {MODEL_PATH} ...")
MODEL = tf.keras.models.load_model(MODEL_PATH)

def predict_frame(frame_bgr):
    """
    Takes a single frame (BGR, OpenCV format), resizes to 30x30,
    runs prediction, and returns (class_index, confidence).
    """
    resized = cv2.resize(frame_bgr, (30, 30), interpolation=cv2.INTER_NEAREST)
    input_batch = np.expand_dims(resized, axis=0)  # shape (1,30,30,3)

    preds = MODEL.predict(input_batch)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    return class_idx, confidence
