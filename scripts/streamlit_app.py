import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import cv2
try:
    from scripts.labels import CLASS_NAMES
except ModuleNotFoundError:
    import sys, os
    _current_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_current_dir, ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from scripts.labels import CLASS_NAMES

# CLASS_NAMES imported from scripts.labels


def run(args):
    model_path = args.model_path
    
    print("----------- Starting Streamlit App -----------")
    st.title("Traffic Sign Recognition App")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Sidebar options
    indexing = st.sidebar.selectbox("Label indexing", ["0-based (0..42)", "1-based (1..43)"])

    # File uploader
    uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Read file bytes -> OpenCV decode (BGR), resize like training
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode image.")
            return
        bgr = cv2.resize(bgr, (30, 30), interpolation=cv2.INTER_NEAREST)

        # Display RGB for user
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption='Uploaded Image', use_column_width=True)
        
        # Match training: raw uint8 values (no normalization)
        input_batch = np.expand_dims(bgr, axis=0)
        
        # Predict
        probs = model.predict(input_batch)
        pred_idx = int(np.argmax(probs, axis=1)[0])
        display_idx = pred_idx + 1 if indexing.startswith("1-based") else pred_idx
        pred_name = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else "Unknown"
        st.write(f"Predicted: {display_idx} ({pred_name})")

        # Top-3 predictions
        top3_idx = np.argsort(probs[0])[::-1][:3]
        top3_vals = probs[0][top3_idx]
        top3_display = [
            {
                "class": int(i + (1 if indexing.startswith("1-based") else 0)),
                "name": CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else "Unknown",
                "prob": float(v)
            }
            for i, v in zip(top3_idx, top3_vals)
        ]
        st.write(top3_display)
        print(f"Predicted Class: {pred_idx}")
    
    print("----------- Streamlit App Running -----------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit App for Traffic Sign Recognition")
    parser.add_argument('--model_path', type=str, default='models/best_model.h5', help='Path of the trained model')
    args = parser.parse_args()
    run(args)
