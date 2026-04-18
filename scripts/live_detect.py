import cv2
import os
import json
import csv
from scripts.predictor import predict_frame
from scripts.labels import CLASS_NAMES
import time

def run(args):
    cam_index = 0 if args is None or not hasattr(args, "camera") else args.camera
    conf_threshold = 0.6 if args is None or not hasattr(args, "threshold") else args.threshold
    labels_path = None if args is None or not hasattr(args, "labels_path") else args.labels_path
    cooldown_seconds = 0.0 if args is None or not hasattr(args, "cooldown") else float(args.cooldown)

    class_id_to_name = {i: name for i, name in enumerate(CLASS_NAMES)}
    if labels_path and os.path.isfile(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                class_id_to_name = {int(k): v for k, v in json.load(f).items()}
        except Exception:
            class_id_to_name = {i: name for i, name in enumerate(CLASS_NAMES)}
    if not class_id_to_name:
        meta_csv = os.path.join("data", "Meta.csv")
        if os.path.isfile(meta_csv):
            try:
                with open(meta_csv, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cid = int(row.get("ClassId"))
                        sign_code = row.get("SignId")
                        class_id_to_name[cid] = f"Sign {sign_code}" if sign_code else f"Class {cid}"
            except Exception:
                class_id_to_name = {i: name for i, name in enumerate(CLASS_NAMES)}

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("----------- Starting Live Detection -----------")
    next_allowed_time = 0.0
    last_text = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if cooldown_seconds > 0 and now < next_allowed_time:
            if last_text:
                cv2.putText(frame, last_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            class_idx, confidence = predict_frame(frame)
            if confidence > conf_threshold:
                label = class_id_to_name.get(class_idx, f"Class {class_idx}")
                last_text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, last_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if cooldown_seconds > 0:
                    next_allowed_time = now + cooldown_seconds

        cv2.imshow("Traffic Sign Live Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("----------- Live Detection Stopped -----------")

def main():
    run(args=None)

if __name__ == "__main__":
    main()