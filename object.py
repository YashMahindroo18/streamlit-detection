import streamlit as st
import cv2
import cvzone
import math
import time
from ultralytics import YOLO

# Load the YOLO model and class names (load only once)
@st.cache_resource
def load_model():
    model = YOLO("../Yolo-Weights/yolov8n.pt")
    return model

@st.cache_data
def load_class_names():
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush", "pen"]
    return classNames

model = load_model()
classNames = load_class_names()

def main():
    st.title("Real-time Object Detection")
    st.subheader("Webcam Feed")
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Detection")

    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not cap.isOpened():
        st.error("Cannot open webcam")
        st.stop()

    prev_frame_time = 0
    run_webcam = True

    while run_webcam:
        new_frame_time = time.time()
        success, img = cap.read()

        if not success:
            st.warning("End of video stream")
            break

        results = model(img, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=5)

        fps = 0 if new_frame_time == prev_frame_time else 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_placeholder.image(img, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
            run_webcam = False

    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    if "run_webcam" not in st.session_state:
        st.session_state["run_webcam"] = True
    main()