import cv2 as cv
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

def run_camera():
    # Initialize Camera 
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("cannot open camera")
        return

    # Initialize InsightFace 
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=1, det_size=(640,640))

    while True:
        # Cature frame by frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to RGB for detection
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_app.get(rgb)

        # 5. Draw bounding boxes on original frame
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, "Face", (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display camera screen in gray
        gray = cv.cvtColor(frame, cv.COLOR_BayerRG2GRAY)
        cv.imshow("frame", gray)

        # Press q to quit
        if cv.waitKey(1) == ord('q'): 
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_camera()


