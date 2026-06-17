from imutils.video import VideoStream
import imutils
import cv2
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import align.detect_face
import sys
from pathlib import Path

# Thêm parent directory để import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":

    nameID = input("Enter Your Name: ").lower()
    
    # Setup parameters and paths
    path = str(config.RAW_DATASET_PATH / nameID)
    create_directory(path)

    with tf.Graph().as_default():
        
        with tf.compat.v1.Session().as_default():

            pnet, rnet, onet = align.detect_face.create_mtcnn(tf.compat.v1.get_default_session(), os.path.join(config.SOURCE_DIR, "align"))

            cap = VideoStream(src=0).start()
            count = 0

            while True:
                frame = cap.read()
                if frame is None:
                    print("Cannot access camera")
                    exit()
                frame = imutils.resize(frame, width=600)
                bounding_boxes, _ = align.detect_face.detect_face(
                    frame, config.MINSIZE, pnet, rnet, onet, config.THRESHOLD, config.FACTOR
                )

                face_count = len(bounding_boxes)

                if face_count == 0:
                    cv2.putText(frame, "No face detected!", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif face_count == 1:
                
                    x1, y1, x2, y2 = bounding_boxes[0][:4].astype(int)
                    face_img = frame[y1:y2, x1:x2]
                    count += 1
                    image_path = f"{path}/User-{nameID}-{count}.jpg"
                    print("Creating Images........." + image_path)
                    cv2.imwrite(image_path, face_img)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Multiple faces detected!", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Face Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if count >= config.CAPTURE_COUNT:
                    break

            cap.stop()
            cv2.destroyAllWindows()