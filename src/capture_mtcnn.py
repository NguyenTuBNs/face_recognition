import pickle
from imutils.video import VideoStream
import imutils
import facenet
import cv2
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import align.detect_face


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":

    nameID = input("Enter Your Name: ").lower()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "..", "Dataset", "FaceData", "raw", nameID)
    create_directory(path)

    BASE_PATH = os.path.join(BASE_DIR, "..", "Models")

    CLASSIFIER_PATH = os.path.join(BASE_PATH, "facemodel.pkl")
    FACENET_MODEL_PATH = os.path.join(BASE_PATH, "20180402-114759.pb")

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    with tf.Graph().as_default():

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)

        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )

        with sess.as_default():

            facenet.load_model(FACENET_MODEL_PATH)
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, os.path.join(BASE_DIR, "align"))

            cap = VideoStream(src=0).start()

            count = 0

            while True:
                frame = cap.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = align.detect_face.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR
                )

                face_count = len(bounding_boxes)

                if face_count == 0:
                    cv2.putText(frame, "No face detected", (20, 40),
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
                    cv2.putText(frame, "Multiple faces", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Face Capture", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if count >= 100:
                    break

            cap.stop()
            cv2.destroyAllWindows()