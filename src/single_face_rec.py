import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from imutils.video import VideoStream
import argparse
import facenet
import imutils
import pickle
import os
import align.detect_face
import numpy as np
import cv2
import time
import sys
from pathlib import Path

# Thêm parent directory để import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Name of the video you want to test on.', default=0)
    args = parser.parse_args()

    VIDEO_NAME = args.name

    if VIDEO_NAME != 0:
        VIDEO_PATH = str(config.VIDEOS_DIR / VIDEO_NAME)
    else:
        VIDEO_PATH = 0  # Sử dụng camera mặc định nếu không có video cụ thể nào được chỉ định

    # Load model classifier đã train để nhận diện khuôn mặt -> tên 
    try:
        with open(config.CLASSIFIER_PATH, 'rb') as file:
            model, class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")
    except FileNotFoundError:
        print(f"Error: Fail to load {config.CLASSIFIER_PATH}")
        return

    with tf.Graph().as_default():

        # Cài đặt GPU
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=config.GPU_MEMORY_FRACTION)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load model Facenet embedding
            print('Loading feature extraction model')
            facenet.load_model(str(config.FACENET_MODEL_PATH))

            # Set up tensor
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Tạo MTCNN để phát hiện khuôn mặt trong video
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            BASE_PATH = os.path.join(BASE_DIR, "align")
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, BASE_PATH)

            # Lấy video từ file video
            cap = cv2.VideoCapture(VIDEO_PATH)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            prev_time = time.time()
            while (cap.isOpened()):
                # Đọc frame
                ret, frame = cap.read()
                if not ret:
                    break
                if VIDEO_PATH == 0 : 
                    frame = cv2.flip(frame, 1) 

                bounding_boxes, _ = align.detect_face.detect_face(frame, config.MINSIZE, pnet, rnet, onet, config.THRESHOLD, config.FACTOR)

                faces_found = bounding_boxes.shape[0]

                if faces_found ==1:
                    x1,y1,x2,y2 = bounding_boxes[0, 0:4].astype(np.int32)

                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or  y2 > frame.shape[0]: 
                        continue

                    cropped = frame[y1:y2, x1:x2, :]
                    scaled = cv2.resize(cropped, (config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]
                    best_name = class_names[best_class_indices[0]]
                    print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                    if best_class_probabilities > config.RECOGNITION_THRESHOLD:
                        cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)

                        name = class_names[best_class_indices[0]]
                        cv2.putText(frame, name, (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                        cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (x1, y2 + 20 + 17),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    else:
                        name = "Unknown"

                elif faces_found > 1:
                        cv2.putText(frame, "Multiple faces detected!", (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), thickness=2, lineType=2)
                elif faces_found == 0:
                        cv2.putText(frame, "No face detected!", (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), thickness=2, lineType=2)
                
                #FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time)
                prev_time = current_time

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


main()
