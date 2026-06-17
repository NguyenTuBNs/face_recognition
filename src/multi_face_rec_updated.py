import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import argparse
import facenet
import os
import pickle
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
        VIDEO_PATH = 0  

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

            # Tạo các lớp (MTCNN)
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            BASE_PATH = os.path.join(BASE_DIR, "align")
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, BASE_PATH)

            # Lấy video từ file video
            cap = cv2.VideoCapture(VIDEO_PATH)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            miss_count = 0
            frame_count = 0
            # Output => Vẽ  
            preds = []
            boxs = []
            names = []
            
            prev_time = time.time()
            # Nếu path video null thì trả về default camera
            while (cap.isOpened()):

                # Đọc frame
                ret, frame = cap.read()
                frame_count += 1

                # Lật khung hình ngang để phù hợp với gương mặt người dùng
                if not ret:
                    break
                if VIDEO_PATH == 0 : 
                    frame = cv2.flip(frame, 1) 

                # Chỉ detect 1 lần mỗi x frame tăng fps
                if frame_count % config.FPD == 0 :
                    
                    # phát hiện khuôn mặt trả về bounding box
                    bounding_boxes, _ = align.detect_face.detect_face(frame, config.MINSIZE, pnet, rnet, onet, config.THRESHOLD, config.FACTOR)

                    faces_found = bounding_boxes.shape[0]

                    if faces_found == 0: 
                        miss_count +=1 
                    elif faces_found > 0:
                        miss_count = 0
                        # reset output  
                        preds = []
                        boxs = []
                        names = []
                        face_batch = []

                        # Xử lý ảnh 
                        for box in bounding_boxes:
                            x1, y1, x2, y2 = box[:4].astype(int)

                            # Kiểm tra nếu bounding box vượt quá kích thước frame thì bỏ qua
                            if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                                continue

                            # Cắt phần khuôn mặt 
                            cropped = frame[y1:y2, x1:x2, :]
                            
                            # Resize về kích thước 160x160
                            scaled = cv2.resize(cropped, (config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)

                            face_batch.append(scaled)
                            boxs.append((x1, y1, x2, y2))

                        if len(face_batch) > 0:

                            # embedding khuôn mặt 
                            face_batch = np.stack(face_batch)
                            emb_array = sess.run(
                                embeddings,
                                feed_dict={
                                    images_placeholder: face_batch,
                                    phase_train_placeholder: False
                                }
                            )

                            # dự đoán tên của khuôn mặt 
                            for i, embedding in enumerate(emb_array):

                                predictions = model.predict_proba([embedding])
                                best_class_index = np.argmax(predictions, axis=1)[0]
                                best_probability = predictions[0][best_class_index]

                                if best_probability > config.RECOGNITION_THRESHOLD:
                                    name = class_names[best_class_index]
                                else:
                                    name = "Unknown"
                                
                                # Trả ra output 
                                preds.append(best_probability)
                                names.append(name) 

                if miss_count < 3 : 
                    #Vẽ từng cái mỗi lần đọc frame
                    for (x1,y1,x2,y2), name ,pred in zip(boxs, names, preds):
                        cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
                        cv2.putText(frame,name,(x1, y2 + 20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255, 255, 255),1)
                        cv2.putText(frame,f"{pred:.3f}",(x1, y2 + 40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255, 255, 255),1)
                else:
                    cv2.putText(frame, "No faces detected!", (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), thickness=2, lineType=2)
                current_time = time.time()
                dt = (current_time - prev_time)
                if dt > 0:
                    fps = 1.0 / dt
                else:
                    fps = 0
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

                # Hien thi frame len man hinh
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()