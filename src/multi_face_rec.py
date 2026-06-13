import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import argparse
import facenet
import os
import pickle
import align.detect_face
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Name of the video you want to test on.', default=0)
    args = parser.parse_args()

    VIDEO_NAME = args.name
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    #Đường dẫn 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.join(BASE_DIR, "..",'Models')
    CLASSIFIER_PATH = os.path.join(BASE_PATH, 'facemodel.pkl')
    FACENET_MODEL_PATH = os.path.join(BASE_PATH, '20180402-114759.pb')

    if VIDEO_NAME != 0:
        VIDEO_BASE_PATH = os.path.join(BASE_DIR,"..", "Videos")
        VIDEO_PATH = os.path.join(VIDEO_BASE_PATH, VIDEO_NAME)
    else:
        VIDEO_PATH = 0  # Sử dụng camera mặc định nếu không có video cụ thể nào được chỉ định

    #Load model classifier đã train để nhận diện khuôn mặt -> tên 
    try:
        with open(CLASSIFIER_PATH, 'rb') as file:
            model, class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")
    except FileNotFoundError:
        print(f"Error: Fail to load {CLASSIFIER_PATH}")
        return

    with tf.Graph().as_default():

        # Cài đặt GPU
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load model Facenet embedding
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Set up tensor
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            #Tạo MTCNN để phát hiện khuôn mặt trong video
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            BASE_PATH = os.path.join(BASE_DIR,"align" )
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, BASE_PATH)

            # Lấy video từ file video
            cap = cv2.VideoCapture(VIDEO_PATH)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            #Nếu path video null thì trả về deafult camera
            while (cap.isOpened()):
                # Đọc frame
                ret, frame = cap.read()
                if VIDEO_PATH == 0 : 
                    frame = cv2.flip(frame, 1)  # Lật khung hình ngang để phù hợp với gương mặt người dùng
                if not ret:
                    break

                # phát hiện khuôn măt trả về bounding box
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]

                if faces_found > 0:
                    face_batch = []
                    valid_boxes = []
                    for box in bounding_boxes:
                        x1, y1, x2, y2 = box[:4].astype(int)

                        # Kiểm tra nếu bounding box vượt quá kích thước frame thì bỏ qua
                        if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                            continue

                        # Cắt phần khuôn mặt được phát hiện
                        cropped = frame[y1:y2, x1:x2, :]
                        
                        # Resize về kích thước 160x160 để đưa vào model Facenet
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)

                        face_batch.append(scaled)
                        valid_boxes.append((x1, y1, x2, y2))

                    # embedding khuôn mặt đã được phát hiện
                    if len(face_batch) > 0:
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

                            if best_probability > 0.5:
                                name = class_names[best_class_index]
                            else:
                                name = "Unknown"


                            x1, y1, x2, y2 = valid_boxes[i]
                            #Vẽ bounding box và tên lên frame
                            cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
                            cv2.putText(frame,name,(x1, y2 + 20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255, 255, 255),1)
                            cv2.putText(frame,f"{best_probability:.3f}",(x1, y2 + 40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255, 255, 255),1)

                # Hien thi frame len man hinh
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()