import cv2
import os
import sys
from pathlib import Path

# Thêm parent directory để import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    nameID = str(input("Enter Your Name: ")).lower()

    # Đường dẫn lưu trữ
    path = str(config.RAW_DATASET_PATH / nameID)
    create_directory(path)

    while True:
        ret, frame = video.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5
            )

        face_count = len(faces)
        if face_count == 0:
            cv2.rectangle(frame,(10, 10),(260, 60),(0, 0, 255),-1)
            cv2.putText(frame,"No face detected!",(20, 45),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)
        elif face_count > 1:
            cv2.rectangle(frame,(10, 10),(350, 60),(0, 0, 255),-1)
            cv2.putText(frame,"Multiple faces detected!",(20, 45),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)
        else:
            x, y, w, h = faces[0]
            count = count + 1
            image_path = f"{path}/User-{nameID}-{count}.jpg"
            print("Creating Images........." + image_path)
            cv2.imwrite(image_path, frame[y:y + h, x:x + w]) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("WindowFrame", frame)
        cv2.waitKey(1)
        if count >= config.CAPTURE_COUNT:
            break

    video.release()
    cv2.destroyAllWindows()
