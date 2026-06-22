# Face Recognition System

Hệ thống nhận diện khuôn mặt bằng FaceNet + MTCNN (TensorFlow, OpenCV).

## Tính năng

- Phát hiện khuôn mặt bằng MTCNN
- Trích xuất embedding bằng FaceNet
- Nhận diện đơn/đa khuôn mặt (realtime, video)

---

## Yêu cầu

- Python 3.10 (đã test trên 3.10.11)
- Webcam (nếu dùng realtime)

---

## Clone & Cài đặt chi tiết

1. Clone repo:

```bash
git clone https://github.com/NguyenTuBNs/face_recognition.git
cd face_recognition
```

2. Tạo virtual environment (Windows):

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Chạy setup (tải model nếu cần):

```bash
python setup.py
```

---

## Cấu trúc dự án 

- Dataset/  : chứa Dataset/FaceData/raw và Dataset/FaceData/processed
- Models/   : chứa `20180402-114759.pb`, `facemodel.pkl`
- src/      : mã nguồn (capture, preprocessing, training, recognition)
- Videos/   : video dùng để test
- config.py : cấu hình tập trung (để ở root)

---

## Cách chạy (từ thư mục gốc project)

1. Capture ảnh thô:

```bash
python src/capture_mtcnn.py
```

2. Align / tiền xử lý (tạo dataset processed):

```bash
python src/align/align_dataset_mtcnn.py
```

3. Huấn luyện classifier (tạo `Models/facemodel.pkl`):

```bash
python src/classifier.py
```

4. Chạy nhận diện (single/multi):

```bash
python src/single_face_rec.py
python src/multi_face_rec.py
```

5. Test bằng video:

```bash
python src/multi_face_rec.py --name sample.mp4
```

---

## Ghi chú 

- Đặt `config.py` ở root; chỉnh tham số ở đó để ảnh hưởng toàn cục (paths, thresholds, sizes, GPU, v.v.).
- Luôn chạy script từ project root để `import config` hoạt động.
- Code chạy ở TF1-compat mode (đã gọi `tf.compat.v1.disable_eager_execution()`) nếu máy có TF2.
- Khi capture, `config.CAPTURE_COUNT` là số ảnh mặc định cần chụp cho mỗi người.
---

## References

Trích yếu các nguồn tham khảo chính:

- FaceNet: https://github.com/davidsandberg/facenet
- MTCNN implementation: https://github.com/kpzhang93/MTCNN_face_detection_alignment
- OpenCV (I/O & preprocessing): https://opencv.org/
- Vietnamese tutorial: https://miai.vn/2019/09/face-recog-2-0-nhan-dien-khuon-mat-trong-video-bang-mtcnn-va-facenet/

---

## Khắc phục lỗi thường gặp

- "ModuleNotFoundError: config": chạy script từ project root (đảm bảo `config.py` ở root)
- Thiếu model `.pb` hoặc `.pkl`: chạy `python setup.py` hoặc đặt files vào `Models/`
- Vấn đề TensorFlow/CUDA: kiểm tra version tương thích và driver

---

## License

Dùng cho mục đích học tập và nghiên cứu.
