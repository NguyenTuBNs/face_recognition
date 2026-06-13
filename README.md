# Face Recognition System

A Python-based face recognition system using FaceNet, MTCNN, OpenCV, and TensorFlow.

## Features

* Face detection using MTCNN
* Face embedding extraction using FaceNet
* Single/multi-face recognition
* Real-time webcam recognition
* Video-based recognition support

---

## Requirements

* Python 3.10 (tested with Python 3.10.11)
* Webcam (for live recognition)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

- Dataset/ : training images, processed images, and input videos
- Models/ : FaceNet model and trained classifier
- src/ : source code
- Videos/ : input videos
---

## Setup

Run the setup script:

```bash
python setup.py
```

The setup script will:

* Create required folders
* Download the FaceNet model (.pb)
* Prepare project directories

---

## Usage

### Capture Face Dataset

```bash
python src/capture.py
```
### Preprocessing

```bash
python src/align/align_dataset_mtcnn.py
```

### Train Classifier

```bash
python src/classifier.py
```

### Single Face Recognition

```bash
python src/single_face_rec.py
```

### Multiple Face Recognition

```bash
python src/multi_face_rec.py
```
### Face Recognition With Video

Place the video file you want to test inside the videos/ folder.
Then run the script and provide the video file name when prompted.
```bash
python src/multi_face_rec.py --name <video_name>
```

---

## Notes

The FaceNet model file (`20180402-114759.pb`) is downloaded automatically during setup.

If the model already exists in the `Models` directory, the download step is skipped.


---

## Acknowledgements

This project was developed for educational and learning purposes.

The implementation was inspired by existing open-source face recognition research and related materials. Several ideas, concepts, and workflows were adapted and modified during the development process to suit the objectives of this project.

### Reference

* https://github.com/davidsandberg/facenet
* https://miai.vn/2019/09/face-recog-2-0-nhan-dien-khuon-mat-trong-video-bang-mtcnn-va-facenet/
## License

This project is intended solely for educational and research purposes.
