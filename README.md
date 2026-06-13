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

* Python 3.10
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



---

## Notes

The FaceNet model file (`20180402-114759.pb`) is downloaded automatically during setup.

If the model already exists in the `Models` directory, the download step is skipped.

---

## Acknowledgements

This project was developed for learning and educational purposes.

The implementation was inspired by and partially based on David Sandberg's FaceNet project. Several components, concepts, and workflows were adapted and further modified during development.

Special thanks to David Sandberg for providing open-source educational resources to the computer vision community.

### Reference

* https://github.com/davidsandberg/facenet
## License

This project is intended for educational and research purposes.
