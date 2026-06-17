"""
config.py
Cấu hình tập trung cho hệ thống nhận diện khuôn mặt dùng FaceNet + MTCNN.
Mọi đường dẫn, ngưỡng và siêu tham số đặt ở một nơi để dễ thay đổi.
"""
from pathlib import Path
import os

# ---- Thư mục gốc của dự án ----
ROOT_DIR = Path(__file__).resolve().parent

# ---- Dữ liệu và mô hình ----
DATA_DIR = ROOT_DIR / "Dataset"
MODELS_DIR = ROOT_DIR / "Models"
VIDEOS_DIR = ROOT_DIR / "Videos"
SOURCE_DIR = ROOT_DIR / "src"

# ---- Dataset paths ----
RAW_DATASET_PATH = DATA_DIR / "FaceData" / "raw"          # Raw images for capture
PROCESSED_DATASET_PATH = DATA_DIR / "FaceData" / "processed"  # Aligned images for training

# ---- Model paths ----
CLASSIFIER_PATH = MODELS_DIR / "facemodel.pkl"  # SVM classifier đã huấn luyện
FACENET_MODEL_PATH = MODELS_DIR / "20180402-114759.pb"  # FaceNet model

# ---- Thiết bị tính toán ----
GPU_MEMORY_FRACTION = 0.6  # Phân bổ bộ nhớ GPU (0.6 = 60%)

# ---- Tham số phát hiện (MTCNN) ----
MINSIZE = 20                        # kích thước tối thiểu của khuôn mặt
THRESHOLD = [0.6, 0.7, 0.7]        # ngưỡng cho 3 stages của MTCNN
FACTOR = 0.709                      # scale factor cho image pyramid

# ---- Tham số trích xuất đặc trưng (FaceNet) ----
INPUT_IMAGE_SIZE = 160              # kích thước ảnh khuôn mặt đưa vào FaceNet (160x160)

# ---- Tham số nhận diện ----
RECOGNITION_THRESHOLD = 0.5         # dưới ngưỡng xác suất này -> gán nhãn "Unknown"

# ---- Cấu hình camera/video ----
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPD = 3                             # Frames per Detect (tối ưu hiệu năng video)

# ---- Tham số capture dataset ----
CAPTURE_COUNT = 100                 # Số lượng ảnh cần capture cho mỗi người

# ---- Tham số classifier training ----
CLASSIFIER_BATCH_SIZE = 1000        # Batch size cho training classifier
CLASSIFIER_SEED = 666               # Random seed
MIN_NROF_IMAGES_PER_CLASS = 20      # Số tối thiểu ảnh per class
NROF_TRAIN_IMAGES_PER_CLASS = 10    # Số ảnh dùng cho training per class

# ---- Tham số align dataset ----
ALIGN_IMAGE_SIZE = 160              # Kích thước ảnh sau alignment
ALIGN_MARGIN = 32                   # Margin khi crop (pixels)
ALIGN_RANDOM_ORDER = True           # Shuffle dataset khi align
ALIGN_DETECT_MULTIPLE_FACES = False # Detect multiple faces per image
ALIGN_GPU_MEMORY_FRACTION = 0.25    # Phân bổ GPU khi align

# Tạo sẵn các thư mục nếu chưa có
MODELS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
RAW_DATASET_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATASET_PATH.mkdir(parents=True, exist_ok=True)
