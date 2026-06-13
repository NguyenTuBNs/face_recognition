import os
import urllib.request

# Thư mục gốc project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tạo Videos
videos_dir = os.path.join(BASE_DIR,"Dataset", "FaceData","Videos")
os.makedirs(videos_dir, exist_ok=True)

# Tạo Models
models_dir = os.path.join(BASE_DIR, "Models")
os.makedirs(models_dir, exist_ok=True)

# URL file pb
pb_url = "https://drive.google.com/uc?export=download&id=1wV13XZ3M9kW4Zl3YSl70GjK-vdKv18eR"

# File đích
pb_path = os.path.join(models_dir, "20180402-114759.pb")

# Tải nếu chưa có
if not os.path.exists(pb_path):
    print("Downloading model...")
    urllib.request.urlretrieve(pb_url, pb_path)
    print("Done.")
else:
    print("Model already exists.")
print("All Done !!!")