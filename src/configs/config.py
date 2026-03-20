from pathlib import Path

# Cấu hình dự án
# Siêu tham số cho Dataset, PCA, KNN, SVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = str(PROJECT_ROOT / "data" / "raw")
PROCESSED_DIR = str(PROJECT_ROOT / "data" / "processed")

ORL_DATA_DIR = str(Path(DATA_DIR) / "ORL")
EXTENDED_YALE_B_DIR = str(Path(DATA_DIR) / "CroppedYale")
LFW_DATA_DIR = str(Path(DATA_DIR) / "lfw")

IMAGE_SIZE = (92, 112)  # PIL/OpenCV convention: (width, height)
IMAGE_SHAPE = (IMAGE_SIZE[1], IMAGE_SIZE[0])  # NumPy convention: (height, width)

# Phân chia tập huấn luyện / kiểm tra
TEST_SIZE = 0.2
RANDOM_STATE = 42

# PCA
PCA_N_COMPONENTS = [10, 20, 40, 60, 80, 100]  # quét để tìm k tối ưu

# KNN
KNN_K_VALUES = [1, 3, 5, 7, 9]
KNN_METRICS = ["euclidean", "cosine"]

# SVM
SVM_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
SVM_KERNELS = ["linear", "rbf", "poly"]
SVM_GAMMA_VALUES = [0.001, 0.01, 0.1, 1.0]  # cho kernel RBF
SVM_DEGREE = 3  # cho kernel đa thức
SVM_MAX_ITER = 1000  # số vòng lặp tối đa của SMO
SVM_TOL = 1e-3  # ngưỡng hội tụ KKT

# Kiểm định chéo
CV_FOLDS = 5

# Kết quả
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"
