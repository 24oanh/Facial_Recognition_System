# Toán Cho Trí Tuệ Nhân Tạo - Nhóm 2

## Đề tài: PCA + KNN vs PCA + SVM cho Nhận Dạng Khuôn Mặt

**Môn học:** Toán Cho Trí Tuệ Nhân Tạo  
**Nhóm:** 2 người  
**Dataset:** ORL/AT&T Face Database (40 người, 10 ảnh/người, tổng 400 ảnh)

---

## Mục tiêu
So sánh hiệu năng của hai pipeline:
- **Pipeline 1:** PCA → KNN
- **Pipeline 2:** PCA → SVM

trên bài toán phân loại nhận dạng khuôn mặt.

---

## Cấu trúc thư mục

```
math_for_ml/
│
├── data/
│   ├── raw/                    # Dataset ORL gốc (chưa xử lý)
│   └── processed/              # Dữ liệu sau khi tiền xử lý
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Khám phá và trực quan hoá dữ liệu
│   ├── 02_PCA.ipynb            # Phân tích PCA, chọn số thành phần
│   ├── 03_PCA_KNN.ipynb        # Pipeline PCA + KNN
│   ├── 04_PCA_SVM.ipynb        # Pipeline PCA + SVM
│   └── 05_Comparison.ipynb     # So sánh kết quả và kết luận
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── loader.py           # Load và chuẩn bị dataset ORL
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pca_knn.py          # Pipeline PCA + KNN
│   │   └── pca_svm.py          # Pipeline PCA + SVM
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # Tính toán các chỉ số đánh giá
│       └── visualization.py    # Vẽ biểu đồ kết quả
│
├── results/
│   ├── figures/                # Biểu đồ, hình ảnh kết quả
│   └── metrics/                # File CSV/JSON lưu kết quả số
│
├── report/
│   └── images/                 # Hình ảnh dùng trong báo cáo
│
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

## Dataset

Tải [ORL/AT&T Face Database](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)  
Giải nén vào thư mục `data/raw/`

---

## Kết quả kỳ vọng

| Metric     | PCA + KNN | PCA + SVM |
|------------|-----------|-----------|
| Accuracy   | ...       | ...       |
| Precision  | ...       | ...       |
| Recall     | ...       | ...       |
| F1-Score   | ...       | ...       |
| Train Time | ...       | ...       |