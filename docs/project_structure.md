## Mục đích các thư mục trong dự án PCA + KNN vs PCA + SVM

- **data/**
  - **raw/**: Chứa dữ liệu gốc tải về (ảnh khuôn mặt, ví dụ Olivetti Faces), không chỉnh sửa.
  - **processed/**: Dữ liệu đã được tiền xử lý (chuẩn hóa, resize, vector hóa, chia train/test) để dùng cho các thuật toán.

- **notebooks/**  
  - Các file `.ipynb` dùng để thử nghiệm, minh họa công thức toán, kiểm tra nhanh PCA/KNN/SVM trước khi đưa vào mã nguồn chính trong `src/`.

- **report/**  
  - Tài liệu báo cáo cuối kỳ (LaTeX, Word, hình minh họa dùng riêng cho báo cáo).

- **results/**
  - **figures/**: Hình vẽ kết quả thực nghiệm (ví dụ: biểu đồ độ chính xác theo số chiều PCA, biểu đồ so sánh KNN vs SVM, hình eigenfaces, v.v.).
  - **metrics/**: Các file `.csv` hoặc `.json` lưu chỉ số đánh giá mô hình (accuracy, precision, recall, F1, thời gian train, v.v.).

- **src/**
  - **datasets/**: Định nghĩa dataset cho bài toán nhận dạng khuôn mặt (ví dụ: `face_dataset.py` để tải/chia dữ liệu Olivetti Faces).
  - **preprocessing/**: Hàm/tiện ích tiền xử lý dữ liệu ảnh (loader, chuẩn hóa, reshape ảnh 2D → vector, chia train/test, v.v.).
  - **features/**: Cài đặt PCA và các phép biến đổi đặc trưng khác (ma trận hiệp phương sai, trị riêng/vector riêng, chiếu dữ liệu xuống không gian thấp chiều).
  - **models/**: Cài đặt mô hình phân loại:
    - `pca_knn.py`: Mô hình PCA + KNN thuần toán.
    - `pca_svm.py`: Mô hình PCA + SVM (ưu tiên thuần toán hoặc giải thích toán học chi tiết).
  - **pipelines/**: Xây dựng pipeline hoàn chỉnh cho từng phương án:
    - Pipeline `PCA → KNN`.
    - Pipeline `PCA → SVM`.
  - **experiments/**: Các script chạy thí nghiệm, so sánh mô hình:
    - Chạy PCA + KNN với nhiều giá trị `k`, nhiều số chiều PCA.
    - Chạy PCA + SVM với nhiều cấu hình (kernel, C, số chiều PCA).
    - Tổng hợp kết quả vào `results/metrics/` và vẽ hình vào `results/figures/`.
  - **configs/**: Cấu hình chung cho thí nghiệm (số chiều PCA, giá trị `k` cho KNN, tham số SVM, đường dẫn dữ liệu, seed ngẫu nhiên, v.v.).
  - **utils/**: Hàm tiện ích dùng chung (ví dụ: tính toán chỉ số đánh giá, trực quan hóa, log kết quả).

- **webapp/**
  - **saved_models/**: Lưu các mô hình đã train (PCA, KNN, SVM) để tải lại khi chạy web demo.
  - `app.py`: Ứng dụng web (ví dụ Flask/FastAPI/Streamlit) để demo nhận dạng khuôn mặt dùng mô hình PCA + KNN / PCA + SVM.

- **docs/**
  - Các file tài liệu phục vụ cho việc quản lý đồ án (kế hoạch, mô tả cấu trúc, v.v.), không ảnh hưởng trực tiếp đến code chạy thực nghiệm.

