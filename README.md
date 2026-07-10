# Face Recognition với PCA + KNN/SVM
Nhận diện khuôn mặt xây dựng từ đầu các thuật toán PCA (Eigenfaces), KNN và SVM để trích xuất đặc trưng và phân loại, kèm theo một ứng dụng web Flask cho phép nhận diện thời gian thực, tải ảnh lên để nhận diện hàng loạt, và xây dựng cơ sở dữ liệu khuôn mặt tùy chỉnh.
## Ý tưởng
1. Ảnh đầu vào → phát hiện khuôn mặt (Haar Cascade hoặc MTCNN)
2. Căn chỉnh, cắt, chuẩn hóa ảnh khuôn mặt
3. Trích xuất đặc trưng bằng PCA → ra vector Eigenfaces
4. Phân loại bằng KNN hoặc SVM → ra danh tính
## Tính năng chính
- **Trích xuất đặc trưng bằng PCA (Eigenfaces)** 
- **Hai pipeline phân loại:** PCA + KNN và PCA + SVM.
- **Tiền xử lý ảnh:** phát hiện khuôn mặt, căn chỉnh, cắt và chuẩn hóa ảnh.
- **Hỗ trợ nhiều bộ dữ liệu:** ORL/AT&T, Extended Yale B, LFW, và bộ dữ liệu tự xây dựng (Custom).
-- 
- **Ứng dụng web (Flask):**
  - Nhận diện qua ảnh tải lên (batch).
  - Nhận diện thời gian thực qua webcam.
  - Công cụ xây dựng cơ sở dữ liệu khuôn mặt từ thư mục ảnh tự chọn.
  - Tải xuống dữ liệu đã xử lý, số liệu đánh giá (metrics) và mô hình đã lưu.
- **Notebook** cho phân tích khám phá dữ liệu (EDA) và huấn luyện mô hình trên từng bộ dữ liệu/điều kiện.
## Cấu trúc thư mục
.
├── src/
│   ├── configs/          # Cấu hình dự án, siêu tham số PCA/KNN/SVM
│   ├── datasets/         # Định nghĩa FaceDataset và loader
│   ├── features/         # Trích xuất đặc trưng PCA
│   ├── models/           # Cài đặt PCA, KNN, SVM và các pipeline PCA+KNN/PCA+SVM
│   ├── pipelines/        # Hàm huấn luyện (train) và đánh giá (eval) pipeline
│   ├── preprocessing/    # Tải dữ liệu thô, phát hiện khuôn mặt, tiền xử lý ảnh
│   ├── process/          # Xử lý bộ dữ liệu theo các preset (balanced, harsh, enhanced, ...)
│   └── utils/            # Hàm toán học, metrics, trực quan hóa
├── webapp/
│   ├── app.py            # Ứng dụng Flask chính
│   ├── templates/        # Giao diện HTML (home, realtime, database builder, about)
│   ├── statics/          # CSS, ảnh tải lên
│   └── saved_models/     # Mô hình PCA+KNN / PCA+SVM đã huấn luyện sẵn (.pkl/.joblib)
└── notebooks/            # EDA và huấn luyện cho ORL, Extended Yale B (điều kiện khắc nghiệt/tăng cường)
