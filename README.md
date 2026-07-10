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
## Cài đặt
pip install flask werkzeug numpy scipy pandas scikit-learn matplotlib seaborn Pillow joblib opencv-python
# Tùy chọn (để dùng MTCNN phát hiện khuôn mặt):
pip install torch facenet-pytorch
## Chạy webapp
cd webapp
python app.py
Mở trình duyệt tại http://127.0.0.1:5000
