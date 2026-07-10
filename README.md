# Công nghệ
Thuật toán PCA (Eigenfaces), KNN và SVM, cùng ứng dụng web Flask để nhận diện thời gian thực.
## Ý tưởng
1. Ảnh đầu vào → phát hiện khuôn mặt
2. Căn chỉnh, cắt, chuẩn hóa ảnh khuôn mặt
3. Trích xuất đặc trưng bằng PCA → ra vector Eigenfaces
4. Phân loại bằng KNN hoặc SVM → ra danh tính
## Tính năng chính
- **Trích xuất đặc trưng bằng PCA (Eigenfaces)** 
- **Hai pipeline phân loại:** PCA + KNN và PCA + SVM.
- **Tiền xử lý ảnh:** phát hiện khuôn mặt, căn chỉnh, cắt và chuẩn hóa ảnh.
- **Hỗ trợ nhiều bộ dữ liệu:** ORL/AT&T, Extended Yale B, LFW, và bộ dữ liệu tự xây dựng.
-- 
- **Ứng dụng web (Flask):**
  - Nhận diện qua ảnh tải lên (batch).
  - Nhận diện thời gian thực qua webcam.
  - Lưu kết quả đánh giá
- **Notebook** cho phân tích khám phá dữ liệu và huấn luyện mô hình trên từng bộ dữ liệu/điều kiện.
## Cài đặt
**pip install flask werkzeug numpy scipy pandas scikit-learn matplotlib seaborn Pillow joblib opencv-python**
_Tùy chọn (để dùng MTCNN phát hiện khuôn mặt):_

**pip install torch facenet-pytorch**
## Chạy webapp
cd webapp
python app.py
Mở trình duyệt tại http://127.0.0.1:5000

