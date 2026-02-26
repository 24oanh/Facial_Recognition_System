# Trích xuất đặc trưng
# Bọc PCA để tạo ra embedding chiều thấp từ ảnh khuôn mặt
#
# Pipeline:
#   ảnh thô -> làm phẳng -> chuẩn hóa -> chiếu PCA -> vector embedding
#
# Giao diện:
#   fit(X_train)     -> tính eigenfaces trên tập huấn luyện
#   transform(X)     -> chiếu X lên top-k eigenfaces
#   fit_transform(X) -> fit rồi transform
#
# Xử lý thêm:
#   - Chọn k tối ưu (ngưỡng phương sai giải thích, ví dụ 95%)
#   - Lưu khuôn mặt trung bình để trừ mean
