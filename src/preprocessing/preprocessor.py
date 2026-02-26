# Tiền xử lý ảnh
# Thay đổi kích thước, chuẩn hóa, làm phẳng ảnh khuôn mặt trước khi đưa vào PCA
#
# Các bước:
#   1. Thay đổi kích thước về kích thước cố định (92x112 cho dataset ORL)
#   2. Chuyển sang ảnh xám (nếu chưa phải)
#   3. Làm phẳng thành vector 1D: shape (92*112,) = (10304,)
#   4. Chuẩn hóa giá trị pixel về [0,1]:  x = x / 255.0
#   5. (Tùy chọn) Cân bằng histogram để chuẩn hóa ánh sáng
