# PCA - Phân tích thành phần chính (cài đặt từ đầu, không dùng thư viện)
# Eigenfaces dùng cho nhận dạng khuôn mặt
#
# Các bước toán học:
#   1. Tính khuôn mặt trung bình: mean = (1/N) * sum(x_i)
#   2. Trừ mean: X_centered = X - mean
#   3. Ma trận hiệp phương sai: C = (1/N) * X_centered.T @ X_centered
#      (Trick: dùng X @ X.T khi N < d (số mẫu < số chiều), rồi chuyển eigenvectors)
#   4. Phân rã trị riêng: C @ v = lambda * v
#   5. Sắp xếp eigenvectors theo eigenvalue giảm dần
#   6. Chiếu xuống: Z = X_centered @ W  (W: d x k, top-k eigenvectors)
#
# Ghi chú:
#   - Eigenvalues = lượng phương sai giải thích bởi mỗi thành phần
#   - Eigenvectors (eigenfaces) = các hướng chính trong không gian khuôn mặt
