# Thí nghiệm: PCA + SVM
# Quét siêu tham số trên n_components, C, kernel, gamma
#
# Quy trình thí nghiệm:
#   1. Tải dataset ORL AT&T
#   2. Tiền xử lý ảnh (làm phẳng, chuẩn hóa)
#   3. Phân chia train/test (80/20 phân tầng theo lớp)
#   4. Với mỗi n_components trong PCA_N_COMPONENTS:
#        - Fit PCA trên X_train, transform X_train và X_test
#        - Với mỗi tổ hợp (kernel, C, gamma):
#            - Fit SVM (SMO) trên PCA(X_train), dự đoán trên PCA(X_test)
#            - Ghi lại: accuracy, precision, recall, F1, confusion matrix
#   5. Lưu kết quả (CSV metrics + biểu đồ confusion matrix)
#   6. Chọn cấu hình tốt nhất
