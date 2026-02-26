# Pipeline huấn luyện
# Toàn bộ quy trình: tải dữ liệu -> tiền xử lý -> PCA -> huấn luyện mô hình
#
# Cách dùng:
#   train_pca_knn(X_train, y_train, n_components, k)              -> (pca, knn) đã fit
#   train_pca_svm(X_train, y_train, n_components, C, kernel, gamma) -> (pca, svm) đã fit
#
# Lưu:
#   - Mô hình đã huấn luyện vào results/metrics/
#   - Log các chỉ số trong quá trình huấn luyện
