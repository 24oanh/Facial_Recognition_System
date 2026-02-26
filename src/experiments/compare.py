# So sánh: PCA+KNN vs PCA+SVM
# Đánh giá trực tiếp mô hình tốt nhất từ cả hai thí nghiệm
#
# Các chiều so sánh:
#   1. Accuracy theo n_components (biểu đồ đường)
#   2. Confusion matrix (cấu hình tốt nhất của từng mô hình)
#   3. Thời gian huấn luyện theo n_components
#   4. Thời gian suy luận trên mỗi mẫu
#   5. Kiểm định thống kê (McNemar's test) về sự khác biệt phân loại sai
#   6. Trực quan hóa eigenfaces (top-k)
#   7. Đường cong phương sai giải thích của PCA
