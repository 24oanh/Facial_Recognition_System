# Pipeline đánh giá
# Tải mô hình đã huấn luyện, chạy suy luận, tính toán và báo cáo các chỉ số
#
# Các chỉ số:
#   - Accuracy  = số dự đoán đúng / tổng số mẫu
#   - Precision (macro) = trung bình precision theo từng lớp
#   - Recall    (macro) = trung bình recall theo từng lớp
#   - F1-score  (macro) = trung bình điều hòa giữa precision và recall
#   - Confusion matrix (ma trận nhầm lẫn)
#
# Công thức:
#   Precision_c = TP_c / (TP_c + FP_c)
#   Recall_c    = TP_c / (TP_c + FN_c)
#   F1_c        = 2 * P_c * R_c / (P_c + R_c)
