# KNN - K Láng giềng gần nhất (cài đặt từ đầu, không dùng thư viện)
# Bộ phân loại cho các embedding được chiếu qua PCA
#
# Toán học cốt lõi:
#   Độ đo khoảng cách:
#     Euclidean:   d(x,y) = sqrt(sum((x_i - y_i)^2))
#     Cosine:      d(x,y) = 1 - (x.y)/(||x|| * ||y||)
#     Manhattan:   d(x,y) = sum(|x_i - y_i|)
#
#   Thuật toán:
#     1. Với mỗi mẫu test x:
#        - Tính d(x, x_i) cho tất cả mẫu huấn luyện x_i
#        - Chọn k láng giềng gần nhất theo khoảng cách nhỏ nhất
#        - Dự đoán nhãn bằng bỏ phiếu đa số trong k láng giềng
#
#   Siêu tham số:
#     - k: số láng giềng
#     - độ đo khoảng cách
