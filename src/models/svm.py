# SVM - Máy vector hỗ trợ (cài đặt từ đầu, không dùng thư viện)
# Đây là trọng tâm toán học chính của đề tài
#
# ============================================================
# SVM NHỊ PHÂN - HARD MARGIN (phân tách tuyến tính hoàn toàn)
# ============================================================
# Bài toán primal:
#   min_{w,b}  (1/2) * ||w||^2
#   ràng buộc: y_i * (w.T @ x_i + b) >= 1  với mọi i
#
# Lề hình học:   gamma = 2 / ||w||
# Siêu phẳng quyết định: w.T @ x + b = 0
# Vector hỗ trợ: các điểm thỏa y_i*(w.T @ x_i + b) = 1
#
# ============================================================
# SVM NHỊ PHÂN - SOFT MARGIN (không phân tách tuyến tính, C-SVM)
# ============================================================
# Bài toán primal:
#   min_{w,b,xi}  (1/2)*||w||^2 + C * sum(xi_i)
#   ràng buộc: y_i*(w.T @ x_i + b) >= 1 - xi_i
#              xi_i >= 0  với mọi i
#
# C: hệ số chính quy hóa (đánh đổi giữa lề rộng và số điểm lỗi)
#
# ============================================================
# CÔNG THỨC DUAL (Lagrangian)
# ============================================================
# Hàm Lagrangian:
#   L = (1/2)*||w||^2 - sum_i alpha_i[y_i*(w.T@x_i+b) - 1]
#
# Điều kiện KKT:
#   w = sum_i alpha_i * y_i * x_i
#   sum_i alpha_i * y_i = 0
#   alpha_i >= 0
#
# Bài toán dual (tối đa hóa):
#   W(alpha) = sum_i alpha_i - (1/2)*sum_i sum_j alpha_i*alpha_j*y_i*y_j*(x_i.T@x_j)
#   ràng buộc: sum_i alpha_i*y_i = 0,  0 <= alpha_i <= C
#
# Dự đoán: f(x) = sign(sum_i alpha_i*y_i*K(x_i,x) + b)
#
# ============================================================
# THỦ THUẬT KERNEL (KERNEL TRICK)
# ============================================================
# K(x_i, x_j) = phi(x_i).T @ phi(x_j)  (không cần tính phi tường minh)
#
# Các kernel:
#   Tuyến tính:   K(x,z) = x.T @ z
#   Đa thức:      K(x,z) = (x.T @ z + c)^d
#   RBF/Gaussian: K(x,z) = exp(-gamma * ||x-z||^2)
#   Sigmoid:      K(x,z) = tanh(alpha*x.T@z + c)
#
# ============================================================
# TỐI ƯU: SMO (Tối ưu hóa tuần tự tối thiểu)
# ============================================================
# John Platt (1998) - giải bài toán QP 2 biến ở mỗi bước
#
# Phác thảo thuật toán SMO:
#   Lặp đến hội tụ:
#     1. Chọn cặp alpha_i, alpha_j (heuristic: vi phạm KKT lớn nhất)
#     2. Tính cận L, H:
#        nếu y_i != y_j: L = max(0, alpha_j - alpha_i), H = min(C, C+alpha_j-alpha_i)
#        nếu y_i == y_j: L = max(0, alpha_i+alpha_j-C), H = min(C, alpha_i+alpha_j)
#     3. Tính eta = 2*K_ij - K_ii - K_jj
#     4. Cập nhật alpha_j: alpha_j_new = alpha_j - y_j*(E_i-E_j)/eta  (cắt về [L,H])
#     5. Cập nhật alpha_i: alpha_i_new = alpha_i + y_i*y_j*(alpha_j-alpha_j_new)
#     6. Cập nhật hệ số bias b
#
# ============================================================
# MỞ RỘNG ĐA LỚP
# ============================================================
# Chiến lược 1: One-vs-Rest (OvR)  - huấn luyện K SVM nhị phân
# Chiến lược 2: One-vs-One  (OvO)  - huấn luyện K*(K-1)/2 SVM nhị phân, bỏ phiếu
