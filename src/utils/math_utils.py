# Các hàm tiện ích toán học
# Hàm kernel, phép toán ma trận, hỗ trợ tính toán số trị
#
# Hàm kernel (dùng bởi SVM):
#   linear_kernel(X, Z)        -> X @ Z.T
#   rbf_kernel(X, Z, gamma)    -> exp(-gamma * ||x_i - z_j||^2)
#   poly_kernel(X, Z, d, c)    -> (X @ Z.T + c)^d
#   sigmoid_kernel(X, Z, a, c) -> tanh(a * X @ Z.T + c)
#
# Tiện ích ma trận:
#   gram_matrix(X, kernel_fn)  -> K[i,j] = K(x_i, x_j)
#   center_kernel(K)           -> căn giữa ma trận kernel trong không gian đặc trưng
#
# Hỗ trợ số trị:
#   normalize_rows(X)          -> chuẩn hóa L2 từng hàng
#   explained_variance_ratio   -> phương sai tích lũy từ các eigenvalue
