# Math for ML

Dự án so sánh hai pipeline nhận dạng khuôn mặt:

- `PCA + KNN`
- `PCA + SVM`

## Cấu trúc chính

- `src/features/`: PCA và lớp trích xuất đặc trưng
- `src/models/`: KNN, SVM và các wrapper tương thích ngược
- `src/pipelines/`: pipeline `PCA -> KNN`, `PCA -> SVM`, train/eval helpers
- `src/preprocessing/`: loader và tiền xử lý ảnh ORL/AT&T
- `src/datasets/`: dataset container
- `results/`: metrics và figures sinh ra khi chạy thí nghiệm
- `webapp/`: nơi dành cho demo app và model artifact

## Chạy nhanh

```bash
pip install -r requirements.txt
python -c "from src.preprocessing import load_and_split; from src.pipelines import train_pca_knn; X_train, X_test, y_train, y_test = load_and_split(); model = train_pca_knn(X_train, y_train, n_components=20, k=3); print(model.evaluate(X_test, y_test)['accuracy'])"
```

## Ghi chú

- `src/models/pca.py`, `src/models/pca_knn.py`, `src/models/pca_svm.py` hiện là compatibility wrapper để không gãy import cũ.
- Đường import khuyến nghị cho code mới là từ `src.features`, `src.models`, `src.pipelines`, `src.preprocessing`.
