# Math for ML

Dự án so sánh hai pipeline nhận dạng khuôn mặt:

- `PCA + KNN`
- `PCA + SVM`

## Cấu trúc chính

- `src/features/`: PCA và lớp trích xuất đặc trưng
- `src/models/`: KNN, SVM và các wrapper tương thích ngược
- `src/pipelines/`: pipeline `PCA -> KNN`, `PCA -> SVM`, train/eval helpers
- `src/process/`: xử lý dataset, lọc theo ngưỡng ảnh, tạo input và lưu artifact vào `data/processed/`
- `src/preprocessing/`: downloader, loader và tiền xử lý ảnh ORL/AT&T, Extended Yale B, LFW
- `src/datasets/`: dataset container
- `results/`: metrics và figures sinh ra khi chạy thí nghiệm
- `webapp/`: nơi dành cho demo app và model artifact

## Chạy nhanh

```bash
pip install -r requirements.txt
python -c "from src.preprocessing import load_and_split; from src.pipelines import train_pca_knn; X_train, X_test, y_train, y_test = load_and_split(); model = train_pca_knn(X_train, y_train, n_components=20, k=3); print(model.evaluate(X_test, y_test)['accuracy'])"
```

## Dataset raw

- ORL/AT&T: `data/raw/ORL/`
- Extended Yale B: `data/raw/CroppedYale/`
- LFW: `data/raw/lfw/`

```bash
python -c "from src.preprocessing import download_extended_yale_b_raw, download_lfw_raw; print(download_extended_yale_b_raw()); print(download_lfw_raw())"
```

```bash
python -c "from src.preprocessing import load_and_split; X_train, X_test, y_train, y_test = load_and_split(dataset_name='extended_yale_b', include_ambient=False); print(X_train.shape, X_test.shape)"
python -c "from src.preprocessing import load_and_split; X_train, X_test, y_train, y_test = load_and_split(dataset_name='lfw', min_images_per_subject=20, max_people=20); print(X_train.shape, X_test.shape)"
```

## Xử lý dataset

Preset đại diện hiện tại:

- `ORL -> balanced`: `10/10`
- `LFW -> many_people_many_images`: `15/15`
- `Extended Yale B -> many_images_few_people`: `59/59`, lấy `20` subject nhiều ảnh nhất, bỏ `ambient`

```bash
python -c "from src.process import process_orl_dataset; bundle = process_orl_dataset(); print(bundle['output_dir'])"
python -c "from src.process import process_extended_yale_b_dataset; bundle = process_extended_yale_b_dataset(); print(bundle['output_dir'])"
python -c "from src.process import process_lfw_dataset; bundle = process_lfw_dataset(); print(bundle['output_dir'])"
```

Preset sẵn có:

- `balanced`: cân bằng số ảnh giữa các người
- `many_people_many_images`: giữ nhiều người nhưng vẫn đảm bảo đủ ảnh mỗi người
- `many_images_few_people`: chọn ít người hơn nhưng ưu tiên người có nhiều ảnh

```bash
python -c "from src.process import process_face_dataset_with_preset; bundle = process_face_dataset_with_preset('lfw', 'balanced'); print(bundle['output_dir'])"
python -c "from src.process import process_face_dataset_with_preset; bundle = process_face_dataset_with_preset('lfw', 'many_images_few_people'); print(bundle['output_dir'])"
```

Kiểm tra ngưỡng trước khi chọn preset:

```bash
python -c "from src.process import analyze_subject_count_thresholds; rows = analyze_subject_count_thresholds('lfw', thresholds=[10,20,30,50]); print(rows)"
```

Mỗi lần xử lý sẽ lưu:

- `inputs.npz`: ma trận input `X`, label `y`, train/test indices
- `manifest.csv`: file nào thuộc người nào, label nào
- `label_mapping.json`: ánh xạ tên người -> label
- `summary.json`: thống kê filter và processing

## Ghi chú

- `src/models/pca.py`, `src/models/pca_knn.py`, `src/models/pca_svm.py` hiện là compatibility wrapper để không gãy import cũ.
- Đường import khuyến nghị cho code mới là từ `src.features`, `src.models`, `src.pipelines`, `src.preprocessing`.
