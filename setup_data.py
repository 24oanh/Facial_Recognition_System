from src.process.dataset_processing import prebuild_representative_face_datasets

print("Đang tự động tải dữ liệu (nếu thiếu) và tạo các bản pre-processed...")
# Hàm này sẽ tự đọc data/raw, cắt mặt, căn chỉnh, cân bằng dữ liệu và lưu vào data/processed
results = prebuild_representative_face_datasets(force_rebuild=True)

print("Hoàn tất! Các bundle đã được tạo thành công:")
for name in results.keys():
    print(f" - {name}")