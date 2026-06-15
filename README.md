# 🧠 Pandatools – SƠN AI DataFrame Cleaner

**Pandatools** là một phần mở rộng (accessor) cho Pandas DataFrame, giúp tự động hóa quá trình phân tích chất lượng dữ liệu, làm sạch (cleaning) và tích hợp AI (SƠN AI) để xử lý dữ liệu thông minh chỉ với vài dòng mã.

---

## 🚀 Tính năng chính

- **`.clean.intoo()`**: Hiển thị bảng phân tích dữ liệu cực đẹp trên Terminal với các gợi ý xử lý và mã code thực thi đi kèm.
- **`.clean.summary()`**: Thống kê chi tiết từng cột: min, max, mean, skewness, unique values.
- **`.clean.info_memory()`**: Kiểm tra memory usage chi tiết từng cột, top 5 cột nặng nhất.
- **`.clean.auto()`**: Tự động làm sạch toàn bộ DataFrame (không cần API) chỉ với 1 dòng lệnh.
- **`.clean.fix_dtypes()`**: Tự động nhận diện và chuyển đổi kiểu dữ liệu (object chứa số → numeric, object chứa ngày → datetime, ít unique → category, float nguyên → Int64, int 0/1 → bool).
- **`.clean.fill_missing()`**: Điền giá trị thiếu thông minh (numeric → median, datetime → forward fill, string → mode hoặc "Unknown").
- **`.clean.strip_strings()`**: Loại bỏ khoảng trắng thừa ở đầu/cuối của tất cả các cột văn bản.
- **`.clean.drop_dupes()`**: Xóa hàng trùng lặp, thông báo chính xác số lượng đã xóa.
- **`.clean.normalize_text()`**: Chuẩn hóa text Unicode NFC (tốt cho tiếng Việt), xóa multiple spaces.
- **`.clean.clip_outliers()`**: Phát hiện và xử lý outliers bằng phương pháp IQR.
- **`.clean.normalize_column_names()`**: Chuẩn hóa tên cột (lowercase, thay space/dấu gạch bằng underscore).
- **`.clean.optimize_memory()`**: Tối ưu memory (downcast int64, float64).
- **`.clean.remove_uninformative()`**: Xóa cột >70% missing hoặc cột chỉ có 1 giá trị duy nhất.
- **🤖 SƠN AI (`.clean.son()`)**: Tích hợp AI (OpenAI GPT-4o, Gemini) để viết code xử lý dữ liệu theo ngữ cảnh thực tế, có cache thông minh, fallback tự động.
- **🐘 Big Data Mode (`.clean.bigdata()`)**: Tự động chunking & sampling cho dataset hàng triệu dòng.
- **📊 Biểu đồ**: 8 loại biểu đồ (auto-detect, bar, line, pie, scatter, hist, box, heatmap).
- **💾 Export**: Lưu CSV (utf-8-sig), Excel, Parquet.

---

## 📦 Cài đặt

Cực kỳ đơn giản, không yêu cầu cấu hình máy phức tạp:

```bash
pip install git+https://github.com/sonbuwin-beep/pandatools.git
