# 🐼 Pandatools: Smart Data Cleaning Accessor

**Pandatools** là một phần mở rộng (accessor) cho Pandas DataFrame, giúp tự động hóa quá trình phân tích chất lượng dữ liệu và thực hiện các bước làm sạch (cleaning) phổ biến chỉ với một dòng mã.

## 🚀 Tính năng chính

- **`.clean.intoo()`**: Hiển thị bảng phân tích dữ liệu cực đẹp trên Terminal với các gợi ý xử lý và mã code thực thi đi kèm.
- **`.clean.fix_dtypes()`**: Tự động nhận diện và chuyển đổi kiểu dữ liệu (vd: chuỗi số -> số, chuỗi ngày tháng -> datetime, danh mục lặp lại -> category).
- **`.clean.fill_missing()`**: Điền giá trị thiếu thông minh dựa trên kiểu dữ liệu của từng cột.
- **`.clean.strip_strings()`**: Loại bỏ khoảng trắng thừa ở đầu/cuối của tất cả các cột văn bản.
- **Hỗ trợ AI**: Xuất báo cáo dưới dạng JSON để tích hợp vào các pipeline tự động hóa hoặc huấn luyện AI.

## 📦 Cài đặt

Cực kỳ đơn giản, không yêu cầu cấu hình máy phức tạp:

```bash
pip install git+https://github.com/sonbuwin-beep/pandatools.git
```

## 🛠 Cách sử dụng
```python
import pandas as pd
import pandatools # Đăng ký accessor .clean

df = pd.read_csv("data.csv")
```

---

## 🛠 Các hàm chi tiết

### 1. Phân tích dữ liệu với `.intoo()`
Đây là tính năng mạnh mẽ nhất. Nó sẽ quét toàn bộ DataFrame và chỉ ra các vấn đề như: Missing values, Outliers, Sai kiểu dữ liệu, Duplicate IDs...

```python
# Hiển thị bảng màu trên Terminal
df.clean.intoo()

# Trả về JSON để dùng cho các ứng dụng khác
report_json = df.clean.intoo(as_json=True)
```

### 2. Tự động sửa kiểu dữ liệu với `.fix_dtypes()`
Hàm này giúp bạn tiết kiệm thời gian ép kiểu thủ công:
- Chuyển `object` sang `numeric` nếu hơn 90% dữ liệu là số.
- Chuyển `object` sang `datetime` nếu định dạng ngày tháng hợp lệ.
- Chuyển `float` sang `Int64` nếu thực chất là số nguyên nhưng có chứa NaN.
- Chuyển sang `bool` cho các cột chỉ chứa 0 và 1.

```python
df = df.clean.fix_dtypes()
```

### 3. Làm sạch chuỗi với `.strip_strings()`
Xóa các khoảng trắng vô hình gây lỗi khi so sánh dữ liệu.

```python
df = df.clean.strip_strings(lowercase=True) # Strip và chuyển về chữ thường
```

### 4. Xử lý giá trị thiếu với `.fill_missing()`
Tự động áp dụng chiến lược phù hợp:
- Cột số: Fill bằng `median` hoặc `mean`.
- Cột Datetime: Fill bằng `ffill` (forward fill).
- Cột Phân loại (Category/Object): Fill bằng nhãn `"Unknown"`.

```python
df = df.clean.fill_missing(numeric_strategy="median")
```

### 5. Loại bỏ trùng lặp với `.drop_dupes()`
Thông báo chính xác số lượng hàng bị xóa.

```python
df = df.clean.drop_dupes(subset=['id'])
```

---

## 💡 Ví dụ thực tế (Pipeline)

Bạn có thể kết chuỗi các lệnh để làm sạch dữ liệu cực nhanh:

```python
df_clean = (df.clean
    .drop_dupes()
    .strip_strings()
    .fix_dtypes()
    .fill_missing()
)

df_clean.clean.intoo() # Kiểm tra lại kết quả
```

## 📝 Lưu ý
- Thư viện yêu cầu `pandas` và `numpy`.
- Các cảnh báo về định dạng ngày tháng cũ đã được tối ưu hóa để chạy mượt mà trên các phiên bản Pandas mới nhất (2.0+).