# 🧠 pandatools – SƠN AI DataFrame Cleaner

**Version 2.0. | Tác giả: Sơn Lê**

<<<<<<< HEAD
`pandatools` là thư viện Python mở rộng Pandas DataFrame với accessor `clean`, cung cấp hơn 25 hàm phân tích, làm sạch, trực quan hóa, và tối ưu dữ liệu. Điểm nổi bật là **SƠN AI Pipeline** – cơ chế gọi AI (OpenAI GPT-4o, Gemini) để sinh code xử lý DataFrame theo ngữ cảnh thực tế, có cache thông minh và fallback khi API lỗi. Phiên bản 2.0 bổ sung Big Data mode (chunking, sampling), phân tích màu trên terminal, Unicode NFC cho tiếng Việt, memory optimization, và hệ thống task có thể mở rộng qua decorator `@son_ai`.
=======
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
>>>>>>> 31d24c1e421edab5b1d93647aba7bb64f62851ec

---

## 📦 Cài đặt

```bash
<<<<<<< HEAD
pip install pandas numpy openai google-generativeai matplotlib seaborn openpyxl
```

Yêu cầu tối thiểu: pandas, numpy. Các gói còn lại tùy chọn theo tính năng sử dụng (AI, vẽ biểu đồ, xuất Excel).

---

## 🚀 Khởi tạo nhanh

```python
import pandas as pd
import pandatools  # Kích hoạt accessor df.clean

df = pd.read_csv('data.csv')
df.clean.intoo()           # Phân tích màu + gợi ý code
df = df.clean.auto()       # Làm sạch tự động, không cần API
df.clean.csv('output.csv') # Lưu kết quả
```

---

## 🤖 SƠN AI Pipeline

SƠN AI là hệ thống gọi mô hình ngôn ngữ lớn (OpenAI/Gemini) để sinh code Python xử lý DataFrame dựa trên prompt mô tả từng task. Mỗi task có fallback là hàm Python thuần, đảm bảo pipeline luôn hoạt động ngay cả khi không có API key hoặc API lỗi.

### Khởi tạo (phải gọi trước khi dùng son)

```python
# Key trực tiếp
df.clean.ai_son(model="gpt-4o", key="sk-abc123...")

# Key từ env (1=OPENAI_API_KEY, 2=SON_AI_KEY, 3=AI_API_KEY)
df.clean.ai_son(model="gpt-4o", key=1)

# Gemini
df.clean.ai_son(model="gemini-1.5-pro", provider="gemini", key="...")
```

Sau khi khởi tạo, `SonConfig.is_ready = True`, có thể gọi pipeline.

### Chạy pipeline

```python
df = df.clean.son([1, 2, 4, 7])                 # Custom tasks
df = df.clean.son([1, 2, 3, 4, 5, 6])           # Basic pipeline
df = df.clean.son([1,2,3,4,5,7,8,6,10])         # Advanced
df = df.clean.son([1,2,3,4,5,7,8,6,9,10])       # ML Ready
```

### Cơ chế hoạt động

1. Với mỗi `task_id`, lấy prompt từ registry `TASKS`.
2. Gửi prompt + context DataFrame (columns, dtypes, shape, sample) lên AI.
3. AI trả về hàm `def clean(df): ...`, được cache ra `~/.son_cache/`.
4. `exec()` code AI sinh, gọi `clean(result)`.
5. Nếu AI lỗi hoặc không khả dụng, chạy fallback `task['fn'](df)` – hàm Python đã đăng ký sẵn.

---

## 📋 Danh sách Task

| ID | Mô tả | Prompt AI (rút gọn) |
|----|--------|---------------------|
| 1 | 🗑️ Xóa duplicate rows | Xóa tất cả hàng trùng lặp dựa trên tất cả columns |
| 2 | 📝 Chuẩn hóa tên cột | lowercase, space→_, bỏ ký tự đặc biệt, giới hạn 50 ký tự |
| 3 | ✂️ Strip text columns | Strip khoảng trắng, chuỗi rỗng/'nan'/'None'→NaN |
| 4 | 🚫 Xóa cột vô nghĩa | Cột >70% missing hoặc chỉ có 1 giá trị duy nhất |
| 5 | 🔧 Auto-fix dtypes | Object→numeric/datetime/category, float→Int64, 0/1→bool |
| 6 | 💉 Fill missing values | Numeric→median, datetime→ffill, string→mode/'Unknown' |
| 7 | 📊 Xử lý outliers (IQR) | Q1, Q3, IQR, clip ngoài [Q1-1.5×IQR, Q3+1.5×IQR] |
| 8 | 🔤 Chuẩn hóa text Unicode | NFC normalization, thay multiple spaces, strip |
| 9 | 📅 Feature engineering datetime | Tạo year, month, day, dayofweek, quarter, is_weekend |
| 10 | 💾 Memory optimization | Downcast int64→int nhỏ hơn, float64→float32 |

---

## 🔧 Hàm làm sạch thủ công (Không cần API)

Tất cả đều có tham số `inplace=False`. Trả về DataFrame mới, nếu `inplace=True` thì ghi đè lên DataFrame gốc.

| Hàm | Chức năng | Tham số chính |
|------|-----------|---------------|
| `auto()` | Gộp task 1→6, chạy tuần tự | inplace |
| `drop_dupes()` | Xóa hàng trùng | subset, keep='first'/'last'/False |
| `strip_strings()` | Strip khoảng trắng string cols | lowercase=True/False |
| `fix_dtypes()` | Tự sửa object→numeric/datetime/category | — |
| `fill_missing()` | Điền missing thông minh | numeric_strategy='median'/'mean', cat_fill='Unknown', datetime_strategy='ffill'/'bfill', add_indicator=True |
| `remove_uninformative()` | Xóa cột >70% missing/constant | missing_threshold=0.7 |
| `normalize_text()` | Unicode NFC (tiếng Việt) | columns=['col1','col2'] |
| `clip_outliers()` | Clip outliers IQR | strategy='iqr', multiplier=1.5 |
| `normalize_column_names()` | Chuẩn hóa tên cột | — |
| `optimize_memory()` | Downcast int/float | — |

---

## 📊 Phân tích & Trực quan hóa

### Phân tích dữ liệu

```python
# Bảng màu terminal + gợi ý code từng cột
df.clean.intoo()

# JSON output
df.clean.intoo(as_json=True, max_rows=20)

# Thống kê nâng cao
df.clean.summary()
df.clean.summary(as_json=True)

# Memory usage
df.clean.info_memory()
```

Bảng `intoo()` hiển thị mỗi cột với: Non-Null count, % thiếu (đỏ nếu >50%, vàng nếu >0%), dtype, gợi ý xử lý, và code snippet tương ứng. Màu sắc trực quan giúp phát hiện vấn đề ngay trên terminal.

### Biểu đồ

| Hàm | Mô tả |
|------|--------|
| `bd('col')` | Tự chọn biểu đồ phù hợp (histogram/boxplot/pie/bar) |
| `bd('x','y')` | Tự detect loại 2 cột (scatter/line/bar) |
| `bd_bar(x, y)` | Bar chart, hỗ trợ group by z |
| `bd_line(x, y)` | Line chart, hỗ trợ group by z |
| `bd_pie(col)` | Pie chart (top 10) |
| `bd_scatter(x, y, z)` | Scatter plot với hue |
| `bd_hist(col, bins=30)` | Histogram + KDE |
| `bd_box(x, y)` | Boxplot |
| `bd_heatmap()` | Correlation heatmap (numeric cols) |

Tất cả hàm biểu đồ hỗ trợ `save='file.png'`, `title='...'`, `figsize=(w,h)`.

---

## 🐘 Big Data Mode

Kích hoạt khi DataFrame quá lớn, tự động giảm tải bộ nhớ:

```python
df.clean.bigdata(start=True, chunk_size=100_000, sample_size=10_000)
```

Khi bật:

- `intoo()` và `auto()` tự động lấy mẫu `sample_size` dòng để phân tích/ làm sạch.
- `optimize_memory()` được gọi tự động trong `auto()`.
- Cảnh báo khi shape thay đổi do sampling.

---

## 💾 Lưu file

```python
df.clean.csv('output.csv')              # UTF-8 BOM (tương thích Excel tiếng Việt)
df.clean.excel('output.xlsx')           # Excel
df.clean.to_parquet('output.parquet')   # Parquet (nhanh hơn CSV ~10x)
```

---

## 🛠️ Tự định nghĩa Task

Dùng decorator `@son_ai` để đăng ký task mới vào registry toàn cục `TASKS`:

```python
from pandatools import son_ai

@son_ai(11, "Prompt mô tả cho AI biết cần làm gì", "📛 Mô tả ngắn")
def my_custom_task(df):
    # Code fallback khi AI không khả dụng
    df['new_col'] = df['old_col'] * 2
    return df

# Sử dụng ngay
df.clean.son([1, 2, 11])
```

Task mới sẽ xuất hiện trong `df.clean.list()` và có thể dùng trong mọi pipeline.

---

## 📚 Hàm tiện ích

| Hàm | Mô tả |
|------|--------|
| `df.clean.hd()` | In hướng dẫn đầy đủ 12 mục ra terminal |
| `df.clean.list()` | In danh sách 10 task mặc định + pipeline gợi ý |

---

## 🎯 Ví dụ đầy đủ

```python
import pandas as pd
import pandatools

# Đọc dữ liệu
df = pd.read_csv('raw_data.csv')

# Xem hướng dẫn
df.clean.hd()

# Phân tích tổng quan
df.clean.intoo()

# Làm sạch tự động
df = df.clean.auto()

# Hoặc dùng Big Data mode
df.clean.bigdata(start=True)
df = df.clean.auto()

# SƠN AI nâng cao
df.clean.ai_son(model="gpt-4o", key=1)
df = df.clean.son([1, 2, 4, 7, 8])

# Trực quan hóa
df.clean.bd('age')
df.clean.bd_scatter('age', 'income', z='gender', save='scatter.png')

# Lưu kết quả
df.clean.csv('clean_data.csv')
df.clean.to_parquet('clean_data.parquet')
```

---

## 📁 Cache

Code do AI sinh được cache tại `~/.son_cache/` dưới dạng file `.py`, keyed bằng MD5 của `(model + prompt + columns)`. Lần chạy sau với cùng input sẽ dùng cache, không gọi API.

---

## 🔑 Biến môi trường

| Vị trí | Tên biến | Dùng với key= |
|--------|----------|---------------|
| 1 | `OPENAI_API_KEY` | key=1 |
| 2 | `SON_AI_KEY` | key=2 |
| 3 | `AI_API_KEY` | key=3 |

Khi không truyền key, hệ thống tự động thử lần lượt 3 biến trên.

---

**License:** MIT
**Author:** Sơn Lê

=======
pip install pandatools
>>>>>>> 31d24c1e421edab5b1d93647aba7bb64f62851ec
