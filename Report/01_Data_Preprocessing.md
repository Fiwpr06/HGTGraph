# Part 1: DATA PREPROCESSING - XỬ LÝ DỮ LIỆU

**File code tương ứng:** `data_preprocessing.py`  
**Input:** `raw/db_job_tuan.xlsx`  
**Output:** `processed/jobs_processed.csv`

---

## 📚 MỤC LỤC

1. [Dữ liệu đầu vào](#1-dữ-liệu-đầu-vào)
2. [Tại sao cần preprocessing?](#2-tại-sao-cần-preprocessing)
3. [Các bước preprocessing](#3-các-bước-preprocessing)
4. [Code chi tiết](#4-code-chi-tiết)
5. [Kết quả](#5-kết-quả)
6. [FAQ](#6-faq)

---

## 1. DỮ LIỆU ĐẦU VÀO

### File Excel: `db_job_tuan.xlsx`

**Cấu trúc:**
- **Sheet name:** `tcv`
- **Số hàng:** 500 job postings
- **Số cột:** 12 columns

### 12 Cột trong Excel:

| # | Tên cột | Ví dụ | Vấn đề |
|---|---------|-------|--------|
| 1 | **JobID** | J001, J002... | ✓ OK |
| 2 | **Title** | "Kế Toán Thuế..." | ✓ OK (text) |
| 3 | **Name company** | "CÔNG TY TNHH..." | ✓ OK |
| 4 | **Job Address** | "Hồ Chí Minh (mới)" | ⚠️ Cần clean "(mới)" |
| 5 | **Job Requirements** | Text dài... | ✓ OK (text) |
| 6 | **Salary** | "18 - 25 triệu" | ⚠️ Cần convert số |
| 7 | **Experience** | "3 năm" | ⚠️ Cần convert số |
| 8 | **Job description** | Text dài... | ✓ OK (text) |
| 9 | **Job type** | "Toàn thời gian" | ✓ OK (categorical) |
| 10 | **company_size** | "25-99 nhân viên" | ✓ OK (categorical) |
| 11 | **quantity** | 1, 50... | ✓ OK (number) |
| 12 | **benefit** | Text dài... | ✓ OK (text) |

### Ví dụ 1 hàng dữ liệu:

```
JobID: J001
Title: Kế Toán Thuế / Kế Toán Tổng Hợp (Ưu Tiên Tiếng Trung Giao Tiếp)
Name company: CÔNG TY TNHH THƯƠNG MẠI YÊU QUẦN ÁO
Job Address: Hồ Chí Minh (mới)
Salary: 18 - 25 triệu
Experience: 3 năm
Job type: Toàn thời gian
company_size: 25-99 nhân viên
quantity: 1
...
```

---

## 2. TẠI SAO CẦN PREPROCESSING?

### Vấn đề với dữ liệu thô:

#### ❌ Problem 1: Salary là text, không phải số
```python
"18 - 25 triệu"        # Làm sao tính toán?
"Thoả thuận"           # Không có thông tin!
"Tới 3,000 USD"        # USD khác VND!
```

→ **Cần:** Convert về dạng số (min, max) để GNN có thể học

#### ❌ Problem 2: Experience là text
```python
"3 năm"                # OK, nhưng vẫn là text
"Dưới 1 năm"           # Làm sao biểu diễn "dưới"?
"Không yêu cầu"        # = 0?
```

→ **Cần:** Convert về số năm (float)

#### ❌ Problem 3: Location không clean
```python
"Hồ Chí Minh (mới)"              # "(mới)" thừa
"Hồ Chí Minh (mới) & 9 nơi khác" # Chỉ lấy địa điểm chính
```

→ **Cần:** Làm sạch, standardize

#### ❌ Problem 4: Text fields để rời rạc
```python
Title: "Kế Toán Thuế..."
Requirements: "- Tốt nghiệp..."
Description: "1. Công việc..."
```

→ **Cần:** Gộp lại thành 1 text để embedding

---

## 3. CÁC BƯỚC PREPROCESSING

### 🔄 Pipeline tổng quan:

```
Excel File (raw)
      │
      ▼
[1] Load data với Pandas
      │
      ▼
[2] Normalize Salary
      "18 - 25 triệu" → (18.0, 25.0)
      │
      ▼
[3] Normalize Experience
      "3 năm" → 3.0
      │
      ▼
[4] Clean Location
      "Hà Nội (mới)" → "Hà Nội"
      │
      ▼
[5] Handle missing values
      Fill empty strings
      │
      ▼
[6] Create combined text
      Title + Requirements + Description
      │
      ▼
CSV File (processed)
```

---

## 4. CODE CHI TIẾT

### 📝 Class `JobDataPreprocessor`

```python
class JobDataPreprocessor:
    """Preprocessor for job posting data"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or config.RAW_DATA_PATH
        self.df = None
```

**Giải thích:**
- Class để tổ chức code gọn gàng
- `data_path`: Đường dẫn đến file Excel
- `self.df`: Sẽ lưu DataFrame sau khi load

---

### 🔹 BƯỚC 1: Load Data

```python
def load_data(self) -> pd.DataFrame:
    """Load data from Excel file"""
    print(f"Loading data from {self.data_path}...")
    self.df = pd.read_excel(self.data_path, sheet_name='tcv')
    print(f"Loaded {len(self.df)} job postings")
    return self.df
```

**Giải thích từng dòng:**

```python
pd.read_excel(self.data_path, sheet_name='tcv')
```
- `pd.read_excel()`: Hàm của Pandas để đọc Excel
- `sheet_name='tcv'`: Chỉ định sheet cụ thể (vì Excel có thể có nhiều sheets)
- → Trả về **DataFrame** (giống như bảng Excel trong Python)

**Output:**
```
Loading data from raw/db_job_tuan.xlsx...
Loaded 500 job postings
```

---

### 🔹 BƯỚC 2: Normalize Salary

#### Vấn đề:
```python
"18 - 25 triệu"        # Range
"Thoả thuận"           # No info
"Tới 3,000 USD"        # USD currency
"15 triệu"             # Single value
```

#### Giải pháp:

```python
def normalize_salary(self, salary_str: str) -> Tuple[float, float]:
    """
    Normalize salary string to (min, max) in million VND
    
    Examples:
        '18 - 25 triệu' -> (18.0, 25.0)
        'Thoả thuận' -> (0.0, 0.0)
        'Tới 3,000 USD' -> (75.0, 75.0)  # Convert to VND
    """
    # Case 1: Missing or "Thoả thuận"
    if pd.isna(salary_str) or salary_str == 'Thoả thuận':
        return (0.0, 0.0)
    
    salary_str = str(salary_str).lower()
    
    # Case 2: Handle USD - convert to VND
    if 'usd' in salary_str:
        numbers = re.findall(r'[\d,]+', salary_str)
        if numbers:
            usd_amount = float(numbers[0].replace(',', ''))
            vnd_amount = usd_amount * 25  # 1 USD ≈ 25 triệu VND
            if 'tới' in salary_str:
                return (0.0, vnd_amount)
            return (vnd_amount, vnd_amount)
    
    # Case 3: Extract numbers from string
    numbers = re.findall(r'\d+', salary_str)
    
    if not numbers:
        return (0.0, 0.0)
    
    numbers = [float(n) for n in numbers]
    
    # Case 4: Range "18 - 25 triệu"
    if len(numbers) >= 2:
        return (min(numbers), max(numbers))
    
    # Case 5: Single value or "Tới X"
    elif len(numbers) == 1:
        if 'tới' in salary_str or 'trên' in salary_str:
            return (0.0, numbers[0])
        return (numbers[0], numbers[0])
    
    return (0.0, 0.0)
```

**Giải thích từng case:**

**Case 1: Missing hoặc "Thoả thuận"**
```python
if pd.isna(salary_str) or salary_str == 'Thoả thuận':
    return (0.0, 0.0)
```
- `pd.isna()`: Check nếu cell Excel trống
- Nếu là "Thoả thuận" → không có info → return (0, 0)

**Case 2: USD currency**
```python
if 'usd' in salary_str:
    numbers = re.findall(r'[\d,]+', salary_str)
    usd_amount = float(numbers[0].replace(',', ''))
    vnd_amount = usd_amount * 25
```
- `re.findall(r'[\d,]+', ...)`: Tìm tất cả số trong string (regex)
- `replace(',', '')`: Bỏ dấu phẩy: "3,000" → "3000"
- `* 25`: Convert USD → triệu VND (1 USD ≈ 25,000 VND = 25 triệu)

**Case 3-5: Extract numbers và xử lý**
```python
numbers = re.findall(r'\d+', salary_str)
numbers = [float(n) for n in numbers]
```
- Tìm tất cả số: "18 - 25 triệu" → ['18', '25']
- Convert sang float: ['18', '25'] → [18.0, 25.0]

```python
if len(numbers) >= 2:
    return (min(numbers), max(numbers))
```
- Nếu có 2+ số → lấy min/max làm range

**Ví dụ thực tế:**

| Input | Output |
|-------|--------|
| "18 - 25 triệu" | (18.0, 25.0) |
| "Thoả thuận" | (0.0, 0.0) |
| "Tới 3,000 USD" | (0.0, 75000.0) |
| "15 triệu" | (15.0, 15.0) |
| "Trên 20 triệu" | (0.0, 20.0) |

---

### 🔹 BƯỚC 3: Normalize Experience

#### Vấn đề:
```python
"3 năm"                # Clear
"Dưới 1 năm"           # < 1 year
"Không yêu cầu"        # No requirement = 0?
```

#### Giải pháp:

```python
def normalize_experience(self, exp_str: str) -> float:
    """
    Normalize experience string to years
    
    Examples:
        '3 năm' -> 3.0
        'Dưới 1 năm' -> 0.5
        'Không yêu cầu' -> 0.0
    """
    if pd.isna(exp_str):
        return 0.0
    
    exp_str = str(exp_str).lower()
    
    # No experience required
    if 'không yêu cầu' in exp_str or 'no experience' in exp_str:
        return 0.0
    
    # Less than 1 year
    if 'dưới' in exp_str or 'under' in exp_str:
        return 0.5
    
    # Extract numbers
    numbers = re.findall(r'\d+', exp_str)
    if numbers:
        return float(numbers[0])
    
    return 0.0
```

**Logic đơn giản hơn Salary:**

1. Check "không yêu cầu" → return 0.0
2. Check "dưới" → return 0.5 (ước lượng < 1 năm)
3. Extract số đầu tiên tìm được → return số đó

**Ví dụ:**

| Input | Output |
|-------|--------|
| "3 năm" | 3.0 |
| "Dưới 1 năm" | 0.5 |
| "Không yêu cầu" | 0.0 |
| "2-3 năm" | 2.0 (lấy số đầu) |

---

### 🔹 BƯỚC 4: Clean Location

#### Vấn đề:
```python
"Hồ Chí Minh (mới)"              # Extra text
"Hồ Chí Minh (mới) & 9 nơi khác" # Multiple locations
```

#### Giải pháp:

```python
def clean_location(self, location_str: str) -> str:
    """
    Clean and standardize location string
    
    Examples:
        'Hồ Chí Minh (mới)' -> 'Hồ Chí Minh'
        'Hồ Chí Minh (mới) & 9 nơi khác' -> 'Hồ Chí Minh'
    """
    if pd.isna(location_str):
        return 'Unknown'
    
    location_str = str(location_str)
    
    # Remove (mới), (new), etc.
    location_str = re.sub(r'\s*\([^)]*\)', '', location_str)
    
    # Take first location if multiple
    if '&' in location_str:
        location_str = location_str.split('&')[0]
    
    return location_str.strip()
```

**Giải thích:**

```python
re.sub(r'\s*\([^)]*\)', '', location_str)
```
- **Regex pattern:** `\s*\([^)]*\)`
  - `\s*`: 0 hoặc nhiều spaces
  - `\(`: Dấu mở ngoặc `(`
  - `[^)]*`: Bất kỳ ký tự nào không phải `)`, lặp lại 0+ lần
  - `\)`: Dấu đóng ngoặc `)`
- → Tìm và xóa mọi thứ trong ngoặc đơn

```python
if '&' in location_str:
    location_str = location_str.split('&')[0]
```
- Nếu có `&` (nhiều locations) → chỉ lấy phần đầu tiên

**Ví dụ:**

| Input | Output |
|-------|--------|
| "Hồ Chí Minh (mới)" | "Hồ Chí Minh" |
| "Hà Nội & Hưng Yên" | "Hà Nội" |
| "Đà Nẵng (new)" | "Đà Nẵng" |

---

### 🔹 BƯỚC 5: Handle Missing Values

```python
# Handle missing values in text fields
text_columns = ['Title', 'Job Requirements', 'Job description', 'benefit']
for col in text_columns:
    df_processed[col] = df_processed[col].fillna('')
```

**Giải thích:**
- `fillna('')`: Thay thế các cell trống bằng empty string `""`
- Tại sao? Vì các bước sau sẽ concatenate text → không muốn có `NaN`

---

### 🔹 BƯỚC 6: Create Combined Text

```python
df_processed['combined_text'] = (
    df_processed['Title'] + ' ' + 
    df_processed['Job Requirements'] + ' ' + 
    df_processed['Job description']
)
```

**Giải thích:**
- Gộp 3 columns text thành 1 column
- Tại sao? 
  - Bước tiếp theo (embedding) sẽ convert text → vector
  - Gộp lại để có **1 vector duy nhất** cho mỗi job
  - Vector này chứa thông tin từ cả Title, Requirements, Description

**Ví dụ:**

```python
Title: "Kế Toán Thuế"
Requirements: "3 năm kinh nghiệm"
Description: "Làm báo cáo thuế"

→ combined_text: "Kế Toán Thuế 3 năm kinh nghiệm Làm báo cáo thuế"
```

---

## 5. KẾT QUẢ

### Before Preprocessing:

```python
JobID: J001
Salary: "18 - 25 triệu"              # Text
Experience: "3 năm"                  # Text
Job Address: "Hồ Chí Minh (mới)"     # Unclean
```

### After Preprocessing:

```python
JobID: J001
salary_min: 18.0                     # Float
salary_max: 25.0                     # Float
experience_years: 3.0                # Float
location_clean: "Hồ Chí Minh"        # Clean text
combined_text: "Kế Toán Thuế / ..."  # Gộp text
```

### Statistics:

```
✓ 500 jobs processed
✓ Salary range: 0.0 - 75,000.0 million VND
✓ Experience range: 0.0 - 5.0 years
✓ Unique locations: 21
✓ Unique companies: 343
✓ No missing values in critical fields
```

### Output File: `processed/jobs_processed.csv`

CSV file với các columns mới:
```
JobID, Title, Name company, ..., 
salary_min, salary_max,           # ← NEW
experience_years,                 # ← NEW
location_clean,                   # ← NEW
combined_text                     # ← NEW
```

---

## 6. FAQ

### Q1: Tại sao salary "Thoả thuận" = (0, 0)?
**A:** Vì không có thông tin cụ thể. Có thể:
- Option 1: Set = 0 (approach hiện tại)
- Option 2: Set = giá trị trung bình của tất cả jobs
- Option 3: Loại bỏ hoàn toàn

Hiện tại dùng Option 1 vì đơn giản, và GNN vẫn có thể học từ các features khác.

### Q2: USD convert * 25 có chính xác không?
**A:** Là ước lượng:
- 1 USD ≈ 25,000 VND
- Trong database, salary VND tính theo **triệu**
- → 1 USD ≈ 25 triệu VND là đúng
- Ví dụ: 3,000 USD = 75,000 triệu VND ✓

### Q3: Tại sao chỉ lấy location đầu tiên?
**A:** Simplification:
- Graph hiện tại: 1 job → 1 location (easier to model)
- Nếu muốn multiple locations → cần thiết kế graph khác (1 job có thể link tới nhiều locations)

### Q4: Combined text có nên include "benefit" không?
**A:** Hiện tại chỉ gộp Title + Requirements + Description vì:
- 3 fields này chứa **job content chính**
- Benefit thường generic ("BHXH, thưởng lễ tết...") → ít discriminative

Có thể experiment với việc thêm benefit sau.

### Q5: Regex phức tạp quá, có cách nào đơn giản hơn?
**A:** Có thể dùng:
```python
# Instead of regex
if "triệu" in salary_str:
    # Extract số manually
```

Nhưng regex **mạnh hơn** vì:
- Handle nhiều cases khác nhau
- Tự động extract numbers
- Code ngắn gọn hơn

---

## 📌 TÓM TẮT

**Input:** Excel file với 500 jobs, 12 columns

**Xử lý:**
1. ✅ Salary: text → (min, max) float
2. ✅ Experience: text → float (years)
3. ✅ Location: clean text
4. ✅ Combined text: gộp 3 fields
5. ✅ Handle missing values

**Output:** CSV file với dữ liệu sạch, sẵn sàng cho bước tiếp theo

---

**👉 Tiếp theo: [Part 2: Text Embedding](02_Text_Embedding.md)**

---

*Part 1 - Data Preprocessing | NCKH Project*
