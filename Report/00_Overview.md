# 📚 BÁO CÁO TỔNG QUAN - XÂY DỰNG HETEROGENEOUS GRAPH CHO JOB RECOMMENDATION

**Tác giả:** NCKH Project  
**Ngày:** 15/01/2026  
**Mục tiêu:** Xây dựng Heterogeneous Graph từ dữ liệu job posting để phục vụ nghiên cứu GNN

---

## 📋 MỤC LỤC BÁO CÁO

### [Part 1: Data Preprocessing](01_Data_Preprocessing.md)
- Giải thích dữ liệu đầu vào
- Cách chuẩn hóa Salary, Experience
- Làm sạch Location
- Tạo combined text

### [Part 2: Text Embedding](02_Text_Embedding.md)
- Embedding là gì?
- Tại sao dùng Sentence Transformers?
- Cách tính similarity matrix
- Tìm similar jobs

### [Part 3: Graph Construction](03_Graph_Construction.md)
- Heterogeneous Graph là gì?
- Node types và features
- Edge types và relationships
- PyTorch Geometric format

### [Part 4: Visualization](04_Visualization.md)
- Cách visualize graph
- Phân tích statistics
- Subgraph visualization

### [Part 5: Code Walkthrough](05_Code_Walkthrough.md)
- Đi qua từng file code
- Giải thích từng function quan trọng
- Flow của toàn bộ pipeline

### [Part 6: Research Directions](06_Research_Directions.md)
- Các hướng nghiên cứu tiếp theo
- So sánh với các paper
- Suggestions cho improvement

---

## 🎯 TỔNG QUAN NGẮN GỌN

### Vấn đề cần giải quyết:
Xây dựng hệ thống **gợi ý công việc** từ CV người dùng sử dụng **Graph Neural Networks (GNN)**

### Approach:
1. **Thu thập dữ liệu**: 500 job postings từ Việt Nam (file Excel)
2. **Xây dựng Graph**: Biểu diễn jobs, companies, locations dưới dạng graph
3. **Chuẩn bị cho GNN**: Format data theo PyTorch Geometric

### Kết quả:
- ✅ Heterogeneous Graph với 864 nodes, 6,364 edges
- ✅ Rich features cho mỗi node
- ✅ Sẵn sàng cho nghiên cứu GNN models

---

## 🔍 GIẢI THÍCH CÁC KHÁI NIỆM CƠ BẢN

### 1. Graph là gì?

**Graph** (đồ thị) gồm 2 thành phần chính:
- **Nodes (đỉnh)**: Các thực thể (ví dụ: jobs, companies, locations)
- **Edges (cạnh)**: Mối quan hệ giữa các nodes (ví dụ: job thuộc company)

**Ví dụ đơn giản:**
```
Job1 ---[posted_by]---> Company_A
Job2 ---[posted_by]---> Company_A
Job1 ---[similar_to]---> Job2
Job1 ---[located_in]---> Hanoi
```

### 2. Heterogeneous Graph là gì?

**Homogeneous Graph**: Chỉ có 1 loại node và 1 loại edge
```
User1 --- User2 --- User3 (tất cả là User nodes)
```

**Heterogeneous Graph**: Nhiều loại nodes và nhiều loại edges
```
Job1 ---[posted_by]---> Company_A
Job1 ---[located_in]---> Hanoi
Job1 ---[similar_to]---> Job2
```

→ **Project này dùng Heterogeneous Graph** vì có 3 loại nodes: Job, Company, Location

### 3. Graph Neural Network (GNN) là gì?

**GNN** là mạng neural học từ cấu trúc graph:
- Mỗi node có **features** (đặc trưng)
- GNN **tổng hợp thông tin** từ các node láng giềng
- Sau nhiều layers, mỗi node có **embedding** (vector đại diện)

**Ứng dụng:**
- Job Recommendation: Dự đoán user thích job nào
- Node Classification: Phân loại job theo ngành
- Link Prediction: Dự đoán user sẽ apply job nào

### 4. Tại sao dùng Graph cho Job Recommendation?

**Cách truyền thống**: Chỉ xem CV và Job description
```
CV (text) --> Model --> Matching Score
```

**Cách dùng Graph**: Khai thác mối quan hệ
```
CV --> User node
Job node có connections:
  - Thuộc Company nào?
  - Ở Location nào?
  - Giống Jobs nào khác?
  - Company đó còn Jobs gì?

--> GNN học từ toàn bộ structure để recommend tốt hơn!
```

---

## 📊 PIPELINE TỔNG QUAN

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Excel File                        │
│              500 jobs với 12 columns                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: Data Preprocessing                     │
│  - Chuẩn hóa Salary: "18-25 triệu" → (18.0, 25.0)         │
│  - Chuẩn hóa Experience: "3 năm" → 3.0                     │
│  - Clean Location: "Hà Nội (mới)" → "Hà Nội"              │
│  - Tạo combined_text để embedding                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: Text Embedding                         │
│  - Dùng Sentence Transformer (multilingual model)          │
│  - Convert text → vectors 384 chiều                        │
│  - Tính similarity giữa các jobs                           │
│  - Tìm top-10 similar jobs cho mỗi job                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│            STEP 3: Graph Construction                       │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Nodes:                                          │      │
│  │    - Job: 500 nodes (399 features)              │      │
│  │    - Company: 343 nodes (10 features)           │      │
│  │    - Location: 21 nodes (8 features)            │      │
│  └──────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Edges:                                          │      │
│  │    - Job → Company: 500 edges                    │      │
│  │    - Job → Location: 500 edges                   │      │
│  │    - Job → Job (similar): 4,364 edges            │      │
│  └──────────────────────────────────────────────────┘      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│            STEP 4: Save & Visualize                         │
│  - Lưu graph dạng PyTorch Geometric format                 │
│  - Tạo biểu đồ thống kê                                    │
│  - Visualize subgraph                                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                OUTPUT: Heterogeneous Graph                  │
│     - hetero_graph.pt (graph file)                         │
│     - job_embeddings.npy (embeddings)                      │
│     - similarity_matrix.npy (similarity scores)            │
│     - Visualization images                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 KẾT QUẢ ĐẠT ĐƯỢC

### 1. Graph Structure:
```
864 nodes tổng cộng:
  ├── 500 Job nodes
  ├── 343 Company nodes
  └── 21 Location nodes

6,364 edges tổng cộng:
  ├── 500 Job → Company
  ├── 500 Company → Job (reverse)
  ├── 500 Job → Location
  ├── 500 Location → Job (reverse)
  └── 4,364 Job ↔ Job (similarity, bidirectional)
```

### 2. Features:
```
Job nodes: 399 dimensions
  ├── Text embedding: 384 dims (multilingual)
  ├── Numerical: 4 dims (salary_min, salary_max, exp, quantity)
  └── Categorical: 11 dims (job_type, company_size one-hot)

Company nodes: 10 dimensions (aggregated stats)
Location nodes: 8 dimensions (aggregated stats)
```

### 3. Memory Usage:
- Total graph size: **0.77 MB** (rất nhẹ!)
- Job embeddings: **0.76 MB**
- Company + Location: **0.01 MB**

### 4. Data Quality:
- **Sparsity**: 1.7% (tốt cho GNN, tránh overfitting)
- **Similarity scores**: 0.6 - 0.8 (reasonable range)
- **No missing values** sau preprocessing

---

## 🎓 KIẾN THỨC CẦN CÓ

Để hiểu rõ báo cáo này, bạn nên biết:

### Cơ bản (bắt buộc):
- ✅ Python programming
- ✅ Pandas (xử lý data)
- ✅ NumPy (arrays, matrices)

### Trung cấp (nên biết):
- 📚 Machine Learning cơ bản
- 📚 Neural Networks
- 📚 Graph theory cơ bản

### Nâng cao (không bắt buộc):
- 🔬 Graph Neural Networks
- 🔬 PyTorch / PyTorch Geometric
- 🔬 Natural Language Processing

**→ Báo cáo sẽ giải thích chi tiết nên không cần lo!**

---

## 📁 CẤU TRÚC FILES

```
Graph/
├── Report/                          # ← BÁO CÁO (bạn đang đọc)
│   ├── 00_Overview.md              # Tổng quan (file này)
│   ├── 01_Data_Preprocessing.md    # Phần 1: Xử lý dữ liệu
│   ├── 02_Text_Embedding.md        # Phần 2: Embedding
│   ├── 03_Graph_Construction.md    # Phần 3: Xây dựng graph
│   ├── 04_Visualization.md         # Phần 4: Visualization
│   ├── 05_Code_Walkthrough.md      # Phần 5: Code chi tiết
│   └── 06_Research_Directions.md   # Phần 6: Hướng nghiên cứu
│
├── raw/                            # Dữ liệu gốc
│   └── db_job_tuan.xlsx
│
├── processed/                      # Dữ liệu đã xử lý
│   ├── jobs_processed.csv
│   ├── job_embeddings.npy
│   └── similarity_matrix.npy
│
├── graph_data/                     # Graph output
│   ├── hetero_graph.pt
│   ├── entity_mappings.pt
│   ├── graph_statistics.png
│   └── graph_subgraph.png
│
└── [Python files]                  # Code
    ├── config.py
    ├── data_preprocessing.py
    ├── text_embedding.py
    ├── graph_construction.py
    ├── visualization.py
    ├── main.py
    └── demo.py
```

---

## 🚀 CÁCH ĐỌC BÁO CÁO

### Đọc theo thứ tự:
1. **00_Overview.md** ← Bạn đang ở đây
2. **01_Data_Preprocessing.md** - Hiểu cách xử lý dữ liệu
3. **02_Text_Embedding.md** - Hiểu cách chuyển text thành vectors
4. **03_Graph_Construction.md** - Hiểu cách xây graph
5. **04_Visualization.md** - Hiểu cách visualize
6. **05_Code_Walkthrough.md** - Đi chi tiết vào code
7. **06_Research_Directions.md** - Hướng phát triển tiếp theo

### Mỗi phần sẽ có:
- ✅ Giải thích lý thuyết
- ✅ Ví dụ cụ thể
- ✅ Code minh họa
- ✅ Hình ảnh (nếu cần)
- ✅ Câu hỏi thường gặp (FAQ)

---

## ❓ CÂU HỎI THƯỜNG GẶP

### Q1: Tại sao cần xây dựng Graph?
**A:** Graph giúp biểu diễn **mối quan hệ** giữa các entities (Job, Company, Location). GNN có thể học từ structure này để recommend tốt hơn so với chỉ xem riêng lẻ từng job.

### Q2: Heterogeneous Graph khác gì Homogeneous?
**A:** 
- **Homogeneous**: 1 loại node, 1 loại edge (ví dụ: social network - tất cả là users)
- **Heterogeneous**: Nhiều loại nodes và edges (ví dụ: Job, Company, Location với các mối quan hệ khác nhau)

### Q3: Tại sao dùng PyTorch Geometric?
**A:** PyTorch Geometric là thư viện **chuẩn** cho GNN research:
- Hỗ trợ heterogeneous graphs
- Nhiều GNN models có sẵn (GCN, GAT, HAN, RGCN...)
- Community lớn, tài liệu đầy đủ
- Dễ implement paper mới

### Q4: 500 jobs có ít không?
**A:** 
- Cho **research/proof-of-concept**: Đủ rồi! ✓
- Cho **production**: Cần nhiều hơn (1000s - 100,000s jobs)
- Quan trọng là **methodology đúng**, scale up sau dễ

### Q5: Graph này có thể dùng cho research paper được không?
**A:** **Hoàn toàn được!** Graph format chuẩn, có thể:
- Implement các GNN models mới
- So sánh với baselines
- Nghiên cứu heterogeneous graph learning
- Publish ở conferences/journals

---

## 📞 LIÊN HỆ & HỖ TRỢ

Nếu có câu hỏi khi đọc báo cáo:
- Đọc phần **FAQ** ở cuối mỗi Part
- Xem **Code Walkthrough** (Part 5)
- Check **SUMMARY.md** để xem tổng quan

---

## ✨ BẮT ĐẦU ĐỌC

**👉 Chuyển sang [Part 1: Data Preprocessing](01_Data_Preprocessing.md)**

---

*Báo cáo được viết ngày 15/01/2026 - NCKH Project*
