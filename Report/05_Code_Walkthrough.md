# Part 5: CODE WALKTHROUGH - ĐI QUA TỪNG FILE CODE

---

## 📁 CÁC FILE PYTHON

### 1. `config.py` - Cấu hình

```python
# File paths
RAW_DATA_PATH = "raw/db_job_tuan.xlsx"
PROCESSED_DATA_PATH = "processed/"
GRAPH_DATA_PATH = "graph_data/"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Graph construction
SIMILARITY_THRESHOLD = 0.6
TOP_K_SIMILAR_JOBS = 10

# Node types & edge types
NODE_TYPES = ["job", "company", "location"]
EDGE_TYPES = [...]
```

**Mục đích:** Centralize configs → dễ thay đổi parameters

---

### 2. `data_preprocessing.py` - Xử lý dữ liệu

**Class:** `JobDataPreprocessor`

**Main methods:**
```python
load_data()              # Load Excel → DataFrame
normalize_salary()       # "18-25 triệu" → (18.0, 25.0)
normalize_experience()   # "3 năm" → 3.0
clean_location()         # "Hà Nội (mới)" → "Hà Nội"
preprocess()            # Main pipeline
save_processed_data()    # Save to CSV
```

**Chi tiết:** Xem [Part 1: Data Preprocessing](01_Data_Preprocessing.md)

---

### 3. `text_embedding.py` - Text → Vectors

**Class:** `TextEmbedder`

**Main methods:**
```python
__init__()                    # Load Sentence Transformer model
embed_texts()                 # Text list → embeddings (500, 384)
compute_similarity_matrix()   # Embeddings → similarity (500, 500)
find_similar_jobs()          # Similarity → edge pairs
```

**Function:** `embed_job_data()`
```python
def embed_job_data(df, save_path):
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(df['combined_text'])
    similarity = embedder.compute_similarity_matrix(embeddings)
    # Save
    np.save(f"{save_path}job_embeddings.npy", embeddings)
    np.save(f"{save_path}similarity_matrix.npy", similarity)
    return df, embeddings, similarity
```

**Chi tiết:** Xem [Part 2: Text Embedding](02_Text_Embedding.md)

---

### 4. `graph_construction.py` - Xây dựng Graph

**Class:** `HeterogeneousJobGraph`

**Main methods:**
```python
_create_entity_mappings()       # Job/Company/Location → indices
_create_job_features()          # 399-dim features
_create_company_features()      # 10-dim features
_create_location_features()     # 8-dim features
_create_edges()                 # All edge types
build_graph()                   # Combine → HeteroData
save_graph()                    # Save to .pt file
```

**Flow:**
```python
1. Load data (CSV, embeddings, similarity)
2. Create mappings (name → index)
3. Create features for each node type
4. Create edges for each edge type
5. Build PyTorch Geometric HeteroData
6. Save graph
```

**Chi tiết:** Xem [Part 3: Graph Construction](03_Graph_Construction.md)

---

### 5. `visualization.py` - Visualize

**Class:** `GraphVisualizer`

**Main methods:**
```python
print_graph_summary()        # Print statistics
plot_graph_statistics()      # 6 charts
plot_subgraph()             # Network visualization
```

**Chi tiết:** Xem [Part 4: Visualization](04_Visualization.md)

---

### 6. `main.py` - Main Pipeline

```python
def main():
    # Step 1: Preprocessing
    preprocessor = JobDataPreprocessor()
    df = preprocessor.preprocess()
    preprocessor.save_processed_data(df)
    
    # Step 2: Embedding
    df, embeddings, similarity = embed_job_data(df, config.PROCESSED_DATA_PATH)
    
    # Step 3: Graph Construction
    graph_builder = HeterogeneousJobGraph(df, embeddings, similarity)
    graph = graph_builder.build_graph()
    graph_builder.save_graph()
    
    # Step 4: Visualization
    visualizer = GraphVisualizer(graph)
    visualizer.print_graph_summary()
    visualizer.plot_graph_statistics()
    visualizer.plot_subgraph()
```

**Chạy:** `python main.py`

---

### 7. `demo.py` - Exploration Demo

```python
def explore_job(graph, mappings, job_idx):
    # Show job details
    # Find company connection
    # Find similar jobs
    
def explore_company(graph, mappings, company_idx):
    # Show company info
    # List all job postings
    
def explore_location(graph, mappings, location_idx):
    # Show location stats
    # Salary distribution
```

**Chạy:** `python demo.py`

---

## 🔄 WORKFLOW TỔNG THỂ

```
                    main.py
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
  preprocessing   embedding   graph_construction
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                 visualization
                       │
                       ▼
                  Output files
```

---

## 🔑 KEY CONCEPTS

### 1. Class-based Organization
```python
# Mỗi module = 1 class
JobDataPreprocessor  # data_preprocessing.py
TextEmbedder         # text_embedding.py
HeterogeneousJobGraph # graph_construction.py
GraphVisualizer      # visualization.py
```

**Lợi ích:**
- Code organized
- Reusable
- Easy to test

### 2. Config Centralization
```python
# Không hardcode values trong code
# BAD:
model = SentenceTransformer("sentence-transformers/...")

# GOOD:
model = SentenceTransformer(config.EMBEDDING_MODEL)
```

### 3. Progressive Data Flow
```python
Excel → CSV → Embeddings → Graph → Visualization
```

Mỗi bước lưu intermediate results → có thể resume từ bất kỳ bước nào

---

## 📌 TÓM TẮT

**7 files chính:**
1. `config.py` - Configs
2. `data_preprocessing.py` - Data cleaning
3. `text_embedding.py` - Text → vectors
4. `graph_construction.py` - Build graph
5. `visualization.py` - Visualize
6. `main.py` - Main pipeline
7. `demo.py` - Exploration

**Design principles:**
- Modular (mỗi file = 1 responsibility)
- Reusable (classes có thể dùng độc lập)
- Progressive (save intermediate results)

---

**👉 Tiếp theo: [Part 6: Research Directions](06_Research_Directions.md)**

---

*Part 5 - Code Walkthrough | NCKH Project*
