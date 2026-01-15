# Part 2: TEXT EMBEDDING - CHUYỂN TEXT THÀNH VECTORS

**File code tương ứng:** `text_embedding.py`  
**Input:** `processed/jobs_processed.csv` (combined_text column)  
**Output:** `processed/job_embeddings.npy`, `processed/similarity_matrix.npy`

---

## 📚 MỤC LỤC

1. [Embedding là gì?](#1-embedding-là-gì)
2. [Tại sao cần Embedding?](#2-tại-sao-cần-embedding)
3. [Sentence Transformers](#3-sentence-transformers)
4. [Tính Similarity Matrix](#4-tính-similarity-matrix)
5. [Code chi tiết](#5-code-chi-tiết)
6. [Kết quả](#6-kết-quả)
7. [FAQ](#7-faq)

---

## 1. EMBEDDING LÀ GÌ?

### Định nghĩa đơn giản:

**Embedding** = Chuyển text (hoặc bất kỳ data nào) thành **vector số**

```
Text                    →    Vector (embedding)
"Kế Toán Thuế"          →    [0.2, -0.5, 0.8, ..., 0.1]  (384 chiều)
```

### Tại sao là vector?

Computer **không hiểu text**, chỉ hiểu **số**!

```
❌ Computer không thể tính toán với text:
   "Kế Toán" + "Nhân Viên" = ???

✓ Computer có thể tính toán với vectors:
   [0.2, 0.5] + [0.3, 0.1] = [0.5, 0.6]
```

### Ví dụ đơn giản:

**One-Hot Encoding** (cách cơ bản nhất):

```python
Vocabulary: ["kế", "toán", "nhân", "viên"]

"kế toán"  → [1, 1, 0, 0]
"nhân viên" → [0, 0, 1, 1]
```

**Vấn đề:** Vector quá dài, không capture meaning!

**Sentence Embedding** (cách hiện đại):

```python
"kế toán"  → [0.2, -0.5, 0.8, 0.3, ...]  (384 chiều)
"nhân viên" → [0.1, -0.4, 0.6, 0.2, ...]  (384 chiều)
```

**Ưu điểm:** 
- Vector ngắn hơn (384 vs 10,000s)
- Capture **semantic meaning** (nghĩa)
- Similar words → similar vectors

---

## 2. TẠI SAO CẦN EMBEDDING?

### Mục tiêu: Tìm jobs tương tự

**Câu hỏi:** Làm sao biết 2 jobs "tương tự" nhau?

#### ❌ Cách 1: So sánh text trực tiếp
```python
job1 = "Kế Toán Thuế"
job2 = "Accountant"

if job1 == job2:  # False!
    print("Similar")
```

**Vấn đề:** 
- Khác ngôn ngữ → không match
- Synonym (từ đồng nghĩa) → không match
- Chỉ match **exact text**

#### ✓ Cách 2: So sánh embeddings
```python
job1_vec = [0.2, -0.5, 0.8, ...]  # "Kế Toán Thuế"
job2_vec = [0.3, -0.4, 0.7, ...]  # "Accountant"

similarity = cosine_similarity(job1_vec, job2_vec)
# → 0.95 (very similar!)
```

**Ưu điểm:**
- Hiểu **nghĩa** (semantic)
- Cross-lingual (nhiều ngôn ngữ)
- Tìm được similar jobs ngay cả khi text khác nhau

---

## 3. SENTENCE TRANSFORMERS

### Model được dùng:

```python
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Đặc điểm:

| Feature | Value |
|---------|-------|
| **Model type** | Sentence Transformer |
| **Base model** | MiniLM (Microsoft) |
| **Languages** | Multilingual (50+ languages) |
| **Vietnamese support** | ✅ YES |
| **Embedding dim** | 384 |
| **Parameters** | ~118M |
| **Speed** | Fast (~500 sentences/sec) |

### Tại sao chọn model này?

#### ✅ 1. Multilingual (hỗ trợ tiếng Việt)
```python
# Model hiểu cả tiếng Việt và tiếng Anh
"Kế Toán Thuế"  ≈  "Tax Accountant"
# → Similar embeddings!
```

Quan trọng vì:
- Job postings ở Việt Nam thường **mix VN-EN**
- Ví dụ: "IT Project Manager", "Kế Toán Tổng Hợp"

#### ✅ 2. Sentence-level (not word-level)
```python
# Word embedding (Word2Vec, GloVe):
"Kế" → vector
"Toán" → vector
# Phải tự combine

# Sentence embedding (Sentence Transformers):
"Kế Toán Thuế" → 1 vector duy nhất
# Đã capture meaning của cả câu!
```

#### ✅ 3. Pre-trained tốt
```python
# Pre-trained on:
# - Paraphrase datasets
# - Translation pairs
# - Q&A pairs
# → Learned semantic similarity!
```

#### ✅ 4. Fast & Efficient
```python
# Small model: 118M parameters
# → Fast inference
# → Không cần GPU cũng chạy được
```

### Các alternatives:

| Model | Pros | Cons |
|-------|------|------|
| **PhoBERT** | Specialized for Vietnamese | Cần fine-tune, larger |
| **mBERT** | Multilingual | Slower, word-level |
| **USE** (Universal Sentence Encoder) | Good quality | English-only |
| **OpenAI Embeddings** | SOTA quality | Cần API key, cost $ |

→ **paraphrase-multilingual-MiniLM** = best balance!

---

## 4. TÍNH SIMILARITY MATRIX

### Mục tiêu:

Tính **similarity** (độ tương tự) giữa **mọi cặp jobs**

### Cosine Similarity:

**Formula:**
$$
\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

Trong đó:
- $A \cdot B$: Dot product
- $\|A\|$: Norm của vector A

**Giá trị:**
- 1.0: Hoàn toàn giống nhau
- 0.0: Không liên quan
- -1.0: Ngược nghĩa (hiếm khi xảy ra với embeddings)

**Ví dụ trực quan:**

```
Vector A = [1, 0]
Vector B = [1, 0]  → similarity = 1.0 (identical)

Vector A = [1, 0]
Vector B = [0, 1]  → similarity = 0.0 (orthogonal)

Vector A = [1, 0]
Vector B = [0.7, 0.7]  → similarity = 0.7 (similar direction)
```

### Similarity Matrix:

**Định nghĩa:** Ma trận 500 × 500 chứa similarity giữa mọi cặp jobs

```
           Job1   Job2   Job3   ...   Job500
Job1    [  1.0    0.77   0.23  ...    0.15 ]
Job2    [  0.77   1.0    0.31  ...    0.42 ]
Job3    [  0.23   0.31   1.0   ...    0.68 ]
...
Job500  [  0.15   0.42   0.68  ...    1.0  ]
```

**Properties:**
- **Diagonal = 1.0**: Job so với chính nó → similarity = 1
- **Symmetric**: similarity(A, B) = similarity(B, A)
- **Range**: 0.0 - 1.0

---

## 5. CODE CHI TIẾT

### 🔹 Class `TextEmbedder`

```python
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    """Generate embeddings for job text data"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
```

**Giải thích:**

```python
self.model = SentenceTransformer(self.model_name)
```
- Load pre-trained model từ HuggingFace
- Lần đầu: Download ~450MB (chỉ 1 lần)
- Lần sau: Load từ cache (fast)

```python
self.model.get_sentence_embedding_dimension()
```
- Lấy số chiều của embedding
- → 384 cho model này

---

### 🔹 Generate Embeddings

```python
def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """
    Generate embeddings for a list of texts
    
    Args:
        texts: List of text strings (500 jobs)
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
        
    Returns:
        numpy array of shape (n_texts, embedding_dim)
        → (500, 384)
    """
    print(f"\nEmbedding {len(texts)} texts...")
    embeddings = self.model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings
```

**Giải thích từng parameter:**

```python
texts = [
    "Kế Toán Thuế 3 năm kinh nghiệm...",
    "IT Project Manager...",
    ...
]  # 500 combined_text strings
```

```python
batch_size=32
```
- **Batch processing**: Xử lý 32 texts cùng lúc thay vì 1
- **Tại sao?** Faster! (GPU/CPU parallelization)
- 500 texts ÷ 32 = ~16 batches

```python
show_progress_bar=True
```
- Hiển thị progress bar:
```
Batches: 100%|██████████| 16/16 [00:11<00:00,  1.41it/s]
```

```python
convert_to_numpy=True
```
- Output format: NumPy array (not Torch tensor)
- Easier to save and manipulate

**Output:**
```python
embeddings.shape = (500, 384)
```
- 500 jobs
- 384 dimensions per job

---

### 🔹 Compute Similarity Matrix

```python
def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between embeddings
    
    Args:
        embeddings: numpy array of shape (n, dim) → (500, 384)
        
    Returns:
        Similarity matrix of shape (n, n) → (500, 500)
    """
    print("\nComputing similarity matrix...")
    
    # Step 1: Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    # Step 2: Compute cosine similarity
    similarity = np.dot(normalized, normalized.T)
    
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    
    return similarity
```

**Giải thích từng bước:**

#### Step 1: Normalize embeddings

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
```

**Norm** = độ dài của vector

Ví dụ:
```python
vector = [3, 4]
norm = sqrt(3² + 4²) = sqrt(9 + 16) = 5
```

```python
normalized = embeddings / (norms + 1e-8)
```

**Normalize** = chia cho norm

Ví dụ:
```python
vector = [3, 4]
norm = 5
normalized = [3/5, 4/5] = [0.6, 0.8]
# Norm of normalized = sqrt(0.6² + 0.8²) = 1.0 ✓
```

**Tại sao normalize?**

Cosine similarity formula:
$$
\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
$$

Nếu đã normalize ($\|A\| = \|B\| = 1$):
$$
\cos(\theta) = A \cdot B
$$

→ Chỉ cần dot product! (faster computation)

#### Step 2: Compute similarity

```python
similarity = np.dot(normalized, normalized.T)
```

**Matrix multiplication:**

```
normalized: (500, 384)
normalized.T: (384, 500)  ← Transpose

Result: (500, 500)
```

**Mỗi element:**
```python
similarity[i][j] = dot(normalized[i], normalized[j])
                 = cosine_similarity(job_i, job_j)
```

**Output:**
```python
similarity.shape = (500, 500)
similarity.min() ≈ -0.091  (ít tương tự)
similarity.max() = 1.000   (giống hệt - diagonal)
```

---

### 🔹 Find Similar Job Pairs

```python
def find_similar_jobs(
    self, 
    similarity_matrix: np.ndarray, 
    threshold: float = 0.6,
    top_k: int = 10
) -> List[Tuple[int, int, float]]:
    """
    Find similar job pairs based on threshold and top-k
    
    Returns:
        List of (job_i, job_j, similarity) tuples
    """
    n_jobs = similarity_matrix.shape[0]  # 500
    edges = []
    
    for i in range(n_jobs):
        # Get similarities for job i (excluding self)
        sims = similarity_matrix[i].copy()
        sims[i] = -1  # Exclude self-loop
        
        # Get top-K most similar jobs
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        for j in top_indices:
            if sims[j] >= threshold and i < j:  # Avoid duplicates
                edges.append((i, j, float(sims[j])))
    
    return edges
```

**Giải thích logic:**

#### Loop qua mỗi job:

```python
for i in range(500):
    sims = similarity_matrix[i]  # Lấy row thứ i
    # sims = [1.0, 0.77, 0.23, ..., 0.15]
    #         ↑    ↑     ↑           ↑
    #       job_i job_1 job_2     job_499
```

#### Exclude self:

```python
sims[i] = -1
```
- Set similarity với chính nó = -1
- Tại sao? Để không lấy job so với chính nó

#### Find top-K:

```python
top_indices = np.argsort(sims)[-top_k:][::-1]
```

**Giải thích `np.argsort`:**

```python
sims = [0.5, 0.9, 0.3, 0.7]

np.argsort(sims) = [2, 0, 3, 1]  # Indices sorted by value
# sims[2]=0.3 < sims[0]=0.5 < sims[3]=0.7 < sims[1]=0.9

[-top_k:]  # Lấy k phần tử cuối (largest)
[::-1]     # Reverse (descending order)
```

**Ví dụ với top_k=3:**
```python
sims = [0.5, 0.9, -1, 0.7, 0.3]  # job[2] = self
argsort = [4, 0, 3, 1, 2]
[-3:] = [3, 1, 2]  # Top 3
[::-1] = [2, 1, 3] = [self, 0.9, 0.7]

# Filter self (similarity[2] = -1 < threshold)
→ Keep [1, 3]  # Jobs with sim 0.9, 0.7
```

#### Filter by threshold và avoid duplicates:

```python
if sims[j] >= threshold and i < j:
    edges.append((i, j, float(sims[j])))
```

- `sims[j] >= threshold`: Chỉ lấy similarity ≥ 0.6
- `i < j`: Tránh duplicate edges
  - Example: (job_1, job_5) và (job_5, job_1) là giống nhau
  - Chỉ lưu (1, 5), không lưu (5, 1)

**Output:**
```python
[
    (0, 23, 0.770),   # Job 0 similar to Job 23 (sim=0.77)
    (0, 40, 0.749),   # Job 0 similar to Job 40 (sim=0.75)
    (1, 15, 0.823),   # Job 1 similar to Job 15 (sim=0.82)
    ...
]
# Total: 2,182 pairs
```

---

## 6. KẾT QUẢ

### Embeddings:

```python
File: processed/job_embeddings.npy
Shape: (500, 384)
Size: ~768 KB

# Mỗi job → 1 vector 384 chiều
job_embeddings[0]  # Vector cho Job J001
→ array([0.0234, -0.1567, 0.0891, ..., 0.0245])
```

### Similarity Matrix:

```python
File: processed/similarity_matrix.npy
Shape: (500, 500)
Size: ~1 MB

# Similarity giữa mọi cặp jobs
similarity_matrix[0, 23]  # Similarity giữa Job 0 và Job 23
→ 0.770
```

### Similar Job Pairs:

```python
Total: 2,182 pairs
Average similarity: 0.717
Range: 0.600 - 0.999

Top 5 examples:
1. Job 0 ↔ Job 23: 0.770
   "Kế Toán Thuế" ↔ "Kế Toán Tổng Hợp"
   
2. Job 0 ↔ Job 40: 0.749
   "Kế Toán Thuế" ↔ "Kế Toán Tổng Hợp"
   
3. Job 1 ↔ Job 5: 0.823
   "Nhân Viên Tín Dụng" ↔ "Nhân Viên Thu Hồi Nợ"
   
...
```

### Visualization:

**Similarity Distribution:**

```
[0.6 - 0.65): ████████ 412 pairs
[0.65 - 0.70): ███████████ 687 pairs
[0.70 - 0.75): █████████ 589 pairs
[0.75 - 0.80): ██████ 312 pairs
[0.80 - 0.85): ███ 128 pairs
[0.85 - 0.90): ██ 45 pairs
[0.90 - 0.95): █ 7 pairs
[0.95 - 1.00): █ 2 pairs
```

→ Most pairs: 0.65 - 0.75 (reasonable similarity)

---

## 7. FAQ

### Q1: Tại sao embedding dimension = 384?
**A:** 
- Trade-off giữa **quality** và **efficiency**
- Nhỏ hơn (128): Faster, nhưng loss information
- Lớn hơn (768, 1024): Better quality, nhưng slower
- 384 = sweet spot cho model này

### Q2: Có thể dùng GPT/OpenAI embeddings không?
**A:** Có, nhưng:
- **Pros**: Quality tốt hơn (1536 dims)
- **Cons**: 
  - Cần API key
  - Cost $$ (pay per request)
  - Phụ thuộc internet
  
→ Sentence Transformers = free, offline, good enough!

### Q3: Tại sao threshold = 0.6?
**A:** 
- Empirical choice (thử nghiệm)
- < 0.6: Quá khác nhau, không similar
- ≥ 0.6: Reasonable similarity
- Có thể adjust: 0.5 (nhiều edges hơn), 0.7 (ít edges hơn)

### Q4: Top-K = 10 có phù hợp không?
**A:**
- 10 = mỗi job connect tới 10 jobs gần nhất
- **Sparse graph**: Tốt cho GNN (tránh overfitting)
- Có thể adjust: 5 (sparser), 20 (denser)

### Q5: Model có hiểu tiếng Việt tốt không?
**A:** 
Khá tốt! Ví dụ:
```python
"Kế Toán Thuế" similar to "Kế Toán Tổng Hợp": 0.77 ✓
"IT Project Manager" similar to "Project Manager IT": 0.95 ✓
```

Nhưng không perfect:
- Slang, abbreviations có thể không hiểu
- Industry-specific terms cần fine-tuning

### Q6: Có cần GPU không?
**A:** **Không bắt buộc!**
- CPU: ~11s cho 500 jobs (acceptable)
- GPU: ~2s cho 500 jobs (faster, nhưng không necessary)

### Q7: Similarity matrix có sparse không?
**A:** 
```python
# Full matrix: 500 × 500 = 250,000 values
# Similarity ≥ 0.6: 2,182 pairs ÷ 250,000 = 0.87%

→ Very sparse! ✓ (good for GNN)
```

---

## 📌 TÓM TẮT

**Input:** 500 combined_text strings

**Process:**
1. ✅ Load Sentence Transformer model (multilingual)
2. ✅ Generate embeddings: 500 × 384 vectors
3. ✅ Compute similarity matrix: 500 × 500
4. ✅ Find similar pairs: 2,182 edges (threshold ≥ 0.6)

**Output:** 
- `job_embeddings.npy`: Vector representations
- `similarity_matrix.npy`: Pairwise similarities
- Similar job pairs: For graph construction

**Key insights:**
- Embeddings capture semantic meaning
- Similar jobs have high cosine similarity
- Sparse similarity graph (1.7% density)

---

**👉 Tiếp theo: [Part 3: Graph Construction](03_Graph_Construction.md)**

---

*Part 2 - Text Embedding | NCKH Project*
