# Part 3: GRAPH CONSTRUCTION - XÂY DỰNG HETEROGENEOUS GRAPH

**File code tương ứng:** `graph_construction.py`  
**Input:** 
- `processed/jobs_processed.csv`
- `processed/job_embeddings.npy`
- `processed/similarity_matrix.npy`

**Output:** 
- `graph_data/hetero_graph.pt`
- `graph_data/entity_mappings.pt`

---

## 📚 MỤC LỤC

1. [Heterogeneous Graph là gì?](#1-heterogeneous-graph-là-gì)
2. [Thiết kế Graph](#2-thiết-kế-graph)
3. [Node Features](#3-node-features)
4. [Edge Construction](#4-edge-construction)
5. [PyTorch Geometric Format](#5-pytorch-geometric-format)
6. [Code chi tiết](#6-code-chi-tiết)
7. [FAQ](#7-faq)

---

## 1. HETEROGENEOUS GRAPH LÀ GÌ?

### So sánh Homogeneous vs Heterogeneous:

#### Homogeneous Graph:
```
       User1 ---- User2
         |          |
       User3 ---- User4

- 1 loại node: User
- 1 loại edge: "friend"
```

#### Heterogeneous Graph:
```
    Job1 --posted_by--> Company_A
      |                     |
   located_in            posts
      |                     |
    Hanoi <------------  Job2
            similar_to

- 3 loại nodes: Job, Company, Location
- 3 loại edges: posted_by, located_in, similar_to
```

### Tại sao dùng Heterogeneous?

**Rich structure** = More information for GNN!

```python
# Homogeneous: Chỉ có Job nodes
Job1 --- Job2 --- Job3
# GNN chỉ học từ job-job relationships

# Heterogeneous: Nhiều loại nodes
Job1 --posted_by--> Company_A
Job1 --located_in--> Hanoi
Job1 --similar_to--> Job2
# GNN học từ:
# - Jobs từ cùng company có thể liên quan
# - Jobs ở cùng location có thể liên quan
# - Similar jobs theo content
```

---

## 2. THIẾT KẾ GRAPH

### 2.1 Node Types (3 loại)

```
┌─────────────────────────────────────┐
│         JOB NODES (500)             │
│  - Các công việc cần tuyển          │
│  - Features: 399 dims               │
│    (embeddings + numerical)         │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      COMPANY NODES (343)            │
│  - Các công ty tuyển dụng           │
│  - Features: 10 dims                │
│    (aggregated statistics)          │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      LOCATION NODES (21)            │
│  - Các địa điểm tuyển dụng          │
│  - Features: 8 dims                 │
│    (aggregated statistics)          │
└─────────────────────────────────────┘
```

### 2.2 Edge Types (5 loại)

```
1. Job --posted_by--> Company  (500 edges)
   Mỗi job được đăng bởi 1 company

2. Company --posts--> Job  (500 edges)
   Reverse edge của (1)

3. Job --located_in--> Location  (500 edges)
   Mỗi job ở 1 location

4. Location --has--> Job  (500 edges)
   Reverse edge của (3)

5. Job --similar_to--> Job  (4,364 edges)
   Jobs tương tự nhau (từ similarity matrix)
   Bidirectional: nếu A similar B → B similar A
```

### 2.3 Graph Structure Diagram

```
                  ┌──────────────┐
                  │  Company A   │
                  └──────┬───────┘
                         │ posts
                   posted_by
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼────┐      ┌───▼────┐      ┌───▼────┐
    │ Job 1  │◄─────┤ Job 2  ├─────►│ Job 3  │
    └───┬────┘ similar └───┬────┘ similar └───┬────┘
        │               │                  │
        │located_in     │located_in        │located_in
        │               │                  │
        │           ┌───▼──────────────────▼───┐
        └──────────►│      Location: Hanoi     │
                    └──────────────────────────┘
```

---

## 3. NODE FEATURES

### 3.1 Job Node Features (399 dims)

```python
Job features = [
    Text embeddings (384 dims),      # From Part 2
    Salary min (1 dim),              # Normalized
    Salary max (1 dim),              # Normalized
    Experience years (1 dim),        # Normalized
    Quantity (1 dim),                # Log-transformed
    Job type one-hot (~3 dims),      # Categorical
    Company size one-hot (~8 dims)   # Categorical
]
```

#### 3.1.1 Text Embeddings (384 dims)

```python
# Already computed in Part 2
job_embeddings = np.load('job_embeddings.npy')  # (500, 384)
text_features = torch.FloatTensor(job_embeddings)
```

#### 3.1.2 Numerical Features

```python
# Salary (2 dims)
salary_min = df['salary_min'].values  # [18.0, 0.0, 12.0, ...]
salary_max = df['salary_max'].values  # [25.0, 0.0, 16.0, ...]

# Normalize: divide by 100
salary_min = salary_min / 100.0  # Scale down
salary_max = salary_max / 100.0
```

**Tại sao normalize?**
- Original range: 0 - 75,000 (very large!)
- Neural networks prefer values ~ 0-1
- Chia 100: 0 - 750 (still large, but better)

```python
# Experience (1 dim)
experience = df['experience_years'].values  # [3.0, 0.5, 2.0, ...]

# Normalize: divide by 10
experience = experience / 10.0  # 0 - 0.5 range
```

```python
# Quantity (1 dim)
quantity = df['quantity'].values  # [1, 50, 1, 12, ...]

# Log transform (handle large values)
quantity = np.log1p(quantity)  # log(1 + x)
```

**Tại sao log transform?**
- Quantity có outliers: [1, 1, 1, 50, 1, ...]
- log1p(1) = 0.69
- log1p(50) = 3.93
- → Giảm impact của outliers

#### 3.1.3 Categorical Features (One-Hot)

```python
# Job type
job_types = pd.get_dummies(df['Job type'])
# "Toàn thời gian" → [1, 0, 0]
# "Bán thời gian" → [0, 1, 0]
# ...

# Company size
company_sizes = pd.get_dummies(df['company_size'])
# "25-99 nhân viên" → [1, 0, 0, 0, ...]
# "1000+ nhân viên" → [0, 1, 0, 0, ...]
# ...
```

#### 3.1.4 Concatenate All

```python
job_features = torch.cat([
    text_embeddings,      # 384
    salary_min,           # 1
    salary_max,           # 1
    experience,           # 1
    quantity,             # 1
    job_type_features,    # ~3
    company_size_features # ~8
], dim=1)

# Result: (500, 399)
```

---

### 3.2 Company Node Features (10 dims)

**Strategy:** Aggregated statistics từ jobs của company đó

```python
For each company:
    company_features[idx, 0] = num_jobs          # Số lượng jobs
    company_features[idx, 1] = avg_salary_max    # Lương TB
    company_features[idx, 2] = avg_salary_min
    company_features[idx, 3] = avg_experience    # Kinh nghiệm TB
    company_features[idx, 4] = total_quantity    # Tổng tuyển
    company_features[idx, 5-9] = size_encoding   # Company size
```

**Ví dụ:**
```python
Company "LG CNS VIỆT NAM":
  - 12 jobs posted
  - Avg salary: 0 (all "Thoả thuận")
  - Avg experience: 2.3 years
  - Total quantity: 15 positions
  - Size: 1000+ nhân viên → [0,0,0,0,1]
```

**Code:**
```python
def _create_company_features(self) -> torch.Tensor:
    n_companies = len(self.company_mapping)
    company_features = torch.zeros(n_companies, 10)
    
    for company, company_idx in self.company_mapping.items():
        # Get all jobs from this company
        company_jobs = self.df[self.df['Name company'] == company]
        
        # Aggregated features
        company_features[company_idx, 0] = len(company_jobs)
        company_features[company_idx, 1] = company_jobs['salary_max'].mean()
        company_features[company_idx, 2] = company_jobs['salary_min'].mean()
        company_features[company_idx, 3] = company_jobs['experience_years'].mean()
        company_features[company_idx, 4] = company_jobs['quantity'].sum()
        # ... size encoding ...
    
    # Normalize
    company_features[:, 1:5] = company_features[:, 1:5] / (company_features[:, 1:5].max(dim=0)[0] + 1e-8)
    
    return company_features
```

---

### 3.3 Location Node Features (8 dims)

**Strategy:** Tương tự Company, aggregate từ jobs tại location đó

```python
For each location:
    location_features[idx, 0] = num_jobs          # Số jobs
    location_features[idx, 1] = avg_salary_max    # Lương TB
    location_features[idx, 2] = avg_salary_min
    location_features[idx, 3] = avg_experience    # Kinh nghiệm TB
    location_features[idx, 4] = total_quantity    # Tổng tuyển
    location_features[idx, 5-7] = reserved        # Future use
```

**Ví dụ:**
```python
Location "Hà Nội":
  - 233 jobs
  - Avg salary max: 848 triệu
  - Avg salary min: 420 triệu
  - Avg experience: 2.1 years
  - Total quantity: 312 positions
```

---

## 4. EDGE CONSTRUCTION

### 4.1 Job → Company Edges

```python
# For each job, link to its company
job_company_edges = []
for idx, row in df.iterrows():
    job_idx = job_mapping[row['JobID']]        # J001 → 0
    company_idx = company_mapping[row['Name company']]  # "CÔNG TY..." → 15
    job_company_edges.append([job_idx, company_idx])

edge_index = torch.tensor(job_company_edges).t()
# Shape: (2, 500)
# [[0,    1,    2,    ...]  ← job indices
#  [15,   20,   15,   ...]] ← company indices
```

**Format giải thích:**
```
Edge (i, j) means: Job i --posted_by--> Company j

Edge list:
[0, 15] → Job 0 posted by Company 15
[1, 20] → Job 1 posted by Company 20
[2, 15] → Job 2 posted by Company 15 (same company as Job 0!)
```

### 4.2 Company → Job Edges (Reverse)

```python
# Simply flip the edge_index
edges[('company', 'posts', 'job')] = edges[('job', 'posted_by', 'company')].flip(0)

# [[15,   20,   15,   ...]  ← company indices
#  [0,    1,    2,    ...]] ← job indices
```

**Tại sao cần reverse edges?**

GNN message passing cần **bidirectional** information flow:
- Job → Company: "Job này thuộc company gì?"
- Company → Job: "Company này có jobs nào?"

### 4.3 Job → Location Edges

```python
# Similar to Job → Company
job_location_edges = []
for idx, row in df.iterrows():
    job_idx = job_mapping[row['JobID']]
    location_idx = location_mapping[row['location_clean']]
    job_location_edges.append([job_idx, location_idx])

edge_index = torch.tensor(job_location_edges).t()
# Shape: (2, 500)
```

### 4.4 Job ↔ Job Similarity Edges

**Key idea:** Sử dụng similarity matrix từ Part 2

```python
def _find_similar_jobs(self) -> List[Tuple[int, int, float]]:
    threshold = 0.6
    top_k = 10
    
    n_jobs = self.similarity_matrix.shape[0]
    edges = []
    
    for i in range(n_jobs):
        sims = self.similarity_matrix[i].copy()
        sims[i] = -1  # Exclude self
        
        # Top-K most similar
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        for j in top_indices:
            if sims[j] >= threshold and i < j:
                edges.append((i, j, float(sims[j])))
    
    return edges  # 2,182 pairs
```

**Convert to bidirectional:**
```python
# Create bidirectional edges
job_job_edges = [[i, j] for (i, j, sim) in similar_pairs]
job_job_edges_reverse = [[j, i] for (i, j, sim) in similar_pairs]

all_edges = job_job_edges + job_job_edges_reverse
# 2,182 × 2 = 4,364 edges

edge_index = torch.tensor(all_edges).t()
# Shape: (2, 4364)
```

**Tại sao bidirectional?**
- Similarity là symmetric: sim(A, B) = sim(B, A)
- GNN cần cả 2 directions để message passing

---

## 5. PYTORCH GEOMETRIC FORMAT

### 5.1 HeteroData Object

```python
from torch_geometric.data import HeteroData

graph = HeteroData()
```

**HeteroData** = Container cho heterogeneous graph

### 5.2 Add Node Features

```python
# Add job nodes
graph['job'].x = job_features  # (500, 399)

# Add company nodes
graph['company'].x = company_features  # (343, 10)

# Add location nodes
graph['location'].x = location_features  # (21, 8)
```

**Syntax:**
```python
graph[node_type].x = features
```

### 5.3 Add Edges

```python
# Add job → company edges
graph[('job', 'posted_by', 'company')].edge_index = job_company_edge_index

# Add reverse edges
graph[('company', 'posts', 'job')].edge_index = company_job_edge_index

# Add job → location edges
graph[('job', 'located_in', 'location')].edge_index = job_location_edge_index

# Add reverse edges
graph[('location', 'has', 'job')].edge_index = location_job_edge_index

# Add job ↔ job edges
graph[('job', 'similar_to', 'job')].edge_index = job_job_edge_index
```

**Syntax:**
```python
graph[(src_type, relation, dst_type)].edge_index = edge_index
```

### 5.4 Add Metadata

```python
# Store entity names for later reference
graph['job'].job_ids = list(job_mapping.keys())
graph['company'].company_names = list(company_mapping.keys())
graph['location'].location_names = list(location_mapping.keys())
```

### 5.5 Final Graph Structure

```python
HeteroData(
  job={
    x=[500, 399],           # Features
    job_ids=[500],          # Metadata
  },
  company={
    x=[343, 10],
    company_names=[343],
  },
  location={
    x=[21, 8],
    location_names=[21],
  },
  (job, posted_by, company)={ edge_index=[2, 500] },
  (company, posts, job)={ edge_index=[2, 500] },
  (job, located_in, location)={ edge_index=[2, 500] },
  (location, has, job)={ edge_index=[2, 500] },
  (job, similar_to, job)={ edge_index=[2, 4364] }
)
```

---

## 6. CODE CHI TIẾT

### Main Pipeline

```python
class HeterogeneousJobGraph:
    def build_graph(self) -> HeteroData:
        # Step 1: Create entity mappings
        self._create_entity_mappings()
        
        # Step 2: Create node features
        self.job_features = self._create_job_features()
        self.company_features = self._create_company_features()
        self.location_features = self._create_location_features()
        
        # Step 3: Create edges
        edges_dict = self._create_edges()
        
        # Step 4: Build HeteroData
        graph = HeteroData()
        
        # Add nodes
        graph['job'].x = self.job_features
        graph['company'].x = self.company_features
        graph['location'].x = self.location_features
        
        # Add edges
        for edge_type, edge_index in edges_dict.items():
            graph[edge_type].edge_index = edge_index
        
        # Add metadata
        graph['job'].job_ids = list(self.job_mapping.keys())
        graph['company'].company_names = list(self.company_mapping.keys())
        graph['location'].location_names = list(self.location_mapping.keys())
        
        return graph
```

### Save Graph

```python
def save_graph(self, path: str = "graph_data/hetero_graph.pt"):
    torch.save(self.graph, path)
    print(f"Graph saved to {path}")
    
    # Also save mappings
    mappings = {
        'job_mapping': self.job_mapping,
        'company_mapping': self.company_mapping,
        'location_mapping': self.location_mapping
    }
    torch.save(mappings, "graph_data/entity_mappings.pt")
```

---

## 7. FAQ

### Q1: Tại sao cần reverse edges?
**A:** GNN message passing là **directional**:
```python
# Without reverse:
Job → Company  # Job can receive info from Company? NO!

# With reverse:
Job → Company
Company → Job  # Now both can exchange information ✓
```

### Q2: Edge features có cần không?
**A:** Optional! Hiện tại chỉ có edge_index (structural info)
```python
# Có thể thêm edge features:
graph[('job', 'similar_to', 'job')].edge_attr = similarity_scores

# GNN có thể dùng edge features để weight message passing
```

### Q3: Tại sao Job features 399 dims, Company chỉ 10?
**A:** 
- Jobs: Có rich text → embeddings 384 dims
- Companies: Chỉ có aggregated stats → 10 dims đủ
- GNN sẽ học update company representations từ connected jobs!

### Q4: Graph này có thể scale không?
**A:** YES!
```python
Current: 500 jobs → 0.77 MB
Scale to: 10,000 jobs → ~15 MB (linear scaling)
          100,000 jobs → ~150 MB (still manageable!)
```

### Q5: Có thể thêm node types khác không?
**A:** Hoàn toàn được!
```python
# Có thể thêm:
- Skill nodes (extracted from requirements)
- Industry nodes
- User nodes (for recommendation)
- ...

# Simply add to HeteroData:
graph['skill'].x = skill_features
graph[('job', 'requires', 'skill')].edge_index = ...
```

### Q6: Format này có tương thích với DGL không?
**A:** PyTorch Geometric và DGL là 2 frameworks khác nhau
```python
# PyTorch Geometric (hiện tại)
from torch_geometric.data import HeteroData

# DGL (nếu muốn convert)
import dgl
dgl_graph = dgl.heterograph({...})
```

Có thể convert qua lại, nhưng PyG là standard cho research.

---

## 📌 TÓM TẮT

**Input:** 
- Processed CSV
- Job embeddings
- Similarity matrix

**Process:**
1. ✅ Define 3 node types: Job, Company, Location
2. ✅ Create rich features for each node type
3. ✅ Build 5 edge types (including reverses)
4. ✅ Format as PyTorch Geometric HeteroData

**Output:**
- `hetero_graph.pt`: Complete graph (0.77 MB)
- `entity_mappings.pt`: ID mappings

**Graph statistics:**
- Nodes: 864 (500 + 343 + 21)
- Edges: 6,364 (500 + 500 + 500 + 500 + 4,364)
- Density: 1.7% (sparse, good for GNN!)

---

**👉 Tiếp theo: [Part 4: Visualization](04_Visualization.md)**

---

*Part 3 - Graph Construction | NCKH Project*
