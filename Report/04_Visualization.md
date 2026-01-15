# Part 4: VISUALIZATION - TRỰC QUAN HÓA GRAPH

**File code tương ứng:** `visualization.py`  
**Input:** `graph_data/hetero_graph.pt`  
**Output:** Images trong `graph_data/`

---

## 📊 CÁC LOẠI VISUALIZATION

### 1. Graph Statistics (`graph_statistics.png`)

6 biểu đồ chính:

#### 1.1 Node Counts
```
Job:      ████████████ 500
Company:  ██████████ 343  
Location: ███ 21
```

#### 1.2 Edge Counts
```
job-posted_by:    ████████ 500
job-located_in:   ████████ 500
job-similar_to:   ████████████████ 4364
```

#### 1.3 Feature Dimensions
```
Job:      ████████ 399
Company:  ██ 10
Location: █ 8
```

#### 1.4 Salary Distribution
Histogram của job salaries (non-zero values)

#### 1.5 Degree Distribution
```
Most jobs có 10 similar connections (top-k=10)
```

#### 1.6 Jobs per Company
```
Most companies: 1-2 jobs
Some big companies: 12+ jobs
```

### 2. Subgraph Visualization (`graph_subgraph.png`)

**Spring layout** của 50 jobs + connected companies

```
     🔴 Job nodes (red, small)
     🔵 Company nodes (blue, larger)
     ─── Job-Company edges (gray)
     === Job-Job edges (green)
```

---

## 💻 CODE HIGHLIGHTS

### Load & Visualize

```python
from visualization import GraphVisualizer

# Load graph
graph = torch.load("graph_data/hetero_graph.pt", weights_only=False)

# Visualize
visualizer = GraphVisualizer(graph)
visualizer.print_graph_summary()
visualizer.plot_graph_statistics()
visualizer.plot_subgraph(num_jobs=50)
```

### Key Functions

```python
def plot_graph_statistics(self):
    """Plot 6 statistical charts"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Plot node counts, edges, features, etc.
    plt.savefig('graph_statistics.png', dpi=300)

def plot_subgraph(self, num_jobs=50):
    """Visualize sample subgraph with NetworkX"""
    G = nx.Graph()
    # Add nodes and edges from HeteroData
    pos = nx.spring_layout(G, k=2)
    nx.draw(G, pos, ...)
    plt.savefig('graph_subgraph.png', dpi=300)
```

---

## 📌 TÓM TẮT

- ✅ 2 types of visualizations
- ✅ Statistics plots (6 charts)
- ✅ Network graph visualization (spring layout)
- ✅ High-res images (300 DPI)

**Tools used:**
- Matplotlib (charts)
- NetworkX (graph layout)
- PyTorch Geometric (data loading)

---

**👉 Tiếp theo: [Part 5: Code Walkthrough](05_Code_Walkthrough.md)**

---

*Part 4 - Visualization | NCKH Project*
