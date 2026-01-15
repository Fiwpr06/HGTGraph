# NCKH Graph - Heterogeneous Job Graph for GNN

Project xay dung heterogeneous graph tu du lieu job postings de phuc vu cho he thong goi y cong viec su dung Graph Neural Networks (GNN).

## Cau truc du lieu

### Node Types
- **Job**: 500 cong viec voi day du thong tin
- **Company**: Cac cong ty dang tuyen
- **Location**: Dia diem tuyen dung

### Edge Types
- `(Job, posted_by, Company)`: Cong viec duoc dang boi cong ty
- `(Job, located_in, Location)`: Cong viec tai dia diem
- `(Job, similar_to, Job)`: Cong viec tuong tu nhau (dua tren text similarity)

### Node Features
- **Job nodes**: Text embeddings (384D) + Salary + Experience + Job type + Company size
- **Company nodes**: Aggregated statistics tu cac job
- **Location nodes**: Aggregated statistics tu cac job

## Installation

```bash
# Tao virtual environment
python -m venv .venv
.venv\Scripts\activate

# Cai dat dependencies
pip install -r requirements.txt
```

## Project Structure

```
Graph/
├── raw/                          # Du lieu goc
│   └── db_job_tuan.xlsx
├── processed/                    # Du lieu da xu ly
│   ├── jobs_processed.csv
│   ├── job_embeddings.npy
│   └── similarity_matrix.npy
├── graph_data/                   # Graph data
│   ├── hetero_graph.pt
│   ├── entity_mappings.pt
│   ├── graph_statistics.png
│   └── graph_subgraph.png
├── Report/                       # Bao cao chi tiet
│   ├── README.md
│   ├── 00_Overview.md
│   ├── 01_Data_Preprocessing.md
│   ├── 02_Text_Embedding.md
│   ├── 03_Graph_Construction.md
│   ├── 04_Visualization.md
│   ├── 05_Code_Walkthrough.md
│   └── 06_Research_Directions.md
├── config.py                     # Configuration
├── data_preprocessing.py         # Data preprocessing
├── text_embedding.py             # Text embedding generation
├── graph_construction.py         # Graph construction
├── visualization.py              # Visualization utilities
├── demo.py                       # Graph exploration
└── main.py                       # Main pipeline
```

## Usage

### Chay toan bo pipeline:

```bash
python main.py
```

### Hoac chay tung buoc:

```bash
# Buoc 1: Preprocessing
python data_preprocessing.py

# Buoc 2: Text Embedding
python text_embedding.py

# Buoc 3: Graph Construction
python graph_construction.py

# Buoc 4: Visualization
python visualization.py
```

## Outputs

### 1. Processed Data
- `jobs_processed.csv`: Du lieu da chuan hoa
- `job_embeddings.npy`: Vector embeddings cho jobs
- `similarity_matrix.npy`: Ma tran similarity giua cac jobs

### 2. Graph Data
- `hetero_graph.pt`: PyTorch Geometric HeteroData object
- `entity_mappings.pt`: Mappings tu entity names den indices

### 3. Visualizations
- `graph_statistics.png`: Thong ke chi tiet ve graph
- `graph_subgraph.png`: Subgraph visualization

## Next Steps - GNN Models

### Potential Tasks:
1. **Job Recommendation**: Goi y cong viec phu hop tu CV
2. **Job Classification**: Phan loai cong viec theo nganh/linh vuc
3. **Link Prediction**: Du doan moi quan he giua job-user

### Suggested Models:
- **Homogeneous**: GCN, GAT, GraphSAGE
- **Heterogeneous**: HAN, RGCN, HGT

### Example Code (Next phase):
```python
import torch
from torch_geometric.nn import HGTConv, Linear

class JobRecommendationGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        # Define HGT layers
        self.convs = torch.nn.ModuleList()
        # ... implementation
        
    def forward(self, x_dict, edge_index_dict):
        # ... forward pass
        return x_dict
```

## Configuration

Chinh sua `config.py` de thay doi:
- Embedding model (default: paraphrase-multilingual-MiniLM-L12-v2)
- Similarity threshold
- Top-K similar jobs
- File paths

## Technical Details

### Text Embedding
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Dimension: 384
- Support tieng Viet

### Graph Statistics
- Nodes: ~500 jobs + ~343 companies + ~21 locations
- Edges: ~500 job-company + ~500 job-location + ~5000 job-job similarity

### Memory Usage
- Total graph size: ~10-20 MB (depending on features)

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `torch`: PyTorch
- `torch-geometric`: GNN library
- `sentence-transformers`: Text embeddings
- `networkx`: Graph algorithms
- `matplotlib`: Visualization
- `scikit-learn`: ML utilities

## Documentation

Xem folder **Report/** de doc bao cao chi tiet tung buoc:
- [Report/README.md](Report/README.md) - Huong dan doc bao cao
- [00_Overview.md](Report/00_Overview.md) - Tong quan project
- [01_Data_Preprocessing.md](Report/01_Data_Preprocessing.md) - Xu ly du lieu
- [02_Text_Embedding.md](Report/02_Text_Embedding.md) - Text embeddings
- [03_Graph_Construction.md](Report/03_Graph_Construction.md) - Xay dung graph
- [04_Visualization.md](Report/04_Visualization.md) - Truc quan hoa
- [05_Code_Walkthrough.md](Report/05_Code_Walkthrough.md) - Giai thich code
- [06_Research_Directions.md](Report/06_Research_Directions.md) - Huong nghien cuu

## Authors

NCKH Project - Graph Neural Networks for Job Recommendation

## License

MIT License
