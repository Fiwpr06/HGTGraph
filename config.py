"""Configuration file for Graph GNN project"""

# File paths
RAW_DATA_PATH = "raw/db_job_tuan.xlsx"
PROCESSED_DATA_PATH = "processed/"
GRAPH_DATA_PATH = "graph_data/"

# Text embedding
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Supports Vietnamese
)
EMBEDDING_DIM = 384

# Graph construction
SIMILARITY_THRESHOLD = 0.6  # Threshold for Job-Job edges
TOP_K_SIMILAR_JOBS = 10  # Number of similar jobs to connect

# Node types
NODE_TYPES = ["job", "company", "location"]

# Edge types
EDGE_TYPES = [
    ("job", "posted_by", "company"),
    ("job", "located_in", "location"),
    ("job", "similar_to", "job"),
]

# Feature dimensions
SALARY_DIM = 2  # [min_salary, max_salary]
EXPERIENCE_DIM = 1  # [years]
CATEGORICAL_DIMS = {
    "job_type": 10,
    "company_size": 10,
}
