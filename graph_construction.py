"""Graph construction module for heterogeneous job graph"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

import config


class HeterogeneousJobGraph:
    """Build heterogeneous graph from job data"""

    def __init__(
        self, df: pd.DataFrame, embeddings: np.ndarray, similarity_matrix: np.ndarray
    ):
        """
        Args:
            df: Processed DataFrame
            embeddings: Job text embeddings (n_jobs, embedding_dim)
            similarity_matrix: Job similarity matrix (n_jobs, n_jobs)
        """
        self.df = df
        self.embeddings = embeddings
        self.similarity_matrix = similarity_matrix

        # Entity mappings
        self.job_mapping = {}  # JobID -> job_idx
        self.company_mapping = {}  # Company name -> company_idx
        self.location_mapping = {}  # Location -> location_idx

        # Node features
        self.job_features = None
        self.company_features = None
        self.location_features = None

        # Graph data
        self.graph = None

        print("Initialized HeterogeneousJobGraph")
        print(f"- Jobs: {len(df)}")
        print(f"- Embedding dim: {embeddings.shape[1]}")

    def _create_entity_mappings(self):
        """Create mappings from entity names to indices"""
        print("\nCreating entity mappings...")

        # Job mapping
        self.job_mapping = {job_id: idx for idx, job_id in enumerate(self.df["JobID"])}

        # Company mapping
        unique_companies = self.df["Name company"].unique()
        self.company_mapping = {
            company: idx for idx, company in enumerate(unique_companies)
        }

        # Location mapping
        unique_locations = self.df["location_clean"].unique()
        self.location_mapping = {
            location: idx for idx, location in enumerate(unique_locations)
        }

        print(f"- Jobs: {len(self.job_mapping)}")
        print(f"- Companies: {len(self.company_mapping)}")
        print(f"- Locations: {len(self.location_mapping)}")

    def _create_job_features(self) -> torch.Tensor:
        """
        Create feature matrix for job nodes

        Features:
            - Text embeddings (384 dim)
            - Salary min, max (2 dim)
            - Experience years (1 dim)
            - Quantity (1 dim)
            - Job type one-hot (categorical)
            - Company size one-hot (categorical)
        """
        print("\nCreating job features...")
        n_jobs = len(self.df)

        # 1. Text embeddings (384)
        text_features = torch.FloatTensor(self.embeddings)

        # 2. Numerical features
        salary_min = torch.FloatTensor(self.df["salary_min"].values).reshape(-1, 1)
        salary_max = torch.FloatTensor(self.df["salary_max"].values).reshape(-1, 1)
        experience = torch.FloatTensor(self.df["experience_years"].values).reshape(
            -1, 1
        )
        quantity = torch.FloatTensor(self.df["quantity"].values).reshape(-1, 1)

        # Normalize numerical features
        salary_min = salary_min / 100.0  # Scale down
        salary_max = salary_max / 100.0
        experience = experience / 10.0
        quantity = torch.log1p(quantity)  # Log transform

        # 3. Categorical features - one-hot encoding
        job_types = pd.get_dummies(self.df["Job type"], prefix="jobtype")
        company_sizes = pd.get_dummies(self.df["company_size"], prefix="size")

        job_type_features = torch.FloatTensor(job_types.values)
        company_size_features = torch.FloatTensor(company_sizes.values)

        # Concatenate all features
        job_features = torch.cat(
            [
                text_features,  # 384
                salary_min,  # 1
                salary_max,  # 1
                experience,  # 1
                quantity,  # 1
                job_type_features,  # variable
                company_size_features,  # variable
            ],
            dim=1,
        )

        print(f"Job feature dimension: {job_features.shape[1]}")
        return job_features

    def _create_company_features(self) -> torch.Tensor:
        """
        Create feature matrix for company nodes

        For simplicity, use aggregated statistics from jobs posted by each company
        """
        print("\nCreating company features...")
        n_companies = len(self.company_mapping)
        feature_dim = 10  # Simple aggregated features

        company_features = torch.zeros(n_companies, feature_dim)

        for company, company_idx in self.company_mapping.items():
            # Get all jobs from this company
            company_jobs = self.df[self.df["Name company"] == company]

            if len(company_jobs) > 0:
                # Aggregated features
                company_features[company_idx, 0] = len(company_jobs)  # Number of jobs
                company_features[company_idx, 1] = company_jobs["salary_max"].mean()
                company_features[company_idx, 2] = company_jobs["salary_min"].mean()
                company_features[company_idx, 3] = company_jobs[
                    "experience_years"
                ].mean()
                company_features[company_idx, 4] = company_jobs["quantity"].sum()

                # Company size encoding (simplified)
                size_counts = company_jobs["company_size"].value_counts()
                if len(size_counts) > 0:
                    # One-hot for most common size
                    most_common_size = size_counts.index[0]
                    if "1000+" in most_common_size:
                        company_features[company_idx, 5] = 1
                    elif "100-499" in most_common_size or "500-999" in most_common_size:
                        company_features[company_idx, 6] = 1
                    elif "25-99" in most_common_size:
                        company_features[company_idx, 7] = 1
                    else:
                        company_features[company_idx, 8] = 1

        # Normalize
        company_features[:, 1:5] = company_features[:, 1:5] / (
            company_features[:, 1:5].max(dim=0)[0] + 1e-8
        )

        print(f"Company feature dimension: {company_features.shape[1]}")
        return company_features

    def _create_location_features(self) -> torch.Tensor:
        """
        Create feature matrix for location nodes

        For simplicity, use aggregated statistics from jobs in each location
        """
        print("\nCreating location features...")
        n_locations = len(self.location_mapping)
        feature_dim = 8

        location_features = torch.zeros(n_locations, feature_dim)

        for location, location_idx in self.location_mapping.items():
            # Get all jobs in this location
            location_jobs = self.df[self.df["location_clean"] == location]

            if len(location_jobs) > 0:
                # Aggregated features
                location_features[location_idx, 0] = len(
                    location_jobs
                )  # Number of jobs
                location_features[location_idx, 1] = location_jobs["salary_max"].mean()
                location_features[location_idx, 2] = location_jobs["salary_min"].mean()
                location_features[location_idx, 3] = location_jobs[
                    "experience_years"
                ].mean()
                location_features[location_idx, 4] = location_jobs["quantity"].sum()

        # Normalize
        location_features[:, 1:5] = location_features[:, 1:5] / (
            location_features[:, 1:5].max(dim=0)[0] + 1e-8
        )

        print(f"Location feature dimension: {location_features.shape[1]}")
        return location_features

    def _create_edges(self) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Create edge indices for all edge types

        Returns:
            Dictionary mapping edge types to edge_index tensors
        """
        print("\nCreating edges...")
        edges = {}

        # 1. Job -> Company edges
        job_company_edges = []
        for idx, row in self.df.iterrows():
            job_idx = self.job_mapping[row["JobID"]]
            company_idx = self.company_mapping[row["Name company"]]
            job_company_edges.append([job_idx, company_idx])

        edges[("job", "posted_by", "company")] = (
            torch.tensor(job_company_edges, dtype=torch.long).t().contiguous()
        )

        # Reverse edge
        edges[("company", "posts", "job")] = edges[
            ("job", "posted_by", "company")
        ].flip(0)

        print(f"- Job-Company edges: {edges[('job', 'posted_by', 'company')].shape[1]}")

        # 2. Job -> Location edges
        job_location_edges = []
        for idx, row in self.df.iterrows():
            job_idx = self.job_mapping[row["JobID"]]
            location_idx = self.location_mapping[row["location_clean"]]
            job_location_edges.append([job_idx, location_idx])

        edges[("job", "located_in", "location")] = (
            torch.tensor(job_location_edges, dtype=torch.long).t().contiguous()
        )

        # Reverse edge
        edges[("location", "has", "job")] = edges[
            ("job", "located_in", "location")
        ].flip(0)

        print(
            f"- Job-Location edges: {edges[('job', 'located_in', 'location')].shape[1]}"
        )

        # 3. Job -> Job similarity edges
        similar_pairs = self._find_similar_jobs()
        if similar_pairs:
            job_job_edges = [[pair[0], pair[1]] for pair in similar_pairs]
            job_job_edges_reverse = [[pair[1], pair[0]] for pair in similar_pairs]
            all_job_edges = job_job_edges + job_job_edges_reverse

            edges[("job", "similar_to", "job")] = (
                torch.tensor(all_job_edges, dtype=torch.long).t().contiguous()
            )

            print(
                f"- Job-Job similarity edges: {len(similar_pairs)} pairs (bidirectional)"
            )

        return edges

    def _find_similar_jobs(self) -> List[Tuple[int, int, float]]:
        """Find similar job pairs from similarity matrix"""
        threshold = config.SIMILARITY_THRESHOLD
        top_k = config.TOP_K_SIMILAR_JOBS

        n_jobs = self.similarity_matrix.shape[0]
        edges = []

        for i in range(n_jobs):
            sims = self.similarity_matrix[i].copy()
            sims[i] = -1  # Exclude self

            # Top-K most similar
            if top_k > 0:
                top_indices = np.argsort(sims)[-top_k:][::-1]
                for j in top_indices:
                    if sims[j] >= threshold and i < j:
                        edges.append((i, j, float(sims[j])))

        return edges

    def build_graph(self) -> HeteroData:
        """
        Build the complete heterogeneous graph

        Returns:
            PyTorch Geometric HeteroData object
        """
        print("\n" + "=" * 60)
        print("BUILDING HETEROGENEOUS GRAPH")
        print("=" * 60)

        # Create mappings
        self._create_entity_mappings()

        # Create node features
        self.job_features = self._create_job_features()
        self.company_features = self._create_company_features()
        self.location_features = self._create_location_features()

        # Create edges
        edges_dict = self._create_edges()

        # Build HeteroData
        print("\nBuilding HeteroData object...")
        graph = HeteroData()

        # Add node features
        graph["job"].x = self.job_features
        graph["company"].x = self.company_features
        graph["location"].x = self.location_features

        # Add edges
        for edge_type, edge_index in edges_dict.items():
            graph[edge_type].edge_index = edge_index

        # Store metadata
        graph["job"].job_ids = list(self.job_mapping.keys())
        graph["company"].company_names = list(self.company_mapping.keys())
        graph["location"].location_names = list(self.location_mapping.keys())

        self.graph = graph

        print("\n" + "=" * 60)
        print("GRAPH CONSTRUCTION COMPLETE!")
        print("=" * 60)
        print("\nGraph structure:")
        print(graph)

        return graph

    def save_graph(self, path: str = None):
        """Save graph to disk"""
        path = path or f"{config.GRAPH_DATA_PATH}hetero_graph.pt"
        print(f"\nSaving graph to {path}...")
        torch.save(self.graph, path)
        print("Graph saved!")

        # Also save mappings
        mappings = {
            "job_mapping": self.job_mapping,
            "company_mapping": self.company_mapping,
            "location_mapping": self.location_mapping,
        }
        mapping_path = f"{config.GRAPH_DATA_PATH}entity_mappings.pt"
        torch.save(mappings, mapping_path)
        print(f"Mappings saved to {mapping_path}")


if __name__ == "__main__":
    # Load processed data
    print("Loading data...")
    df = pd.read_csv(f"{config.PROCESSED_DATA_PATH}jobs_processed.csv")
    embeddings = np.load(f"{config.PROCESSED_DATA_PATH}job_embeddings.npy")
    similarity_matrix = np.load(f"{config.PROCESSED_DATA_PATH}similarity_matrix.npy")

    # Build graph
    graph_builder = HeterogeneousJobGraph(df, embeddings, similarity_matrix)
    graph = graph_builder.build_graph()

    # Save graph
    graph_builder.save_graph()

    print("\n✅ Graph construction pipeline complete!")
