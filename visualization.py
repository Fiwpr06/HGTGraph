"""Visualization module for graph analysis"""

import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

import config

warnings.filterwarnings("ignore")


class GraphVisualizer:
    """Visualize heterogeneous job graph"""

    def __init__(self, graph: HeteroData):
        """
        Args:
            graph: PyTorch Geometric HeteroData object
        """
        self.graph = graph

    def plot_graph_statistics(self, save_path: str = None):
        """Plot basic graph statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Heterogeneous Job Graph Statistics", fontsize=16, fontweight="bold"
        )

        # 1. Node counts
        ax = axes[0, 0]
        node_types = []
        node_counts = []
        for node_type in self.graph.node_types:
            node_types.append(node_type)
            node_counts.append(self.graph[node_type].x.shape[0])

        ax.bar(node_types, node_counts, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
        ax.set_title("Number of Nodes by Type", fontweight="bold")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(node_counts):
            ax.text(
                i, v + max(node_counts) * 0.02, str(v), ha="center", fontweight="bold"
            )

        # 2. Edge counts
        ax = axes[0, 1]
        edge_types = []
        edge_counts = []
        for edge_type in self.graph.edge_types:
            edge_label = f"{edge_type[0]}-{edge_type[1]}"
            edge_types.append(edge_label)
            edge_counts.append(self.graph[edge_type].edge_index.shape[1])

        ax.barh(edge_types, edge_counts, color=["#95E1D3", "#F38181", "#AA96DA"])
        ax.set_title("Number of Edges by Type", fontweight="bold")
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(edge_counts):
            ax.text(
                v + max(edge_counts) * 0.02, i, str(v), va="center", fontweight="bold"
            )

        # 3. Feature dimensions
        ax = axes[0, 2]
        feature_dims = {}
        for node_type in self.graph.node_types:
            feature_dims[node_type] = self.graph[node_type].x.shape[1]

        ax.bar(
            feature_dims.keys(),
            feature_dims.values(),
            color=["#FECA57", "#48C9B0", "#5F27CD"],
        )
        ax.set_title("Feature Dimensions by Node Type", fontweight="bold")
        ax.set_ylabel("Dimension")
        ax.grid(axis="y", alpha=0.3)
        for i, (k, v) in enumerate(feature_dims.items()):
            ax.text(
                i,
                v + max(feature_dims.values()) * 0.02,
                str(v),
                ha="center",
                fontweight="bold",
            )

        # 4. Job salary distribution
        if "job" in self.graph.node_types:
            ax = axes[1, 0]
            # Assuming salary is in columns 384-385 (after embeddings)
            job_features = self.graph["job"].x.numpy()
            if job_features.shape[1] > 385:
                salary_max = job_features[:, 385] * 100  # Denormalize
                salary_max = salary_max[salary_max > 0]  # Remove zeros

                ax.hist(
                    salary_max, bins=30, color="#FF6B6B", alpha=0.7, edgecolor="black"
                )
                ax.set_title("Job Salary Distribution (Max)", fontweight="bold")
                ax.set_xlabel("Salary (Million VND)")
                ax.set_ylabel("Frequency")
                ax.grid(axis="y", alpha=0.3)
                ax.axvline(
                    salary_max.mean(),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {salary_max.mean():.1f}M",
                )
                ax.legend()

        # 5. Degree distribution for jobs
        ax = axes[1, 1]
        if ("job", "similar_to", "job") in self.graph.edge_types:
            edge_index = self.graph[("job", "similar_to", "job")].edge_index.numpy()
            degrees = np.bincount(edge_index[0])

            ax.hist(degrees, bins=20, color="#4ECDC4", alpha=0.7, edgecolor="black")
            ax.set_title("Job-Job Similarity Degree Distribution", fontweight="bold")
            ax.set_xlabel("Degree (Number of Similar Jobs)")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", alpha=0.3)
            ax.axvline(
                degrees.mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {degrees.mean():.1f}",
            )
            ax.legend()

        # 6. Company size distribution
        ax = axes[1, 2]
        if "company" in self.graph.node_types:
            company_features = self.graph["company"].x.numpy()
            num_jobs_per_company = company_features[
                :, 0
            ]  # First feature is number of jobs

            ax.hist(
                num_jobs_per_company,
                bins=20,
                color="#45B7D1",
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_title("Jobs per Company Distribution", fontweight="bold")
            ax.set_xlabel("Number of Jobs")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", alpha=0.3)
            ax.axvline(
                num_jobs_per_company.mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {num_jobs_per_company.mean():.1f}",
            )
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved graph statistics to {save_path}")
        else:
            plt.savefig(
                f"{config.GRAPH_DATA_PATH}graph_statistics.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Saved graph statistics to {config.GRAPH_DATA_PATH}graph_statistics.png"
            )

        plt.close()

    def plot_subgraph(self, num_jobs: int = 50, save_path: str = None):
        """
        Plot a subgraph with limited nodes for visualization

        Args:
            num_jobs: Number of job nodes to include
            save_path: Path to save figure
        """
        print(f"\nCreating subgraph visualization with {num_jobs} jobs...")

        # Create a NetworkX graph
        G = nx.Graph()

        # Add job nodes
        job_ids = self.graph["job"].job_ids[:num_jobs]
        for i, job_id in enumerate(job_ids):
            G.add_node(f"J{i}", node_type="job", label=job_id)

        # Add company nodes (only those connected to selected jobs)
        if ("job", "posted_by", "company") in self.graph.edge_types:
            edge_index = self.graph[("job", "posted_by", "company")].edge_index.numpy()
            companies_in_subgraph = set()

            for src, dst in edge_index.T:
                if src < num_jobs:
                    company_name = self.graph["company"].company_names[dst]
                    company_node = f"C{dst}"
                    if company_node not in G:
                        G.add_node(
                            company_node, node_type="company", label=company_name
                        )
                    G.add_edge(f"J{src}", company_node)
                    companies_in_subgraph.add(dst)

        # Add job-job similarity edges
        if ("job", "similar_to", "job") in self.graph.edge_types:
            edge_index = self.graph[("job", "similar_to", "job")].edge_index.numpy()

            for src, dst in edge_index.T:
                if src < num_jobs and dst < num_jobs:
                    G.add_edge(f"J{src}", f"J{dst}")

        # Plot
        plt.figure(figsize=(20, 16))

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Separate nodes by type
        job_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "job"]
        company_nodes = [
            n for n, d in G.nodes(data=True) if d.get("node_type") == "company"
        ]

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=job_nodes,
            node_color="#FF6B6B",
            node_size=300,
            alpha=0.8,
            label="Jobs",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=company_nodes,
            node_color="#4ECDC4",
            node_size=500,
            alpha=0.8,
            label="Companies",
        )

        # Draw edges
        job_company_edges = [
            (u, v)
            for u, v in G.edges()
            if (
                G.nodes[u].get("node_type") == "job"
                and G.nodes[v].get("node_type") == "company"
            )
            or (
                G.nodes[v].get("node_type") == "job"
                and G.nodes[u].get("node_type") == "company"
            )
        ]
        job_job_edges = [
            (u, v)
            for u, v in G.edges()
            if G.nodes[u].get("node_type") == "job"
            and G.nodes[v].get("node_type") == "job"
        ]

        nx.draw_networkx_edges(
            G, pos, edgelist=job_company_edges, edge_color="gray", alpha=0.3, width=1
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=job_job_edges, edge_color="#95E1D3", alpha=0.5, width=2
        )

        # Labels (only for companies to avoid clutter)
        company_labels = {
            n: d["label"][:20]
            for n, d in G.nodes(data=True)
            if d.get("node_type") == "company"
        }
        nx.draw_networkx_labels(G, pos, labels=company_labels, font_size=8)

        plt.title(
            f"Heterogeneous Job Graph Subgraph (Top {num_jobs} Jobs)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.legend(loc="upper left", fontsize=12)
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved subgraph to {save_path}")
        else:
            plt.savefig(
                f"{config.GRAPH_DATA_PATH}graph_subgraph.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Saved subgraph to {config.GRAPH_DATA_PATH}graph_subgraph.png")

        plt.close()

    def print_graph_summary(self):
        """Print detailed graph summary"""
        print("\n" + "=" * 70)
        print(" " * 20 + "HETEROGENEOUS GRAPH SUMMARY")
        print("=" * 70)

        print("\n📊 NODE STATISTICS:")
        print("-" * 70)
        for node_type in self.graph.node_types:
            num_nodes = self.graph[node_type].x.shape[0]
            feat_dim = self.graph[node_type].x.shape[1]
            print(
                f"  • {node_type.upper():12s}: {num_nodes:6d} nodes  |  {feat_dim:4d} features"
            )

        print("\n🔗 EDGE STATISTICS:")
        print("-" * 70)
        for edge_type in self.graph.edge_types:
            num_edges = self.graph[edge_type].edge_index.shape[1]
            src, rel, dst = edge_type
            print(f"  • ({src} --{rel}--> {dst}):  {num_edges:6d} edges")

        print("\n💾 MEMORY USAGE:")
        print("-" * 70)
        total_params = 0
        for node_type in self.graph.node_types:
            params = self.graph[node_type].x.numel()
            total_params += params
            size_mb = params * 4 / (1024**2)  # Assuming float32
            print(f"  • {node_type.upper():12s} features: {size_mb:8.2f} MB")

        print(f"\n  Total: {total_params * 4 / (1024**2):.2f} MB")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Load graph
    print("Loading graph...")
    graph = torch.load(f"{config.GRAPH_DATA_PATH}hetero_graph.pt", weights_only=False)

    # Initialize visualizer
    visualizer = GraphVisualizer(graph)

    # Print summary
    visualizer.print_graph_summary()

    # Plot statistics
    print("Generating statistics plots...")
    visualizer.plot_graph_statistics()

    # Plot subgraph
    print("Generating subgraph visualization...")
    visualizer.plot_subgraph(num_jobs=50)

    print("\n✅ Visualization complete!")
