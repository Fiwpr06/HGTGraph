"""Text embedding module using sentence transformers"""

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config

warnings.filterwarnings("ignore")


class TextEmbedder:
    """Generate embeddings for job text data"""

    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: Name of sentence-transformer model
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(
            f"Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
        )

    def embed_texts(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        print(f"\nEmbedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix between embeddings

        Args:
            embeddings: numpy array of shape (n, dim)

        Returns:
            Similarity matrix of shape (n, n)
        """
        print("\nComputing similarity matrix...")
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Compute cosine similarity
        similarity = np.dot(normalized, normalized.T)
        print(f"Similarity matrix shape: {similarity.shape}")
        print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")

        return similarity

    def find_similar_jobs(
        self, similarity_matrix: np.ndarray, threshold: float = None, top_k: int = None
    ) -> List[Tuple[int, int, float]]:
        """
        Find similar job pairs based on similarity threshold or top-k

        Args:
            similarity_matrix: Similarity matrix (n x n)
            threshold: Minimum similarity threshold (0-1)
            top_k: Number of most similar jobs per job

        Returns:
            List of (job_i, job_j, similarity) tuples
        """
        threshold = threshold or config.SIMILARITY_THRESHOLD
        top_k = top_k or config.TOP_K_SIMILAR_JOBS

        n_jobs = similarity_matrix.shape[0]
        edges = []

        print(f"\nFinding similar job pairs (threshold={threshold}, top_k={top_k})...")

        for i in range(n_jobs):
            # Get similarities for job i (excluding self)
            sims = similarity_matrix[i].copy()
            sims[i] = -1  # Exclude self

            # Method 1: Top-K most similar
            if top_k > 0:
                top_indices = np.argsort(sims)[-top_k:][::-1]
                for j in top_indices:
                    if sims[j] >= threshold and i < j:  # Avoid duplicates
                        edges.append((i, j, float(sims[j])))

            # Method 2: All above threshold
            else:
                indices = np.where(sims >= threshold)[0]
                for j in indices:
                    if i < j:  # Avoid duplicates
                        edges.append((i, j, float(sims[j])))

        print(f"Found {len(edges)} similar job pairs")
        if edges:
            avg_sim = np.mean([e[2] for e in edges])
            print(f"Average similarity: {avg_sim:.3f}")

        return edges


def embed_job_data(
    df: pd.DataFrame, save_path: str = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Main function to embed job data

    Args:
        df: Processed DataFrame with 'combined_text' column
        save_path: Path to save embeddings

    Returns:
        (df, embeddings, similarity_matrix)
    """
    # Initialize embedder
    embedder = TextEmbedder()

    # Generate embeddings
    texts = df["combined_text"].tolist()
    embeddings = embedder.embed_texts(texts)

    # Compute similarity
    similarity_matrix = embedder.compute_similarity_matrix(embeddings)

    # Save if path provided
    if save_path:
        print(f"\nSaving embeddings to {save_path}...")
        np.save(f"{save_path}job_embeddings.npy", embeddings)
        np.save(f"{save_path}similarity_matrix.npy", similarity_matrix)
        print("Embeddings saved!")

    return df, embeddings, similarity_matrix


if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv(f"{config.PROCESSED_DATA_PATH}jobs_processed.csv")
    print(f"Loaded {len(df)} jobs")

    # Generate embeddings
    df, embeddings, similarity = embed_job_data(
        df, save_path=config.PROCESSED_DATA_PATH
    )

    # Find similar jobs
    embedder = TextEmbedder()
    similar_pairs = embedder.find_similar_jobs(similarity)

    print("\n" + "=" * 60)
    print("SAMPLE SIMILAR JOB PAIRS:")
    print("=" * 60)
    for i, (job1, job2, sim) in enumerate(similar_pairs[:5]):
        print(f"\n{i + 1}. Similarity: {sim:.3f}")
        print(f"   Job {job1}: {df.iloc[job1]['Title'][:60]}...")
        print(f"   Job {job2}: {df.iloc[job2]['Title'][:60]}...")
