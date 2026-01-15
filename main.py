"""Main pipeline to build heterogeneous job graph"""

import sys
import warnings

warnings.filterwarnings("ignore")

import config
from data_preprocessing import JobDataPreprocessor
from graph_construction import HeterogeneousJobGraph
from text_embedding import embed_job_data
from visualization import GraphVisualizer


def main():
    """Complete pipeline for building heterogeneous job graph"""

    print("\n" + "=" * 70)
    print(" " * 15 + "HETEROGENEOUS JOB GRAPH CONSTRUCTION")
    print("=" * 70)

    # Step 1: Data Preprocessing
    print("\n[STEP 1/5] DATA PREPROCESSING")
    print("-" * 70)
    preprocessor = JobDataPreprocessor()
    df_processed = preprocessor.preprocess()
    preprocessor.save_processed_data(df_processed)

    # Step 2: Text Embedding
    print("\n[STEP 2/5] TEXT EMBEDDING")
    print("-" * 70)
    df_processed, embeddings, similarity_matrix = embed_job_data(
        df_processed, save_path=config.PROCESSED_DATA_PATH
    )

    # Step 3: Graph Construction
    print("\n[STEP 3/5] GRAPH CONSTRUCTION")
    print("-" * 70)
    graph_builder = HeterogeneousJobGraph(df_processed, embeddings, similarity_matrix)
    graph = graph_builder.build_graph()
    graph_builder.save_graph()

    # Step 4: Visualization
    print("\n[STEP 4/5] VISUALIZATION")
    print("-" * 70)
    visualizer = GraphVisualizer(graph)
    visualizer.print_graph_summary()
    visualizer.plot_graph_statistics()
    visualizer.plot_subgraph(num_jobs=50)

    # Step 5: Summary
    print("\n[STEP 5/5] PIPELINE SUMMARY")
    print("-" * 70)
    print("✅ Data preprocessing complete")
    print("✅ Text embeddings generated")
    print("✅ Heterogeneous graph constructed")
    print("✅ Visualizations created")
    print("✅ Graph saved to:", config.GRAPH_DATA_PATH)

    print("\n" + "=" * 70)
    print(" " * 20 + "🎉 PIPELINE COMPLETE! 🎉")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  📁 {config.PROCESSED_DATA_PATH}jobs_processed.csv")
    print(f"  📁 {config.PROCESSED_DATA_PATH}job_embeddings.npy")
    print(f"  📁 {config.PROCESSED_DATA_PATH}similarity_matrix.npy")
    print(f"  📁 {config.GRAPH_DATA_PATH}hetero_graph.pt")
    print(f"  📁 {config.GRAPH_DATA_PATH}entity_mappings.pt")
    print(f"  📁 {config.GRAPH_DATA_PATH}graph_statistics.png")
    print(f"  📁 {config.GRAPH_DATA_PATH}graph_subgraph.png")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
