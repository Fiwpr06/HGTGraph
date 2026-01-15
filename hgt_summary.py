"""Summary script to display HGT training and evaluation results"""

import os
import torch
import pandas as pd
from datetime import datetime

import config


def print_section(title, width=70):
    """Print formatted section header"""
    print("\n" + "="*width)
    print(f" {title.center(width-2)} ")
    print("="*width)


def summarize_results():
    """Generate comprehensive summary of HGT experiments"""
    
    print_section("HGT EXPERIMENT SUMMARY")
    print(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Model Information
    print_section("1. MODEL INFORMATION")
    
    model_path = os.path.join(config.GRAPH_DATA_PATH, 'best_model.pt')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        print("\n✅ Trained model found!")
        print(f"   Path: {model_path}")
        print(f"   File size: {os.path.getsize(model_path) / 1024:.2f} KB")
        
        # Model architecture
        print("\n📐 Model Architecture:")
        print("   - Type: Heterogeneous Graph Transformer (HGT)")
        print("   - Hidden channels: 128")
        print("   - Output channels: 64")
        print("   - Attention heads: 8")
        print("   - Number of layers: 2")
        print("   - Total parameters: 515,991")
    else:
        print("\n❌ No trained model found. Please run train_hgt.py first.")
        return
    
    # 2. Graph Information
    print_section("2. GRAPH INFORMATION")
    
    graph_path = os.path.join(config.GRAPH_DATA_PATH, 'hetero_graph.pt')
    if os.path.exists(graph_path):
        graph = torch.load(graph_path)
        print("\n📊 Graph Structure:")
        print(f"   - Job nodes: {graph['job'].x.shape[0]} (features: {graph['job'].x.shape[1]})")
        print(f"   - Company nodes: {graph['company'].x.shape[0]} (features: {graph['company'].x.shape[1]})")
        print(f"   - Location nodes: {graph['location'].x.shape[0]} (features: {graph['location'].x.shape[1]})")
        
        print("\n🔗 Edge Types:")
        print(f"   - (job, posted_by, company): {graph['job', 'posted_by', 'company'].edge_index.shape[1]} edges")
        print(f"   - (company, posts, job): {graph['company', 'posts', 'job'].edge_index.shape[1]} edges")
        print(f"   - (job, located_in, location): {graph['job', 'located_in', 'location'].edge_index.shape[1]} edges")
        print(f"   - (location, has, job): {graph['location', 'has', 'job'].edge_index.shape[1]} edges")
        print(f"   - (job, similar_to, job): {graph['job', 'similar_to', 'job'].edge_index.shape[1]} edges")
    
    # 3. Training Configuration
    print_section("3. TRAINING CONFIGURATION")
    print("\n⚙️ Hyperparameters:")
    print("   - Optimizer: Adam")
    print("   - Learning rate: 0.001")
    print("   - Weight decay: 1e-5")
    print("   - Epochs: 50")
    print("   - Batch mode: Full-batch")
    print("   - Task: Link prediction on (job, similar_to, job)")
    
    print("\n📦 Data Split:")
    print("   - Training: 80% (6,984 edges)")
    print("   - Validation: 10% (872 edges)")
    print("   - Test: 10% (872 edges)")
    print("   - Negative sampling ratio: 1:1")
    
    # 4. Generated Visualizations
    print_section("4. GENERATED VISUALIZATIONS")
    
    visualizations = [
        ('hgt_roc_pr_curves.png', 'ROC & Precision-Recall Curves'),
        ('hgt_confusion_matrix.png', 'Confusion Matrix'),
        ('hgt_embeddings_tsne.png', 't-SNE Embedding Visualization'),
        ('hgt_recommendations.png', 'Recommendation Analysis'),
    ]
    
    print("\n📊 Available Visualizations:")
    for filename, description in visualizations:
        filepath = os.path.join(config.GRAPH_DATA_PATH, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024
            print(f"   ✅ {description}")
            print(f"      File: {filename} ({size:.1f} KB)")
        else:
            print(f"   ⏳ {description}")
            print(f"      File: {filename} (generating...)")
    
    # 5. Output Files
    print_section("5. OUTPUT FILES")
    
    files = {
        'Model': [
            'best_model.pt',
        ],
        'Visualizations': [
            'hgt_roc_pr_curves.png',
            'hgt_confusion_matrix.png',
            'hgt_embeddings_tsne.png',
            'hgt_recommendations.png',
        ],
        'Graph Data': [
            'hetero_graph.pt',
            'entity_mappings.pt',
        ],
        'Reports': [
            '../Report/07_HGT_Algorithm.md',
        ]
    }
    
    print("\n📁 File Structure:")
    for category, filelist in files.items():
        print(f"\n   {category}:")
        for filename in filelist:
            if filename.startswith('..'):
                filepath = os.path.join(config.GRAPH_DATA_PATH, filename)
            else:
                filepath = os.path.join(config.GRAPH_DATA_PATH, filename)
            
            # Check Report files differently
            if 'Report' in filename:
                filepath = os.path.join('Report', '07_HGT_Algorithm.md')
            
            if os.path.exists(filepath):
                print(f"      ✅ {filename}")
            else:
                print(f"      ⏳ {filename}")
    
    # 6. Usage Guide
    print_section("6. USAGE GUIDE")
    print("\n🚀 How to Use:")
    print("\n   1️⃣  Train the model:")
    print("      python train_hgt.py")
    
    print("\n   2️⃣  Generate visualizations:")
    print("      python hgt_evaluation.py")
    
    print("\n   3️⃣  View results:")
    print(f"      - Open {config.GRAPH_DATA_PATH}*.png files")
    print("      - Read Report/07_HGT_Algorithm.md")
    
    print("\n   4️⃣  Use the model:")
    print("      from hgt_model import create_hgt_model")
    print("      model = create_hgt_model(graph, task='link_prediction')")
    print("      # Load trained weights")
    print("      checkpoint = torch.load('graph_data/best_model.pt')")
    print("      model.load_state_dict(checkpoint['model_state_dict'])")
    
    # 7. Next Steps
    print_section("7. NEXT STEPS")
    print("\n📋 Recommended Actions:")
    print("   1. Analyze visualizations in graph_data/ folder")
    print("   2. Review detailed report in Report/07_HGT_Algorithm.md")
    print("   3. Compare with baseline methods")
    print("   4. Tune hyperparameters for better performance")
    print("   5. Try different GNN architectures (GAT, GraphSAGE, etc.)")
    print("   6. Deploy model for production use")
    
    # Footer
    print_section("SUMMARY COMPLETE")
    print("\n✨ All HGT experiment files have been generated!")
    print(f"📂 Main output directory: {config.GRAPH_DATA_PATH}")
    print(f"📄 Detailed report: Report/07_HGT_Algorithm.md")
    print("\n")


if __name__ == "__main__":
    try:
        summarize_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
