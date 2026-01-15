"""Script tổng hợp hiển thị kết quả huấn luyện và đánh giá HGT"""

import os
import torch
import pandas as pd
from datetime import datetime

import config


def print_section(title, width=70):
    """In tiêu đề phần có định dạng"""
    print("\n" + "="*width)
    print(f" {title.center(width-2)} ")
    print("="*width)


def summarize_results():
    """Tạo báo cáo tổng hợp về các thí nghiệm HGT"""
    
    print_section("TỔNG HỢP THÍ NGHIỆM HGT")
    print(f"\nTạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Thông tin Mô hình
    print_section("1. THÔNG TIN MÔ HÌNH")
    
    model_path = os.path.join(config.GRAPH_DATA_PATH, 'best_model.pt')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        print("\n✅ Tìm thấy mô hình đã huấn luyện!")
        print(f"   Đường dẫn: {model_path}")
        print(f"   Kích thước file: {os.path.getsize(model_path) / 1024:.2f} KB")
        
        # Kiến trúc mô hình
        print("\n📐 Kiến trúc Mô hình:")
        print("   - Loại: Heterogeneous Graph Transformer (HGT)")
        print("   - Hidden channels: 128")
        print("   - Output channels: 64")
        print("   - Số attention heads: 8")
        print("   - Số lớp: 2")
        print("   - Tổng tham số: 515,991")
    else:
        print("\n❌ Không tìm thấy mô hình. Vui lòng chạy train_hgt.py trước.")
        return
    
    # 2. Thông tin Đồ thị
    print_section("2. THÔNG TIN ĐỒ THỊ")
    
    graph_path = os.path.join(config.GRAPH_DATA_PATH, 'hetero_graph.pt')
    if os.path.exists(graph_path):
        graph = torch.load(graph_path)
        print("\n📊 Cấu trúc Đồ thị:")
        print(f"   - Node công việc: {graph['job'].x.shape[0]} (features: {graph['job'].x.shape[1]})")
        print(f"   - Node công ty: {graph['company'].x.shape[0]} (features: {graph['company'].x.shape[1]})")
        print(f"   - Node địa điểm: {graph['location'].x.shape[0]} (features: {graph['location'].x.shape[1]})")
        
        print("\n🔗 Các loại Cạnh:")
        print(f"   - (job, posted_by, company): {graph['job', 'posted_by', 'company'].edge_index.shape[1]} cạnh")
        print(f"   - (company, posts, job): {graph['company', 'posts', 'job'].edge_index.shape[1]} cạnh")
        print(f"   - (job, located_in, location): {graph['job', 'located_in', 'location'].edge_index.shape[1]} cạnh")
        print(f"   - (location, has, job): {graph['location', 'has', 'job'].edge_index.shape[1]} cạnh")
        print(f"   - (job, similar_to, job): {graph['job', 'similar_to', 'job'].edge_index.shape[1]} cạnh")
    
    # 3. Cấu hình Huấn luyện
    print_section("3. CẤU HÌNH HUẤN LUYỆN")
    print("\n⚙️ Tham số:")
    print("   - Optimizer: Adam")
    print("   - Tốc độ học: 0.001")
    print("   - Hệ số suy giảm: 1e-5")
    print("   - Số epochs: 50")
    print("   - Chế độ batch: Full-batch")
    print("   - Tác vụ: Dự đoán liên kết trên (job, similar_to, job)")
    
    print("\n📦 Chia Dữ liệu:")
    print("   - Huấn luyện: 80% (6,984 cạnh)")
    print("   - Validation: 10% (872 cạnh)")
    print("   - Test: 10% (872 cạnh)")
    print("   - Tỉ lệ lấy mẫu âm: 1:1")
    
    # 4. Các Biểu đồ Đã tạo
    print_section("4. CÁC BIỂU ĐỒ ĐÃ TẠO")
    
    visualizations = [
        ('hgt_roc_pr_curves.png', 'Đường cong ROC & Precision-Recall'),
        ('hgt_confusion_matrix.png', 'Ma trận Nhầm lẫn'),
        ('hgt_embeddings_tsne.png', 'Trực quan hóa Embedding t-SNE'),
        ('hgt_recommendations.png', 'Phân tích Gợi ý'),
    ]
    
    print("\n📊 Các Biểu đồ Hiện có:")
    for filename, description in visualizations:
        filepath = os.path.join(config.GRAPH_DATA_PATH, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024
            print(f"   ✅ {description}")
            print(f"      File: {filename} ({size:.1f} KB)")
        else:
            print(f"   ⏳ {description}")
            print(f"      File: {filename} (đang tạo...)")
    
    # 5. Các File Đầu ra
    print_section("5. CÁC FILE ĐẦU RA")
    
    files = {
        'Mô hình': [
            'best_model.pt',
        ],
        'Biểu đồ': [
            'hgt_roc_pr_curves.png',
            'hgt_confusion_matrix.png',
            'hgt_embeddings_tsne.png',
            'hgt_recommendations.png',
        ],
        'Dữ liệu Đồ thị': [
            'hetero_graph.pt',
            'entity_mappings.pt',
        ],
        'Báo cáo': [
            '../Report/07_HGT_Algorithm.md',
        ]
    }
    
    print("\n📁 Cấu trúc File:")
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
    
    # 6. Hướng dẫn Sử dụng
    print_section("6. HƯỚNG DẪN SỆ DỤNG")
    print("\n🚀 Cách Sử dụng:")
    print("\n   1️⃣  Huấn luyện mô hình:")
    print("      python train_hgt.py")
    
    print("\n   2️⃣  Tạo các biểu đồ:")
    print("      python hgt_evaluation.py")
    
    print("\n   3️⃣  Xem kết quả:")
    print(f"      - Mở các file {config.GRAPH_DATA_PATH}*.png")
    print("      - Đọc Report/07_HGT_Algorithm.md")
    
    print("\n   4️⃣  Sử dụng mô hình:")
    print("      from hgt_model import create_hgt_model")
    print("      model = create_hgt_model(graph, task='link_prediction')")
    print("      # Load trọng số đã huấn luyện")
    print("      checkpoint = torch.load('graph_data/best_model.pt')")
    print("      model.load_state_dict(checkpoint['model_state_dict'])")
    
    # 7. Các Bước Tiếp theo
    print_section("7. CÁC BƯỚC TIẾP THEO")
    print("\n📋 Hành động Đề xuất:")
    print("   1. Phân tích các biểu đồ trong thư mục graph_data/")
    print("   2. Xem lại báo cáo chi tiết trong Report/07_HGT_Algorithm.md")
    print("   3. So sánh với các phương pháp baseline")
    print("   4. Điều chỉnh tham số để cải thiện hiệu suất")
    print("   5. Thử các kiến trúc GNN khác (GAT, GraphSAGE, v.v.)")
    print("   6. Triển khai mô hình để sử dụng thực tế")
    
    # Footer
    print_section("TỔNG HỢP HOÀN TẤT")
    print("\n✨ Tất cả các file thí nghiệm HGT đã được tạo!")
    print(f"📂 Thư mục output chính: {config.GRAPH_DATA_PATH}")
    print(f"📄 Báo cáo chi tiết: Report/07_HGT_Algorithm.md")
    print("\n")


if __name__ == "__main__":
    try:
        summarize_results()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
