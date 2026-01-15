"""Công cụ đánh giá và trực quan hóa cho mô hình HGT"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd

import config
from hgt_model import create_hgt_model

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class HGTEvaluator:
    """Đánh giá toàn diện và trực quan hóa cho mô hình HGT"""
    
    def __init__(self, model, graph, test_data, edge_type, device='cpu'):
        self.model = model.to(device)
        self.graph = graph
        self.test_data = test_data.to(device)
        self.edge_type = edge_type
        self.device = device
        
    @torch.no_grad()
    def get_predictions(self):
        """Lấy dự đoán của mô hình trên tập test"""
        self.model.eval()
        
        edge_label_index = self.test_data[self.edge_type].edge_label_index
        edge_label = self.test_data[self.edge_type].edge_label
        
        x_dict = {
            'job': self.test_data['job'].x,
            'company': self.test_data['company'].x,
            'location': self.test_data['location'].x,
        }
        
        edge_index_dict = {
            key: self.test_data[key].edge_index
            for key in self.test_data.edge_types
        }
        
        pred = self.model(x_dict, edge_index_dict, edge_label_index, self.edge_type)
        pred_probs = torch.sigmoid(pred).cpu().numpy()
        labels = edge_label.cpu().numpy()
        
        return pred_probs, labels
    
    @torch.no_grad()
    def get_embeddings(self):
        """Lấy embeddings của các node từ HGT encoder"""
        self.model.eval()
        
        x_dict = {
            'job': self.graph['job'].x.to(self.device),
            'company': self.graph['company'].x.to(self.device),
            'location': self.graph['location'].x.to(self.device),
        }
        
        edge_index_dict = {
            ('job', 'posted_by', 'company'): self.graph['job', 'posted_by', 'company'].edge_index.to(self.device),
            ('company', 'posts', 'job'): self.graph['company', 'posts', 'job'].edge_index.to(self.device),
            ('job', 'located_in', 'location'): self.graph['job', 'located_in', 'location'].edge_index.to(self.device),
            ('location', 'has', 'job'): self.graph['location', 'has', 'job'].edge_index.to(self.device),
            ('job', 'similar_to', 'job'): self.graph['job', 'similar_to', 'job'].edge_index.to(self.device),
        }
        
        embeddings = self.model.encode(x_dict, edge_index_dict)
        
        # Chuyển sang numpy
        embeddings_np = {
            key: emb.cpu().numpy() 
            for key, emb in embeddings.items()
        }
        
        return embeddings_np
    
    def plot_roc_pr_curves(self, save_path=None):
        """Vẽ đường cong ROC và Precision-Recall"""
        pred_probs, labels = self.get_predictions()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Đường cong ROC
        fpr, tpr, _ = roc_curve(labels, pred_probs)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, pred_probs)
        
        axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'HGT (AUC = {auc:.4f})')
        axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ngẫu nhiên')
        axes[0].set_xlabel('Tỉ lệ Dương Giả (FPR)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Tỉ lệ Dương Thực (TPR)', fontsize=12, fontweight='bold')
        axes[0].set_title('Đường cong ROC - Dự đoán Liên kết', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Đường cong Precision-Recall
        precision, recall, _ = precision_recall_curve(labels, pred_probs)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(labels, pred_probs)
        
        axes[1].plot(recall, precision, 'b-', linewidth=2, label=f'HGT (AP = {ap:.4f})')
        axes[1].axhline(y=labels.mean(), color='r', linestyle='--', linewidth=2, label='Ngẫu nhiên')
        axes[1].set_xlabel('Độ phủ (Recall)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Độ chính xác (Precision)', fontsize=12, fontweight='bold')
        axes[1].set_title('Đường cong Precision-Recall', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower left', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Đường cong ROC & PR đã lưu tại {save_path}")
        
        plt.close()
        
    def plot_confusion_matrix(self, threshold=0.5, save_path=None):
        """Vẽ ma trận nhầm lẫn (confusion matrix)"""
        pred_probs, labels = self.get_predictions()
        pred_labels = (pred_probs >= threshold).astype(int)
        
        cm = confusion_matrix(labels, pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Âm tính', 'Dương tính'],
                    yticklabels=['Âm tính', 'Dương tính'],
                    cbar_kws={'label': 'Số lượng'})
        plt.xlabel('Nhãn dự đoán', fontsize=12, fontweight='bold')
        plt.ylabel('Nhãn thực tế', fontsize=12, fontweight='bold')
        plt.title(f'Ma trận Nhầm lẫn (ngưỡng={threshold})', fontsize=14, fontweight='bold')
        
        # Thêm các chỉ số
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f"Độ chính xác: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}"
        plt.text(2.5, 0.5, metrics_text, fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Ma trận nhầm lẫn đã lưu tại {save_path}")
        
        plt.close()
        
    def plot_embeddings_tsne(self, save_path=None):
        """Trực quan hóa embeddings của node bằng t-SNE"""
        embeddings = self.get_embeddings()
        
        # Load dữ liệu đã xử lý cho nhãn
        df = pd.read_csv(f"{config.PROCESSED_DATA_PATH}jobs_processed.csv")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Biểu đồ 1: Embeddings công việc theo danh mục
        job_embeddings = embeddings['job']
        
        print("Đang tính toán t-SNE cho embeddings công việc...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        job_2d = tsne.fit_transform(job_embeddings)
        
        # Trích xuất danh mục công việc từ tiêu đề
        def extract_category(title):
            title_lower = str(title).lower()
            # Định nghĩa các danh mục
            if any(x in title_lower for x in ['developer', 'lập trình', 'it', 'software', 'backend', 'frontend', 'fullstack', 'kỹ sư', 'engineer']):
                return 'IT/Developer'
            elif any(x in title_lower for x in ['kế toán', 'accountant', 'tax', 'thuế', 'finance', 'tài chính']):
                return 'Accounting/Finance'
            elif any(x in title_lower for x in ['sale', 'bán hàng', 'kinh doanh', 'marketing', 'market']):
                return 'Sales/Marketing'
            elif any(x in title_lower for x in ['hr', 'nhân sự', 'talent', 'recruitment', 'tuyển dụng']):
                return 'HR/Recruitment'
            elif any(x in title_lower for x in ['thiết kế', 'design', 'ux', 'ui', 'đồ họa']):
                return 'Design'
            elif any(x in title_lower for x in ['tư vấn', 'consultant', 'advisor', 'cố vấn']):
                return 'Consulting'
            elif any(x in title_lower for x in ['quản lý', 'manager', 'trưởng phòng', 'giám đốc', 'director']):
                return 'Management'
            elif any(x in title_lower for x in ['nhân viên', 'staff', 'chuyên viên', 'specialist']):
                return 'Staff/Specialist'
            else:
                return 'Other'
        
        job_categories = df['Title'].apply(extract_category).values
        unique_categories = np.unique(job_categories)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        
        for i, category in enumerate(unique_categories):
            mask = job_categories == category
            axes[0].scatter(job_2d[mask, 0], job_2d[mask, 1], 
                          c=[colors[i]], label=category, alpha=0.6, s=50)
        
        axes[0].set_xlabel('Chiều t-SNE 1', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Chiều t-SNE 2', fontsize=12, fontweight='bold')
        axes[0].set_title('Embeddings Công việc (theo Danh mục)', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Biểu đồ 2: Embeddings công việc theo mức lương
        salary_avg = (df['salary_min'] + df['salary_max']) / 2
        scatter = axes[1].scatter(job_2d[:, 0], job_2d[:, 1], 
                                 c=salary_avg, cmap='viridis', alpha=0.6, s=50)
        
        axes[1].set_xlabel('Chiều t-SNE 1', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Chiều t-SNE 2', fontsize=12, fontweight='bold')
        axes[1].set_title('Embeddings Công việc (theo Mức lương)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Mức lương TB (triệu VNĐ)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Trực quan hóa embeddings đã lưu tại {save_path}")
        
        plt.close()
        
        # In thống kê danh mục
        print("\n📊 Phân bố Danh mục Công việc:")
        category_counts = pd.Series(job_categories).value_counts()
        for cat, count in category_counts.items():
            percentage = (count / len(job_categories)) * 100
            print(f"   {cat:20s}: {count:3d} công việc ({percentage:5.1f}%)")
        print(f"   {'='*40}")
        print(f"   {'Tổng':20s}: {len(job_categories):3d} công việc")
        
        return job_categories
        
    def analyze_recommendations(self, job_idx=0, top_k=10, save_path=None):
        """Phân tích top-K gợi ý cho một công việc cụ thể"""
        embeddings = self.get_embeddings()
        job_embeddings = embeddings['job']
        
        # Tính độ tương đồng
        target_emb = job_embeddings[job_idx]
        similarities = np.dot(job_embeddings, target_emb) / (
            np.linalg.norm(job_embeddings, axis=1) * np.linalg.norm(target_emb)
        )
        
        # Lấy top-K công việc tương đồng nhất (không bao gồm chính nó)
        similarities[job_idx] = -1
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Load dữ liệu công việc
        df = pd.read_csv(f"{config.PROCESSED_DATA_PATH}jobs_processed.csv")
        
        # Tạo biểu đồ trực quan hóa
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Biểu đồ trên: Biểu đồ cột độ tương đồng
        top_sims = similarities[top_indices]
        job_titles = [df.iloc[i]['Title'][:30] for i in top_indices]
        
        axes[0].barh(range(top_k), top_sims, color='steelblue')
        axes[0].set_yticks(range(top_k))
        axes[0].set_yticklabels(job_titles, fontsize=9)
        axes[0].set_xlabel('Độ tương đồng Cosine', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Top-{top_k} Công việc Tương đồng với: {df.iloc[job_idx]["Title"][:50]}', 
                         fontsize=13, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Biểu đồ dưới: So sánh đặc trưng
        target_job = df.iloc[job_idx]
        comparison_data = {
            'Công việc Đích': [
                target_job['salary_min'], 
                target_job['salary_max'],
                target_job['experience_years'],
                target_job['quantity']
            ]
        }
        
        for i, idx in enumerate(top_indices[:5]):  # Hiển thị top 5
            job = df.iloc[idx]
            comparison_data[f'Tương đồng #{i+1}'] = [
                job['salary_min'],
                job['salary_max'],
                job['experience_years'],
                job['quantity']
            ]
        
        comparison_df = pd.DataFrame(comparison_data, 
                                     index=['Lương Tối thiểu', 'Lương Tối đa', 'Kinh nghiệm (năm)', 'Số lượng'])
        
        x = np.arange(len(comparison_df.index))
        width = 0.15
        
        for i, col in enumerate(comparison_df.columns):
            offset = width * (i - len(comparison_df.columns)/2 + 0.5)
            axes[1].bar(x + offset, comparison_df[col], width, label=col)
        
        axes[1].set_xlabel('Đặc trưng', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Giá trị', fontsize=12, fontweight='bold')
        axes[1].set_title('So sánh Đặc trưng', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(comparison_df.index)
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Phân tích gợi ý đã lưu tại {save_path}")
        
        plt.close()
        
        return top_indices, similarities[top_indices]


def main():
    """Đánh giá toàn diện"""
    print("\n" + "="*70)
    print(" "*15 + "ĐÁNH GIÁ & TRỰC QUAN HÓA MÔ HÌNH HGT")
    print("="*70)
    
    # Tải đồ thị và dữ liệu
    print("\n[1/6] Đang tải dữ liệu...")
    graph = torch.load(f"{config.GRAPH_DATA_PATH}hetero_graph.pt")
    
    # Tải dữ liệu test (cần tạo lại split)
    from torch_geometric.transforms import RandomLinkSplit
    edge_type = ('job', 'similar_to', 'job')
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        edge_types=[edge_type],
        rev_edge_types=[edge_type],
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    _, _, test_data = transform(graph)
    print("✅ Dữ liệu đã tải")
    
    # Tải mô hình
    print("\n[2/6] Đang tải mô hình đã huấn luyện...")
    from hgt_model import create_hgt_model
    model = create_hgt_model(graph, task='link_prediction', 
                            hidden_channels=128, out_channels=64, 
                            num_heads=8, num_layers=2)
    
    checkpoint = torch.load(f"{config.GRAPH_DATA_PATH}best_model.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Mô hình đã tải")
    
    # Tạo evaluator
    print("\n[3/6] Đang tạo evaluator...")
    evaluator = HGTEvaluator(model, graph, test_data, edge_type, device='cpu')
    print("✅ Evaluator đã tạo")
    
    # Tạo các biểu đồ trực quan
    print("\n[4/6] Đang tạo đường cong ROC & PR...")
    evaluator.plot_roc_pr_curves(save_path=f"{config.GRAPH_DATA_PATH}hgt_roc_pr_curves.png")
    
    print("\n[5/6] Đang tạo ma trận nhầm lẫn...")
    evaluator.plot_confusion_matrix(save_path=f"{config.GRAPH_DATA_PATH}hgt_confusion_matrix.png")
    
    print("\n[6/6] Đang tạo trực quan embeddings...")
    evaluator.plot_embeddings_tsne(save_path=f"{config.GRAPH_DATA_PATH}hgt_embeddings_tsne.png")
    
    print("\n[7/7] Đang phân tích gợi ý...")
    evaluator.analyze_recommendations(job_idx=0, top_k=10, 
                                     save_path=f"{config.GRAPH_DATA_PATH}hgt_recommendations.png")
    
    print("\n" + "="*70)
    print(" "*20 + "🎉 ĐÁNH GIÁ HOÀN TẤT! 🎉")
    print("="*70)
    print(f"\nCác biểu đồ đã tạo:")
    print(f"  📊 {config.GRAPH_DATA_PATH}hgt_roc_pr_curves.png")
    print(f"  📊 {config.GRAPH_DATA_PATH}hgt_confusion_matrix.png")
    print(f"  📊 {config.GRAPH_DATA_PATH}hgt_embeddings_tsne.png")
    print(f"  📊 {config.GRAPH_DATA_PATH}hgt_recommendations.png")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
