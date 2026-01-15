# Tài liệu Chi tiết Mã nguồn HGT (Heterogeneous Graph Transformer)

Tài liệu này cung cấp phân tích chi tiết về các tệp tin nguồn liên quan đến mô hình HGT trong dự án. Tài liệu được biên soạn nhằm mục đích giáo dục, giải thích rõ ràng từng phương thức, luồng xử lý và vai trò của chúng trong hệ thống.

---

## 1. hgt_evaluation.py

Tệp này chịu trách nhiệm đánh giá hiệu suất của mô hình và trực quan hóa kết quả. Nó sử dụng các thư viện như `matplotlib`, `seaborn` và `sklearn` để tạo biểu đồ và tính toán các chỉ số thống kê.

### 1.1 `HGTEvaluator.__init__`

```python
    def __init__(self, model, graph, test_data, edge_type, device='cpu'):
        self.model = model.to(device)
        self.graph = graph
        self.test_data = test_data.to(device)
        self.edge_type = edge_type
        self.device = device
```

- **Mục đích:** Khởi tạo đối tượng `HGTEvaluator`, thiết lập môi trường đánh giá.
- **Giải thích chi tiết:**
  - Đây là hàm khởi tạo (constructor) của lớp.
  - **Tham số:**
    - `model`: Mô hình HGT đã được huấn luyện.
    - `graph`: Toàn bộ cấu trúc đồ thị (HeteroData).
    - `test_data`: Tập dữ liệu kiểm thử (đã được tách ra từ quá trình training).
    - `edge_type`: Loại cạnh (quan hệ) mà chúng ta muốn đánh giá dự đoán (ví dụ: `('job', 'similar_to', 'job')`).
    - `device`: Thiết bị chạy tính toán ('cpu' hoặc 'cuda').
  - **Logic:**
    - `self.model = model.to(device)`: Chuyển mô hình sang thiết bị tính toán (GPU/CPU).
    - Lưu trữ các tham số còn lại vào thuộc tính của instance để sử dụng trong các phương thức khác.

### 1.2 `HGTEvaluator.get_predictions`

```python
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
```

- **Mục đích:** Thực hiện dự đoán trên tập dữ liệu kiểm thử (test set) và trả về xác suất dự đoán cùng nhãn thực tế.
- **Giải thích chi tiết:**
  - `@torch.no_grad()`: Decorator báo cho PyTorch không cần tính toán gradient, giúp tiết kiệm bộ nhớ và tăng tốc độ vì đây là bước đánh giá (inference), không phải huấn luyện.
  - `self.model.eval()`: Chuyển mô hình sang chế độ đánh giá (tắt Dropout, Batch Norm hoạt động theo thống kê toàn cục).
  - **Chuẩn bị dữ liệu đầu vào:**
    - `edge_label_index`: Chỉ số các cạnh cần dự đoán trong tập test.
    - `edge_label`: Nhãn thực tế (1 là có liên kết, 0 là không có) của các cạnh đó.
    - `x_dict`: Dictionary chứa đặc trưng (features) của từng loại node (job, company, location).
    - `edge_index_dict`: Dictionary chứa cấu trúc kết nối của đồ thị trong tập test.
  - **Thực thi mô hình:**
    - `pred = self.model(...)`: Gọi hàm `forward` của mô hình để lấy kết quả (logits).
    - `torch.sigmoid(pred)`: Áp dụng hàm Sigmoid để chuyển đổi logits thành xác suất (0.0 đến 1.0).
    - `.cpu().numpy()`: Chuyển Tensor từ GPU về CPU và đổi sang định dạng NumPy array để dễ xử lý với thư viện sklearn.
  - **Giá trị trả về:**
    - `pred_probs`: Mảng xác suất dự đoán.
    - `labels`: Mảng nhãn thực tế.

### 1.3 `HGTEvaluator.get_embeddings`

```python
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
```

- **Mục đích:** Trích xuất vector đặc trưng (embeddings) của tất cả các node sau khi đã đi qua mô hình HGT. Embeddings này chứa thông tin ngữ nghĩa tổng hợp từ đồ thị.
- **Giải thích chi tiết:**
  - Khác với `get_predictions` chỉ chạy trên tập test, hàm này sử dụng `self.graph` (toàn bộ đồ thị) để tạo embeddings cho tất cả các nút.
  - **Chuẩn bị dữ liệu:**
    - Tạo `x_dict` và `edge_index_dict` từ toàn bộ đồ thị gốc, đảm bảo chuyển dữ liệu sang đúng `device`.
  - **Trích xuất:**
    - `self.model.encode(...)`: Gọi phương thức `encode` riêng của mô hình HGT (chỉ chạy phần Encoder, không chạy phần Predictor head).
  - **Xử lý kết quả:**
    - Vòng lặp `for` chuyển đổi kết quả từ Tensor sang NumPy array cho từng loại node.
  - **Giá trị trả về:** `embeddings_np` là một dictionary chứa các ma trận embedding cho 'job', 'company', 'location'.

### 1.4 `HGTEvaluator.plot_roc_pr_curves`

```python
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
```

- **Mục đích:** Vẽ hai biểu đồ quan trọng để đánh giá mô hình phân loại nhị phân: ROC Curve và Precision-Recall Curve.
- **Giải thích chi tiết:**
  - **IO/GUI:** Sử dụng `matplotlib` để vẽ hình.
  - **Logic:**
    1.  Gọi `get_predictions()` để lấy kết quả dự đoán.
    2.  **Biểu đồ ROC (Receiver Operating Characteristic):**
        - Tính `roc_curve` (False Positive Rate vs True Positive Rate) và `auc` (Area Under Curve). AUC càng gần 1.0 càng tốt.
        - Vẽ đường cơ sở (Random) màu đỏ đứt nét (đại diện cho việc đoán mò).
    3.  **Biểu đồ Precision-Recall:**
        - Tính `precision`, `recall` và điểm `Average Precision (AP)`. Biểu đồ này quan trọng khi dữ liệu mất cân bằng.
  - **Lưu file:** Nếu có tham số `save_path`, biểu đồ sẽ được lưu thành file ảnh (PNG/JPG).

### 1.5 `HGTEvaluator.plot_confusion_matrix`

```python
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
```

- **Mục đích:** Hiển thị Confusion Matrix để xem chi tiết số lượng True Positive, False Positive, True Negative, và False Negative.
- **Giải thích chi tiết:**
  - **Tham số:** `threshold`: Ngưỡng để quyết định phân lớp (mặc định 0.5). Nếu xác suất >= 0.5 thì coi là Positive.
  - **Logic:**
    1.  Chuyển xác suất thành nhãn 0/1 dựa trên ngưỡng.
    2.  Sử dụng `seaborn.heatmap` để vẽ ma trận màu sắc trực quan.
    3.  Tính toán thủ công các chỉ số Accuracy, Precision, Recall, F1-Score từ các giá trị TN, FP, FN, TP và hiển thị trực tiếp lên biểu đồ.

### 1.6 `HGTEvaluator.plot_embeddings_tsne`

```python
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
```

- **Mục đích:** Sử dụng thuật toán t-SNE (t-Distributed Stochastic Neighbor Embedding) để giảm chiều dữ liệu embedding (từ 128 chiều xuống 2 chiều) nhằm mục đích hiển thị lên mặt phẳng.
- **Giải thích chi tiết:**
  - **Data Science Logic:** t-SNE là thuật toán mạnh mẽ để trực quan hóa dữ liệu cao chiều, giúp ta thấy được liệu các điểm dữ liệu giống nhau có đứng gần nhau trong không gian vector hay không.
  - **IO:** Đọc file CSV `jobs_processed.csv` để lấy thông tin nhãn (Title, Salary).
  - **Hàm `extract_category`:** Phân loại công việc thủ công dựa trên từ khóa trong tiêu đề (ví dụ: 'java' -> 'IT/Developer') để tô màu cho các điểm trên biểu đồ.
  - **Biểu đồ 1:** Tô màu theo Danh mục công việc. Ta kỳ vọng các công việc cùng ngành sẽ cụm lại với nhau.
  - **Biểu đồ 2:** Tô màu theo Mức lương trung bình.
  - **Kết quả:** Biểu đồ giúp xác nhận trực quan rằng mô hình đã học được ngữ nghĩa của dữ liệu hay chưa.

### 1.7 `HGTEvaluator.analyze_recommendations`

```python
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
```

- **Mục đích:** Kiểm thử thực tế khả năng gợi ý của mô hình bằng cách chọn 1 công việc bất kỳ và tìm ra các công việc tương tự nhất.
- **Giải thích chi tiết:**
  - **Thuật toán Cosine Similarity:**
    - Công thức: `dot(A, B) / (norm(A) * norm(B))`.
    - Dùng để đo góc giữa hai vector embedding. Góc càng nhỏ (giá trị càng gần 1) thì càng tương đồng.
  - **Logic:**
    1.  Tính Cosine Similarity giữa công việc mục tiêu (`job_idx`) và TẤT CẢ các công việc khác.
    2.  Sắp xếp (`argsort`) và lấy ra `top_k` chỉ số có điểm cao nhất.
    3.  Hiển thị tên các công việc gợi ý.
    4.  So sánh các thông số (Lương, Kinh nghiệm) giữa công việc gốc và công việc gợi ý để xem liệu gợi ý có hợp lý về mặt logic nghiệp vụ không.

### 1.8 `main` (hgt_evaluation.py)

```python
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
```

- **Mục đích:** File thực thi chính (Entry Point) cho quá trình đánh giá.
- **Luồng xử lý:**
  1.  Load file đồ thị `hetero_graph.pt`.
  2.  Tái tạo lại việc chia dữ liệu `RandomLinkSplit` để có tập `test_data` giống như lúc train (lưu ý: để chính xác tuyệt đối, hạt giống ngẫu nhiên (seed) cần phải giống nhau).
  3.  Khởi tạo kiến trúc mô hình HGT (cấu trúc phải khớp với lúc train).
  4.  Load trọng số (weights) từ file `best_model.pt`.
  5.  Khởi tạo `HGTEvaluator`.
  6.  Gọi lần lượt các hàm vẽ biểu đồ.

---

## 2. hgt_model.py

Tệp này định nghĩa kiến trúc cốt lõi của mạng nơ-ron Heterogeneous Graph Transformer. Đây là "trái tim" của hệ thống trí tuệ nhân tạo dự án.

### 2.1 `HGT.__init__`

```python
    def __init__(
        self,
        metadata,
        hidden_channels=128,
        out_channels=64,
        num_heads=8,
        num_layers=2,
        node_type_dims=None,
    ):
        """
        Tham số:
            metadata: Metadata của PyG HeteroData (các loại node, edge)
            hidden_channels: Kích thước chiều ẩn
            out_channels: Kích thước embedding đầu ra
            num_heads: Số lượng attention heads
            num_layers: Số lượng lớp HGT
            node_type_dims: Dict ánh xạ loại node đến kích thước features đầu vào
        """
        super().__init__()

        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Lớp projection đầu vào cho mỗi loại node
        self.lin_dict = nn.ModuleDict()
        for node_type, dim in node_type_dims.items():
            self.lin_dict[node_type] = Linear(dim, hidden_channels)

        # Các lớp HGT Convolution
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                hidden_channels,
                hidden_channels,
                metadata,
                num_heads,
            )
            self.convs.append(conv)

        # Lớp projection đầu ra
        self.lin_out = Linear(hidden_channels, out_channels)
```

- **Mục đích:** Khởi tạo kiến trúc mạng HGT.
- **Giải thích chi tiết (OOP/Deep Learning):**
  - Kế thừa từ `nn.Module` (lớp cơ sở của mọi mạng nơ-ron trong PyTorch).
  - **Input Projection (`lin_dict`):** Vì các loại node khác nhau (job, company, location) có số lượng features đầu vào khác nhau, chúng ta cần các lớp Linear riêng biệt để chiếu tất cả về cùng một không gian vector `hidden_channels` trước khi đưa vào HGT.
  - **HGT Layers (`convs`):** Sử dụng `HGTConv` từ thư viện `torch_geometric`. Đây là lớp thực hiện cơ chế Attention trên đồ thị dị thể.
  - **Output Projection (`lin_out`):** Lớp Linear cuối cùng để đưa vector về kích thước mong muốn `out_channels`.

### 2.2 `HGT.forward`

```python
    def forward(self, x_dict, edge_index_dict):
        """
        Lan truyền xuôi (Forward pass)

        Tham số:
            x_dict: Dictionary chứa features của các node {loại_node: features}
            edge_index_dict: Dictionary chứa chỉ số các cạnh {loại_edge: edge_index}

        Trả về:
            Dictionary chứa embeddings của các node {loại_node: embeddings}
        """
        # Projection đầu vào
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        # Các lớp HGT convolution
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        # Projection đầu ra
        out_dict = {
            node_type: self.lin_out(x)
            for node_type, x in x_dict.items()
        }

        return out_dict
```

- **Mục đích:** Định nghĩa luồng dữ liệu đi qua mạng.
- **Luồng xử lý:**
  1.  **Input Linear:** Features thô -> Linear -> ReLU -> Hidden Features.
  2.  **HGT Message Passing:** Lặp qua số lớp `num_layers`. Mỗi lớp `conv` sẽ tổng hợp thông tin từ các node lân cận dựa trên metadata của đồ thị. Sau mỗi lớp đều dùng hàm kích hoạt `relu`.
  3.  **Output Linear:** Hidden Features -> Linear -> Output Embeddings.
  4.  Trả về dictionary chứa embeddings mới cho từng loại node.

### 2.3 `HGTLinkPredictor` (Các methods)

Lớp này bao bọc lớp `HGT` để thực hiện nhiệm vụ cụ thể là dự đoán liên kết.

#### `__init__`

```python
    def __init__(
        self,
        metadata,
        hidden_channels=128,
        out_channels=64,
        num_heads=8,
        num_layers=2,
        node_type_dims=None,
    ):
        super().__init__()

        # HGT encoder
        self.hgt = HGT(
            metadata,
            hidden_channels,
            out_channels,
            num_heads,
            num_layers,
            node_type_dims,
        )

        # Đầu dự đoán liên kết
        self.predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 1)
        )
```

- **Giải thích:**
  - Chứa thành phần `self.hgt` để tạo embeddings.
  - Chứa thành phần `self.predictor`: Đây là một mạng MLP (Multi-Layer Perceptron) nhỏ. Đầu vào là `out_channels * 2` vì nó sẽ ghép nối (concat) embedding của 2 node lại với nhau để dự đoán mối quan hệ.

#### `forward`

```python
    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_type):
        """
        Tham số:
            x_dict: Features của các node
            edge_index_dict: Chỉ số các cạnh
            edge_label_index: Các cạnh cần dự đoán (2, num_edges)
            edge_type: Loại cạnh cần dự đoán ('job', 'similar_to', 'job')

        Trả về:
            Điểm số dự đoán liên kết
        """
        # Lấy embeddings của các node
        node_emb_dict = self.hgt(x_dict, edge_index_dict)

        # Lấy loại node nguồn và đích
        src_type, _, dst_type = edge_type

        # Lấy embeddings cho các cạnh cần dự đoán
        src_emb = node_emb_dict[src_type][edge_label_index[0]]
        dst_emb = node_emb_dict[dst_type][edge_label_index[1]]

        # Ghép nối và dự đoán
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        pred = self.predictor(edge_emb).squeeze(-1)

        return pred
```

- **Luồng xử lý:**
  1.  Chạy `self.hgt` để lấy vector đặc trưng cho TẤT CẢ các node trong đồ thị.
  2.  `edge_label_index` chứa các cặp node cần kiểm tra (ví dụ: Job A - Job B).
  3.  Lấy vector của Job A (`src_emb`) và vector của Job B (`dst_emb`).
  4.  `torch.cat`: Nối đuôi hai vector này lại.
  5.  Đưa qua `predictor` để tính ra một điểm số (score) duy nhất thể hiện khả năng tồn tại liên kết.

#### `encode`

```python
    def encode(self, x_dict, edge_index_dict):
        """Lấy embeddings của các node"""
        return self.hgt(x_dict, edge_index_dict)
```

- **Mục đích:** Hàm wrapper tiện ích để chỉ lấy embeddings mà không cần dự đoán.

### 2.4 `HGTNodeClassifier` (Các methods)

Lớp này bao bọc `HGT` cho nhiệm vụ phân loại node (ví dụ: phân loại mức lương của công việc).

#### `__init__`

```python
    def __init__(
        self,
        metadata,
        num_classes,
        hidden_channels=128,
        out_channels=64,
        num_heads=8,
        num_layers=2,
        node_type_dims=None,
        target_node_type='job',
    ):
        super().__init__()

        self.target_node_type = target_node_type

        # HGT encoder
        self.hgt = HGT(
            metadata,
            hidden_channels,
            out_channels,
            num_heads,
            num_layers,
            node_type_dims,
        )

        # Đầu phân loại
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, num_classes)
        )
```

- **Giải thích:**
  - Tương tự `LinkPredictor` nhưng `classifier` đầu ra có kích thước là `num_classes` (số lượng lớp phân loại, ví dụ: thấp/trung bình/cao).

#### `forward`

```python
    def forward(self, x_dict, edge_index_dict):
        """
        Tham số:
            x_dict: Features của các node
            edge_index_dict: Chỉ số các cạnh

        Trả về:
            Logits phân loại cho loại node đích
        """
        # Lấy embeddings của các node
        node_emb_dict = self.hgt(x_dict, edge_index_dict)

        # Phân loại các node đích
        target_emb = node_emb_dict[self.target_node_type]
        logits = self.classifier(target_emb)

        return logits
```

- **Luồng xử lý:** Lấy embedding của node mục tiêu -> đưa qua lớp phân loại -> trả về xác suất thuộc về từng lớp.

### 2.5 `create_hgt_model`

```python
def create_hgt_model(graph, task='link_prediction', **kwargs):
    """
    Hàm factory để tạo mô hình HGT

    Tham số:
        graph: Đối tượng PyG HeteroData
        task: 'link_prediction' hoặc 'node_classification'
        **kwargs: Các tham số mô hình bổ sung

    Trả về:
        Instance của mô hình HGT
    """
    # Lấy metadata
    metadata = graph.metadata()

    # Lấy kích thước của các loại node
    node_type_dims = {
        'job': graph['job'].x.shape[1],
        'company': graph['company'].x.shape[1],
        'location': graph['location'].x.shape[1],
    }

    # Tham số mặc định
    default_params = {
        'hidden_channels': 128,
        'out_channels': 64,
        'num_heads': 8,
        'num_layers': 2,
    }
    default_params.update(kwargs)

    # Tạo mô hình dựa trên tác vụ
    if task == 'link_prediction':
        model = HGTLinkPredictor(
            metadata=metadata,
            node_type_dims=node_type_dims,
            **default_params
        )
    elif task == 'node_classification':
        if 'num_classes' not in kwargs:
            raise ValueError("num_classes required for node classification")
        model = HGTNodeClassifier(
            metadata=metadata,
            node_type_dims=node_type_dims,
            **default_params
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return model
```

- **Mục đích:** Áp dụng mẫu thiết kế **Factory Pattern**. Thay vì khởi tạo class trực tiếp, ta dùng hàm này để tự động trích xuất metadata từ graph và tạo đúng loại class (LinkPredictor hoặc NodeClassifier) dựa trên tham số `task`.

---

## 3. hgt_summary.py

Tệp tiện ích đơn giản để in ra báo cáo tổng kết sau khi chạy huấn luyện, giúp người dùng nắm bắt nhanh trạng thái hệ thống.

### 3.1 `print_section`

```python
def print_section(title, width=70):
    """In tiêu đề phần có định dạng"""
    print("\n" + "="*width)
    print(f" {title.center(width-2)} ")
    print("="*width)
```

- **Mục đích:** Hàm hỗ trợ định dạng chuỗi, in ra các tiêu đề được căn giữa với đường viền dấu bằng, giúp log file dễ đọc hơn.

### 3.2 `summarize_results`

```python
def summarize_results():
    """Tạo báo cáo tổng hợp về các thí nghiệm HGT"""

    print_section("TỔNG HỢP THÍ NGHIỆM HGT")
    print(f"\nTạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Thông tin Mô hình
    # ... (Code kiểm tra file best_model.pt và in kích thước)

    # 2. Thông tin Đồ thị
    # ... (Code load graph và in số lượng node/edge)

    # 3. Cấu hình Huấn luyện
    # ... (In hardcode các tham số config)

    # 4. Các Biểu đồ Đã tạo
    # ... (Kiểm tra sự tồn tại của các file png)

    # 5. Các File Đầu ra
    # ... (Liệt kê đường dẫn file)

    # 6. Hướng dẫn Sử dụng
    # ... (In hướng dẫn command line)

    # Footer
    # ...
```

- **Mục đích:** Cung cấp cái nhìn toàn cảnh (Dashboard dạng text) về project.
- **IO:** Kiểm tra sự tồn tại của file (`os.path.exists`), lấy kích thước file, đọc file `.pt` để lấy metadata.
- **Logic:** Tuần tự kiểm tra từng thành phần (Model, Graph, Plot) và in trạng thái (✅ hoặc ❌).

---

## 4. train_hgt.py

Tệp này quản lý quy trình huấn luyện (Training Loop) cho mô hình.

### 4.1 `HGTTrainer.__init__`

```python
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.001,
        weight_decay=1e-5,
    ):
        """
        Tham số:
            model: Mô hình HGT
            device: Thiết bị để huấn luyện
            lr: Tốc độ học (learning rate)
            weight_decay: Hệ số suy giảm trọng số
        """
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        print(f"\n{'='*60}")
        print(f"Khởi tạo HGT Trainer")
        print(f"{'='*60}")
        print(f"Thiết bị: {device}")
        print(f"Tốc độ học: {lr}")
        print(f"Hệ số suy giảm: {weight_decay}")
        print(f"Số tham số mô hình: {sum(p.numel() for p in model.parameters()):,}")
```

- **Mục đích:** Thiết lập môi trường huấn luyện.
- **Logic:**
  - Chuyển model sang GPU/CPU.
  - Khởi tạo `Adam Optimizer`: Thuật toán tối ưu hóa phổ biến nhất cho Deep Learning hiện nay. Nó sẽ cập nhật trọng số của model để giảm thiểu hàm mất mát.

### 4.2 `HGTTrainer.train_epoch`

```python
    def train_epoch(self, data, edge_type):
        """Huấn luyện một epoch"""
        self.model.train()
        self.optimizer.zero_grad()

        data = data.to(self.device)

        # Lấy các cạnh dương và âm
        edge_label_index = data[edge_type].edge_label_index
        edge_label = data[edge_type].edge_label

        # Lan truyền xuôi
        x_dict = {
            'job': data['job'].x,
            'company': data['company'].x,
            'location': data['location'].x,
        }

        edge_index_dict = {
            key: data[key].edge_index
            for key in data.edge_types
        }

        pred = self.model(x_dict, edge_index_dict, edge_label_index, edge_type)

        # Hàm mất mát binary cross entropy
        loss = F.binary_cross_entropy_with_logits(pred, edge_label.float())

        loss.backward()
        self.optimizer.step()

        return loss.item()
```

- **Mục đích:** Thực hiện 1 vòng lặp huấn luyện (1 epoch).
- **Quy trình chuẩn trong PyTorch:**
  1.  `model.train()`: Bật chế độ training (quan trọng cho Dropout/BatchNorm).
  2.  `optimizer.zero_grad()`: Xóa sạch các gradient cũ để không bị cộng dồn.
  3.  **Forward Pass:** Chạy dữ liệu qua model để lấy dự đoán `pred`.
  4.  **Loss Calculation:** Tính sai số giữa dự đoán và thực tế dùng `binary_cross_entropy_with_logits` (thích hợp cho bài toán phân loại nhị phân như dự đoán liên kết).
  5.  **Backward Pass (`loss.backward()`):** Tính đạo hàm (gradient) ngược từ loss về các trọng số (Backpropagation).
  6.  **Optimizer Step (`optimizer.step()`):** Cập nhật trọng số dựa trên gradient vừa tính.

### 4.3 `HGTTrainer.evaluate`

```python
    @torch.no_grad()
    def evaluate(self, data, edge_type):
        """Đánh giá mô hình"""
        self.model.eval()

        data = data.to(self.device)

        edge_label_index = data[edge_type].edge_label_index
        edge_label = data[edge_type].edge_label

        x_dict = {
            'job': data['job'].x,
            'company': data['company'].x,
            'location': data['location'].x,
        }

        edge_index_dict = {
            key: data[key].edge_index
            for key in data.edge_types
        }

        pred = self.model(x_dict, edge_index_dict, edge_label_index, edge_type)
        pred = torch.sigmoid(pred)

        preds = pred.cpu().numpy()
        labels = edge_label.cpu().numpy()

        # Tính các chỉ số
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)

        return auc, ap
```

- **Mục đích:** Kiểm tra độ chính xác của model hiện tại trên tập validation hoặc test.
- **Lưu ý:** Hàm này tương tự hàm trong `HGTEvaluator` nhưng đơn giản hơn, chỉ trả về chỉ số AUC và AP để phục vụ việc theo dõi quá trình training.

### 4.4 `HGTTrainer.train`

```python
    def train(
        self,
        train_data,
        val_data,
        test_data,
        edge_type,
        epochs=50,
        eval_every=5,
    ):
        """
        Vòng lặp huấn luyện đầy đủ
        ...
        """
        print(f"\n{'='*60}")
        print("Bắt đầu huấn luyện")
        # ... (In thông tin)

        best_val_auc = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            # Huấn luyện
            loss = self.train_epoch(train_data, edge_type)

            # Đánh giá
            if epoch % eval_every == 0:
                train_auc, train_ap = self.evaluate(train_data, edge_type)
                val_auc, val_ap = self.evaluate(val_data, edge_type)

                print(f"\nEpoch {epoch:03d}:")
                print(f"  Mất mát: {loss:.4f}")
                print(f"  Huấn luyện - AUC: {train_auc:.4f}, AP: {train_ap:.4f}")
                print(f"  Validation - AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

                # Lưu mô hình tốt nhất
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    self.save_model('best_model.pt')
                    print(f"  ✅ Mô hình tốt nhất đã được lưu!")
            else:
                print(f"Epoch {epoch:03d}: Mất mát = {loss:.4f}")

        # ... (Load best model và test cuối cùng)

        return { ... }
```

- **Mục đích:** Điều phối toàn bộ quá trình training qua nhiều epochs.
- **Logic:**
  - Vòng lặp chạy từ 1 đến `epochs`.
  - Gọi `train_epoch` mỗi vòng.
  - Mỗi `eval_every` epoch, gọi `evaluate` để kiểm tra trên tập validation.
  - **Cơ chế Checkpoint:** Nếu AUC trên tập validation cao hơn kỷ lục cũ (`best_val_auc`), lưu model lại (`save_model`). Điều này đảm bảo ta luôn giữ lại phiên bản tốt nhất chứ không phải phiên bản cuối cùng (tránh overfitting).

### 4.5 `HGTTrainer.save_model`, `HGTTrainer.load_model`

```python
    def save_model(self, filename):
        """Lưu checkpoint của mô hình"""
        path = os.path.join(config.GRAPH_DATA_PATH, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, filename):
        """Tải checkpoint của mô hình"""
        path = os.path.join(config.GRAPH_DATA_PATH, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

- **IO:** Sử dụng `torch.save` và `torch.load` để ghi/đọc file nhị phân.
- **Lưu ý:** Ta lưu cả `optimizer_state_dict` để có thể tiếp tục training từ điểm dừng nếu cần thiết (resume training).

### 4.6 `prepare_data`

```python
def prepare_data(graph, edge_type=('job', 'similar_to', 'job'), split_ratio=[0.8, 0.1, 0.1]):
    """
    Chuẩn bị chia dữ liệu train/val/test
    ...
    """
    # ... (In thông tin)

    # Chia các cạnh thành train/val/test
    transform = RandomLinkSplit(
        num_val=split_ratio[1],
        num_test=split_ratio[2],
        edge_types=[edge_type],
        rev_edge_types=[edge_type],  # Vì có cạnh hai chiều
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,  # Tỉ lệ mẫu dương:mẫu âm = 1:1
    )

    train_data, val_data, test_data = transform(graph)

    # ... (In kết quả)

    return train_data, val_data, test_data
```

- **Mục đích:** Chia tập dữ liệu đồ thị thành 3 phần.
- **Logic:**
  - Sử dụng `RandomLinkSplit` của PyG. Đây là công cụ chuyên dụng cho bài toán Link Prediction.
  - Nó sẽ giấu đi một số cạnh trong `edge_index` để làm dữ liệu kiểm thử (positive samples).
  - Nó cũng tự động sinh ra các cạnh giả (negative samples - nối 2 node thực tế không liên kết với nhau) với tỉ lệ 1:1 để model học cách phân biệt liên kết thật và giả.

### 4.7 `main` (train_hgt.py)

```python
def main():
    """Pipeline huấn luyện chính"""
    # ... (In tiêu đề)

    # Load đồ thị
    # ... (Code load graph file)

    # Chuẩn bị dữ liệu
    # ... (Gọi prepare_data)

    # Tạo mô hình
    # ... (Gọi create_hgt_model)

    # Huấn luyện
    # ... (Khởi tạo HGTTrainer và gọi train)

    # Tóm tắt
    # ... (In kết quả)
```

- **Mục đích:** Hàm main điều phối toàn bộ file `train_hgt.py`. Kết nối các bước từ load dữ liệu -> xử lý -> tạo model -> training -> báo cáo.
