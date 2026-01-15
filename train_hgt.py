"""Script huấn luyện mô hình HGT cho tác vụ gợi ý công việc"""

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.transforms import RandomLinkSplit

import config
from hgt_model import create_hgt_model

warnings.filterwarnings("ignore")


class HGTTrainer:
    """Lớp huấn luyện mô hình HGT cho tác vụ dự đoán liên kết"""

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

        Tham số:
            train_data: Dữ liệu huấn luyện
            val_data: Dữ liệu validation
            test_data: Dữ liệu test
            edge_type: Loại cạnh để dự đoán
            epochs: Số epoch
            eval_every: Đánh giá sau mỗi N epochs
        """
        print(f"\n{'='*60}")
        print("Bắt đầu huấn luyện")
        print(f"{'='*60}")
        print(f"Số epochs: {epochs}")
        print(f"Loại cạnh: {edge_type}"))

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

        # Load mô hình tốt nhất và đánh giá trên tập test
        print(f"\n{'='*60}")
        print("Huấn luyện hoàn tất!")
        print(f"{'='*60}")
        print(f"AUC validation tốt nhất: {best_val_auc:.4f} (Epoch {best_epoch})")

        self.load_model('best_model.pt')
        test_auc, test_ap = self.evaluate(test_data, edge_type)
        print(f"\nKết quả Test cuối cùng:")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  AP:  {test_ap:.4f}")

        return {
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'test_auc': test_auc,
            'test_ap': test_ap,
        }

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


def prepare_data(graph, edge_type=('job', 'similar_to', 'job'), split_ratio=[0.8, 0.1, 0.1]):
    """
    Chuẩn bị chia dữ liệu train/val/test

    Tham số:
        graph: Đối tượng PyG HeteroData
        edge_type: Loại cạnh dùng cho dự đoán liên kết
        split_ratio: Tỉ lệ [train, val, test]

    Trả về:
        train_data, val_data, test_data
    """
    print(f"\n{'='*60}")
    print("Chuẩn bị dữ liệu")
    print(f"{'='*60}")
    print(f"Loại cạnh cho dự đoán: {edge_type}")
    print(f"Tỉ lệ chia: Train={split_ratio[0]}, Val={split_ratio[1]}, Test={split_ratio[2]}")

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

    print(f"\nChia dữ liệu hoàn tất:")
    print(f"  Cạnh train: {train_data[edge_type].edge_label_index.size(1)}")
    print(f"  Cạnh val:   {val_data[edge_type].edge_label_index.size(1)}")
    print(f"  Cạnh test:  {test_data[edge_type].edge_label_index.size(1)}")

    return train_data, val_data, test_data


def main():
    """Pipeline huấn luyện chính"""
    print("\n" + "="*70)
    print(" "*15 + "PIPELINE HUẤN LUYỆN HGT")
    print("="*70)

    # Load đồ thị
    print("\n[BƯỚC 1/4] Đang tải đồ thị...")
    print("-"*70)
    graph_path = os.path.join(config.GRAPH_DATA_PATH, 'hetero_graph.pt')
    
    if not os.path.exists(graph_path):
        print(f"❌ Không tìm thấy file đồ thị: {graph_path}")
        print("Vui lòng chạy main.py trước để xây dựng đồ thị!")
        return
    
    graph = torch.load(graph_path)
    print(f"✅ Đã tải đồ thị từ {graph_path}")
    print(f"\nCấu trúc đồ thị:")
    print(graph)

    # Chuẩn bị dữ liệu
    print("\n[BƯỚC 2/4] Đang chuẩn bị dữ liệu...")
    print("-"*70)
    edge_type = ('job', 'similar_to', 'job')
    train_data, val_data, test_data = prepare_data(graph, edge_type)

    # Tạo mô hình
    print("\n[BƯỚC 3/4] Đang tạo mô hình...")
    print("-"*70)
    model = create_hgt_model(
        graph,
        task='link_prediction',
        hidden_channels=128,
        out_channels=64,
        num_heads=8,
        num_layers=2,
    )
    print(f"✅ Mô hình HGT đã được tạo")
    print(f"   Tham số: {sum(p.numel() for p in model.parameters()):,}")

    # Huấn luyện
    print("\n[BƯỚC 4/4] Đang huấn luyện mô hình...")
    print("-"*70)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = HGTTrainer(model, device=device, lr=0.001)
    
    results = trainer.train(
        train_data,
        val_data,
        test_data,
        edge_type,
        epochs=50,
        eval_every=5,
    )

    # Tóm tắt
    print("\n" + "="*70)
    print(" "*25 + "🎉 HUẤN LUYỆN HOÀN TẤT! 🎉")
    print("="*70)
    print("\nKết quả cuối cùng:")
    print(f"  AUC Validation tốt nhất: {results['best_val_auc']:.4f} (Epoch {results['best_epoch']})")
    print(f"  AUC Test:                {results['test_auc']:.4f}")
    print(f"  AP Test:                 {results['test_ap']:.4f}")
    print(f"\nMô hình đã lưu tại: {config.GRAPH_DATA_PATH}best_model.pt")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Huấn luyện bị ngắt bởi người dùng")
    except Exception as e:
        print(f"\n\n❌ Lỗi xảy ra: {str(e)}")
        import traceback
        traceback.print_exc()
