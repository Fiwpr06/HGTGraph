"""Triển khai Heterogeneous Graph Transformer (HGT) cho gợi ý công việc"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGT(nn.Module):
    """
    Mô hình Heterogeneous Graph Transformer (HGT) cho gợi ý công việc
    
    Paper: "Heterogeneous Graph Transformer" (WWW 2020)
    https://arxiv.org/abs/2003.01332
    
    Kiến trúc:
    - Lớp projection đầu vào cho mỗi loại node
    - Nhiều lớp HGT convolution
    - Lớp projection đầu ra cho các tác vụ cụ thể
    """
    
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


class HGTLinkPredictor(nn.Module):
    """
    Mô hình dự đoán liên kết dựa trên HGT (gợi ý công việc)
    
    Tác vụ: Dự đoán công việc nào người dùng có thể quan tâm dựa trên cấu trúc đồ thị
    Cho dự án này: Dự đoán độ tương đồng job-job hoặc kết nối job-location
    """
    
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
    
    def encode(self, x_dict, edge_index_dict):
        """Get node embeddings"""
        return self.hgt(x_dict, edge_index_dict)


class HGTNodeClassifier(nn.Module):
    """
    Mô hình phân loại node dựa trên HGT
    
    Tác vụ: Phân loại các node công việc (ví dụ: dự đoán danh mục công việc, mức lương, v.v.)
    """
    
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
    
    def encode(self, x_dict, edge_index_dict):
        """Get node embeddings"""
        return self.hgt(x_dict, edge_index_dict)


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


if __name__ == "__main__":
    # Kiểm tra tạo mô hình
    import torch
    from torch_geometric.data import HeteroData
    
    # Tạo đồ thị giả để test
    print("Đang tạo đồ thị giả để kiểm tra...")
    graph = HeteroData()
    
    # Features của các node
    graph['job'].x = torch.randn(100, 400)
    graph['company'].x = torch.randn(20, 10)
    graph['location'].x = torch.randn(10, 8)
    
    # Chỉ số các cạnh
    graph['job', 'posted_by', 'company'].edge_index = torch.randint(0, 100, (2, 200))
    graph['company', 'posts', 'job'].edge_index = torch.randint(0, 100, (2, 200))
    graph['job', 'located_in', 'location'].edge_index = torch.randint(0, 100, (2, 200))
    graph['location', 'has', 'job'].edge_index = torch.randint(0, 100, (2, 200))
    graph['job', 'similar_to', 'job'].edge_index = torch.randint(0, 100, (2, 300))
    
    print("\nCấu trúc đồ thị:")
    print(graph)
    
    # Kiểm tra mô hình dự đoán liên kết
    print("\n" + "="*60)
    print("Kiểm tra mô hình HGT dự đoán liên kết")
    print("="*60)
    
    model = create_hgt_model(graph, task='link_prediction', hidden_channels=64, num_layers=2)
    print(f"\nMô hình đã tạo với {sum(p.numel() for p in model.parameters())} tham số")
    
    # Lan truyền xuôi
    x_dict = {
        'job': graph['job'].x,
        'company': graph['company'].x,
        'location': graph['location'].x,
    }
    edge_index_dict = {
        ('job', 'posted_by', 'company'): graph['job', 'posted_by', 'company'].edge_index,
        ('company', 'posts', 'job'): graph['company', 'posts', 'job'].edge_index,
        ('job', 'located_in', 'location'): graph['job', 'located_in', 'location'].edge_index,
        ('location', 'has', 'job'): graph['location', 'has', 'job'].edge_index,
        ('job', 'similar_to', 'job'): graph['job', 'similar_to', 'job'].edge_index,
    }
    
    # Kiểm tra encoding
    node_emb_dict = model.encode(x_dict, edge_index_dict)
    print("\nEmbeddings của các node:")
    for node_type, emb in node_emb_dict.items():
        print(f"  {node_type}: {emb.shape}")
    
    # Kiểm tra dự đoán liên kết
    edge_label_index = torch.randint(0, 100, (2, 50))
    pred = model(x_dict, edge_index_dict, edge_label_index, ('job', 'similar_to', 'job'))
    print(f"\nKích thước output dự đoán liên kết: {pred.shape}")
    
    print("\n✅ Kiểm tra mô hình thành công!")
