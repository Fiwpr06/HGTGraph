"""Heterogeneous Graph Transformer (HGT) Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for job recommendation
    
    Paper: "Heterogeneous Graph Transformer" (WWW 2020)
    https://arxiv.org/abs/2003.01332
    
    Architecture:
    - Input projection layers for each node type
    - Multiple HGT convolution layers
    - Output projection for task-specific predictions
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
        Args:
            metadata: PyG HeteroData metadata (node types, edge types)
            hidden_channels: Hidden dimension size
            out_channels: Output embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of HGT layers
            node_type_dims: Dict mapping node type to input feature dimension
        """
        super().__init__()
        
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type, dim in node_type_dims.items():
            self.lin_dict[node_type] = Linear(dim, hidden_channels)
        
        # HGT Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                hidden_channels,
                hidden_channels,
                metadata,
                num_heads,
            )
            self.convs.append(conv)
        
        # Output projection
        self.lin_out = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass
        
        Args:
            x_dict: Dictionary of node features {node_type: features}
            edge_index_dict: Dictionary of edge indices {edge_type: edge_index}
            
        Returns:
            Dictionary of node embeddings {node_type: embeddings}
        """
        # Input projection
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }
        
        # HGT convolution layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Output projection
        out_dict = {
            node_type: self.lin_out(x)
            for node_type, x in x_dict.items()
        }
        
        return out_dict


class HGTLinkPredictor(nn.Module):
    """
    HGT-based model for link prediction (job recommendation)
    
    Task: Predict which jobs a user might be interested in based on graph structure
    For this project: Predict job-job similarity or job-location connections
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
        
        # Link prediction head
        self.predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_type):
        """
        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            edge_label_index: Edges to predict (2, num_edges)
            edge_type: Type of edge to predict ('job', 'job', 'similar_to')
            
        Returns:
            Link prediction scores
        """
        # Get node embeddings
        node_emb_dict = self.hgt(x_dict, edge_index_dict)
        
        # Get source and target node type
        src_type, _, dst_type = edge_type
        
        # Get embeddings for the edges we want to predict
        src_emb = node_emb_dict[src_type][edge_label_index[0]]
        dst_emb = node_emb_dict[dst_type][edge_label_index[1]]
        
        # Concatenate and predict
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        pred = self.predictor(edge_emb).squeeze(-1)
        
        return pred
    
    def encode(self, x_dict, edge_index_dict):
        """Get node embeddings"""
        return self.hgt(x_dict, edge_index_dict)


class HGTNodeClassifier(nn.Module):
    """
    HGT-based model for node classification
    
    Task: Classify job nodes (e.g., predict job category, salary range, etc.)
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, num_classes)
        )
        
    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            
        Returns:
            Classification logits for target node type
        """
        # Get node embeddings
        node_emb_dict = self.hgt(x_dict, edge_index_dict)
        
        # Classify target nodes
        target_emb = node_emb_dict[self.target_node_type]
        logits = self.classifier(target_emb)
        
        return logits
    
    def encode(self, x_dict, edge_index_dict):
        """Get node embeddings"""
        return self.hgt(x_dict, edge_index_dict)


def create_hgt_model(graph, task='link_prediction', **kwargs):
    """
    Factory function to create HGT model
    
    Args:
        graph: PyG HeteroData object
        task: 'link_prediction' or 'node_classification'
        **kwargs: Additional model parameters
        
    Returns:
        HGT model instance
    """
    # Get metadata
    metadata = graph.metadata()
    
    # Get node type dimensions
    node_type_dims = {
        'job': graph['job'].x.shape[1],
        'company': graph['company'].x.shape[1],
        'location': graph['location'].x.shape[1],
    }
    
    # Default parameters
    default_params = {
        'hidden_channels': 128,
        'out_channels': 64,
        'num_heads': 8,
        'num_layers': 2,
    }
    default_params.update(kwargs)
    
    # Create model based on task
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
    # Test model creation
    import torch
    from torch_geometric.data import HeteroData
    
    # Create dummy graph for testing
    print("Creating dummy graph for testing...")
    graph = HeteroData()
    
    # Node features
    graph['job'].x = torch.randn(100, 400)
    graph['company'].x = torch.randn(20, 10)
    graph['location'].x = torch.randn(10, 8)
    
    # Edge indices
    graph['job', 'posted_by', 'company'].edge_index = torch.randint(0, 100, (2, 200))
    graph['company', 'posts', 'job'].edge_index = torch.randint(0, 100, (2, 200))
    graph['job', 'located_in', 'location'].edge_index = torch.randint(0, 100, (2, 200))
    graph['location', 'has', 'job'].edge_index = torch.randint(0, 100, (2, 200))
    graph['job', 'similar_to', 'job'].edge_index = torch.randint(0, 100, (2, 300))
    
    print("\nGraph structure:")
    print(graph)
    
    # Test link prediction model
    print("\n" + "="*60)
    print("Testing HGT Link Prediction Model")
    print("="*60)
    
    model = create_hgt_model(graph, task='link_prediction', hidden_channels=64, num_layers=2)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Forward pass
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
    
    # Test encoding
    node_emb_dict = model.encode(x_dict, edge_index_dict)
    print("\nNode embeddings:")
    for node_type, emb in node_emb_dict.items():
        print(f"  {node_type}: {emb.shape}")
    
    # Test link prediction
    edge_label_index = torch.randint(0, 100, (2, 50))
    pred = model(x_dict, edge_index_dict, edge_label_index, ('job', 'similar_to', 'job'))
    print(f"\nLink prediction output shape: {pred.shape}")
    
    print("\n✅ Model test successful!")
