"""Training script for HGT model on job recommendation task"""

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
    """Trainer for HGT model on link prediction task"""

    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.001,
        weight_decay=1e-5,
    ):
        """
        Args:
            model: HGT model
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        print(f"\n{'='*60}")
        print(f"HGT Trainer Initialized")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {weight_decay}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, data, edge_type):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()

        data = data.to(self.device)

        # Get positive and negative edges
        edge_label_index = data[edge_type].edge_label_index
        edge_label = data[edge_type].edge_label

        # Forward pass
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

        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(pred, edge_label.float())

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, edge_type):
        """Evaluate model"""
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

        # Compute metrics
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
        Full training loop

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            edge_type: Edge type to predict
            epochs: Number of epochs
            eval_every: Evaluate every N epochs
        """
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Edge type: {edge_type}")

        best_val_auc = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            # Train
            loss = self.train_epoch(train_data, edge_type)

            # Evaluate
            if epoch % eval_every == 0:
                train_auc, train_ap = self.evaluate(train_data, edge_type)
                val_auc, val_ap = self.evaluate(val_data, edge_type)

                print(f"\nEpoch {epoch:03d}:")
                print(f"  Loss: {loss:.4f}")
                print(f"  Train - AUC: {train_auc:.4f}, AP: {train_ap:.4f}")
                print(f"  Val   - AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

                # Save best model
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    self.save_model('best_model.pt')
                    print(f"  ✅ New best model saved!")
            else:
                print(f"Epoch {epoch:03d}: Loss = {loss:.4f}")

        # Load best model and evaluate on test set
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best validation AUC: {best_val_auc:.4f} (Epoch {best_epoch})")

        self.load_model('best_model.pt')
        test_auc, test_ap = self.evaluate(test_data, edge_type)
        print(f"\nFinal Test Results:")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  AP:  {test_ap:.4f}")

        return {
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'test_auc': test_auc,
            'test_ap': test_ap,
        }

    def save_model(self, filename):
        """Save model checkpoint"""
        path = os.path.join(config.GRAPH_DATA_PATH, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, filename):
        """Load model checkpoint"""
        path = os.path.join(config.GRAPH_DATA_PATH, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def prepare_data(graph, edge_type=('job', 'similar_to', 'job'), split_ratio=[0.8, 0.1, 0.1]):
    """
    Prepare train/val/test splits

    Args:
        graph: PyG HeteroData object
        edge_type: Edge type to use for link prediction
        split_ratio: [train, val, test] ratio

    Returns:
        train_data, val_data, test_data
    """
    print(f"\n{'='*60}")
    print("Preparing Data")
    print(f"{'='*60}")
    print(f"Edge type for prediction: {edge_type}")
    print(f"Split ratio: Train={split_ratio[0]}, Val={split_ratio[1]}, Test={split_ratio[2]}")

    # Split edges into train/val/test
    transform = RandomLinkSplit(
        num_val=split_ratio[1],
        num_test=split_ratio[2],
        edge_types=[edge_type],
        rev_edge_types=[edge_type],  # Since we have bidirectional edges
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,  # 1:1 ratio of positive to negative samples
    )

    train_data, val_data, test_data = transform(graph)

    print(f"\nData split complete:")
    print(f"  Train edges: {train_data[edge_type].edge_label_index.size(1)}")
    print(f"  Val edges:   {val_data[edge_type].edge_label_index.size(1)}")
    print(f"  Test edges:  {test_data[edge_type].edge_label_index.size(1)}")

    return train_data, val_data, test_data


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print(" "*15 + "HGT TRAINING PIPELINE")
    print("="*70)

    # Load graph
    print("\n[STEP 1/4] Loading Graph...")
    print("-"*70)
    graph_path = os.path.join(config.GRAPH_DATA_PATH, 'hetero_graph.pt')
    
    if not os.path.exists(graph_path):
        print(f"❌ Graph file not found: {graph_path}")
        print("Please run main.py first to build the graph!")
        return
    
    graph = torch.load(graph_path)
    print(f"✅ Graph loaded from {graph_path}")
    print(f"\nGraph structure:")
    print(graph)

    # Prepare data
    print("\n[STEP 2/4] Preparing Data...")
    print("-"*70)
    edge_type = ('job', 'similar_to', 'job')
    train_data, val_data, test_data = prepare_data(graph, edge_type)

    # Create model
    print("\n[STEP 3/4] Creating Model...")
    print("-"*70)
    model = create_hgt_model(
        graph,
        task='link_prediction',
        hidden_channels=128,
        out_channels=64,
        num_heads=8,
        num_layers=2,
    )
    print(f"✅ HGT model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n[STEP 4/4] Training Model...")
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

    # Summary
    print("\n" + "="*70)
    print(" "*25 + "🎉 TRAINING COMPLETE! 🎉")
    print("="*70)
    print("\nFinal Results:")
    print(f"  Best Validation AUC: {results['best_val_auc']:.4f} (Epoch {results['best_epoch']})")
    print(f"  Test AUC:            {results['test_auc']:.4f}")
    print(f"  Test AP:             {results['test_ap']:.4f}")
    print(f"\nModel saved to: {config.GRAPH_DATA_PATH}best_model.pt")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
