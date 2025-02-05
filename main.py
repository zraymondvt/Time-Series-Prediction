import os
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models import HybridTransformerESN
from dataset import TimeSeriesDataset
from utils import log_cosh_loss
from train import train_and_evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

def main(args):
    # Load data
    X = np.load(os.path.join(args.input_folder, 'X.npy'))
    y = np.load(os.path.join(args.input_folder, 'y.npy'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridTransformerESN(
        input_dim=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        ff_dim=args.ff_dim,
        reservoir_size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        sparsity=args.sparsity
    ).to(device)

    criterion = log_cosh_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Train and evaluate
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, device, args.epochs, args.patience)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hybrid Transformer-ESN Model on 5G Traffic Data")
    parser.add_argument('--input_folder', type=str, default='ds/npy-data', help='input directory')
    # parser.add_argument('--dataset_name', type=str, default="kimdaegyeom/5g-traffic-datasets",
    #                     help="Kaggle dataset identifier (default: 'kimdaegyeom/5g-traffic-datasets')")
    # parser.add_argument('--file_relative_path', type=str, default="5G_Traffic_Datasets/Video_Conferencing/Zoom/Zoom_1.csv",
    #                     help="Relative file path inside the downloaded dataset (default: '5G_Traffic_Datasets/Video_Conferencing/Zoom/Zoom_1.csv')")
    parser.add_argument('--sequence_length', type=int, default=65, help="Sequence length for time series data")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--d_model', type=int, default=64, help="Transformer model dimension")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--num_encoder_layers', type=int, default=3, help="Number of encoder layers")
    parser.add_argument('--ff_dim', type=int, default=128, help="Feedforward network dimension")
    parser.add_argument('--reservoir_size', type=int, default=100, help="ESN reservoir size")
    parser.add_argument('--spectral_radius', type=float, default=0.85, help="ESN spectral radius")
    parser.add_argument('--sparsity', type=float, default=0.1, help="Reservoir sparsity")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--max_data', type=int, default=150000, help="Maximum number of data points to use")
    
    args = parser.parse_args()
    main(args)