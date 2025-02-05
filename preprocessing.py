import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
from dataset import create_sequences

def proc(args):
    file_path = os.path.join(args.input_folder, args.dataset_name, args.file_relative_path)
    print("Loading and preprocessing data...")
    data = pd.read_csv(file_path)['Length'].values[:args.max_data]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    X, y = create_sequences(data_scaled, args.sequence_length) # applying windowing to the data
    print(f'Data Shape: X:{X.shape}, y:{y.shape}')

    npy_path = os.path.join(args.input_folder, 'npy-data')
    os.makedirs(npy_path, exist_ok=True)
    print(f'Saving .npy dataset to {npy_path}')
    np.save(os.path.join(npy_path, 'X.npy'), X)
    np.save(os.path.join(npy_path, 'y.npy'), y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hybrid Transformer-ESN Model on 5G Traffic Data")
    parser.add_argument('--input_folder', type=str, default='ds', help='input directory')
    parser.add_argument('--dataset_name', type=str, default="kimdaegyeom/5g-traffic-datasets",
                        help="Kaggle dataset identifier (default: 'kimdaegyeom/5g-traffic-datasets')")
    parser.add_argument('--file_relative_path', type=str, default="5G_Traffic_Datasets/Video_Conferencing/Zoom/Zoom_1.csv",
                        help="Relative file path inside the downloaded dataset (default: '5G_Traffic_Datasets/Video_Conferencing/Zoom/Zoom_1.csv')")
    parser.add_argument('--sequence_length', type=int, default=65, help="Sequence length for time series data") # think of it as the window size i guess? 
    parser.add_argument('--max_data', type=int, default=150000, help="Maximum number of data points to use")
    
    args = parser.parse_args()
    proc(args)