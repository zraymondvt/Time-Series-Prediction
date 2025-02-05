import os
import argparse
import kagglehub
import shutil

def get_dataset(args):
    os.makedirs(args.input_folder, exist_ok=True)

    # Download dataset
    print("Downloading dataset...")
    default_path = kagglehub.dataset_download(args.dataset_name)

    # Move the dataset to the desired folder
    dataset_target_path = os.path.join(args.input_folder, args.dataset_name)

    if not os.path.exists(dataset_target_path):
        shutil.move(default_path, dataset_target_path)  # Move dataset to input_folder
        print(f"Moved dataset to: {dataset_target_path}")
    else:
        print(f"Dataset already exists at: {dataset_target_path}")

    # Ensure correct file path
    file_path = os.path.join(dataset_target_path, args.file_relative_path)
    print(f"Dataset ready at: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hybrid Transformer-ESN Model on 5G Traffic Data")
    parser.add_argument('--input_folder', type=str, default='ds', help='input directory')
    parser.add_argument('--dataset_name', type=str, default="kimdaegyeom/5g-traffic-datasets",
                        help="Kaggle dataset identifier (default: 'kimdaegyeom/5g-traffic-datasets')")
    parser.add_argument('--file_relative_path', type=str, default="5G_Traffic_Datasets/Video_Conferencing/Zoom/Zoom_1.csv",
                        help="Relative file path inside the downloaded dataset (default: '5G_Traffic_Datasets/Video_Conferencing/Zoom/Zoom_1.csv')")
    
    args = parser.parse_args()
    get_dataset(args)