# Time Series Prediction for Large Scale Data

## Overview
This project is a Python-based framework designed for efficient and flexible time series forecasting. It employs a hybrid architecture combining **Transformers** and **Echo State Networks (ESN)** to process large-scale datasets effectively. The project is structured for easy customization and experimentation with various hyperparameters.

The work is contribution on below PoC and Demo: 

1) F. Rezazadeh, S. Barrachina-Muñoz, H. Chergui, J. Mangues, M. Bennis, D. Niyato, H. Song, and L. Liu, “Toward Explainable Reasoning in 6G: A Proof of Concept Study on Radio Resource Allocation”, IEEE Open Journal of the Communications Society, 2024. [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10689363) [[arxiv]](https://arxiv.org/abs/2407.10186)
   
3) F. Rezazadeh, S. Barrachina-Muñoz, E. Zeydan, H. Song, K.P. Subbalakshmi, and J. Mangues-Bafalluy, “X-GRL: An Empirical Assessment of Explainable GNN-DRL in B5G/6G Networks”, IEEE NFV-SDN, 2023. [[IEEE Xplore]](https://ieeexplore.ieee.org/abstract/document/10329778/authors#authors) [[arxiv]](https://arxiv.org/abs/2311.08798)
---

## Requirements
- **Python Version**: 3.11.11  

Ensure you are using Python 3.11 for compatibility. Install the necessary dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Main Script
The project is run using the `main.py` script, which is fully configurable through command-line arguments.

### Example Usage
```bash
python main.py \
    --dataset_name "kimdaegyeom/5g-traffic-datasets" \
    --file_relative_path "5G_Traffic_Datasets/Video_Conferencing/Zoom/Zoom_1.csv" \
    --sequence_length 65 \
    --batch_size 128 \
    --d_model 64 \
    --n_heads 8 \
    --num_encoder_layers 3 \
    --ff_dim 128 \
    --reservoir_size 100 \
    --spectral_radius 0.85 \
    --sparsity 0.1 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --epochs 50 \
    --patience 10 \
    --max_data 150000
```

### Argument Details

- `--dataset_name`: **(str)** Kaggle dataset identifier (default: `"kimdaegyeom/5g-traffic-datasets"`).
- `--file_relative_path`: **(str)** Relative file path inside the dataset (default: `"5G_Traffic_Datasets/Video_Conferencing/Zoom/Zoom_1.csv"`).
- `--sequence_length`: **(int)** Sequence length for time series data (default: `65`).
- `--batch_size`: **(int)** Batch size for training (default: `128`).
- `--d_model`: **(int)** Transformer model dimension (default: `64`).
- `--n_heads`: **(int)** Number of attention heads (default: `8`).
- `--num_encoder_layers`: **(int)** Number of Transformer encoder layers (default: `3`).
- `--ff_dim`: **(int)** Feedforward network dimension (default: `128`).
- `--reservoir_size`: **(int)** ESN reservoir size (default: `100`).
- `--spectral_radius`: **(float)** ESN spectral radius (default: `0.85`).
- `--sparsity`: **(float)** Reservoir sparsity (default: `0.1`).
- `--lr`: **(float)** Learning rate (default: `0.001`).
- `--weight_decay`: **(float)** Weight decay for optimizer (default: `1e-5`).
- `--epochs`: **(int)** Number of training epochs (default: `50`).
- `--patience`: **(int)** Early stopping patience (default: `10`).
- `--max_data`: **(int)** Maximum number of data points to use (default: `150000`).

---

## Example Workflow

1. **Download Dataset**:
   The dataset is downloaded using the `kagglehub` library. Specify the `--dataset_name` and `--file_relative_path` to locate the desired dataset file.

2. **Preprocessing**:
   The script automatically scales the dataset and creates sequences for training.

3. **Model Training**:
   The hybrid Transformer-ESN model is trained on the processed dataset with specified hyperparameters.
---

## Results
Below is a visualization of the model's performance:

<img src="hybrid_transformer_plots.png"/>

---

## Contributions
Feel free to fork this repository, report issues, or submit pull requests to improve the framework.

---

## License
This project is licensed under the MIT License.

