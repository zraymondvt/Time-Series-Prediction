import matplotlib.pyplot as plt

# Create a single figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(35, 18))

# First subplot: Zoom on residuals (first 2500 samples)
axs[0].plot(residual_actuals[:2500], label='Actual', alpha=0.7)
axs[0].plot(residual_predictions[:2500], label='Predicted', alpha=0.7)
axs[0].set_title(f"Zoom on Hybrid Transformer-ESN ({mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f})")
axs[0].legend()

# Second subplot: Zoom in residuals (first 200 samples)
axs[1].plot(residual_actuals[:200], label='Actual', alpha=0.7)
axs[1].plot(residual_predictions[:200], label='Predicted', alpha=0.7)
axs[1].set_title(f"Zoom in Hybrid Transformer-ESN ({mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f})")
axs[1].legend()

# Third subplot: Loss During Training
axs[2].plot(train_losses, label='Train Loss')
axs[2].plot(test_losses, label='Test Loss')
axs[2].set_title('Train Loss vs. Test Loss')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Loss')
axs[2].legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure as PNG with 300 dpi
plt.savefig('hybrid_transformer_plots.png', dpi=300)

# Show the combined plot
plt.show()
