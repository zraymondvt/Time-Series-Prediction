import torch
from tqdm import tqdm

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, device, n_epoch, patience):
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    early_stop_counter = 0

    print("Training the model...")
    for epoch in tqdm(range(n_epoch)):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).unsqueeze(-1), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device).unsqueeze(-1), y_batch.to(device)
                outputs = model(X_batch)
                test_loss += criterion(outputs.squeeze(), y_batch).item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        scheduler.step(test_loss)

        print(f"Epoch {epoch + 1}/{n_epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

    return train_losses, test_losses
