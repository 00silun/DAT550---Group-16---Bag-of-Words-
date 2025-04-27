import torch
import csv
import os
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, name="Model", model_type="FFNN"):
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []

    # CSV setup
    csv_filename = f"{name.lower()}_training_log.csv"
    write_header = not os.path.exists(csv_filename)

    with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['model_type', 'activation', 'epoch', 'train_loss', 'val_loss', 'epoch_time'])

        activation = getattr(model, "activation_name", "unknown")

        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            total_train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            epoch_time = time.time() - start_time

            print(f"[{name}] Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")

            # Write to CSV
            writer.writerow([model_type, activation, epoch + 1, avg_train_loss, avg_val_loss, round(epoch_time, 2)])

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

    torch.save(best_model_state, f"best_{name.lower()}_model.pt")
    return train_losses, val_losses, best_model_state
