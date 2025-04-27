import torch
from tqdm import tqdm
from config import MODEL_WEIGHTS_PATH,TRAIN_WEIGHTS_PATH  # Import the save path from config

# Automatically select device inside train.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    """
    Trains the model over the training data and saves the final weights.

    Args:
        model: PyTorch model to be trained
        num_epochs: Number of epochs to train
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch, labels in progress_bar:
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=total_loss / (total / batch.size(0)), accuracy=100 * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%\n")

    # Save final model weights
    torch.save(model.state_dict(),TRAIN_WEIGHTS_PATH)
    print(f"Model weights saved at {TRAIN_WEIGHTS_PATH}")


def evaluate_model(model, data_loader):
    """
    Evaluates the model over a dataset (validation or test).

    Args:
        model: Trained model
        data_loader: DataLoader for validation or test data

    Returns:
        Accuracy in percentage
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch, labels in tqdm(data_loader, desc="Evaluating"):
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy
