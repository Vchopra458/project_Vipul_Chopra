import torch
import os
from torch.utils.data import random_split
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_WEIGHTS_PATH
from dataset import get_data_loader, PlantDiseaseDataset
from model import CNNModel
from train import train_model, evaluate_model
from torch import optim
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Initialize dataset
    dataset = PlantDiseaseDataset()

    # Splitting dataset into train and validation sets 
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader for train and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # dataset split information
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Initialize model
    model = CNNModel(num_classes=38).to(device)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    print("Starting training\n")
    train_model(model, EPOCHS, train_loader, loss_fn, optimizer)

    # Evaluating the model on the validation set
    print("Evaluating model on validation set\n")
    evaluate_model(model, val_loader)

if __name__ == "__main__":
    main()
