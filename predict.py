import os
import torch
from PIL import Image
from torchvision import transforms
from model import CNNModel
from config import IMG_SIZE, NUM_CLASSES, MODEL_WEIGHTS_PATH
from dataset import get_data_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def inferloader(list_of_img_paths):
    """
    Prepares a batch tensor from a list of image paths.

    Args:
        list_of_img_paths: List of paths to images.

    Returns:
        Torch tensor of shape [batch_size, 3, IMG_SIZE, IMG_SIZE]
    """
    batch = []
    for img_path in list_of_img_paths:
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        batch.append(image)
    batch = torch.stack(batch)  
    return batch

def classify_plants(list_of_img_paths):
    """
    Predicts the labels for a batch of image paths.

    Args:
        list_of_img_paths: List of image file paths.

    Returns:
        List of predicted class names (labels).
    """
    # Initialize model
    model = CNNModel(num_classes=NUM_CLASSES).to(device)

    # Load weights 
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print(f"✅ Loaded model weights from {MODEL_WEIGHTS_PATH}")
    else:
        print("⚠️ No trained weights found, using untrained (raw) model!")

    model.eval()

    # Prepare batch using inferloader
    plant_batch = inferloader(list_of_img_paths).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(plant_batch)
        _, preds = torch.max(outputs, 1)

    # Mapping index to class names
    loader, class_to_idx = get_data_loader(batch_size=1, shuffle=False)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    predicted_labels = [idx_to_class[pred.item()] for pred in preds]

    return predicted_labels
