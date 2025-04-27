import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import SAMPLE_IMAGE_DIR, IMG_SIZE

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transform
default_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class PlantDiseaseDataset(Dataset):
    """
    Custom dataset for plant disease classification.
    """

    def __init__(self, image_folder=SAMPLE_IMAGE_DIR, transform=default_transform):
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        self.transform = transform

        classes = sorted(os.listdir(image_folder))
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(image_folder, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                images = glob.glob(os.path.join(class_path, "*.*"))
                images = [img for img in images if img.lower().endswith((".jpg", ".jpeg", ".png"))]
                for img_path in images:
                    self.samples.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loader(image_folder=SAMPLE_IMAGE_DIR, batch_size=32, shuffle=False, transform=default_transform):
    """
    Returns a DataLoader for PlantDiseaseDataset.

    Args:
        image_folder: Root folder where labeled subfolders exist
        batch_size: Batch size for DataLoader
        shuffle: Shuffle the data
        transform: Transformations to apply

    Returns:
        DataLoader
    """
    dataset = PlantDiseaseDataset(image_folder=image_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Found {len(dataset)} images across {len(dataset.class_to_idx)} classes.")
    return loader, dataset.class_to_idx
