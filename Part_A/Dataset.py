import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def load_data(data_dir, val_fraction=0.2, image_size=(224, 224), batch_size=32):
    """
    Loads train, validation, and test datasets from the iNaturalist_12K dataset structure.

    Args:
        data_dir (str): Path to the dataset folder (e.g., '../../inaturalist_12K')
        val_fraction (float): Fraction of training data to use as validation
        image_size (tuple): Target image size for resizing
        batch_size (int): Batch size for DataLoader

    Returns:
        train_set, val_set, test_set: Split datasets
        class_names: List of class names
    """

    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    full_train_set = datasets.ImageFolder(train_path, transform=transform)
    test_set = datasets.ImageFolder(test_path, transform=transform)

    # Split full_train_set into train and val
    total_size = len(full_train_set)
    val_size = int(total_size * val_fraction)
    train_size = total_size - val_size

    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    class_names = full_train_set.classes

    return train_set, val_set, test_set, class_names
